//! Compiles the vendored ExecuTorch runtime and the shim into one static
//! archive with the `cc` crate (issue #315, step R1). No CMake, no flatc, no
//! Python and no network at build time: every generated file is already in
//! `torchnative/rust/vendor/executorch/generated/` (issue #312, step R0).
//!
//! # What is compiled
//!
//! The file list is read from the vendored tree's `MANIFEST.tsv`, by group,
//! not typed here, so a re-vendoring that adds or removes a source changes
//! the build without an edit to this file:
//!
//! | MANIFEST group | taken | what |
//! |---|---|---|
//! | `core` | every `.cpp` except the PALs not chosen | program loader, method executor, operator registry, platform layer |
//! | `prim_ops` | every `.cpp` | `executorch_prim::*` and `aten::sym_size` (symbolic sizes) |
//! | `portable` | every `.cpp` | the whole portable kernel library (leader decision 2026-11-14: no selective build, the program is unknown when the wheel is built) |
//! | `generated` | every `.cpp` | the codegen'd kernel registration |
//!
//! plus `shim/tn_et.cpp`. The PAL is `runtime/platform/default/posix.cpp` on
//! macOS (`android.cpp` needs `<android/log.h>`; `minimal.cpp` has no clock).
//! The shim then overrides the log and abort hooks through `register_pal()`
//! at run time, not through weak symbols.
//!
//! # Registration must survive the archive
//!
//! The portable kernels and the prim ops register themselves from static
//! initializers (`static auto success_with_kernel_reg = register_kernels(..)`
//! at the end of `RegisterCodegenUnboxedKernelsEverything.cpp`), and nothing
//! references those objects by symbol. A linker pulls an archive member only
//! to resolve an undefined symbol, so without help every kernel object would
//! be dropped and the registry would be empty at run time, silently. The
//! archive is therefore linked `+whole-archive` (rustc passes `-force_load`
//! to ld64), and `-bundle` so that the modifier is applied at the final
//! `cdylib` link of `torch_c` rather than inside this rlib. The proof that it
//! worked is a test, not this comment: `tests/export/test_et_runtime.py`
//! asserts that every kernel the generated registration file names is in
//! `torch._C._et_registered_kernels()`.
//!
//! # Missing inputs are named
//!
//! The generated schema headers include `<flatbuffers/flatbuffers.h>`
//! (flatbuffers v24.3.25), vendored by step R0 (issue #312). It is read from one
//! include root, `torchnative/rust/vendor/flatbuffers/include`; when that
//! directory is missing the build stops here with what to do, instead of 8
//! compiler errors deep inside `program.cpp`.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

/// The only target R1 builds for. `torch_c/Cargo.toml` pulls this crate on
/// exactly this target; the check here is for a direct build of the crate.
const SUPPORTED_TARGET: &str = "aarch64-apple-darwin";

/// Archive name. `links` in Cargo.toml carries the same name so two copies of
/// the runtime can never end up in one artefact.
const ARCHIVE: &str = "tn_executorch";

fn main() {
    let manifest_dir = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let vendor = manifest_dir.join("../vendor");
    let et = vendor.join("executorch");
    let flatbuffers = vendor.join("flatbuffers/include");
    let target = env("TARGET");

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=shim");
    println!("cargo::rerun-if-changed={}", et.join("MANIFEST.tsv").display());
    println!("cargo::rerun-if-changed={}", flatbuffers.display());

    if target != SUPPORTED_TARGET {
        panic!(
            "executorch_rt: target {target} is not supported in step R1 (issue #315); \
             only {SUPPORTED_TARGET} is. torch_c pulls this crate only on that target, \
             so reaching this message means it was built directly or the cfg table in \
             torchnative/rust/torch_c/Cargo.toml was widened without porting this build."
        );
    }

    let manifest_path = et.join("MANIFEST.tsv");
    let manifest = fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "executorch_rt: cannot read {}: {e}. The vendored ExecuTorch tree is \
             produced by scripts/vendor/vendor_executorch.py (issue #312).",
            manifest_path.display()
        )
    });
    let (version, commit) = upstream_pin(&manifest);
    println!("cargo::rustc-env=TN_ET_UPSTREAM_VERSION={version}");
    println!("cargo::rustc-env=TN_ET_UPSTREAM_COMMIT={commit}");

    if !flatbuffers.join("flatbuffers/flatbuffers.h").is_file() {
        panic!(
            "executorch_rt: flatbuffers C++ headers not found at {fb}/flatbuffers/flatbuffers.h.\n\
             The generated ExecuTorch schema headers \
             (vendor/executorch/generated/include/executorch/schema/program_generated.h) \
             include <flatbuffers/flatbuffers.h> from flatbuffers v24.3.25 (commit 595bf000), \
             and that tree is not vendored yet (issue #315, decision pending).\n\
             Either vendor flatbuffers' include/ directory at that tag to {fb}, or build \
             torch_c without the `executorch` feature: TORCHNATIVE_SHIM_FEATURES=none for \
             scripts/vendor/install_shim.sh and tests/run.sh, or plain `cargo build --release` \
             in torchnative/rust/torch_c.",
            fb = flatbuffers.display()
        );
    }

    let groups = manifest_groups(&manifest);
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for group in ["core", "prim_ops", "portable", "generated"] {
        let paths = groups.get(group).unwrap_or_else(|| {
            panic!(
                "executorch_rt: MANIFEST.tsv has no `{group}` group; the vendored tree and \
                 this build script disagree about its layout"
            )
        });
        for rel in paths.iter().filter(|p| p.ends_with(".cpp")) {
            if group == "core" && is_unchosen_pal(rel) {
                continue;
            }
            let path = et.join(rel);
            if !path.is_file() {
                panic!(
                    "executorch_rt: MANIFEST.tsv lists {rel} but {} is missing; run \
                     `python scripts/vendor/vendor_executorch.py --check`",
                    path.display()
                );
            }
            *counts.entry(group).or_default() += 1;
            sources.push(path);
        }
    }
    // A manifest that lost a group's sources would still "build"; with an
    // empty kernel library every program then fails with OperatorMissing.
    for (group, at_least) in [("core", 20), ("prim_ops", 3), ("portable", 150), ("generated", 1)] {
        let got = counts.get(group).copied().unwrap_or(0);
        if got < at_least {
            panic!(
                "executorch_rt: MANIFEST group `{group}` gave {got} .cpp files, expected at \
                 least {at_least}; refusing to build a partial runtime"
            );
        }
    }
    let pal = et.join("runtime/platform/default/posix.cpp");
    if !sources.contains(&pal) {
        panic!("executorch_rt: the posix PAL ({}) is not in MANIFEST.tsv's core group", pal.display());
    }
    sources.push(manifest_dir.join("shim/tn_et.cpp"));

    let release = std::env::var("PROFILE").map(|p| p == "release").unwrap_or(false);

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        // Upstream's release flags (CMakeLists.txt:300-308). The unwind-table
        // flags upstream also passes are left out: they only shrink the
        // archive and on Apple targets they remove the compact unwind info
        // that crash reports and the Rust panic machinery read.
        .flag("-fno-exceptions")
        .flag("-fno-rtti")
        .flag("-Wno-deprecated-declarations")
        .warnings(false)
        .include(&vendor)
        .include(et.join("runtime/core/portable_type/c10"))
        .include(et.join("generated/include"))
        .include(&flatbuffers)
        .include(manifest_dir.join("shim"))
        // Upstream's definitions (CMakeLists.txt:212-233, :573,
        // schema/CMakeLists.txt, tools/cmake/preset/default.cmake).
        .define("C10_USING_CUSTOM_GENERATED_MACROS", None)
        .define("FLATBUFFERS_MAX_ALIGNMENT", "1024")
        .define("ET_ENABLE_PROGRAM_VERIFICATION", "1")
        .define("ET_ENABLE_DEPRECATED_CONSTANT_BUFFER", "0")
        .define("ET_LOG_ENABLED", "1")
        .define("ET_MIN_LOG_LEVEL", "Info")
        // The counting tracer (shim) needs the runtime's tracer hooks; with
        // this off, operator_calls would stay empty and the suite says so.
        .define("ET_EVENT_TRACER_ENABLED", None)
        // We emit every link line ourselves (below).
        .cargo_metadata(false);
    if release {
        // ET_DCHECK and friends compile out, as in upstream's Release build.
        build.define("NDEBUG", None);
    }
    build.files(&sources);
    build.compile(ARCHIVE);

    let out_dir = env("OUT_DIR");
    println!("cargo::rustc-link-search=native={out_dir}");
    println!("cargo::rustc-link-lib=static:+whole-archive,-bundle={ARCHIVE}");
    // Apple's C++ runtime is a system dylib; nothing is shipped for it.
    println!("cargo::rustc-link-lib=dylib=c++");
    println!(
        "cargo::warning=executorch_rt: compiled {} translation units ({} core, {} prim_ops, {} portable, {} generated, 1 shim) for ExecuTorch {version}",
        sources.len(),
        counts["core"],
        counts["prim_ops"],
        counts["portable"],
        counts["generated"]
    );
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("executorch_rt: cargo did not set {name}"))
}

/// `runtime/platform/default/{android,minimal}.cpp`: the PALs not used on
/// this target. Exactly one PAL may be linked (each defines every et_pal_*).
fn is_unchosen_pal(rel: &str) -> bool {
    rel.starts_with("runtime/platform/default/") && !rel.ends_with("/posix.cpp")
}

/// `group -> [path]` from the tab-separated manifest
/// (`group  sha256  bytes  path  upstream`); `#` lines are comments.
fn manifest_groups(manifest: &str) -> BTreeMap<String, Vec<String>> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for line in manifest.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 4 {
            panic!("executorch_rt: malformed MANIFEST.tsv line: {line:?}");
        }
        groups
            .entry(fields[0].to_string())
            .or_default()
            .push(fields[3].to_string());
    }
    groups
}

/// The pin from the manifest's first line:
/// `# ExecuTorch v1.4.1 (e4d02f41...) subset for torchnative, issue #312.`
fn upstream_pin(manifest: &str) -> (String, String) {
    let first = manifest.lines().next().unwrap_or_default();
    let version = first
        .split_whitespace()
        .find(|w| w.starts_with('v') && w[1..].starts_with(|c: char| c.is_ascii_digit()))
        .map(|w| w.trim_start_matches('v').to_string());
    let commit = first
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(sha, _)| sha.to_string());
    match (version, commit) {
        (Some(v), Some(c)) if c.len() == 40 => (v, c),
        _ => panic!(
            "executorch_rt: cannot read the upstream pin from MANIFEST.tsv's first line {first:?}"
        ),
    }
}

