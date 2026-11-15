//! Compiles the vendored ExecuTorch runtime, its XNNPACK delegate and the shim
//! into two static archives with the `cc` crate (issue #315, step R1; issue
//! #337, step R2b). No CMake, no flatc, no Python and no network at build
//! time: every generated file is already in the vendored trees (issues #312
//! and #319).
//!
//! # What is compiled
//!
//! The runtime's file list is read from the ExecuTorch tree's `MANIFEST.tsv`,
//! by group, not typed here, so a re-vendoring that adds or removes a source
//! changes the build without an edit to this file:
//!
//! | MANIFEST group | taken | what |
//! |---|---|---|
//! | `core` | every `.cpp` except the PALs not chosen | program loader, method executor, operator registry, platform layer |
//! | `prim_ops` | every `.cpp` | `executorch_prim::*` and `aten::sym_size` (symbolic sizes) |
//! | `portable` | every `.cpp` | the whole portable kernel library (leader decision 2026-11-14: no selective build, the program is unknown when the wheel is built) |
//! | `generated` | every `.cpp` | the codegen'd kernel registration |
//!
//! plus `shim/tn_et.cpp`. The PAL is `runtime/platform/default/posix.cpp` on
//! macOS (`android.cpp` needs `<android/log.h>`; `minimal.cpp` has no clock)
//! and `runtime/platform/default/android.cpp` on Android (step R3, below).
//! The shim then overrides the log and abort hooks through `register_pal()`
//! at run time, not through weak symbols.
//!
//! The delegate (R2b) is read from each vendored tree's `BUILD_GROUPS.json`
//! (`torchnative-build-groups/1`, written by `scripts/vendor/vendor_executorch.py`):
//! per group its files, flags, definitions, include directories and language
//! standard for this target. `.c` goes to the C compiler, `.cc`/`.cpp` to the
//! C++ compiler, `.S` to the C compiler driver as assembler-with-cpp, each
//! with the group's recorded flags.
//!
//! | tree | groups | what |
//! |---|---|---|
//! | `executorch` | `xnnpack_backend`, `extension_threadpool` | the backend's runtime and the threadpool it runs on |
//! | `xnnpack` | 40 | operators, subgraph, configs, production microkernels (C and aarch64 assembly) |
//! | `cpuinfo` | the common group and this target's | CPU detection for XNNPACK and the threadpool size |
//! | `pthreadpool` | 1 | the pool itself |
//! | `fxdiv` | 0 | headers only (pthreadpool and XNNPACK include it) |
//! | `fp16` | 0 | headers only, and no vendored file includes it: read, checked to have no group, on no include path |
//!
//! The JSON is parsed by the small parser at the end of this file, not by a
//! crate: this crate's lock holds `cc` and nothing that reads JSON, and a new
//! build dependency is a decision this step does not need.
//!
//! # Anything not understood is refused by name
//!
//! Every key of a `BUILD_GROUPS.json` is either acted on or listed in
//! `INFORMATIONAL_KEYS` with the reason it is not. An unknown key, a flag
//! outside `check_flag`'s vocabulary, a source extension other than
//! `.c`/`.cc`/`.cpp`/`.S`, a link library without a rule for this target or a
//! tree that does not list this target stops the build with its name. Nothing
//! is skipped silently.
//!
//! # Judgement calls of step R2b
//!
//! * **Optimisation levels.** `cc` puts its own `-O<n>` (from cargo's profile)
//!   first on the command line and the flags given here after it, and clang
//!   applies the last `-O`. For XNNPACK, cpuinfo and pthreadpool the recorded
//!   flags are appended in upstream's order (`build_type.flags` `-O3`, then
//!   `common_flags`, then the group's `-Os`/`-O2`), so each group gets exactly
//!   upstream's level in every cargo profile: a microkernel library at `-O0`
//!   is not a useful debug build, and `NDEBUG` and no `-g` come with the same
//!   recorded Release build type. The ExecuTorch tree's two groups instead
//!   follow the runtime core they share inline code with: cargo's profile
//!   level, `NDEBUG` only in release, as step R1 chose.
//! * **One configuration for all ExecuTorch code.** The recorded definitions
//!   of `xnnpack_backend` and `extension_threadpool` say `ET_LOG_ENABLED=0`
//!   and `ET_ENABLE_PROGRAM_VERIFICATION=0` (upstream's Release default). The
//!   core is built with both on (R1: the shim's messages are the runtime's log
//!   lines, and verification is what turns a garbage buffer into
//!   `InvalidProgram`). Those macros change inline code in shared headers, so
//!   the two groups are compiled with the core's values. Only a definition the
//!   JSON itself lists under `configurable` may be replaced; any other
//!   disagreement with the core's definitions stops the build by name. They
//!   also get the core's `-fno-rtti`: the backend class derives from
//!   `BackendInterface`, and with RTTI on its type info would reference a base
//!   type info the core never emitted. They do not get `-fno-exceptions`:
//!   `XNNWeightsCache.cpp` has a `try` block around an allocation
//!   (`cc -fsyntax-only` refuses it with exceptions off), and upstream
//!   compiles this target with exceptions on. The exception it catches is
//!   thrown and caught inside that one function, so none crosses into the
//!   core's exception-free frames from there.
//! * **`ET_EVENT_TRACER_ENABLED` for the backend.** Not in the recorded
//!   definitions (upstream turns it on with devtools). With it the backend
//!   creates XNNPACK runtimes with `XNN_FLAG_BASIC_PROFILING` and reports
//!   XNNPACK's own per-operator profile to the tracer after every execute
//!   (`XNNCompiler.cpp`, `profiling/XNNProfiler.cpp`): the `delegate_ops`
//!   counter, the only evidence here that XNNPACK itself ran.
//! * **Weight cache off, shared workspace on, logging on.** The recorded
//!   definitions follow `tools/cmake/preset/default.cmake`
//!   (`ENABLE_XNNPACK_SHARED_WORKSPACE`, no `ENABLE_XNNPACK_WEIGHTS_CACHE`),
//!   not the `macos.cmake` preset, which turns the weight cache on. Kept: the
//!   weight cache is process-wide state keyed by named-data names that
//!   outlives a method, and it pays off for several methods sharing weights or
//!   a cache file, neither of which an in-memory, one-method program has. The
//!   choice is reported in `torch._C._et_runtime_info()` from what was
//!   compiled, not from this comment.
//! * **SME groups compiled.** KleidiAI and SME2 are off (leader decision
//!   2026-11-14). `microkernels-prod/neonsme` is still compiled, as recorded:
//!   every group carries `XNN_ENABLE_ARM_SME=1`, without KleidiAI those 16
//!   files are wrappers whose bodies are an `assert` (no SME instruction is
//!   emitted, checked with `cc -fsyntax-only` and by reading them), and
//!   leaving the group out while the definition stays 1 would rest on a
//!   reading of `src/configs/` that no link has confirmed. Cost: 16 small
//!   objects the linker can discard.
//! * **FP16 is not compiled and not on an include path.** It has no group and
//!   no vendored source includes it.
//! * **Registration: whole-archive, not a shim-owned call.** Upstream's
//!   `XnnpackBackend` class is defined inside `XNNPACKBackend.cpp` and
//!   registered there from a static initializer; no header declares it. A
//!   shim-owned `register_backend` call would need that class, so either an
//!   edit of the vendored file (forbidden: the tree is generated and checked
//!   by sha256) or a second copy of its 270 lines in the shim, which would
//!   drift and, with the vendored file still compiled, register twice. What
//!   is certain to survive archiving is what already does for the 237
//!   kernels: the object goes into the archive that is linked
//!   `+whole-archive` (observed in CI for step R1 with this same link line).
//!   The proof is a test: `_et_registered_backends()` must name the backend.
//! * **Hidden visibility.** XNNPACK, cpuinfo and pthreadpool are compiled with
//!   `-fvisibility=hidden` (upstream records it for pthreadpool on Apple
//!   only). Everything is linked into one image, where hidden symbols resolve
//!   normally, and none of `xnn_*`, `cpuinfo_*`, `pthreadpool_*` can then be
//!   exported from the final `cdylib` even on a toolchain whose link does not
//!   restrict exports. On macOS rustc's exported-symbols list already does;
//!   the test reads the export table with `nm`. The ExecuTorch objects keep
//!   R1's default visibility and rely on that list.
//! * **Link libraries.** `c++` (a system dylib on Apple) as in R1. XNNPACK's
//!   recorded `m` is part of libSystem on Apple targets, so nothing is
//!   emitted for it; a name without such a rule is refused.
//!
//! # Judgement calls of step R3 (aarch64-linux-android, issue #350)
//!
//! Every target this crate builds is in `SUPPORTED_TARGETS`, each with its
//! `TargetRules`; any other target is refused by name. What differs on
//! Android, and why (agent decisions in the R3 round, not the owner's):
//!
//! * **PAL: `android.cpp`.** `posix.cpp` would print to stderr, which an
//!   Android process does not show; `android.cpp` writes to logcat through
//!   `__android_log_print` (hence `liblog`) and keeps the same steady clock.
//!   The shim replaces the log and abort hooks at run time either way, so
//!   only the boot-time default differs. `_et_runtime_info()["pal"]` reports
//!   which one was compiled (`TN_ET_PAL`, set here).
//! * **The recorded Android groups.** cpuinfo's `cpuinfo/aarch64-linux-android`
//!   group (12 files, `only_targets`) replaces the Apple one, with the
//!   recorded `CPUINFO_LOG_TO_STDIO=0`, `_GNU_SOURCE=1`; every group's
//!   `flags_by_target` adds `-pthread`; pthreadpool gets no GCD and no
//!   `-fvisibility=hidden` from the JSON on Android (this build adds the
//!   latter to every third-party group anyway).
//! * **BTI.** XNNPACK records `-mbranch-protection=bti` for C and C++,
//!   `-mmark-bti-property` for assembly and `-Wl,-z,force-bti` for the link.
//!   The compile flags are applied as recorded: `bti` landing pads are hint
//!   instructions, harmless where BTI is not enforced, and clang accepts
//!   `-mmark-bti-property` on a `.S` under `-c` (checked with NDK r27 clang
//!   18; only `-fsyntax-only` rejects it, which is why R2b saw a refusal).
//!   The linker flag is **not** applied, by name (`DECLINED_LINKER_FLAGS`):
//!   this crate does not link, `torch_c`'s Rust objects carry no BTI note,
//!   and `force-bti` would mark the whole `_C.abi3.so` as BTI anyway. On a
//!   BTI-enforcing CPU (the Galaxy Z Flip 4's Armv9 cores are) the kernel
//!   would then fault the first indirect branch into Rust code that has no
//!   landing pad. Without it, lld ANDs the inputs' notes and the image is
//!   not BTI-marked, so nothing is enforced.
//! * **Link libraries.** `log` and `m` are bionic shared libraries and are
//!   linked as such. `pthread` has no library on Android (bionic's libc
//!   holds it; the NDK ships no `libpthread`), so nothing is emitted.
//! * **libc++: static.** The artefact must not need `libc++_shared.so`,
//!   which an Android Python process does not have unless an app ships it.
//!   The NDK's `libc++_static.a` and `libc++abi.a` are linked as static
//!   libraries (`-bundle`, so at `torch_c`'s final link). NDK r27 has no
//!   `libc++.a` linker script to name instead, and `-static-libstdc++` is a
//!   driver flag this crate cannot put on `torch_c`'s link line. The
//!   unwinder they need is the NDK's `libunwind`, which Rust's std already
//!   links on Android. `torch_c` without this feature links no C++ at all,
//!   so there is no second C++ runtime to collide with. The CI job
//!   `android-build` reads `NEEDED` and fails on `libc++_shared.so`.
//! * **Exports.** rustc links a `cdylib` with a version script that exports
//!   only the symbols Rust marks exported (`PyInit__C`), so `xnn_*`,
//!   `cpuinfo_*`, `pthreadpool_*`, the ExecuTorch C++ symbols and the static
//!   libc++ are all local. The third-party objects are hidden as on macOS;
//!   the ExecuTorch objects keep R1's default visibility, as on macOS,
//!   because the version script already makes them local and hiding them
//!   would change the compile of the tree the macOS build has proven. The CI
//!   job reads the dynamic symbol table (`llvm-nm -D --defined-only`) and
//!   fails on any of those names: that check, not this paragraph, is the
//!   evidence.
//! * **16 KB pages.** A load-time property of `torch_c`'s link, not of this
//!   archive: `torch_c/build.rs` passes `-z max-page-size=16384` to the
//!   cdylib link on Android when this runtime is linked, and the CI job
//!   checks every `LOAD` segment's alignment.
//! * **API 21 floor.** The NDK's headers hide every function introduced
//!   after the `--target` API level, so a vendored source that calls one
//!   fails to compile here (no silent weak reference: the NDK makes them weak
//!   only with `__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__`, which nothing sets).
//!   Compiling the whole tree with `aarch64-linux-android21-clang` (the wheel
//!   job's `--platform 21`) is the check.
//!
//! # Registration must survive the archive
//!
//! The portable kernels, the prim ops and the XNNPACK backend register
//! themselves from static initializers (`static auto success_with_kernel_reg =
//! register_kernels(..)`, `static auto success_with_compiler =
//! register_backend(backend)`), and nothing references those objects by
//! symbol. A linker pulls an archive member only to resolve an undefined
//! symbol, so without help every such object would be dropped and the
//! registries would be empty at run time, silently. The archive that holds
//! them, `tn_executorch`, is therefore linked `+whole-archive` (rustc passes
//! `-force_load` to ld64, `--whole-archive` to lld on Android), and `-bundle` so that the modifier is applied at the
//! final `cdylib` link of `torch_c` rather than inside this rlib. XNNPACK,
//! cpuinfo and pthreadpool go into a second archive, `tn_et_delegates`, linked
//! as an ordinary static library: nothing in them registers itself, so the
//! linker takes what the backend references and drops the rest.
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
use std::path::{Path, PathBuf};

/// The targets this crate builds for (macOS arm64: steps R1 and R2b; Android
/// arm64: step R3), each with its rules. `torch_c/Cargo.toml` pulls this
/// crate on exactly these targets; the check here is for a direct build of
/// the crate.
const SUPPORTED_TARGETS: &[&str] = &["aarch64-apple-darwin", "aarch64-linux-android"];

/// What differs per target (module docs, "Judgement calls of step R3").
struct TargetRules {
    /// The PAL file under `runtime/platform/default/`, without `.cpp`.
    pal: &'static str,
    /// `cargo::rustc-link-lib` values for the C++ runtime, in link order.
    cxx_runtime: &'static [&'static str],
    /// A recorded link library: `Some(link-lib value)` to emit, `Some("")`
    /// for "already provided, nothing to emit", `None` for "no rule".
    link_library: fn(&str) -> Option<&'static str>,
}

fn target_rules(target: &str) -> TargetRules {
    match target {
        "aarch64-apple-darwin" => TargetRules {
            pal: "posix",
            // Apple's C++ runtime is a system dylib; nothing is shipped for it.
            cxx_runtime: &["dylib=c++"],
            link_library: |lib| match lib {
                // libm and libpthread are part of libSystem on Apple targets,
                // which rustc links already.
                "m" | "pthread" => Some(""),
                _ => None,
            },
        },
        "aarch64-linux-android" => TargetRules {
            pal: "android",
            // Static, so the artefact never needs libc++_shared.so (module
            // docs). NDK r27 ships libc++_static.a and libc++abi.a, no libc++.a.
            cxx_runtime: &["static:-bundle=c++_static", "static:-bundle=c++abi"],
            link_library: |lib| match lib {
                "log" => Some("dylib=log"),
                "m" => Some("dylib=m"),
                // bionic's libc holds pthreads; there is no libpthread.
                "pthread" => Some(""),
                _ => None,
            },
        },
        other => panic!(
            "executorch_rt: target {other} is not supported (issues #315, #337, #350); only \
             {SUPPORTED_TARGETS:?} are. torch_c pulls this crate only on those targets, so \
             reaching this message means it was built directly or the cfg table in \
             torchnative/rust/torch_c/Cargo.toml was widened without porting this build."
        ),
    }
}

/// Recorded linker flags this build declines, by target and flag, with the
/// reason (module docs, "BTI"). Any other non-empty linker flag list stops
/// the build: this crate emits no linker argument.
const DECLINED_LINKER_FLAGS: &[(&str, &str, &str)] = &[(
    "aarch64-linux-android",
    "-Wl,-z,force-bti",
    "torch_c's Rust objects carry no BTI note, so force-bti would mark _C.abi3.so BTI-enforced \
     and fault the first indirect branch into Rust code on a BTI CPU",
)];

/// The whole-archive: runtime core, kernels, shim, XNNPACK backend runtime
/// and threadpool. `links` in Cargo.toml carries the same name so two copies
/// of the runtime can never end up in one artefact.
const ARCHIVE: &str = "tn_executorch";

/// The ordinary static archive: XNNPACK, cpuinfo, pthreadpool.
const DELEGATE_ARCHIVE: &str = "tn_et_delegates";

/// `schema` of every `BUILD_GROUPS.json` this script reads.
const BUILD_GROUPS_SCHEMA: &str = "torchnative-build-groups/1";

/// The ExecuTorch tree: its groups share inline code with the runtime core.
const ET_TREE: &str = "executorch";

/// Trees whose groups are compiled as upstream's Release build records them.
const THIRD_PARTY_TREES: &[&str] = &["xnnpack", "cpuinfo", "pthreadpool"];

/// Header-only trees: read and checked to have no group.
const HEADER_ONLY_TREES: &[&str] = &["fxdiv", "fp16"];

/// The definitions the runtime core is compiled with (upstream's
/// CMakeLists.txt:212-233, :573, schema/CMakeLists.txt,
/// tools/cmake/preset/default.cmake, with logging, verification and the event
/// tracer on: step R1). Every ExecuTorch translation unit gets exactly these.
const CORE_DEFINES: &[(&str, Option<&str>)] = &[
    ("C10_USING_CUSTOM_GENERATED_MACROS", None),
    ("FLATBUFFERS_MAX_ALIGNMENT", Some("1024")),
    ("ET_ENABLE_PROGRAM_VERIFICATION", Some("1")),
    ("ET_ENABLE_DEPRECATED_CONSTANT_BUFFER", Some("0")),
    ("ET_LOG_ENABLED", Some("1")),
    ("ET_MIN_LOG_LEVEL", Some("Info")),
    // The counting tracer (shim) needs the runtime's tracer hooks; with this
    // off, operator_calls and delegate_ops would stay empty and the suite
    // says so.
    ("ET_EVENT_TRACER_ENABLED", None),
];

/// Recorded definitions of the ExecuTorch tree's groups that this build
/// replaces, with the replacement. Each must be listed in the JSON's
/// `configurable` map (see "One configuration for all ExecuTorch code").
const CONFIGURED_DEFINES: &[(&str, &str)] = &[
    ("ET_LOG_ENABLED=0", "ET_LOG_ENABLED=1"),
    ("ET_ENABLE_PROGRAM_VERIFICATION=0", "ET_ENABLE_PROGRAM_VERIFICATION=1"),
];

/// Top-level keys of a `BUILD_GROUPS.json` that describe where the file came
/// from or repeat what the groups already carry. Read by nobody here, on
/// purpose:
///
/// * `upstream`, `generated_by`, `derived_from`, `evidence`, `settings`,
///   `seed`, `configuration`, `undefined_variables_read`: provenance of the
///   description (pins, the CMake values it was evaluated with).
/// * `interface`: what a consumer of the tree must add. The vendoring script
///   already expanded it into each consuming group's own `defines` and
///   `include_dirs`.
/// * `archives`, `not_built_targets`: upstream's archive layout. This build
///   has its own (two archives, see the module docs).
/// * `unbuilt`, `not_described`: files and groups deliberately outside the
///   description.
const INFORMATIONAL_KEYS: &[&str] = &[
    "upstream",
    "generated_by",
    "derived_from",
    "evidence",
    "settings",
    "seed",
    "configuration",
    "undefined_variables_read",
    "interface",
    "archives",
    "not_built_targets",
    "unbuilt",
    "not_described",
];

fn main() {
    let manifest_dir = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let vendor = manifest_dir.join("../vendor");
    let et = vendor.join(ET_TREE);
    let flatbuffers = vendor.join("flatbuffers/include");
    let target = env("TARGET");

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=shim");
    println!("cargo::rerun-if-changed={}", flatbuffers.display());
    for tree in std::iter::once(ET_TREE).chain(THIRD_PARTY_TREES.iter().copied()).chain(HEADER_ONLY_TREES.iter().copied())
    {
        println!("cargo::rerun-if-changed={}", vendor.join(tree).join("MANIFEST.tsv").display());
        println!("cargo::rerun-if-changed={}", vendor.join(tree).join("BUILD_GROUPS.json").display());
    }

    // Refuses every target outside SUPPORTED_TARGETS by name.
    let rules = target_rules(&target);
    debug_assert!(SUPPORTED_TARGETS.contains(&target.as_str()));
    println!("cargo::rustc-env=TN_ET_PAL={}", rules.pal);

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

    let release = std::env::var("PROFILE").map(|p| p == "release").unwrap_or(false);
    let core = CoreConfig {
        vendor: vendor.clone(),
        et: et.clone(),
        flatbuffers: flatbuffers.clone(),
        shim: manifest_dir.join("shim"),
        release,
    };

    // ------------------------------------------------ the runtime core (R1)

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
            if group == "core" && is_unchosen_pal(rel, rules.pal) {
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
    let pal = et.join(format!("runtime/platform/default/{}.cpp", rules.pal));
    if !sources.contains(&pal) {
        panic!(
            "executorch_rt: the {} PAL ({}) is not in MANIFEST.tsv's core group",
            rules.pal,
            pal.display()
        );
    }
    let runtime_units = sources.len();

    let mut whole_objects: Vec<PathBuf> = Vec::new();
    {
        let mut build = core.build(Exceptions::Off);
        build.files(&sources);
        whole_objects.extend(build.compile_intermediates());
    }

    // ------------------------------------------------- the delegate (R2b)

    let et_tree = read_tree(&vendor, ET_TREE, &target);
    let mut group_counts: Vec<(String, String, usize)> = Vec::new();
    for group in &et_tree.groups {
        let objects = compile_et_group(&core, &et_tree, group);
        group_counts.push((et_tree.name.clone(), group.name.clone(), objects.len()));
        whole_objects.extend(objects);
    }
    let backend = et_tree.group("xnnpack_backend");
    if !backend.files.iter().any(|f| f.file_name().is_some_and(|n| n == "XNNPACKBackend.cpp")) {
        panic!(
            "executorch_rt: group `xnnpack_backend` of {} does not list XNNPACKBackend.cpp, the \
             file whose static initializer registers the backend",
            et_tree.json.display()
        );
    }
    let has = |define: &str| backend.defines.iter().any(|d| d == define || d.starts_with(&format!("{define}=")));
    println!(
        "cargo::rustc-env=TN_ET_XNNPACK_WEIGHT_CACHE={}",
        u8::from(has("ENABLE_XNNPACK_WEIGHTS_CACHE"))
    );
    println!(
        "cargo::rustc-env=TN_ET_XNNPACK_SHARED_WORKSPACE={}",
        u8::from(has("ENABLE_XNNPACK_SHARED_WORKSPACE"))
    );

    // The shim includes the threadpool's header, which includes pthreadpool's.
    let threadpool = et_tree.group("extension_threadpool");
    {
        let mut build = core.build(Exceptions::Off);
        for dir in &threadpool.include_dirs {
            build.include(dir);
        }
        build.file(manifest_dir.join("shim/tn_et.cpp"));
        whole_objects.extend(build.compile_intermediates());
    }

    let mut delegate_objects: Vec<PathBuf> = Vec::new();
    let mut link_libraries: Vec<(String, String)> = Vec::new();
    for name in THIRD_PARTY_TREES {
        let tree = read_tree(&vendor, name, &target);
        if tree.groups.is_empty() {
            panic!(
                "executorch_rt: {} describes no group for {target}; the delegate cannot be built without {name}",
                tree.json.display()
            );
        }
        for group in &tree.groups {
            let objects = compile_third_party_group(&tree, group);
            group_counts.push((tree.name.clone(), group.name.clone(), objects.len()));
            delegate_objects.extend(objects);
        }
        for lib in &tree.link_libraries {
            link_libraries.push((tree.name.clone(), lib.clone()));
        }
    }
    for lib in &et_tree.link_libraries {
        link_libraries.push((et_tree.name.clone(), lib.clone()));
    }
    for name in HEADER_ONLY_TREES {
        let tree = read_tree(&vendor, name, &target);
        if !tree.groups.is_empty() {
            panic!(
                "executorch_rt: {} now describes {} group(s) for {target}; this build treats {name} as \
                 header-only and compiles nothing from it. Decide what to do with them here.",
                tree.json.display(),
                tree.groups.len()
            );
        }
        for lib in &tree.link_libraries {
            link_libraries.push((tree.name.clone(), lib.clone()));
        }
    }

    archive(ARCHIVE, &whole_objects);
    archive(DELEGATE_ARCHIVE, &delegate_objects);

    let out_dir = env("OUT_DIR");
    println!("cargo::rustc-link-search=native={out_dir}");
    println!("cargo::rustc-link-lib=static:+whole-archive,-bundle={ARCHIVE}");
    println!("cargo::rustc-link-lib=static:-bundle={DELEGATE_ARCHIVE}");
    for lib in rules.cxx_runtime {
        println!("cargo::rustc-link-lib={lib}");
    }
    let mut emitted: Vec<&str> = Vec::new();
    for (tree, lib) in &link_libraries {
        match (rules.link_library)(lib) {
            Some("") => {}
            Some(spec) => {
                if !emitted.contains(&spec) {
                    emitted.push(spec);
                    println!("cargo::rustc-link-lib={spec}");
                }
            }
            None => panic!(
                "executorch_rt: tree `{tree}` asks to link `{lib}`, and this build has no rule for \
                 that library on {target}"
            ),
        }
    }

    let mut per_tree: BTreeMap<&str, (usize, Vec<String>)> = BTreeMap::new();
    for (tree, group, n) in &group_counts {
        let entry = per_tree.entry(tree.as_str()).or_default();
        entry.0 += n;
        entry.1.push(format!("{group} {n}"));
    }
    let delegate_summary: Vec<String> = per_tree
        .iter()
        .map(|(tree, (total, groups))| format!("{tree} {total} [{}]", groups.join(", ")))
        .collect();
    let delegate_units: usize = group_counts.iter().map(|(_, _, n)| n).sum();
    println!(
        "cargo::warning=executorch_rt: compiled {} translation units for ExecuTorch {version}: runtime {} ({} core, {} prim_ops, {} portable, {} generated), 1 shim, delegate {} in {} groups: {}",
        runtime_units + 1 + delegate_units,
        runtime_units,
        counts["core"],
        counts["prim_ops"],
        counts["portable"],
        counts["generated"],
        delegate_units,
        group_counts.len(),
        delegate_summary.join("; ")
    );
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("executorch_rt: cargo did not set {name}"))
}

/// One archive from already compiled objects.
fn archive(name: &str, objects: &[PathBuf]) {
    if objects.is_empty() {
        panic!("executorch_rt: archive `{name}` would be empty");
    }
    let mut build = cc::Build::new();
    // We emit every link line ourselves.
    build.cargo_metadata(false).warnings(false);
    for object in objects {
        build.object(object);
    }
    build.compile(name);
}

/// How every ExecuTorch translation unit is compiled (step R1's choices).
struct CoreConfig {
    vendor: PathBuf,
    et: PathBuf,
    flatbuffers: PathBuf,
    shim: PathBuf,
    release: bool,
}

/// Whether a translation unit may contain `try`/`catch`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exceptions {
    /// The runtime core, the kernels and the shim (step R1).
    Off,
    /// The XNNPACK backend runtime and the threadpool (module docs).
    On,
}

impl CoreConfig {
    fn build(&self, exceptions: Exceptions) -> cc::Build {
        let mut build = cc::Build::new();
        build.cpp(true).std("c++17");
        if exceptions == Exceptions::Off {
            // Upstream's release flags (CMakeLists.txt:300-308). The unwind-table
            // flags upstream also passes are left out: they only shrink the
            // archive and on Apple targets they remove the compact unwind info
            // that crash reports and the Rust panic machinery read.
            build.flag("-fno-exceptions");
        }
        build
            .flag("-fno-rtti")
            .flag("-Wno-deprecated-declarations")
            .warnings(false)
            .include(&self.vendor)
            .include(self.et.join("runtime/core/portable_type/c10"))
            .include(self.et.join("generated/include"))
            .include(&self.flatbuffers)
            .include(&self.shim)
            // We emit every link line ourselves.
            .cargo_metadata(false);
        for (name, value) in CORE_DEFINES {
            build.define(name, *value);
        }
        if self.release {
            // ET_DCHECK and friends compile out, as in upstream's Release build.
            build.define("NDEBUG", None);
        }
        build
    }
}

/// `runtime/platform/default/*.cpp` other than `pal`: the PALs not used on
/// this target. Exactly one PAL may be linked (each defines every et_pal_*).
fn is_unchosen_pal(rel: &str, pal: &str) -> bool {
    rel.starts_with("runtime/platform/default/") && !rel.ends_with(&format!("/{pal}.cpp"))
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

// ------------------------------------------------------- BUILD_GROUPS.json

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lang {
    C,
    Cxx,
    Asm,
}

impl Lang {
    /// The key of this language in `target_flags.<target>`.
    fn key(self) -> &'static str {
        match self {
            Lang::C => "c",
            Lang::Cxx => "c++",
            Lang::Asm => "asm",
        }
    }
}

/// One group of a tree, resolved for the target being built.
struct Group {
    name: String,
    /// Absolute source paths, in the JSON's order.
    files: Vec<PathBuf>,
    std_c: Option<String>,
    std_cxx: Option<String>,
    /// The group's own flags, then its flags for this target.
    flags: Vec<String>,
    /// The group's own definitions, then its definitions for this target.
    defines: Vec<String>,
    /// Absolute include directories.
    include_dirs: Vec<PathBuf>,
}

/// One tree's build description, resolved for the target being built.
struct Tree {
    name: String,
    json: PathBuf,
    groups: Vec<Group>,
    /// `build_type.flags` then `common_flags`: before every group's own.
    base_flags: Vec<String>,
    /// `build_type.defines`.
    base_defines: Vec<String>,
    /// `target_flags.<target>` by language key.
    lang_flags: BTreeMap<String, Vec<String>>,
    link_libraries: Vec<String>,
    /// The keys of `configurable`.
    configurable: Vec<String>,
}

impl Tree {
    fn group(&self, name: &str) -> &Group {
        self.groups.iter().find(|g| g.name == name).unwrap_or_else(|| {
            panic!("executorch_rt: {} has no group `{name}` for this target", self.json.display())
        })
    }
}

fn read_tree(vendor: &Path, tree: &str, target: &str) -> Tree {
    let dir = vendor.join(tree);
    let json = dir.join("BUILD_GROUPS.json");
    let at = json.display().to_string();
    let text = fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "executorch_rt: cannot read {at}: {e}. It is written by \
             scripts/vendor/vendor_executorch.py (issue #319)."
        )
    });
    let root = json::parse(&text, &at);
    let top = root.object(&at);

    let mut out = Tree {
        name: tree.to_string(),
        json: json.clone(),
        groups: Vec::new(),
        base_flags: Vec::new(),
        base_defines: Vec::new(),
        lang_flags: BTreeMap::new(),
        link_libraries: Vec::new(),
        configurable: Vec::new(),
    };
    let mut seen_schema = false;
    let mut seen_targets = false;
    let mut raw_groups: Option<&json::Value> = None;
    let mut build_type_flags: Vec<String> = Vec::new();
    let mut common_flags: Vec<String> = Vec::new();

    for (key, value) in top {
        let ctx = format!("{at}: `{key}`");
        match key.as_str() {
            "schema" => {
                let schema = value.string(&ctx);
                if schema != BUILD_GROUPS_SCHEMA {
                    panic!("executorch_rt: {ctx} is {schema:?}; this build reads {BUILD_GROUPS_SCHEMA:?} only");
                }
                seen_schema = true;
            }
            "tree" => {
                let named = value.string(&ctx);
                if named != tree {
                    panic!("executorch_rt: {ctx} is {named:?}, but the file is in vendor/{tree}");
                }
            }
            "targets" => {
                let targets = value.strings(&ctx);
                if !targets.iter().any(|t| t == target) {
                    panic!(
                        "executorch_rt: {at} describes {targets:?}, not {target}; the tree was not \
                         vendored for this target"
                    );
                }
                seen_targets = true;
            }
            "build_type" => {
                for (k, v) in value.object(&ctx) {
                    let ctx = format!("{ctx}.`{k}`");
                    match k.as_str() {
                        "name" => {
                            let name = v.string(&ctx);
                            if name != "Release" {
                                panic!("executorch_rt: {ctx} is {name:?}; only upstream's Release configuration is understood");
                            }
                        }
                        "flags" => build_type_flags = checked_flags(v, &ctx),
                        "defines" => out.base_defines = checked_defines(v, &ctx),
                        "note" => {}
                        other => panic!("executorch_rt: {ctx}: key `{other}` is not understood"),
                    }
                }
            }
            "common_flags" => common_flags = checked_flags(value, &ctx),
            "target_flags" => {
                for (t, per_lang) in value.object(&ctx) {
                    if t != target {
                        continue;
                    }
                    for (lang, flags) in per_lang.object(&format!("{ctx}.`{t}`")) {
                        let ctx = format!("{ctx}.`{t}`.`{lang}`");
                        match lang.as_str() {
                            "c" | "c++" | "asm" => {
                                out.lang_flags.insert(lang.clone(), checked_flags(flags, &ctx));
                            }
                            // Flags for the final link. This crate emits no
                            // linker argument, so a non-empty list is
                            // something it cannot honour, except a flag
                            // DECLINED_LINKER_FLAGS names with its reason.
                            "shared_linker" | "exe_linker" => {
                                for flag in flags.strings(&ctx) {
                                    let declined = DECLINED_LINKER_FLAGS
                                        .iter()
                                        .find(|(t, f, _)| *t == target && *f == flag);
                                    match declined {
                                        Some((_, _, reason)) => println!(
                                            "cargo::warning=executorch_rt: {ctx}: linker flag {flag} not \
                                             applied: {reason}"
                                        ),
                                        None => panic!(
                                            "executorch_rt: {ctx} asks for linker flag {flag:?}; this \
                                             build has no rule for linker flags on {target}"
                                        ),
                                    }
                                }
                            }
                            other => panic!("executorch_rt: {ctx}: language `{other}` is not understood"),
                        }
                    }
                }
            }
            "link_libraries" => out.link_libraries.extend(value.strings(&ctx)),
            "link_libraries_by_target" => {
                for (t, libs) in value.object(&ctx) {
                    if t == target {
                        out.link_libraries.extend(libs.strings(&format!("{ctx}.`{t}`")));
                    }
                }
            }
            "configurable" => {
                out.configurable = value.object(&ctx).iter().map(|(k, _)| k.clone()).collect();
            }
            "groups" => raw_groups = Some(value),
            other if INFORMATIONAL_KEYS.contains(&other) => {}
            other => panic!(
                "executorch_rt: {at}: top-level key `{other}` is not understood by this build \
                 script; teach torchnative/rust/executorch_rt/build.rs what it means"
            ),
        }
    }
    if !seen_schema || !seen_targets {
        panic!("executorch_rt: {at} lacks `schema` or `targets`");
    }
    out.base_flags = build_type_flags;
    out.base_flags.extend(common_flags);

    let raw_groups = raw_groups.unwrap_or_else(|| panic!("executorch_rt: {at} has no `groups`"));
    for (index, raw) in raw_groups.array(&format!("{at}: `groups`")).iter().enumerate() {
        if let Some(group) = read_group(raw, &dir, tree, target, &format!("{at}: groups[{index}]")) {
            if out.groups.iter().any(|g| g.name == group.name) {
                panic!("executorch_rt: {at}: two groups are named `{}`", group.name);
            }
            out.groups.push(group);
        }
    }
    out
}

/// One group, or `None` when its `only_targets` leaves this target out.
fn read_group(raw: &json::Value, dir: &Path, tree: &str, target: &str, at: &str) -> Option<Group> {
    let fields = raw.object(at);
    let name = fields
        .iter()
        .find(|(k, _)| k == "name")
        .map(|(_, v)| v.string(&format!("{at}.`name`")).to_string())
        .unwrap_or_else(|| panic!("executorch_rt: {at} has no `name`"));
    let at = format!("{at} (`{name}`)");

    let mut group = Group {
        name: name.clone(),
        files: Vec::new(),
        std_c: None,
        std_cxx: None,
        flags: Vec::new(),
        defines: Vec::new(),
        include_dirs: Vec::new(),
    };
    let mut for_target = true;
    let mut target_flags: Vec<String> = Vec::new();
    let mut target_defines: Vec<String> = Vec::new();
    let mut seen_files = false;

    for (key, value) in fields {
        let ctx = format!("{at}.`{key}`");
        match key.as_str() {
            "name" => {}
            // Upstream's CMake target and archive: provenance. This build's
            // archive layout is in the module docs.
            "cmake_target" | "archive" => {
                value.string(&ctx);
            }
            "only_targets" => {
                for_target = value.strings(&ctx).iter().any(|t| t == target);
            }
            "files" => {
                seen_files = true;
                for rel in value.strings(&ctx) {
                    let path = dir.join(&rel);
                    if !path.is_file() {
                        panic!(
                            "executorch_rt: {ctx} lists {rel}, but {} is missing; run \
                             `python scripts/vendor/vendor_executorch.py --check`",
                            path.display()
                        );
                    }
                    lang_of(&path, &ctx);
                    group.files.push(path);
                }
            }
            "std" => {
                for (lang, standard) in value.object(&ctx) {
                    let ctx = format!("{ctx}.`{lang}`");
                    let standard = value_std(standard.string(&ctx), lang, &ctx);
                    match lang.as_str() {
                        "c" => group.std_c = Some(standard),
                        "c++" => group.std_cxx = Some(standard),
                        other => panic!("executorch_rt: {ctx}: language `{other}` is not understood"),
                    }
                }
            }
            "flags" => group.flags = checked_flags(value, &ctx),
            "flags_by_target" => {
                for (t, flags) in value.object(&ctx) {
                    let flags = checked_flags(flags, &format!("{ctx}.`{t}`"));
                    if t == target {
                        target_flags = flags;
                    }
                }
            }
            "defines" => group.defines = checked_defines(value, &ctx),
            "defines_by_target" => {
                for (t, defines) in value.object(&ctx) {
                    let defines = checked_defines(defines, &format!("{ctx}.`{t}`"));
                    if t == target {
                        target_defines = defines;
                    }
                }
            }
            "include_dirs" => {
                for rel in value.strings(&ctx) {
                    let path = dir.join(&rel);
                    if !path.is_dir() {
                        panic!(
                            "executorch_rt: {ctx} lists {rel}, but {} is not a directory",
                            path.display()
                        );
                    }
                    group.include_dirs.push(path);
                }
            }
            // How upstream links the group's archive. The only rule this
            // build has is the one it applies to the whole ExecuTorch tree.
            "link" => {
                let link = value.string(&ctx);
                if tree != ET_TREE || !link.starts_with("whole-archive") {
                    panic!(
                        "executorch_rt: {ctx} is {link:?}; the only link rule this build understands \
                         is `whole-archive` on a group of the `{ET_TREE}` tree"
                    );
                }
            }
            other => panic!(
                "executorch_rt: {at}: key `{other}` is not understood by this build script; teach \
                 torchnative/rust/executorch_rt/build.rs what it means"
            ),
        }
    }
    if !seen_files {
        panic!("executorch_rt: {at} has no `files`");
    }
    if !for_target {
        return None;
    }
    if group.files.is_empty() {
        panic!("executorch_rt: {at} lists no file for {target}");
    }
    group.flags.extend(target_flags);
    group.defines.extend(target_defines);
    Some(group)
}

fn lang_of(path: &Path, ctx: &str) -> Lang {
    match path.extension().and_then(|e| e.to_str()) {
        Some("c") => Lang::C,
        Some("cc") | Some("cpp") => Lang::Cxx,
        // Capital S: assembler-with-cpp, preprocessed and assembled by the C
        // compiler driver.
        Some("S") => Lang::Asm,
        _ => panic!(
            "executorch_rt: {ctx}: {} is not a .c, .cc, .cpp or .S file; this build has no rule \
             for it",
            path.display()
        ),
    }
}

/// A language standard the compilers of this build accept as `-std=<value>`.
fn value_std(std: &str, lang: &str, ctx: &str) -> String {
    let stem = match lang {
        "c" => std.strip_prefix("gnu").or_else(|| std.strip_prefix('c')),
        "c++" => std.strip_prefix("gnu++").or_else(|| std.strip_prefix("c++")),
        _ => None,
    };
    match stem {
        Some(digits) if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_alphanumeric()) && !digits.starts_with('+') => {
            std.to_string()
        }
        _ => panic!("executorch_rt: {ctx}: standard {std:?} is not understood for language `{lang}`"),
    }
}

/// The compiler flags this build understands. Anything else is refused: a
/// flag that changes the ABI or the instruction set must not ride in unseen.
fn check_flag(flag: &str, ctx: &str) {
    const EXACT: &[&str] = &[
        "-O0",
        "-O1",
        "-O2",
        "-O3",
        "-Os",
        "-Oz",
        "-fPIC",
        "-fno-math-errno",
        "-fvisibility=hidden",
        "-pthread",
        "-mmark-bti-property",
    ];
    const PREFIX: &[&str] = &["-march=", "-Wno-", "-mbranch-protection="];
    let known = EXACT.contains(&flag) || PREFIX.iter().any(|p| flag.len() > p.len() && flag.starts_with(p));
    if !known || flag.contains(char::is_whitespace) {
        panic!(
            "executorch_rt: {ctx}: flag {flag:?} is not in the vocabulary of this build script \
             (check_flag in torchnative/rust/executorch_rt/build.rs)"
        );
    }
}

fn checked_flags(value: &json::Value, ctx: &str) -> Vec<String> {
    let flags = value.strings(ctx);
    for flag in &flags {
        check_flag(flag, ctx);
    }
    flags
}

/// `NAME` or `NAME=VALUE`.
fn split_define(define: &str) -> (&str, Option<&str>) {
    match define.split_once('=') {
        Some((name, value)) => (name, Some(value)),
        None => (define, None),
    }
}

fn checked_defines(value: &json::Value, ctx: &str) -> Vec<String> {
    let defines = value.strings(ctx);
    for define in &defines {
        let (name, _) = split_define(define);
        let ok = !name.is_empty()
            && !name.starts_with(|c: char| c.is_ascii_digit())
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !ok {
            panic!("executorch_rt: {ctx}: {define:?} is not a NAME or NAME=VALUE definition");
        }
    }
    defines
}

/// The group's files by language, in the JSON's order within a language.
fn by_language(group: &Group) -> Vec<(Lang, Vec<&PathBuf>)> {
    let mut out: Vec<(Lang, Vec<&PathBuf>)> = Vec::new();
    for lang in [Lang::C, Lang::Cxx, Lang::Asm] {
        let files: Vec<&PathBuf> = group.files.iter().filter(|f| lang_of(f, &group.name) == lang).collect();
        if !files.is_empty() {
            out.push((lang, files));
        }
    }
    out
}

/// A group of XNNPACK, cpuinfo or pthreadpool: upstream's recorded Release
/// configuration, exactly, plus hidden visibility.
fn compile_third_party_group(tree: &Tree, group: &Group) -> Vec<PathBuf> {
    let mut objects = Vec::new();
    for (lang, files) in by_language(group) {
        let mut build = cc::Build::new();
        build
            .cpp(lang == Lang::Cxx)
            .warnings(false)
            // The recorded build type is Release: no debug information, in
            // any cargo profile (see "Optimisation levels").
            .debug(false)
            .cargo_metadata(false);
        match lang {
            Lang::C => {
                let std = group.std_c.as_deref().unwrap_or_else(|| {
                    panic!(
                        "executorch_rt: {}: group `{}` has C sources and no `std.c`",
                        tree.json.display(),
                        group.name
                    )
                });
                build.std(std);
            }
            Lang::Cxx => {
                let std = group.std_cxx.as_deref().unwrap_or_else(|| {
                    panic!(
                        "executorch_rt: {}: group `{}` has C++ sources and no `std.c++`",
                        tree.json.display(),
                        group.name
                    )
                });
                build.std(std);
            }
            // No language standard applies to assembler-with-cpp.
            Lang::Asm => {}
        }
        // Upstream's order: build type, common, the group's own (and its
        // target's), then the target's per-language flags. The last -O wins.
        for flag in tree.base_flags.iter().chain(&group.flags) {
            build.flag(flag);
        }
        for flag in tree.lang_flags.get(lang.key()).into_iter().flatten() {
            build.flag(flag);
        }
        // This build's own addition (module docs, "Hidden visibility").
        build.flag("-fvisibility=hidden");
        for define in tree.base_defines.iter().chain(&group.defines) {
            let (name, value) = split_define(define);
            build.define(name, value);
        }
        for dir in &group.include_dirs {
            build.include(dir);
        }
        for file in files {
            build.file(file);
        }
        objects.extend(build.compile_intermediates());
    }
    objects
}

/// A group of the ExecuTorch tree (the XNNPACK backend runtime, the
/// threadpool): the core's configuration, plus the group's own flags,
/// definitions and include directories.
fn compile_et_group(core: &CoreConfig, tree: &Tree, group: &Group) -> Vec<PathBuf> {
    let at = format!("{}: group `{}`", tree.json.display(), group.name);
    for file in &group.files {
        if lang_of(file, &at) != Lang::Cxx {
            panic!(
                "executorch_rt: {at}: {} is not C++; the ExecuTorch tree's groups are compiled with \
                 the runtime core's C++ configuration only",
                file.display()
            );
        }
    }
    match group.std_cxx.as_deref() {
        Some("c++17") => {}
        other => panic!(
            "executorch_rt: {at}: `std.c++` is {other:?}; the runtime core is compiled as c++17 and \
             its groups must match"
        ),
    }

    let mut build = core.build(Exceptions::On);
    for flag in &group.flags {
        build.flag(flag);
    }
    for define in &group.defines {
        let define = match CONFIGURED_DEFINES.iter().find(|(recorded, _)| recorded == define) {
            Some((recorded, replacement)) => {
                if !tree.configurable.iter().any(|c| c == recorded) {
                    panic!(
                        "executorch_rt: {at}: this build replaces `{recorded}` with `{replacement}`, \
                         but {} no longer lists `{recorded}` under `configurable`",
                        tree.json.display()
                    );
                }
                *replacement
            }
            None => define.as_str(),
        };
        let (name, value) = split_define(define);
        match CORE_DEFINES.iter().find(|(core_name, _)| *core_name == name) {
            // The core defines it too: the values must be the same, or two
            // translation units would see different inline code.
            Some((_, core_value)) => {
                if *core_value != value {
                    panic!(
                        "executorch_rt: {at}: `{define}` disagrees with the runtime core's \
                         `{name}{}`; every ExecuTorch translation unit must be compiled with one \
                         configuration (CORE_DEFINES and CONFIGURED_DEFINES in \
                         torchnative/rust/executorch_rt/build.rs)",
                        core_value.map(|v| format!("={v}")).unwrap_or_default()
                    );
                }
            }
            None => {
                build.define(name, value);
            }
        }
    }
    for dir in &group.include_dirs {
        build.include(dir);
    }
    for file in &group.files {
        build.file(file);
    }
    build.compile_intermediates()
}

// ------------------------------------------------------------------- JSON

/// A JSON reader for the vendored `BUILD_GROUPS.json` files: the whole
/// grammar (RFC 8259), values kept as a tree, object members in file order.
/// Every accessor names what it was reading when the shape is not the
/// expected one.
mod json {
    pub enum Value {
        Null,
        /// Booleans and numbers are recognised and validated, and their
        /// values dropped: no build description field is one today, so an
        /// accessor for them would be code nothing calls.
        Bool,
        Number,
        String(String),
        Array(Vec<Value>),
        Object(Vec<(String, Value)>),
    }

    impl Value {
        fn kind(&self) -> &'static str {
            match self {
                Value::Null => "null",
                Value::Bool => "a boolean",
                Value::Number => "a number",
                Value::String(_) => "a string",
                Value::Array(_) => "an array",
                Value::Object(_) => "an object",
            }
        }

        pub fn object(&self, ctx: &str) -> &[(String, Value)] {
            match self {
                Value::Object(members) => members,
                other => panic!("executorch_rt: {ctx}: expected an object, found {}", other.kind()),
            }
        }

        pub fn array(&self, ctx: &str) -> &[Value] {
            match self {
                Value::Array(items) => items,
                other => panic!("executorch_rt: {ctx}: expected an array, found {}", other.kind()),
            }
        }

        pub fn string(&self, ctx: &str) -> &str {
            match self {
                Value::String(s) => s,
                other => panic!("executorch_rt: {ctx}: expected a string, found {}", other.kind()),
            }
        }

        pub fn strings(&self, ctx: &str) -> Vec<String> {
            self.array(ctx).iter().map(|v| v.string(ctx).to_string()).collect()
        }
    }

    struct Parser<'a> {
        bytes: &'a [u8],
        pos: usize,
        file: &'a str,
    }

    pub fn parse(text: &str, file: &str) -> Value {
        let mut parser = Parser { bytes: text.as_bytes(), pos: 0, file };
        let value = parser.value(0);
        parser.skip_space();
        if parser.pos != parser.bytes.len() {
            parser.fail("trailing data after the JSON value");
        }
        value
    }

    impl Parser<'_> {
        fn fail(&self, what: &str) -> ! {
            let line = 1 + self.bytes[..self.pos.min(self.bytes.len())].iter().filter(|b| **b == b'\n').count();
            panic!("executorch_rt: {} is not valid JSON at byte {} (line {line}): {what}", self.file, self.pos)
        }

        fn skip_space(&mut self) {
            while let Some(b) = self.bytes.get(self.pos) {
                if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn expect(&mut self, byte: u8) {
            if self.peek() != Some(byte) {
                self.fail(&format!("expected `{}`", byte as char));
            }
            self.pos += 1;
        }

        fn literal(&mut self, word: &str) {
            if !self.bytes[self.pos..].starts_with(word.as_bytes()) {
                self.fail(&format!("expected `{word}`"));
            }
            self.pos += word.len();
        }

        fn value(&mut self, depth: usize) -> Value {
            if depth > 64 {
                self.fail("nested deeper than 64 levels");
            }
            self.skip_space();
            match self.peek() {
                Some(b'{') => {
                    self.pos += 1;
                    let mut members: Vec<(String, Value)> = Vec::new();
                    self.skip_space();
                    if self.peek() == Some(b'}') {
                        self.pos += 1;
                        return Value::Object(members);
                    }
                    loop {
                        self.skip_space();
                        let key = self.string();
                        if members.iter().any(|(k, _)| *k == key) {
                            self.fail(&format!("duplicate key {key:?}"));
                        }
                        self.skip_space();
                        self.expect(b':');
                        let value = self.value(depth + 1);
                        members.push((key, value));
                        self.skip_space();
                        match self.peek() {
                            Some(b',') => self.pos += 1,
                            Some(b'}') => {
                                self.pos += 1;
                                return Value::Object(members);
                            }
                            _ => self.fail("expected `,` or `}` in an object"),
                        }
                    }
                }
                Some(b'[') => {
                    self.pos += 1;
                    let mut items = Vec::new();
                    self.skip_space();
                    if self.peek() == Some(b']') {
                        self.pos += 1;
                        return Value::Array(items);
                    }
                    loop {
                        items.push(self.value(depth + 1));
                        self.skip_space();
                        match self.peek() {
                            Some(b',') => self.pos += 1,
                            Some(b']') => {
                                self.pos += 1;
                                return Value::Array(items);
                            }
                            _ => self.fail("expected `,` or `]` in an array"),
                        }
                    }
                }
                Some(b'"') => Value::String(self.string()),
                Some(b't') => {
                    self.literal("true");
                    Value::Bool
                }
                Some(b'f') => {
                    self.literal("false");
                    Value::Bool
                }
                Some(b'n') => {
                    self.literal("null");
                    Value::Null
                }
                Some(b'-') | Some(b'0'..=b'9') => {
                    let start = self.pos;
                    while let Some(b) = self.peek() {
                        if matches!(b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                    let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or_default();
                    if text.parse::<f64>().is_err() {
                        self.fail("malformed number");
                    }
                    Value::Number
                }
                _ => self.fail("expected a JSON value"),
            }
        }

        fn hex4(&mut self) -> u32 {
            let Some(digits) = self.bytes.get(self.pos..self.pos + 4) else {
                self.fail("truncated \\u escape");
            };
            let text = std::str::from_utf8(digits).unwrap_or_default();
            let Ok(code) = u32::from_str_radix(text, 16) else {
                self.fail("malformed \\u escape");
            };
            self.pos += 4;
            code
        }

        fn string(&mut self) -> String {
            self.expect(b'"');
            let mut out: Vec<u8> = Vec::new();
            loop {
                let Some(b) = self.peek() else {
                    self.fail("unterminated string");
                };
                self.pos += 1;
                match b {
                    b'"' => break,
                    b'\\' => {
                        let Some(escape) = self.peek() else {
                            self.fail("unterminated escape");
                        };
                        self.pos += 1;
                        let ch = match escape {
                            b'"' => '"',
                            b'\\' => '\\',
                            b'/' => '/',
                            b'b' => '\u{0008}',
                            b'f' => '\u{000C}',
                            b'n' => '\n',
                            b'r' => '\r',
                            b't' => '\t',
                            b'u' => {
                                let mut code = self.hex4();
                                if (0xD800..0xDC00).contains(&code) {
                                    // A surrogate pair: the low half must follow.
                                    if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u".as_slice()) {
                                        self.fail("unpaired surrogate in a \\u escape");
                                    }
                                    self.pos += 2;
                                    let low = self.hex4();
                                    if !(0xDC00..0xE000).contains(&low) {
                                        self.fail("unpaired surrogate in a \\u escape");
                                    }
                                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                }
                                match char::from_u32(code) {
                                    Some(ch) => ch,
                                    None => self.fail("\\u escape is not a character"),
                                }
                            }
                            _ => self.fail("unknown escape"),
                        };
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                    0x00..=0x1F => self.fail("control character in a string"),
                    other => out.push(other),
                }
            }
            match String::from_utf8(out) {
                Ok(s) => s,
                Err(_) => self.fail("string is not UTF-8"),
            }
        }
    }
}
