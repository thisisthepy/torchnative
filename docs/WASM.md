# WASM — feasibility, layer by layer

Status: **complete for all four layers, and §7 now closes the one gap the first pass left
open** — Emscripten was "not attempted" and is now *executed*. It was written as the
investigation ran, one entry per step, so that it would survive an interrupted session.
Anything not answered still says so explicitly.

**Read §7 before §2d/§3c/§5.** §7 supersedes them where they disagree: §2d said emscripten was
not attempted; it has now been built and run under Node. §3c's conclusion — that Emscripten
voids `abi3` — is **not** overturned by §7 and is restated there with the measurement that
confirms it.

Question this document answers: the user names six supported platforms, but **WASM is
absent from the `docs/DESIGN.md` §722 matrix.** Is the matrix stale, or is WASM a
different kind of thing? This is a feasibility determination, **not an implementation.**

Vocabulary used throughout, kept strict:

| word | meaning |
|---|---|
| **works** | built/ran here, exit 0, output inspected |
| **blocked** | built/ran here, failed, and the failing thing is named |
| **not attempted** | no execution — reasoning or reading only |

## 0. Environment as found

- `rustc 1.98.0 (88d9e12ae 2026-08-18)`
- wasm targets **already installed**, nothing added:
  `wasm32-unknown-unknown`, `wasm32-wasip1`, `wasm32-unknown-emscripten`
- No `node`, `wasmtime` or `wasmer` **on `PATH`** — but §6a corrects this: a complete emsdk
  5.0.3 with `emcc` and a bundled Node 24 is already on this machine. Nothing was installed
  and the emsdk was not used; only inspected.
- `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wasm2`
- All probing is done in a **separate experiment crate**, following the `rust/vk_probe`
  precedent. No dependency is added to the shipping crate.

## Layer 1 — does candle build for `wasm32`? **Yes, and better than expected.**

**Verdict: works.** The candle surface `rust/torch_c/src/` actually calls compiles for
both `wasm32-wasip1` and `wasm32-unknown-unknown`, including the entire quantised path.
One qualification, in §1d: SIMD is broken upstream, so it compiles *scalar-only*.

### 1a. The probe

`rust/wasm_probe/` — a separate crate, **not** a dependency of `rust/torch_c` and not in a
workspace with it, following the `rust/vk_probe` precedent for exactly the stated reason:
putting a wasm target's constraints on the shipping crate before the question is answered
risks the three platforms that currently build, for nothing.

Its `src/lib.rs` names every `candle_core` item the shipping crate imports. The list was
not invented — it is the output of

```
grep -rhoE "use candle_core::\{?[^;]*" rust/torch_c/src/
```

which is:

```
CpuStorage  DType  Layout  Shape  Tensor  Device  Module
quantized::{GgmlDType, QMatMul, QStorage, QTensor}
Error::{Msg, MatMulUnexpectedStriding, WithBacktrace}
```

plus the twelve `GgmlDType` variants `quant.rs` enumerates and the `half::{f16, bf16}`
constructors `reduced.rs` uses. `candle-core` is pinned to the same version and the same
`default-features = false` as `rust/torch_c/Cargo.toml:36`.

It is a library and not a `#[test]`, deliberately: **there is no wasm runtime on this
machine** (§0), so the only question answerable here is compilation. Every reference is
written so that a missing item is a *compile* error.

**Control:** the probe builds for the host (`aarch64-apple-darwin`) first, exit 0. Without
that, "it compiles for wasm32" would not distinguish a working target from a probe that
had quietly stopped referring to anything.

### 1b. Results — measured

`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wasm2`, `rustc 1.98.0`.

| target | command | result |
|---|---|---|
| `aarch64-apple-darwin` (control) | `cargo build --release` | **exit 0**, 0 errors |
| `wasm32-wasip1` | `cargo build --release --target wasm32-wasip1` | **exit 0**, 0 errors, `libwasm_probe.rlib` produced |
| `wasm32-unknown-unknown` | bare | **exit 101** — `getrandom`, see §1c |
| `wasm32-unknown-unknown` | with the §1c pin | **exit 0**, 0 errors |
| `wasm32-wasip1` **+ `-C target-feature=+simd128`** | | **exit 101, 38 errors** — see §1d |

The prompt asked specifically whether `Tensor`, `DType`, matmul and `QTensor` survive.
**All four survive, on both wasm targets.** `QTensor::quantize`, `QTensor::dequantize`,
`QTensor::data`, `QStorage::from_data`, `QMatMul::from_arc`, `QMatMul::forward` and all
twelve `GgmlDType` variants compile. The GGUF/k-quant machinery is *not* behind the
`wasm32` gate.

### 1c. What `wasm32` drops — the `tokenizers` problem solves itself

`docs/CANDLE_DEPS.md` is **still accurate**; re-verified against the same
`candle-core 0.11.0` source. `Cargo.toml.orig:37` still reads

```
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
tokenizers = { workspace = true, features = ["onig"] }
```

and `src/quantized/mod.rs:17` still gates `pub mod tokenizer;` on
`#[cfg(not(target_arch = "wasm32"))]`. Neither has a feature gate; both key on the target
architecture alone.

The consequence is that **on wasm32 the entire `tokenizers`/`onig` subtree drops out for
free** — no patch, no fork, no `[patch.crates-io]`. Measured with `cargo tree` on the probe's
own graph:

| target | crates |
|---|---|
| `aarch64-apple-darwin` | 129 |
| `wasm32-wasip1` | **80** (−49) |
| `wasm32-unknown-unknown` | **84** (−45) |

The 49 that vanish are exactly the subtree `docs/CANDLE_DEPS.md` §3a costed: `tokenizers`,
`onig`, `onig_sys`, `regex`/`regex-automata`/`regex-syntax`, `compact_str`,
`derive_builder`(+core/macro), `darling`(+core/macro), `esaxx-rs`, `spm_precompiled`,
`unicode-normalization-alignments`, `monostate`, `cc`, `pkg-config`, `find-msvc-tools` and
friends. (49 here versus that document's −44 is not a discrepancy: this probe's graph is
not `torch_c`'s — it has no PyO3 and names `half` directly.)

**So CANDLE_DEPS.md §6's last row — "WASM target: irrelevant, candle already filters it" —
is correct as a statement about relevance but understates the direction.** On wasm32 the
dependency problem that document spends 400 lines on does not exist. If WASM ever became a
target, it is the one platform that needs none of §8's vendored patch.

`wasm32-unknown-unknown` needs four crates `wasip1` does not, and they name the environment:
`wasm-bindgen`(+macro/support/shared) and `bumpalo`. That is the browser. The cause is
`getrandom`, which refuses `wasm32-unknown-unknown` outright:

```
error: The wasm32-unknown-unknown targets are not supported by default; you may need to
enable the "wasm_js" configuration flag.
```

Getting past it needs **both** `--cfg getrandom_backend="wasm_js"` in `RUSTFLAGS` **and** an
explicit `getrandom` dependency with `features = ["wasm_js"]` — the cfg alone gives a second,
different error. Two major versions sit in candle's graph simultaneously (`rand` 0.8 pulls
`getrandom` 0.3, `rand` 0.9 pulls 0.4), so both must be pinned. `rust/wasm_probe/Cargo.toml`
carries that block with a comment saying it is a finding and not a wanted dependency.

**This is a real, if small, structural fact: `wasm32-unknown-unknown` forces a browser
dependency (Web Crypto via `js-sys`) into the build, and `wasm32-wasip1` needs nothing.**
Anything server- or CLI-side should target `wasip1`.

### 1d. The one thing that is blocked — WASM SIMD does not compile

This is the load-bearing negative result of layer 1.

candle **has** hand-written WASM SIMD kernels: `src/cpu/simd128.rs` and
`src/quantized/simd128.rs`, both `use core::arch::wasm32::*`. `k_quants.rs` dispatches to
them at six sites (`vec_dot_q4_0_q8_0`, `q8_0_q8_0`, `q2k_q8k`, `q4k_q8k`, `q6k_q8k`,
`q8k_q8k`). All of it is gated on `#[cfg(target_feature = "simd128")]`, which is **off by
default** on both wasm targets.

Turning it on does not compile:

```
RUSTFLAGS='-C target-feature=+simd128' cargo build --release --target wasm32-wasip1
  -> exit 101, 38 errors, all E0433, all in candle-core-0.11.0/src/cpu/mod.rs
     19x cannot find type `CurrentCpuF16` in this scope
     19x cannot find type `CurrentCpuBF16` in this scope
```

The cause is an upstream inconsistency, confirmed by reading the three CPU backends:

| module | defines |
|---|---|
| `src/cpu/avx.rs` | `CurrentCpu`, `CurrentCpuF16`, `CurrentCpuBF16` |
| `src/cpu/neon.rs` | `CurrentCpu`, `CurrentCpuF16`, `CurrentCpuBF16` |
| `src/cpu/simd128.rs` | **`CurrentCpu` only** |

and `src/cpu/mod.rs` gates the reduced-precision helpers (`vec_add_f16` and the `bf16`
equivalents) on `#[cfg(any(target_feature = "neon", target_feature = "avx2",
target_feature = "simd128"))]` — a three-way list — while the scalar fallbacks beneath them
are gated on `#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]`, a
**two-way** list that forgets `simd128`. So enabling `simd128` selects the SIMD branch of a
helper whose SIMD types were never written for this backend.

Named precisely: **`simd128` is a maintained-in-name-only backend in candle 0.11.0. Nobody
compiles it with `+simd128`, because it does not build.**

What this costs, concretely: WASM would run candle's **scalar** kernels. The
comparison is not "WASM is a bit slower" but "WASM gets none of the vectorised
quantised dot products that `docs/PERF.md`/`docs/PERF_ANDROID.md` treat as the
baseline on NEON". It is a fixable bug — the fix is to write the two missing types in
`simd128.rs`, or to add `simd128` to the fallback's `not(any(...))` list, which is a
one-line change that would at least make `+simd128` build and pick up the six quantised
kernels. **Not attempted here**: this is a feasibility study and the fix belongs upstream
or in a vendored patch, neither of which is this task's area.

### 1e. Also survives, but unverified at runtime

`rayon`, `memmap2` and `num_cpus` all *compile* for `wasm32-wasip1` — they are in the 80.
candle uses `rayon` in `cpu_backend/mod.rs`, `sort.rs`, `conv2d.rs` and `utils.rs`, and
`utils.rs:317` calls `num_cpus::get_physical()` to size its thread pool. **Whether any of
that behaves at runtime is not attempted** — wasm32 without the threads proposal has one
thread, and `std::thread::Builder::new().spawn()` on `wasip1` is a runtime error rather than
a compile error. This is the most likely place for a "compiles but does not run" surprise,
and it cannot be checked without a runtime (§ execution requirements).

## Layer 2 — does PyO3 work on `wasm32`? **`abi3-py313` + `extension-module` builds and links.**

**Verdict: works, for `wasm32-wasip1`, with one added linker flag.** The result is a real
extension module in shape. Whether anything can *load* it is layer 3, and that is where the
answer turns.

### 2a. Getting the control right first

The first attempt at this layer produced a **false negative that would have ended the
investigation**, and it is worth recording because it is the §5.5 failure mode: a check that
cannot pass is not a check.

- The probe was an `rlib`. `cargo build --features pyo3-route --target wasm32-wasip1` gave
  **exit 0** — but an `rlib` build never invokes the linker, so that exit 0 said nothing.
- Changing it to `cdylib` (which is what `rust/torch_c/Cargo.toml:10` is) made **the host
  build fail too**, with undefined `_Py*` symbols. Had that not been checked, "wasm fails to
  link" would have been reported as a wasm finding when it was a miswired probe.

The cause: `rust/torch_c/.cargo/config.toml` supplies `-C link-arg=-undefined -C
link-arg=dynamic_lookup` on Apple targets, and the probe had no such file.
`rust/wasm_probe/.cargo/config.toml` now mirrors it. **Control: host `cdylib` with
`extension-module` + `abi3-py313` links, exit 0, 1.5 MB `libwasm_probe.dylib`.**

### 2b. Results — measured

| target | crate type | flags | result |
|---|---|---|---|
| `aarch64-apple-darwin` (control) | cdylib | `-undefined dynamic_lookup` | **exit 0**, 1,502,176 B dylib |
| `wasm32-wasip1` | cdylib | none | **exit 101** — `rust-lld: undefined symbol: _Py_DecRef`, `_Py_IncRef`, `PyErr_GetRaisedException`, `PyList_Type`, `PyType_GetFlags`, … |
| `wasm32-wasip1` | cdylib | `-C link-arg=--allow-undefined` | **exit 0**, 1,073,660 B `wasm_probe.wasm` |

PyO3's build script raised **no objection at all** to being cross-compiled to wasm with
`abi3-py313`. That is the abi3 forward-compatibility path: with `extension-module` on, no
`libpython` is linked and no interpreter needs to be found at build time, so there is
nothing for the build script to fail on. The failure, when it came, was purely at link.

`--allow-undefined` tells `wasm-ld` to turn unresolved symbols into **module imports**
rather than errors. That is the wasm analogue of the note already in
`rust/torch_c/.cargo/config.toml`: *"Android needs no extra flags: ELF shared libraries may
carry undefined symbols and the interpreter resolves them at load time."* wasm can express
the same thing; it just will not do it by default.

**Second false-positive caught.** The first `--allow-undefined` build produced a 128 KB
`.wasm`, which is far too small to contain candle. `--gc-sections` had discarded all of it,
because nothing reachable from `PyInit_wasm_probe` called the layer-1 functions. The probe
now exposes `probe_all()`, which calls every one of them, and the artefact is **1.07 MB** —
candle and PyO3 in one wasm module together.

### 2c. The artefact has the right shape — verified, not assumed

The `.wasm` was parsed directly (import and export sections, `/tmp/wimp.py`, a ~30-line
reader written for this):

```
total imports: 50
  Py*/_Py* imports: 45      all from module "env"
  wasi_snapshot_preview1:  5   environ_get environ_sizes_get fd_write proc_exit sched_yield
exports containing PyInit: ['PyInit_wasm_probe']
```

This is exactly the shape of a CPython extension module: it **exports `PyInit_<name>`** and
**imports the 45 `Py*` symbols it needs from its host**. Nothing about `abi3-py313` +
`extension-module` is rejected by the wasm target.

**But read the import list again, because it is the whole of layer 3.** Those 45 symbols are
imports of a *core wasm module*, resolved by whoever instantiates it. For this to load into
a Python interpreter, something has to play the role `dlopen` plays on Linux and Android —
and `wasm32-wasip1` core modules have no dynamic linking. See §3.

### 2d. Emscripten — **not attempted**, and the reason is not technical

`wasm32-unknown-emscripten` is the target PyO3 actually supports on purpose:
`pyo3-build-config-0.29.2/src/lib.rs:73` special-cases it to emit `-sSIDE_MODULE=2
-sWASM_BIGINT` for rustc < 1.95, and `:304` exempts it from the "wasm has no rpath" rule.
It is the Pyodide target. The rust target is installed here.

The toolchain is present too — `/Volumes/macMini/caches/emsdk` is a complete emsdk **5.0.3**
with `upstream/emscripten/emcc` and a bundled `node 24.19.0_64bit`.

**It was not used.** The brief said to verify that directory and not touch it, and
verification turned up a specific reason to obey: `find` shows **22 files under it modified
today**, and it holds a live `upstream/emscripten/cache/cache.lock`. It is in active use by
another project. `emcc` writes to that cache on every invocation, so building through it
risks another workstream's build for a datapoint that layer 3 makes secondary anyway.

So: **emscripten is "not attempted", not "does not work".** What is known about it is read
from PyO3's source, not measured here.

## Layer 3 — what shape does CPython take on WASM? **This is where it splits in two.**

**Verdict: `wasm32-wasip1` is blocked. `wasm32-unknown-emscripten` is open but costs the
project's central ABI decision.** Everything in this section is read from CPython's own
policy documents and PEPs, **not measured here** — there is no target CPython for either
wasm platform on this machine.

The layers above quietly assumed one thing: that the `.wasm` produced in §2c, which exports
`PyInit_wasm_probe` and imports 45 `Py*` symbols, can be *loaded by an interpreter*. That is
the same assumption `rust/torch_c/.cargo/config.toml` states for Android — *"ELF shared
libraries may carry undefined symbols and the interpreter resolves them at load time."*
**Whether that sentence has a wasm translation is the whole of layer 3, and the answer
differs between the two wasm platforms.**

### 3a. The two are not variants of one platform

| | `wasm32-wasip1` | `wasm32-unknown-emscripten` |
|---|---|---|
| CPython support tier | **tier 2** (PEP 11) | **tier 3** (PEP 776, from Python 3.14) |
| what it is | server/CLI, POSIX-ish capability sandbox | browser / Node — Pyodide, PyScript, JupyterLite |
| `dlopen`/`dlsym` | **absent** | **provided by Emscripten** |
| our extension shape (`_C.abi3.so` loaded at import) | **cannot work** | works — Pyodide loads side modules this way |
| PyO3 support | incidental | **explicit** (`pyo3-build-config` special-cases the triple) |
| wheel platform tag | **none exists** | `pyemscripten_*_wasm32` (PEP 783) |
| threads | n/a here | **forbidden** — PEP 783: *"libraries cannot use `-pthread`"* |

Note the inversion: **WASI has the higher CPython support tier and is the one that cannot
take our artefact.** Tier is about whether CPython itself builds and passes tests, not about
whether third-party extension modules can be loaded. Reading the tier alone would give the
wrong answer.

### 3b. WASI — blocked, and the blocker has a name

The blocker is **`dlopen` does not exist in WASI preview 1**. It is not a gap in CPython;
it is absent from the platform. CPython's own WASI build has to skip building the test
modules that need it (`_testimportmultiple`, `_testmultiphase`, `_testsinglephase`,
`xxlimited`, `xxlimited_35`), and the documented approach for extension modules on WASI is
to compile them **statically into the interpreter** as builtins.

This is exactly the sentence the brief asked for — *what does our embedding shape become
there*. The answer:

> On WASI, `torch._C` cannot be a wheel. It would have to be a **CPython fork built with
> `torch._C` as a builtin module**, shipped as a whole interpreter binary.

That is a different product from the one `tools/wheel/build.py` makes. Note also that the
§2c artefact *already showed this*: it imports its `Py*` symbols from a module literally
named `env`, meaning the instantiating host must supply all 45 — which is not something a
Python interpreter does for a module it imports.

`--enable-wasm-dynamic-linking` exists in CPython's configure, but it is the Emscripten
path, not a WASI one.

### 3c. Emscripten — open, but it invalidates `abi3-py313`

Emscripten implements `dlopen`/`dlsym`, which is why it is *the* wasm target that supports
native extension modules, why Pyodide works, and why PyO3 special-cases it. Our shape
survives.

**The cost lands squarely on this project's most load-bearing decision.**
`rust/torch_c/Cargo.toml:13-23` spends ten lines justifying `abi3-py313`, and `docs/ABI3.md`
§7 recommends it, on the grounds that one artefact loads into many interpreter versions and
that a version-pinned `.so` is a silent failure mode. **On Emscripten that argument does not
hold**:

- Emscripten **makes no ABI stability guarantee between its own versions** (PEP 783 says so
  outright). The `pyemscripten` platform therefore pins a specific Emscripten compiler
  version, the set of statically linked libraries, and specific linker flags.
- Pyodide's own position, from its maintainers' discussion, is that abi3 *works* but is
  *useless* until the underlying ABI is stabilised, and that building with the limited API
  should be disabled for Pyodide.

So on Emscripten the wheel is pinned to a Python feature release **and** an Emscripten
version anyway. abi3 buys nothing, and the one thing it does buy elsewhere — a single
artefact across interpreter versions — is unavailable. **This does not block WASM. It means
WASM would be the one platform where the project's ABI strategy does not apply**, and that
is a design fact worth knowing before anyone commits.

### 3d. `-pthread` is forbidden — and layer 1 §1e is where that bites

PEP 783: *"libraries cannot use `-pthread`."* §1e recorded that `rayon`, `num_cpus` and
candle's thread-pool code in `utils.rs` all compile for wasm. **They compile; they cannot be
used as intended.** `utils.rs:317` sizes a pool from `num_cpus::get_physical()` and
`utils.rs:127` spawns with `std::thread::Builder`.

Combined with §1d — SIMD does not compile — the honest performance statement for WASM is:
**scalar kernels, single thread.** That is not "somewhat slower than Android"; it is the
slowest configuration candle has. No number is offered here because none was measured.

## Layer 4 — is there a distribution path? **Yes, for Emscripten. None for WASI.**

The brief's framing — *"Pyodide uses its own index; can it go to PyPI?"* — was true until
recently and is **now out of date.**

### 4a. PEP 783 exists and PyPI accepts the tag

**PEP 783 (Emscripten Packaging)** defines the platform tag series

```
pyemscripten_${YEAR}_${PATCH}_wasm32        e.g. pyemscripten_2026_0_wasm32   (Python 3.14)
```

and states that package indexes **SHOULD accept any wheel whose platform tag matches
`pyemscripten_[0-9]+_[0-9]+_wasm32`**. PyPI supports these uploads; packages built for
Pyodide can be published to PyPI directly and installed at runtime, rather than living only
in Pyodide's own index. The tag replaced the earlier `pyodide_${YEAR}_${PATCH}_wasm32`, and
before that the form was `emscripten_3_1_45_wasm32`-style, versioned on the compiler.

`maturin` already emits the tag. Our wheel builder does not — `tools/wheel/build.py` has
`AndroidTarget` (PEP 738, `android_<api>_<abi>`), `IOSTarget` (`ios_<major>_<minor>_...`) and
the macOS path, and nothing for wasm. Adding a `PyEmscriptenTarget` is the same shape of work
as the two that exist. **That file is another workstream's area and was not touched.**

So layer 4 is the *easiest* of the four, and it is the one the original question expected to
be hardest.

### 4b. WASI has no tag at all

There is no platform tag for `wasm32-wasi` in any PEP, which is consistent with §3b: a
platform that cannot load extension-module wheels has no need of a tag for them.

## Recommendation on `DESIGN.md` §722

**Recommendation: do not add WASM to the §722 matrix. Add a WASM row to the README platform
table instead, and reconcile the two lists inside `DESIGN.md` that currently disagree.**

### 5a. The matrix is not stale — but `DESIGN.md` contradicts itself, and that is the real finding

The brief asked whether the matrix is stale or WASM is a different kind of thing. It is
neither, exactly. **The six platforms and the five in the matrix come from two different
lists in the same document.**

- `DESIGN.md` §861 (§10, repository layout) says the build tool `pypackpack` targets
  **Android · iOS · macOS · Linux · Windows · WASM** — that is the six.
- `DESIGN.md` §722 lists **five** platforms across six rows (Android appears twice, CPU and
  GPU). No WASM.

They are not the same kind of table. §722 is a ***`kernels` backend* matrix** — its columns
are "which `kernels` backend", "which actual kernel", "resolved when". It answers where
optimised kernels come from. §861 is a **build-target list**.

**On its own terms, §722 is correct to omit WASM.** `kernels`' backend enumeration is
`cpu` · `cuda` · `metal` · `rocm` · `xpu` (+`cann`, `neuron`). It has no `wasm` backend — the
same gap §722 already documents at length for `vulkan` on Android GPU. And per §1d/§3d there
would be no vectorised kernel to point the entry at even if a backend existed. A WASM row
in §722 would read `cpu` / scalar / build-time, which is what "no entry" already means.

**What is genuinely wrong is that §861 promises a platform the rest of the document never
mentions again.** That is the mismatch the user noticed. The fix is not to add a row to
§722; it is to say in §861, or beside it, that WASM is a pypackpack capability that
torchnative has not adopted, with a pointer here.

### 5b. What to do with the README

`README.md:264` currently says WASM's absence is *"an open question rather than an
oversight"* and gives two reasons: candle excludes `wasm32` from parts of itself, and CPython
on WASM is a different embedding shape. **This investigation resolves both, and one of them
was wrong:**

| README's reason | status after this investigation |
|---|---|
| "candle excludes `wasm32` from parts of itself" | **Not a blocker — it is a benefit.** What candle excludes is `quantized::tokenizer`, which `torch_c` never used, and excluding it drops 49 crates for free (§1c). Everything `torch_c` calls survives, `QTensor` included (§1b). |
| "CPython on WASM is a different embedding shape" | **Correct, and sharper than stated.** It is two different shapes: WASI cannot load our artefact at all (§3b), Emscripten can but voids `abi3-py313` (§3c). |

Suggested replacement rows for the README platform table, using its own legend
(⚠️ = built, never executed):

```
| in the target matrix | ... | ❌ *not listed* — see docs/WASM.md, correctly so |
| rust target installed | ... | ✅ (wasip1, unknown-unknown, emscripten all present) |
| target CPython        | ... | 🔲 |
| extension builds      | ... | ⚠️ wasip1 only; emscripten not attempted |
```

The row `can be run *here*` should stay ❌ but its footnote at `README.md:206` needs a
correction: it lists `node` among the runtimes this machine lacks, and that is **no longer
true** — see below.

### 5c. If someone does pursue it, the order is fixed

1. **Emscripten, not WASI.** WASI needs a CPython fork with `torch._C` as a builtin; that is
   a different product.
2. **Fix candle's `simd128` first** (§1d), or accept scalar kernels. This is a small upstream
   patch and it gates whether the result is worth shipping.
3. **Drop `abi3` for that target only** (§3c) and expect a wheel per Pyodide release.
4. **Then** add a `PyEmscriptenTarget` to `tools/wheel/build.py` (§4a).

Steps 1-3 are all cheaper than step 4 is misleading: the packaging works, which makes it
tempting to start there.

## What would be needed to actually verify by execution

Nothing below was run. This section exists so the gap is a list rather than an adjective.

### 6a. What is actually on this machine — one correction

`node`, `wasmtime` and `wasmer` are **not** on `PATH`; confirmed. But the claim in
`README.md:206` that this machine has no `node` is **wrong**, and the same would be said of
this document if it were not checked:

```
/Volumes/macMini/caches/emsdk/node/24.19.0_64bit/bin/node      exists
/Volumes/macMini/caches/emsdk/upstream/emscripten/emcc         exists (emsdk 5.0.3)
/Volumes/macMini/caches/emsdk/python/3.13.3_64bit/bin/python3  exists
```

**A full Emscripten toolchain and a Node 24 are already here.** Nothing was installed and
nothing was used: the brief said to verify that directory and not touch it, and verification
gave an independent reason to obey — 22 files under it were modified today and it holds a
live `upstream/emscripten/cache/cache.lock`, so another workstream is using it. `emcc` writes
to that cache on every run.

So the honest status of layer 2's emscripten half is **"not attempted for a scheduling
reason, not a capability reason."** If that directory is free, or `EM_CACHE` is redirected to
a scratch path, `cargo build --target wasm32-unknown-emscripten --features pyo3-route` is
runnable **today**.

### 6b. To execute, in increasing order of cost

| to answer | what is needed | have it? |
|---|---|---|
| does the wasip1 `.wasm` instantiate | `wasmtime` or `wasmer` | **no** — and §3b says the answer would be "no host supplies those 45 imports", so this is low value |
| does the emscripten build link | `emcc` + `EM_CACHE` pointed away from the shared emsdk | **yes**, gated on not disturbing another workstream |
| does `import torch` work in Pyodide | a Pyodide distribution matching the Emscripten version, plus Node (present) | **no Pyodide** — it is a download, not a build |
| do the golden tests pass on WASM | Pyodide + `numpy`/`torch` reference wheels for `pyemscripten`, and a harness that does not assume a local CPython | **no** — `tools/golden/compare.py` runs against a host interpreter |
| is it fast enough to matter | all of the above **plus** the `simd128` fix (§1d) | **no** — and without §1d the measurement would only restate "scalar is slow" |

The cheapest meaningful next step is the second row, and it needs no installation.

### 6c. Reproducing what *was* run

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wasm2
cd rust/wasm_probe

# layer 1 -- candle only
cargo build --release                                          # host control, exit 0
cargo build --release --target wasm32-wasip1                    # exit 0
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  cargo build --release --target wasm32-unknown-unknown         # exit 0
RUSTFLAGS='-C target-feature=+simd128' \
  cargo build --release --target wasm32-wasip1                  # exit 101, 38 errors (§1d)

# layer 2 -- with PyO3
cargo build --release --features pyo3-route                     # host control, exit 0
RUSTFLAGS='-C link-arg=--allow-undefined' \
  cargo build --release --features pyo3-route --target wasm32-wasip1   # exit 0, 1.07 MB

# crate-count comparison (§1c)
cargo tree --target <triple> --prefix none | grep -oE '^[a-z0-9_-]+ v[0-9.]+' | sort -u | wc -l
```

`rust/wasm_probe/` is an investigation crate. It is **not** a dependency of `rust/torch_c`
and shares no workspace with it; `rust/torch_c/` and `tools/wheel/` were not modified.

## Summary table

| layer | question | verdict |
|---|---|---|
| 1 | candle on `wasm32` | **works** — `Tensor` `DType` matmul `QTensor` all survive, both targets; 49 fewer crates |
| 1d | candle WASM SIMD | **blocked** — `+simd128` fails to build, `simd128.rs` lacks `CurrentCpuF16`/`CurrentCpuBF16` |
| 2 | PyO3 `abi3-py313` + `extension-module` | **works** on wasip1 with `--allow-undefined`; exports `PyInit_*`, imports 45 `Py*` |
| 2d | same, emscripten | **not attempted** — shared emsdk in use by another workstream |
| 3 | CPython on WASI | **blocked** — no `dlopen`; would need `torch._C` as a CPython builtin |
| 3 | CPython on Emscripten | **open**, but `abi3-py313` buys nothing there; no `-pthread` |
| 4 | distribution | **works** — PEP 783 `pyemscripten_*_wasm32`, accepted by PyPI |
| — | `DESIGN.md` §722 | **leave WASM out** — it is a `kernels` backend matrix and there is no wasm backend. Fix the §861/§722 contradiction instead |

Sources for the layer 3/4 claims, none of which were measured here:
[PEP 776](https://peps.python.org/pep-0776/) ·
[PEP 783](https://peps.python.org/pep-0783/) ·
[Pyodide PyEmscripten ABI](https://pyodide.org/en/stable/development/abi.html) ·
[abi3-on-Pyodide discussion](https://github.com/pyodide/pyodide/discussions/4377) ·
[CPython WASI dlopen issue](https://github.com/python/cpython/issues/115983) ·
[tier promotion](https://discuss.python.org/t/wasm32-emscripten-and-wasm32-wasi-have-been-promoted-to-tier-3-platforms-for-cpython/17590)

---

# 7. Emscripten, actually executed

Everything in §1–§6 above was a *compile* result. §2d and §6a left one thing open for a
scheduling reason: `emcc` was never run. This section runs it. **New vocabulary rule for this
section: "runs" means a `.wasm` was executed by Node on this machine and its stdout was read.**

## 7.0 Setup — what was used, and what was protected

```sh
emsdk   /Volumes/macMini/caches/emsdk                                    (5.0.3, shared)
emcc    /Volumes/macMini/caches/emsdk/upstream/emscripten/emcc
node    /Volumes/macMini/caches/emsdk/node/24.19.0_64bit/bin/node        v24.19.0
export EM_CACHE=/Volumes/macMini/caches/emcc-scratch      # NOT the shared emsdk cache
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-emcc
```

The shared emsdk is redirected away from with `EM_CACHE`, which is the whole of §6a's concern:
`emcc` writes its sysroot and port cache on every invocation, and that is the only part of the
emsdk it writes to. Baseline before starting: **24431 files under the emsdk, 22 modified in the
last 24h** — the same 22 §2d saw. The count is re-checked at the end of this section.

`emcc` did populate the scratch cache on first use, as expected:

```
cache:INFO: generating system headers: sysroot_install.stamp...
   (this will be cached in "/Volumes/macMini/caches/emcc-scratch/sysroot_install.stamp")
```

Nothing was installed. `wasm32-unknown-emscripten` was already an installed rustc target (§0).

## 7.1 Layer 1 on Emscripten — candle compiles, and the crate graph is identical to WASI

| target | command | result |
|---|---|---|
| `wasm32-unknown-emscripten` | `cargo build --release` | **exit 0**, 0 errors |

Crate counts, re-measured on this branch (numbers differ by +2 from §1c because the lockfile
has moved since; the *comparison* is what matters):

| target | crates |
|---|---|
| `aarch64-apple-darwin` | 131 |
| `wasm32-wasip1` | 82 |
| `wasm32-unknown-emscripten` | **82** |

`comm` on the two sorted crate lists is **empty in both directions**: the emscripten and wasip1
dependency graphs are *the same set of 82 crates*. So §1c's finding — the `tokenizers`/`onig`
subtree drops out for free on wasm32 — holds on Emscripten too, and Emscripten costs nothing
extra. It also does **not** drag in the `wasm-bindgen`/`bumpalo` browser subtree that
`wasm32-unknown-unknown` forces (§1c); the `getrandom` `wasm_js` pin in the probe's
`Cargo.toml` is inert here.

**One thing this table does not say, and §5.5 requires saying it.** The layer-1 `cdylib` link
also exits 0, and it produces a **65-byte `.wasm`**:

```
0061 736d 01000000  000f 08 "dylink.0" ... 0715 01 11 "__wasm_call_ctors"
```

That is a side module containing one empty function. `--gc-sections` discarded all of candle
because, with no PyO3 feature on, nothing is reachable from an export. **A layer-1 emscripten
`cdylib` exit 0 proves nothing on its own** — it is the same false positive §2b caught on
wasip1, in a new place. The load-bearing artefact is §7.2's, and it is 857 KB.

Note also what the 65-byte header already tells us: rustc's `wasm32-unknown-emscripten` cdylib
link emits a **`dylink.0` section**, i.e. `-sSIDE_MODULE`. That section is the difference from
wasip1 in one word, and §7.3 is about what it buys.

## 7.2 Layer 2 on Emscripten — `abi3-py313` + `extension-module` links with **no extra flag**

```sh
cargo build --release --features pyo3-route --target wasm32-unknown-emscripten
```

**exit 0. 857,161-byte `wasm_probe.wasm`.** Candle and PyO3 in one module — the size is the
control that §2b established: 128 KB would mean candle had been stripped.

**This is the sharpest difference from WASI, and it is not a matter of degree.**

| | `wasm32-wasip1` | `wasm32-unknown-emscripten` |
|---|---|---|
| link with no extra flags | **exit 101** — `rust-lld: undefined symbol: _Py_DecRef`, … | **exit 0** |
| what makes it link | `-C link-arg=--allow-undefined`, added by hand | nothing — `dylink.0`/`SIDE_MODULE` is the target default |
| unresolved `Py*` become | imports of module `"env"` on a **core** module | imports of `"env"` **and `GOT.mem`** on a **side** module |

The second row is the answer to "how does Emscripten resolve CPython symbols — same as
wasip1's `--allow-undefined` or not?" **Not the same.** On wasip1 the flag is a way of telling
`wasm-ld` to stop complaining, and it produces a core module whose imports must be supplied by
whoever *instantiates* it — which, as §3b says, a Python interpreter never does for a module it
imports. On Emscripten the same unresolved symbols are recorded in a `dylink.0` section, which
is a **defined interchange format for dynamic linking**: the loader is expected to resolve them
against an already-loaded main module. That is `dlopen`'s job, and it is the thing WASI has no
equivalent of.

### 7.2a The artefact's shape, read from the binary

Disassembled with the emsdk's own `wasm-dis` (binaryen) and counted:

```
sections:  dylink.0  TYPE  IMPORT  FUNCTION  GLOBAL  EXPORT  START  ELEM  DATACOUNT  CODE  DATA
exports:   PyInit_wasm_probe            (llvm-nm: 00003735 T PyInit_wasm_probe)
imports:   198 total   =  88 "env"  +  70 "GOT.func"  +  40 "GOT.mem"
```

Of those, **54 are CPython**: 45 functions from `"env"` and **9 data symbols from `"GOT.mem"`**:

```
GOT.mem:  PyExc_AttributeError  PyExc_BaseException  PyExc_SystemError  PyExc_TypeError
          PyList_Type  PyModule_Type  PyTuple_Type  PyType_Type  PyUnicode_Type
```

The `GOT.mem` half is new information relative to §2c, which counted 45 and stopped. Those nine
are **data**, not functions — type objects and exception singletons — and on wasip1 they were
folded into the same undifferentiated `"env"` import list. Emscripten separates them because
data relocation is a distinct problem from function relocation, and it is one more reason the
two platforms are not variants of each other.

### 7.2b The build imports **20 pthread symbols** — §1e/§3d, now measured rather than predicted

The non-CPython `"env"` imports include:

```
pthread_create  pthread_detach  pthread_attr_init  pthread_attr_setstacksize
pthread_mutex_{init,lock,trylock,unlock,destroy}  pthread_mutexattr_{init,settype,destroy}
pthread_cond_{init,wait,signal,broadcast,destroy}  pthread_condattr_{init,setclock,destroy}
sched_yield  sysconf
```

§1e said "`rayon`, `memmap2` and `num_cpus` compile; whether they *behave* is not attempted",
and §3d said PEP 783 forbids `-pthread`. **This import list is the two statements meeting.**
The artefact was built **without** `-pthread`, and it still names `pthread_create`, because
candle's thread-pool code is linked in and only the *linker* knows it is unreachable at
runtime. Emscripten without `-pthread` supplies a `pthread_create` that fails rather than one
that spawns. So the shape is: **it links, it loads, and any code path that actually reaches
`pthread_create` fails at runtime rather than at build time.** That is the "compiles but does
not run" surprise §1e predicted, and it is now located precisely.

## 7.3 Layer 3, part one — **candle executes on Emscripten under Node**

This is the result the first pass could not reach at all, and it is the one thing about WASM
that nobody in this project knew: **our layer-1 code runs.**

`rust/wasm_probe/src/main.rs` is a new `[[bin]]` in the same probe crate. It is deliberately
*not* an "ok / not ok" harness — it prints computed **values** and compares each to a number
worked out by hand, because an exit code alone cannot tell "candle ran" apart from "the runtime
started and the code had been gc'd away". That is the same false positive as §2b and §7.1, and
it is the specific way a runtime probe lies.

**Control first**, on the host, since a probe that passes everywhere proves nothing:

```
$ cargo run --release --bin wasm_probe                       # aarch64-apple-darwin
target_arch=aarch64 target_os=macos
tensor  sum(ones(2,3)) = 6  expect 6  PASS
matmul  dims=[2, 4] sum=48  expect [2,4] 48  PASS
reduced f16sum=24 bf16sum=24  expect 24 24  PASS
q4_0    bytes=1152 maxerr=0 matmulsum=511.96875  expect 1152 ~0 512  PASS
simd128 enabled = false
== failures = 0 ==                                            exit 0
```

Then the same source, compiled to `wasm32-unknown-emscripten` (998,294-byte `.wasm` plus a
55,653-byte `wasm_probe.js` loader, both emitted by rustc via `emcc`) and executed by the
emsdk's bundled Node:

```
$ node wasm_probe.js                                          # node v24.19.0
target_arch=wasm32 target_os=emscripten
tensor  sum(ones(2,3)) = 6  expect 6  PASS
matmul  dims=[2, 4] sum=48  expect [2,4] 48  PASS
reduced f16sum=24 bf16sum=24  expect 24 24  PASS
q4_0    bytes=1152 maxerr=0 matmulsum=511.96875  expect 1152 ~0 512  PASS
simd128 enabled = false
== failures = 0 ==                                            exit 0
```

**Verdict: runs.** Read the two blocks side by side — they are *identical below the header
line*, including the last digit of the quantised matmul.

What that costs to say precisely, item by item:

| checked | host | emscripten/node |
|---|---|---|
| `Tensor::zeros` + broadcast add + `sum_all` | 6 | 6 |
| `matmul`, shape and value | `[2,4]`, 48 | `[2,4]`, 48 |
| `f16` and `bf16` round trip through `to_dtype` | 24, 24 | 24, 24 |
| `QTensor::quantize(Q4_0)` block packing | 1152 B | 1152 B |
| `QTensor::dequantize` max abs error | 0 | 0 |
| `QMatMul::forward` | **511.96875** | **511.96875** |
| `cfg!(target_feature = "simd128")` | false | **false** |

The Q4_0 row is the load-bearing one. 2048 elements at 32 weights per 18-byte block is 64
blocks is 1152 bytes — the *block layout* is identical, so this is not "some quantiser ran",
it is candle's GGML-compatible one. And `511.96875` rather than 512 is the quantisation error
itself: it is the *same wrong answer* on both platforms, which is a much stronger statement
than agreement on a round number would have been.

**The `simd128 = false` row is the honest half.** §1d predicted this and it is now confirmed at
runtime rather than inferred: what executed above is candle's **scalar** kernel path. §1d's
upstream bug (`simd128.rs` lacks `CurrentCpuF16`/`CurrentCpuBF16`) is unchanged and was not
fixed here.

**No timing is reported, and that is deliberate.** Between §1d (scalar only) and §3d/§7.2b
(single-threaded, `-pthread` forbidden by PEP 783) this is the slowest configuration candle
has. A number here would be read as "WASM performance" when it is "WASM performance with both
accelerators off", and the interesting question — what it costs *after* the simd128 fix — is
not answerable on this machine.

## 7.4 Layer 3, part two — **`dlopen` works on our actual `cdylib`**

§3a's whole argument turns on one row of its table: WASI has no `dlopen`, Emscripten does. That
row was **read from CPython and Pyodide policy documents, not measured** — §3 says so. It is
also the row that decides whether `torch._C` can ever be a wheel on WASM, so it is worth more
than a citation.

`rust/wasm_probe/dlopen_host.c` is a 60-line C program: `dlopen` a path, `dlsym`
`wasm_probe_run`, call it, compare the returned bitfield against 31. It loads the
**candle-only** side module, not the PyO3 one, on purpose — that separates "does the dynamic
loader work" from "can 54 CPython symbols be resolved" (§7.5), so a failure can only be blamed
on one of them.

`wasm_probe_run` is a new `#[no_mangle] pub extern "C"` in the probe's `lib.rs`. It exists
because of §7.1: without an export reachable from outside, `--gc-sections` reduces the
emscripten cdylib to 65 bytes. With it the candle-only side module is **865,573 bytes**.

**Host control**, `cc dlopen_host.c` against `libwasm_probe.dylib`:

```
dlopen  PASS (handle=0x745d8a40)
dlsym   PASS (wasm_probe_run=0x106afb6a0)
call    wasm_probe_run() = 31  expect 31  PASS      exit 0
```

**Emscripten**, `emcc -sMAIN_MODULE=1`, side module loaded from disk via `-sNODERAWFS=1`, run
under Node 24:

```
dlopen  PASS (handle=0x7fab0)
dlsym   PASS (wasm_probe_run=0x14aa)
call    wasm_probe_run() = 31  expect 31  PASS      exit 0
```

**Verdict: runs.** Same bitfield, 31, on both — all five candle checks including the quantised
`QMatMul::forward`, executing from inside a module that was loaded **at runtime, by name, out of
a file**, sharing the main module's linear memory and heap. That is the exact mechanism CPython
uses to import a native extension, and it is exactly what §3b says WASI cannot do.

**This is the single most load-bearing new fact in this document.** Everything §3c and §5c say
about "Emscripten is the one to pursue" rested on a citation; it now rests on an execution.

### 7.4a The first attempt failed, and the failure is a real constraint

The `-sMAIN_MODULE=1` build linked fine and then died at load:

```
dlopen FAIL: could not load dynamic lib: side_nopyo3.wasm
LinkError: WebAssembly.Instance(): Import #136 "env" "__cpp_exception":
           tag import requires a WebAssembly.Tag
```

rustc's `wasm32-unknown-emscripten` side module uses **native wasm exception handling**, so it
imports `__cpp_exception` as a wasm **tag**. A main module compiled with emcc's default
exception mode exports no such tag, and the two cannot be linked. Adding `-fwasm-exceptions` to
the *main* module fixed it, and that is the only change between the failing and passing runs
above.

Recorded because it is not a quirk of this probe: **the main module and every side module must
agree on the exception ABI, and rustc picks native wasm EH for you.** Any real host — a Pyodide
distribution, a custom CPython build — has to have been built with `-fwasm-exceptions` for a
Rust extension to load into it. It is one more thing pinned by the platform, alongside the
Emscripten version and linker flags PEP 783 already pins (§3c), and it belongs on that list.

## 7.5 Layer 3, part three — **the PyO3 extension loads, and `PyInit_` runs**

§7.4 loaded the candle-only module. This loads the real shape: the `cdylib` built with
`abi3-py313` + `extension-module`, 893,771 bytes, exporting **both** `PyInit_wasm_probe` and
`wasm_probe_run`, and importing the 54 CPython symbols of §7.2a.

There is no CPython for this target on this machine, so the host supplies those 54 itself.
`rust/wasm_probe/gen_pystubs.py` generates them **from the side module's own import table**
(`wasm-dis` output), not from CPython headers — there are none to take, and guessing is not an
option: a WebAssembly import whose type does not match the exporting module is a `LinkError` at
*instantiation*, so an arity wrong by one gives "will not load" rather than "misbehaves".
45 function stubs + 9 `char[512]` data symbols, one special case: `PyModuleDef_Init` returns
its argument, because PyO3's multi-phase init is literally `return PyModuleDef_Init(&MODULE_DEF)`
and that is the smallest thing that lets `PyInit_` run to completion.

`rust/wasm_probe/pyinit_host.c`, `emcc -fwasm-exceptions -sMAIN_MODULE=1`, Node 24:

```
loading: side_pyo3.wasm
dlopen  PASS -- module instantiated
candle  wasm_probe_run() = 31  expect 31  PASS
dlsym   PASS PyInit_wasm_probe=0x14fc
PyInit  returned 0x160900, 1 stub call(s), last=PyModuleDef_Init
PyInit  PASS -- PyModuleDef.m_name == "wasm_probe" at +20
== failures = 0 ==                                              exit 0
```

**Verdict: runs.** Four things, each worth naming separately:

1. **The extension instantiated** with all 54 CPython imports bound to a host module — including
   the nine `GOT.mem` *data* symbols, which are the half wasip1 has no mechanism for.
2. **candle still returns 31 from inside the PyO3 build.** PyO3's presence did not change the
   numerical result or let `--gc-sections` take anything.
3. **`PyInit_wasm_probe` was found by `dlsym` and executed.** It made exactly **one** call into
   the host, and that call was `PyModuleDef_Init` — which independently confirms PyO3 0.29 uses
   *multi-phase* module initialisation, since single-phase would have called `PyModule_Create2`
   and a dozen others.
4. **It returned the right module definition.** The returned pointer was dereferenced from the
   *main* module and its `m_name` reads `"wasm_probe"`, at offset **+20** — which also measures
   `sizeof(PyModuleDef_Base)` on wasm32 as 20 bytes (8-byte `PyObject` header + `m_init` +
   `m_index` + `m_copy`, all 4-byte). That a pointer stored in the side module's data segment is
   dereferenceable from the main module is the proof that **data relocation into the shared
   linear memory worked**, not just function relocation.

This is as far as it is possible to go without a target CPython, and it is further than §3
assumed was reachable.

### 7.5a Three negative controls — including one that made this section weaker

§5.5 of CLAUDE.md: a check that cannot fail is not a check. Three were run.

| control | change | result |
|---|---|---|
| **A** | `PyModuleDef_Init` stub returns `0` instead of its argument | `PyInit returned 0` → **FAIL, exit 1** |
| **B** | delete `PyType_IsSubtype` from the host entirely | `dlopen` **still succeeds**, run still exits 0 |
| **C** | delete `PyModuleDef_Init` — the one symbol `PyInit_` calls | loads, then **aborts** on the call |

**A** proves the `PyInit PASS` line is not vacuous. **C** proves resolution is real for symbols
that are actually called.

**B is the one that matters, because it falsified something this section originally claimed.**
The first draft of `pyinit_host.c` printed *"all 54 CPython imports resolved against this
host"*. **That was false.** Emscripten's dynamic loader does not refuse a side module with an
unresolvable import — it substitutes a stub that aborts if reached, exactly as **C** then
demonstrated:

```
thread '<unnamed>' panicked at .../panicking.rs: panic in a function that cannot unwind
Aborted()   RuntimeError: unreachable
```

So **on Emscripten, "the extension loaded" does not mean "the interpreter has every symbol it
needs"; a missing one surfaces as a runtime abort at first use.** The probe's output was
corrected to say so.

That is not a detail — it is the same failure mode as the 20 `pthread` imports in §7.2b, and it
generalises into a deployment property: **an ABI mismatch between a Rust extension and its
Emscripten CPython host presents as an abort deep in a call, not as an import error.** On Linux
or Android the loader rejects the `.so` up front. This is a *worse* diagnostic than the
"silent failure mode" `docs/ABI3.md` §7 warns about for version-pinned `.so` files, and it is
one more argument on the same side as §3c/§7.6.

## 7.6 Layer 4 — what a real verdict still needs, and how close it actually is

§7.5 is the ceiling without a target CPython. The remaining question is what it would take to
run `import torch` for real, and the answer turned out to be **much closer than §6b estimated**.

`docs/WASM.md` §6b listed "a Pyodide distribution matching the Emscripten version" as a blocker
and left it at that, as if the matching were the hard part. It is not. Pyodide's own
`Makefile.envs` on `main`:

```
export PYODIDE_EMSCRIPTEN_VERSION ?= 5.0.3
export PYVERSION                  ?= 3.14.2
export PYODIDE_ABI_VERSION        ?= 2026_0
```

**The emsdk on this machine is 5.0.3 — the exact version current Pyodide pins.** And
`docs/development/abi/314.md` requires *"Rust version 1.93.0 or later"*; this machine has 1.98.0
(§0). The toolchain half of the gap is already closed, by coincidence rather than by anyone's
plan.

What is missing is only the artefact:

| to answer | needed | status |
|---|---|---|
| does the extension load into a real CPython | a built Pyodide distribution (`pyodide-core`, a download, not a build) | **absent** — nothing Pyodide-shaped anywhere on this machine |
| do the golden tests pass | the above **plus** `numpy`/reference wheels for `pyemscripten_2026_0`, and a harness not assuming a host interpreter | absent; `tools/golden/compare.py` runs against a host CPython |
| is it worth shipping | all of the above **plus** the §1d `simd128` fix | absent, and §1d is upstream |

**The next step is a download and is not attempted here.** Fetching and unpacking a Pyodide
distribution is outside the immediate request (CLAUDE.md §5.7) and the brief said to install
nothing. It is recorded as the one remaining step rather than done.

Two things that would otherwise be found the hard way, if someone does take that step:

- **`-fwasm-exceptions` is mandatory, at compile *and* link time**, for the main module as well
  as the side module. §7.4a discovered this empirically — the first `dlopen` died with
  `tag import requires a WebAssembly.Tag` — and Pyodide's ABI spec lists "the stack unwinding
  ABI" as one of the five things a PyEmscripten platform pins. The empirical finding and the
  spec agree.
- **`-sSUPPORT_LONGJMP=wasm` is also part of the platform**, and the probe's host did *not* pass
  it. It got away with that only because nothing in the side module uses `setjmp`. A build that
  does would fail; the emcc invocations in §7.0 are a probe, not a conforming build recipe.

## 7.7 What this does **not** change: `abi3` is still void on Emscripten

§3c concluded that Emscripten invalidates `abi3-py313` on the strength of a PEP 783 sentence
and a Pyodide discussion thread. **Nothing in §7 overturns that, and the platform's own
documentation now makes it sharper than §3c stated.** This is the part not to over-read:
`abi3-py313` *built* (§7.2), *linked* (§7.2), *loaded* (§7.5) and its `PyInit_` *ran* (§7.5).
None of that buys what `abi3` exists to buy.

From Pyodide's ABI documentation:

> The Emscripten compiler makes no ABI stability guarantees between versions, and several
> linker flags can adjust the ABI. […] Pyodide adopts a **new PyEmscripten platform for each
> feature release of Python.**

And the platforms, read off their own specifications:

| platform tag | CPython | **Emscripten** |
|---|---|---|
| `pyemscripten_2025_0` | 3.13 | **4.0.9** |
| `pyemscripten_2026_0` | 3.14 | **5.0.3** |
| `pyemscripten_2026_5` | 3.15 | **6.0.5** |

**Read the right-hand column.** Three CPython feature releases, three *different compilers*. The
entire premise of `abi3` — `rust/torch_c/Cargo.toml:13-23` spends ten lines on it, and
`docs/ABI3.md` §7 recommends it — is that one artefact serves many interpreter versions. On
Emscripten, supporting 3.13, 3.14 and 3.15 means **building three times with three different
Emscripten toolchains**, and the `abi3` tag changes nothing about that count. The limited API
would still be *used*; it would simply buy nothing.

Restated as a deployment consequence, since that is what the question is for:

> **On every other platform this project ships, "one binary per platform" is achieved by
> `abi3`. On WASM it is not achievable at all.** WASM would be one binary per *PyEmscripten
> platform*, i.e. one per CPython feature release, each built with the Emscripten compiler that
> platform names. That is the same cardinality as dropping `abi3` everywhere, and it is a
> different distribution model from the other five platforms — not a variation on it.

Three further consequences, each measured above rather than argued:

1. **Mismatches abort, they do not fail to load** (§7.5a). A side module whose host lacks a
   symbol still `dlopen`s, and aborts at first use. So the wrong-platform wheel is not caught
   by the loader. That is *worse* than the version-pinned-`.so` failure mode `docs/ABI3.md` §7
   already calls silent.
2. **Scalar, single-threaded, and that is structural.** §1d (candle's `simd128` does not
   compile) and §7.2b/§3d (PEP 783 and Pyodide both forbid `-pthread`; the artefact imports 20
   `pthread` symbols that can only abort) are not tuning problems.
3. **The ABI pins the unwinding mode too** (§7.4a, §7.6), so an extension is tied to its host's
   exception ABI on top of everything else.

**None of this says WASM is impossible — §7.3/§7.4/§7.5 say the opposite, by execution.** It
says WASM is a platform whose distribution model this project does not currently have, and that
the cost is a wheel per CPython feature release rather than the one-artefact story the other
five platforms get.

## 7.8 Corrections to earlier sections of this document

| section | said | now |
|---|---|---|
| header | "complete for all four layers" | true only for *compilation*; nothing had been executed |
| §0, §1a, §6a | "there is no wasm runtime on this machine" | **wrong** — the emsdk's Node 24 is a runtime, and §7.3 used it |
| §2d, §6a | emscripten "not attempted", shared emsdk in use | **attempted and passed**; `EM_CACHE` redirection was sufficient (§7.9) |
| §3a table, row `dlopen` | "provided by Emscripten" (read from PEPs) | **measured** — §7.4, on our own `cdylib` |
| §3a table, row "our extension shape" | "works — Pyodide loads side modules this way" (inferred) | **measured** — §7.5, `PyInit_` runs and returns the right `PyModuleDef` |
| §1e | rayon/threads "compiles but unverified at runtime" | **located**: 20 `pthread` imports present in the artefact, unusable at runtime (§7.2b) |
| §6b row 2 | "does the emscripten build link — gated on not disturbing another workstream" | **done**, exit 0, and the emsdk was not disturbed (§7.9) |
| §6b row 3 | "no Pyodide — it is a download, not a build" | still true, but the toolchain gap is **closed**: Pyodide pins emscripten 5.0.3, which is what is here (§7.6) |
| §5b README rows | "extension builds: ⚠️ wasip1 only; emscripten not attempted" | should read **✅ emscripten builds, loads and runs under Node**; `can be run here` flips from ❌ to ✅ |

## 7.9 Reproducing §7, and the emsdk was left as found

```sh
export PATH="/Volumes/macMini/caches/emsdk/upstream/emscripten:\
/Volumes/macMini/caches/emsdk/node/24.19.0_64bit/bin:$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-emcc
export EM_CACHE=/Volumes/macMini/caches/emcc-scratch          # not the shared emsdk cache
cd rust/wasm_probe

# 7.1 / 7.2 -- compile and link
cargo build --release --target wasm32-unknown-emscripten                        # exit 0 (65 B .wasm!)
cargo build --release --features pyo3-route --target wasm32-unknown-emscripten  # exit 0, 857 KB

# 7.3 -- candle executes
cargo run   --release --bin wasm_probe                                          # host control
cargo build --release --bin wasm_probe --target wasm32-unknown-emscripten
node $CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/wasm_probe.js          # exit 0

# 7.4 -- dlopen a side module
cargo build --release --lib --target wasm32-unknown-emscripten
cc dlopen_host.c -o host_dl && ./host_dl $CARGO_TARGET_DIR/release/libwasm_probe.dylib   # control
emcc dlopen_host.c -O1 -fwasm-exceptions -sMAIN_MODULE=1 -sALLOW_MEMORY_GROWTH=1 \
     -sNODERAWFS=1 -o dl.js
node dl.js wasm_probe.wasm                                                      # exit 0
#   NB: without -fwasm-exceptions this fails at load, see 7.4a

# 7.5 -- the PyO3 module, with generated CPython stubs
cargo build --release --lib --features pyo3-route --target wasm32-unknown-emscripten
wasm-dis wasm_probe.wasm -o side.wat && python3 gen_pystubs.py side.wat > pystubs_gen.h
emcc pyinit_host.c -O1 -fwasm-exceptions -sMAIN_MODULE=1 -sALLOW_MEMORY_GROWTH=1 \
     -sNODERAWFS=1 -o py.js
node py.js wasm_probe.wasm                                                      # exit 0
```

New files, all inside `rust/wasm_probe/`: `src/main.rs`, `dlopen_host.c`, `pyinit_host.c`,
`gen_pystubs.py`, plus a `wasm_probe_run` export added to `src/lib.rs`. `rust/torch_c/` and
`tools/wheel/` were not touched.

**The shared emsdk.** §2d declined to run `emcc` because 22 files under
`/Volumes/macMini/caches/emsdk` had been modified that day and a `cache.lock` was present.
`EM_CACHE=/Volumes/macMini/caches/emcc-scratch` was sufficient: every cache write `emcc` made
went there (`sysroot_install.stamp`, `libc.a`, `libc++-noexcept.a`, `libcompiler_rt.a`,
`pic/*`, …). Rechecked afterwards:

| | before | after |
|---|---|---|
| files under `/Volumes/macMini/caches/emsdk` | 24431 | **24431** |
| of those, modified in the last 24 h | 22 | **22** |
| modified in the last 70 min (i.e. by this session) | — | **0** |
| files under `EM_CACHE=/…/emcc-scratch` | 0 | 1605 (55 MB) |

The 55 MB of system libraries `emcc` generated all landed in the scratch directory; the shared
emsdk was not written to at all. **§2d's caution was right and its conclusion was too strong**:
the correct response to a shared toolchain is to redirect its cache, not to skip the experiment.

### 7.9a Regressions, re-run after these changes

The changes are confined to `rust/wasm_probe/` and this file, but the shipping crate's suites
were re-run anyway, because "I only touched X" is a claim and not a check:

| check | result |
|---|---|
| `PYTHON=$PY sh rust/torch_c/pytests/run.sh` | **exit 0** — 197 ok, 0 not ok |
| `$PY tools/golden/compare.py` | **exit 0** — 2811/2811 cases, 0 failed, ops covered = 119 |

## 7.10 Summary of §7 — the four layers, executed

| layer | question | first pass | §7 |
|---|---|---|---|
| 1 | candle on emscripten | not attempted | **works** — exit 0, same 82 crates as wasip1, identical set |
| 1 | candle *runs* | impossible, "no runtime" | **runs** — Node 24, all 5 checks, `511.96875` identical to host |
| 1d | wasm SIMD | blocked upstream | **unchanged** — confirmed off at runtime (`simd128 = false`) |
| 2 | PyO3 `abi3` + `extension-module` link | wasip1 only, needs `--allow-undefined` | **works on emscripten with no extra flag** — side module, 857 KB |
| 3 | `dlopen` our `cdylib` | read from PEPs | **runs** — dlopen + dlsym + call, bitfield 31 = host |
| 3 | `PyInit_` on the real extension | read from PEPs | **runs** — returns `PyModuleDef` with `m_name == "wasm_probe"` |
| 3 | `import torch` in a real interpreter | not attempted | **still not attempted** — needs a Pyodide download (§7.6) |
| 4 | distribution | PEP 783 works | **works, and costs `abi3`** — one wheel per CPython release, per §7.7 |

**The headline, stated so it cannot be over-read:** *our extension shape builds, loads and
executes on Emscripten under Node — and doing so costs the one-binary-per-platform property
that `abi3` gives this project everywhere else.* Both halves are measured. Neither cancels the
other.
