# WASM — feasibility, layer by layer

Status: **complete for all four layers.** It was written as the investigation ran, one entry
per step, so that it would survive an interrupted session. Anything not answered still says
so explicitly — see the summary table at the end for the three items that are
"not attempted" rather than "does not work".

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
