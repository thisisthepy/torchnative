# 0.1.0b6: release notes

**torchnative now builds wheels for free-threaded CPython 3.15t, and transformers'
continuous batching greedy on mps is token-equal to upstream mps.** The rest of
the release is a correctness round: each defect in §2 was measured against
upstream, fixed, and pinned by a test that was red before the fix.

The tracking issue is #105. Every entry names its issue and pull request.

---

## 1. Features added

- **CPython 3.15t (free-threaded) wheels** (#51, #70, #96). The tag is
  `cp315-cp315t`, for macOS arm64, Linux x86_64 and Linux aarch64. They are
  built against python-build-standalone's CPython 3.15.0 final. Publish dry run
  38867632114 (ref `develop`) built cp315t on all three, and the run was green:
  16 jobs succeeded and the upload job was skipped, as a dry run does. The abi3 wheels are
  unchanged: `abi3` is now a default cargo feature, and the 3.15t build turns it
  off.
  - The module **keeps the GIL** (`#[pymodule(gil_used = true)]`). On 3.15t
    CPython re-enables the GIL and warns. `Py_mod_gil` is not declared until the
    thread audit lands (#51, #76). Rebuilt without `gil_used = true`,
    `test_module_keeps_the_gil_until_the_audit_lands` went red.
  - Grade on the wheel: **reaches** for macOS arm64 (installed with
    `uv pip install` into a 3.15t venv, `torch.ones(2)+1` gives `[2., 2.]`; #51,
    measured on the 3.15.0rc1 build). The Linux cp315t wheels are **builds**:
    they have not been run.
  - Windows, iOS, Android and wasm cp315t refuse by name: there is no
    free-threaded target CPython for them (#51).
- **Continuous batching greedy on mps** (#32; #84, #89, #100, #104).
  transformers' `generate_batch` greedy on mps, #13's 8 configurations (Qwen3
  and Llama, paged|sdpa and paged|eager, float32 and bfloat16), is token-equal
  to upstream mps in all 8. Before #100 the shim stopped at
  `aten.embedding.default` (int32) in all 8. Measured locally on this Mac
  (#100). What it needed:
  - **int32 on Metal** (#100). The candle forks gained i32 cast, copy and fill
    kernels. Buffer ops keep I32 storage. Claimed arithmetic widens to int64 on
    the device and narrows back, so overflow wraps exactly like int32.
    `test_dtypedev`'s int32 mps column moved from 2 to 18 of 23.
  - **Strided in-place writes, `cumsum`, `any` and bool copy on Metal** (#84),
    one contiguous run per `slice_set`, with no host download.
  - **`index.Tensor` and `gather` on Metal** (#89), and **`index_put_`** (#104),
    each with a device-side bounds check. Out of range raises upstream cpu's
    type and text. One small vector is read back per call for the check.
  - `x[i] = scalar` and `x[mask] = scalar` on the device (#99, #102).
- **`torch.result_type` and c10 promotion** (#31, #80; #93, #94). The result
  dtype follows `torch.result_type` for dimensioned, 0-d and Python-scalar
  operands. A 0-d operand no longer raises the result dtype. On a 6144-cell
  upstream table, 790 cells failed before #80.
- **Tie order on Linux** (#81 Linux half, #92). `sort`, `argsort` and `topk` on
  linux-gnu order ties as GCC 13.3 libstdc++ does, which is what upstream's
  manylinux wheel is built with. The oracle was measured from upstream torch
  2.13.0 in Docker on linux/arm64 and linux/amd64 (byte-identical); 8542 of its
  22728 cells differ from libc++. The transcription agrees on 22168 of 22168
  non-error cells. On Apple and Android the libc++ order landed in #33 (#82).
- **`aten.sort.stable`** (#82). It raised `NotImplementedError` before.
- **candle-kernels is vendored** (#86, #88), like candle-core and
  candle-metal-kernels. A one-hunk patch guards `__hmax_nan`/`__hmin_nan` on the
  CUDA toolkit version, so the CUDA build compiles again. This is a **builds**
  claim only.
- **mps on macOS 11 to 14** (#65, #85). The candle fork builds an
  `MTLResidencySet` only where the OS has one, so creating an mps device no
  longer panics before macOS 15. `_C._metal_residency()` reports what was
  decided. A `macos14` CI job runs the fork's gate test, `cargo test` and the
  suite. On PR #85 it passed on macOS 14.8.9, including the `cargo test` that
  used to panic there (`host_const_refuses_an_f64_constant_on_metal`). A hosted
  VM may lack a usable GPU, so green there may mean "refuses by name without a
  panic", not "computes" (#85).

## 2. Defects fixed

- **#49** (#83): mps `amax` of an all `-inf` row gave `-FLT_MAX`, float compares
  on mps died at `to_dtype F32 F64`, and cpu `min`/`max` lost upstream's zero
  sign; all three now equal upstream.
- **#50** (#87): mps f16 `add`/`sub`/`rsub.Scalar` rounded the scalar once,
  where upstream mps rounds through f32 (0.031280517578125 against 0.03125);
  it now equals upstream mps, and `intelnpu.pack_f16` narrows as upstream x86.
- **#65** (#85): creating an mps device panicked on macOS 11 to 14 while the
  wheel is tagged `macosx_11_0_arm64`; it now runs without a residency set.
- **#93** (#94): cpu `float32([0.1]) == 0.1` was False and a 0-d float64 operand
  made the result float64; both now equal upstream (True; float32).
- **#97** (#98): mps `clone()` aliased its source, so `y = x.clone(); y.add_(1)`
  changed `x`; clone now copies into a fresh buffer. 20 of 22 cases failed
  before.
- **#99** (#102): on mps, `w[0] = 9` and `w[w > 1] = 0` raised device-mismatch
  errors; both now compute on the device and equal upstream mps.
- **#33** (#82): cpu `sort`/`argsort`/`topk` ordered ties stably on Apple and
  Android; the order now equals upstream's libc++ order (10,719 of 22,728 cells
  were red before).
- **#78** (#79): a forward on one thread and `.backward()` on another raised a
  misnamed "backward through the graph a second time"; it now gives upstream's
  gradient.
- **#76, part 1** (#77): grad mode and the `torch.device(...)` mode stack were
  process-wide; they are now per thread, as upstream's are.
- **#31** (#80): a 0-d tensor operand raised the result dtype; the dtype now
  equals `torch.result_type`.
- **#69** (#90, #103, and the pinning commits): the x86_64 Linux golden run
  died or failed without naming its cases; CI now prints every failing and
  every pinned case by name, and 72 x86_64 divergences are pinned by exact
  case name (§3).
- **#71, #86** (#88): the CUDA wheel build failed on every branch since
  2026-10-04 (`__hmax_nan` redefined under CUDA 12.6); it builds again. #71 was
  closed as the same defect as #86.
- **#72** (#73): the publish gate ran on `main`'s reduced layout, where
  `AGENTS.md` is absent, and 112 suites failed; it now gates the release's
  source commit on develop. #73 is already in the `v0.1.0b5` tag; it is listed
  here because the 0.1.0b5 notes predate it.

## 3. Measured but not implemented

- **f32 `sin`/`cos`/`exp` on mps** differ from upstream cpu by about 1 ulp
  (#101). The open question is the oracle: upstream mps also runs Metal
  transcendental kernels, so the reference may be upstream mps, with a derived
  tolerance. Not graded in this release.
- **#76 remainder:** the `CAPTURING` flag is still process-wide, and the RNG
  stream free race is still open. The 3.15t thread hammer does not catch the
  race (it stays green on the old code); a cargo test does.
- **#81, Windows half:** MSVC STL tie order is not transcribed. Windows keeps
  the stable order, which diverges from Windows upstream on tied rows of 24 or
  more elements.
- **`index_put_` into an int32 `self` on mps** still refuses by name: it is not
  in the int32 widened-op table (#104).
- **72 x86_64 golden pins** (`KNOWN_X86_64_LINUX` in `tests/golden/compare.py`
  on develop). Each is a **pinned divergence, not an agreement**: on x86_64
  Linux the shim differs from upstream there, and the pin keeps the gate
  readable without hiding the case name (#69). In CI run 38982163472 at the
  develop head, 60 of the 72 fired. None of the three `sort`/`argsort` tie pins
  fired: after #92 the shim follows libstdc++'s order on Linux, so those three
  are stale and are to be retired.
- **`torch.compile` is still a permanent refusal**, for the structural reason in
  [`../graph/COMPILE.md`](../graph/COMPILE.md).
- **82 of 297 architectures** were never numerically judged. That count is
  [`../architectures/ARCH100.md`](../architectures/ARCH100.md)'s, it measured
  reachability, and it was not re-measured for this release.
- **Metal argsort on int8** runs the i16 kernel on a 1-byte buffer in the
  candle fork (#91). It is latent: `sort` is refused on mps before candle is
  reached.
- **`torch.compile` is still a permanent refusal**, for the structural reason in
  [`../graph/COMPILE.md`](../graph/COMPILE.md).
- The 3.15t gate has not run, and `Py_mod_gil` is not declared (#51).

## 4. Documentation corrected

- `docs/devices/CUDA.md` §6 step 9 records upstream CUDA's f16 narrowing rule,
  and that candle-kernels' casts are unmeasured (#87).
- The mps host-readback set is 90, since `aten.sort.stable` reads back like
  `sort.default`; README, README_ko and STATUS were updated (#82).
- `docs/devices/matrix.md` §7.21 records `index_put_` on mps (#104).

## 5. Gate at this head

For this release the local gate is replaced by the CI numbers below: they were
measured at the same head, develop 44e53e9.

GitHub Actions, CI run 38982163472:

```
Linux (gate, device-free, x86_64):
  cargo test (torch_c)  51 passed, 0 failed
  SUITE LEDGER          suites=139/139 ok=1805 FAIL=0 SKIP=326
  golden                12455/12515 cases passed, 0 failed, x86_known=60
  golden self-test      PASS (26 comparators x 12 fault modes, 0 problems)
  DOCWATCH              1557 markers: PASS=1527 SKIP=30
macOS (VM, no GPU claim, arm64):
  cargo test (torch_c)  51 passed, 0 failed
  SUITE LEDGER          suites=139/139 ok=1959 FAIL=0 SKIP=172
  golden self-test      PASS (26 comparators x 12 fault modes, 0 problems)
  DOCWATCH              1557 markers: PASS=1527 SKIP=30
macos-14 job (mps without MTLResidencySet): success
```

## 6. Platform status

Unchanged from [`RELEASE_0_1_0b3.md`](RELEASE_0_1_0b3.md) §6 for the abi3
wheels, except for the rows below.

| platform | grade | what changed in 0.1.0b6 |
|---|---|---|
| macOS arm64, cp315t | **reaches** | installed into a 3.15t venv and computed `torch.ones(2)+1`; GIL kept (#51) |
| Linux x86_64, cp315t | **builds** | built in dry run 38867632114; not run |
| Linux aarch64, cp315t | **builds** | built in dry run 38867632114; not run |
| Windows, iOS, Android, wasm, cp315t | refused by name | no free-threaded target CPython (#51) |
| mps (macOS 15+) | **agrees** on the measured cells | continuous batching greedy token-equal to upstream mps, 8 of 8 configurations (#100) |
| mps (macOS 11 to 14) | no panic | device created without a residency set; the `macos14` CI job may lack a GPU (#85) |
| CUDA | **builds** | the build is green again (build-cuda-wheel run 38789561664 on #88); no CUDA value was measured |
