# `aten.mm` in LLM decode on the CPU: why 211 small matmuls cost 1.6x upstream, and the kernel that answers it

Written 2026-11-21 on branch `feat/mm-decode-perf`, from source and from one CI measurement.
**Nothing in this document was measured on this machine.** The only numbers are Gemstone's
(issue #74, run 44792524320, AMD EPYC 7763, `ubuntu` hosted runner, SmolLM2-135M `float32`,
torchnative 0.1.0b9 on CPython 3.15.2t against upstream torch on 3.13), and everything the fix
claims about speed is a prediction until `tests/bench/bench_skinny_mm.py` runs in CI (§6).
Owner's direction behind the task: torchnative is the LLM path; LiteRT and ONNX are benchmark
competitors only. The kernel below is an agent proposal to reach that goal, not an owner
decision about the means (AGENTS.md §7).

---

## 0. The evidence, and what it does and does not say

| layer | torchnative | upstream | ratio |
|---|---:|---:|---:|
| raw forward, one decode step, batch 1 | 72.6 ms | 44.7 ms | 1.6x |
| raw forward, one decode step, batch 4 | 122.8 ms | 76.8 ms | 1.6x |

The dispatch timer on torchnative attributes **90.8 ms of 122.9 ms** at batch 4 to `aten.mm`
(211 calls per step; next SDPA 8.6 ms, `view` 4.1, `mul` 3.1). The `t` and `_unsafe_view`
counts are 211 each too: every `F.linear` is lowered to `t, view, mm, _unsafe_view`, upstream's
own no-bias N-D branch (`bootstrap.py::linear`, `_cia_matmul`'s fold).

What this does not say: upstream's own `mm` time on that runner. The 76.8 ms is upstream's whole
step; its matmul share was not broken down. So "1.6x per call" is an upper bound on the kernel
gap, not a measurement of it.

## 1. How `aten.mm` reaches the CPU kernel on that runner

`mm_default` (`aten.rs`) widens nothing for `float32`, calls `gemm_with_layout_fallback` with
`gemm_multiply`, which is `Tensor::matmul`. In candle's `cpu_backend/mod.rs` the `MatMul` map has
three bodies, chosen at build time: Accelerate (`feature = "accelerate"`, every Apple build of
this crate), MKL (never enabled here), and otherwise the **`gemm` crate** (`gemm = "0.19.0"`,
runtime-dispatched microkernels, rayon threading). Linux, Windows and Android are the third.

The operands at decode time, per projection of SmolLM2-135M (hidden 576, intermediate 1536,
vocabulary 49152, 30 layers), after the fold:

| projection | `lhs` rows x k | `rhs` k x n (a `t(weight)` view: row stride 1, column stride k) | calls per step |
|---|---|---|---:|
| q_proj, o_proj | batch x 576 | 576 x 576 | 60 |
| k_proj, v_proj | batch x 576 | 576 x 192 | 60 |
| gate_proj, up_proj | batch x 576 | 576 x 1536 | 60 |
| down_proj | batch x 1536 | 1536 x 576 | 30 |
| lm_head | batch x 576 | 576 x 49152 | 1 |

Weight bytes streamed per step: 538 MB (134.5 M parameters). At batch 4 that is 4 multiply-adds
per weight element: the step is a memory stream, not arithmetic.

What upstream does with the same call (from source, torch 2.15 CPU wheel, not verified on the
runner): `addmm_impl_cpu_` hands `float32` to `cpublas::gemm`, which is MKL's `sgemm` in the
pip wheel, with `transb` set from the transposed stride (no copy) and MKL's own OpenMP
threading over the intra-op thread count.

## 2. The concrete reasons, read from `gemm-common-0.19.0/src/gemm.rs` and `gemm-f32-0.19.0`

1. **Batch 1 is single-threaded by construction.** `m <= 1 && rhs_rs == 1 && lhs_cs == 1` goes to
   `gemv::mixed_gemv_rowmajor`, which takes no `Parallelism` argument. All 211 calls at batch 1
   run on the calling thread. docs/perf/PERF_ANDROID.md §4.5 recorded this ("gemv has no
   threading path") and concluded decode shapes were out of scope of the threading threshold;
   that is true of the threshold and is exactly the problem here.
2. **Batch 4 is single-threaded by the threshold.** The blocked GEMM threads a chunk only when
   `m * n_chunk * k_chunk >= get_threading_threshold()`, and `lib.rs` sets that to 4 M (measured
   on square shapes, where threading below it loses). At batch 4 every projection is below it:
   q/o 1.3 M, k/v 0.44 M, gate/up/down 3.5 M. Only `lm_head` (113 M) is split. So 210 of 211
   calls stream their weight on one core. The threshold is a cost model for arithmetic applied
   to a shape whose cost is bandwidth.
3. **The microkernel tile is mostly padding.** `gemm-f32`'s `fma` kernels are `MR = 2 * 8 = 16`
   rows by up to `NR = 6` columns (`MR = 64` with AVX-512; the EPYC 7763 has AVX2, not
   AVX-512). With 4 rows, 12 of 16 row lanes of every fused multiply-add are zero, and
   `do_prepack_lhs` packs the 4 rows into that tile on every call.

What is **not** a reason, checked rather than assumed:

* The transposed weight is not materialised. `rhs_rs == 1` is passed through; on x86 the
  `do_pack_rhs` condition is `rhs_rs == 1 && m > 128 * MR`, false at any decode row count.
  (docs/perf/LINEAR.md removed the copies at the shim level in 2026-08.)
* The output allocation is `rows * n` floats (24 KB for gate/up at batch 4); `t` and
  `_unsafe_view` are about a microsecond each (docs/perf/DISPATCH2.md §2): under 1 ms per step
  together.
* There is no fused bias to lose: SmolLM2's `Linear` layers have no bias, so no `addmm` runs.

## 3. What was changed

`torchnative/rust/torch_c/src/skinny_gemm.rs` (new), reached from `gemm_multiply` in `aten.rs`
before `Tensor::matmul`, so `mm`, `addmm` and the fold of `matmul` all take it and `bmm`,
`baddbmm` and the broadcasting `matmul` do not.
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/skinny_gemm.rs try_mm present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/aten.rs gemm_multiply present -->

| what | how |
|---|---|
| claimed operands | both rank 2, `float32`, CPU, 1 to 8 rows, depth >= 32, `lhs` column stride 1, `rhs` row stride 1 (the `t(weight)` layout; a row-major `rhs` is not claimed). Everything else: candle, unchanged |
| routed where | candle reports neither `accelerate` nor `mkl` (`candle_core::utils::has_accelerate/has_mkl`), i.e. the `gemm` builds. Apple builds with Accelerate keep `sgemm`, already at 0.94 to 1.05x upstream at these shapes (docs/perf/LINEAR.md §5.1) |
| the stream | per output column, the weight column (contiguous, k floats) is read once from memory while the 1 to 8 activation rows (L1-resident) are reused: 8-lane accumulators per row, product `t` into lane `t mod 8`, fused multiply-add where the target has it (`avx2,fma` detected at runtime on x86_64, NEON on aarch64; separate multiply and add otherwise), lanes added in order at the end |
| threads | candle's persistent `BarrierPool` (`utils::barrier_pool`), the pool the quantised matmul already tiles over (issue #177). Output columns are split into one contiguous range per participant when the weight is at least `TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES` (default 256 KiB: below every SmolLM2 projection, 442 KB for k/v). The calling thread is a participant |
| counters | `_C._skinny_gemm_counters()`: `routed`, `enabled`, `calls`, `threaded_calls`, `threads`, `thread_bytes`, `simd`. A faster number is not evidence the kernel ran (AGENTS.md §16); the counts are |
| off switch | `TORCHNATIVE_SKINNY_GEMM=0`, read once at import. Same binary, general path, for an A/B |

### 3.1 What this cannot change, and what it changes

The arithmetic is stated so a test can pin it (`tests/ops/test_skinny_gemm.py`):

* an element's value depends on its row and column only, so an m-row call equals the m
  one-row calls **bit for bit**; the general path does not have this property (one row is a
  gemv kernel, more rows the blocked GEMM);
* the column split decides which thread computes an element, never how, so **the thread count
  cannot move a bit** (checked in subprocesses at 1 thread, the default, and a forced split of
  every call);
* it is **not `gemm`'s summation order** (`gemm` chains one fused multiply-add per `t` within a
  `k` chunk and adds chunks), so against the path it replaces, and against upstream, the
  agreement grade is the golden harness's scale-aware GEMM bound (`_gemm_scale_check`,
  `4 * u * sqrt(k) + 2 * u` of the largest output), which is the grade upstream's own x86 wheel
  meets against the shim at these depths (the `_GEMM` x86 pins in `tests/golden/compare.py`).
  The depth floor of 32 keeps every bit-exact golden case (`k` of 3 and 4, transposed and
  folded) on the path it was on, so those cannot move.

Golden on the x86_64 runner: the one model-scale case this path claims is `matmul(float16,
(1, 6, 512) x t((8, 512)))` (6 rows, widened to `float32`), checked against the bound. The
`float32` model-scale cases have 32 folded rows and stay on `gemm`.

## 4. What is predicted, and from what

Not measured. Reasoning only, to be replaced by §6's numbers:

* At batch 4 the 210 projection calls go from one core's stream to `threads` cores' streams,
  where `threads` is candle's participant count (physical cores, `CANDLE_NUM_THREADS`). On a
  hosted runner with two physical cores that is at most 2x on 85 % of the `mm` time; on a
  larger runner more.
* At batch 1 the same, where before nothing was split at all.
* The lane use goes from 4 of 16 to 4 of 8 (AVX2) and the per-call `lhs` packing disappears.
  This matters only where a call is not bandwidth-bound (weights resident in L2/L3, which 538
  MB per step are not); expect little from it.
* The barrier costs one wake-up per split call. Workers spin 10 000 iterations before parking,
  so between consecutive projections of one step they are still spinning; between steps they
  may have parked. The 256 KiB floor is where a few tens of microseconds of streaming meet
  that cost. It is a reading, not a measurement; the bench sweeps it.

## 5. What is left

| # | wall | what it needs |
|---|---|---|
| 1 | **The measurement.** This document has none of its own | §6 in CI, then rewrite §4 as a table |
| 2 | 9 to 64 rows (prefill of a short prompt, batch 16 decode) still take the `gemm` tile at 4 to 64 of `MR` rows and the 4 M arithmetic threshold | a row-blocked variant of this kernel, or a bandwidth-aware threading threshold; decide after #1 |
| 3 | `bfloat16`/`float16` weights are widened to `float32` per call (docs/perf/LINEAR.md §6.1) before they reach this path | unchanged by this round |
| 4 | Accelerate builds are not routed | measure `sgemm` against this kernel at these shapes on an M1 before deciding; the LINEAR.md numbers say there is nothing to gain at batch 1 |
| 5 | Concurrent callers (free-threaded Python, Gemstone's engine) serialise on the pool's call lock, as the quantised matmul does | a measurement with two generating threads; nothing here makes it worse than the quantised path |
| 6 | Gemstone's step is 235.7 ms against the raw forward's 122.8 ms: about 104 ms per step above the model on torchnative against 22 ms upstream (#74) | not a kernel; Gemstone's own per-step work, named in the issue as the first target |

## 6. How to measure it (CI, x86_64 Linux, alone)

```sh
# the five decode shapes x batch 1 and 4, upstream vs shim vs shim with the kernel off,
# interleaved rounds, counters per shape; JSON for the artefact
PYTHONPATH= python tests/bench/bench_skinny_mm.py --python "$(which python)" --rounds 3 \
    --out "$RUNNER_TEMP/skinny_mm.json"

# the split floor, without a rebuild
PYTHONPATH= python tests/bench/bench_skinny_mm.py --python "$(which python)" \
    --sides shim --thread-bytes 0,262144,1048576,4194304

# the decode step itself (SmolLM2-135M from the HF cache), before and after
python tests/bench/ab_upstream.py --suite model
TORCHNATIVE_SKINNY_GEMM=0 python tests/bench/ab_upstream.py --suite model
```

Read `taken(shim)` before the times: a shape whose count is 0 was measured on the general path.
Record `uptime` and the runner's CPU model with the numbers.

## 7. Report by category

| category | what |
|---|---|
| feature added | the routed `float32` small-row CPU matmul (`skinny_gemm.rs`), `_C._skinny_gemm_counters()`, two environment switches |
| defect fixed | none claimed: the general path was correct; it was single-threaded at the decode shapes |
| test added | `tests/ops/test_skinny_gemm.py` (6 tests: routing by counter, the fold door, bound agreement, bit-for-bit row independence, thread-count and off-switch subprocess checks); `tests/bench/bench_skinny_mm.py` (bench, not in the gate) |
| documentation corrected | docs/SPEC.md S7.5; this file |
| deleted | nothing |
