# `aten._scaled_dot_product_flash_attention_for_cpu` in LLM decode on the CPU: 220 us per call for 340 thousand multiply-adds, and the fused row that answers it

Written 2026-11-21 on branch `feat/sdpa-decode-cpu`, from source and from one CI measurement.
**Nothing in this document was measured on this machine.** The numbers are Gemstone's (issue
#74, run 44874740472, Intel Xeon Platinum 8573C, hosted `ubuntu` runner with 4 vCPUs,
SmolLM2-135M `float32`, torchnative 0.1.0b11 on CPython 3.15t against upstream torch 2.15 on
3.13), and everything the kernel claims about speed is a prediction until
`tests/bench/bench_sdpa_decode.py` runs in CI (§6). The kernel is an agent proposal on the
LLM performance line the owner asked for (torchnative is the LLM path), not an owner decision
about the means (AGENTS.md §7).

---

## 0. The evidence

Dispatch timer, one batch-4 decode step on the Xeon, top operators by self time (run
44874740472, round 1, torchnative 0.1.0b11):

| operator | self ms | calls |
|---|---:|---:|
| `aten.mm.default` | 50.6 | 211 |
| `aten._scaled_dot_product_flash_attention_for_cpu.default` | 6.4 | 30 |
| `aten.view.default` | 2.9 | 361 |
| `aten.mul.Tensor` | 2.4 | 273 |

The whole step is 75.0 ms (upstream: 34.4 ms). At batch 1 the thirty attention calls are 4.9 ms
of 48.0 ms. The matmul line is docs/perf/MM_DECODE.md's; this document is the second line.

What one call computes: SmolLM2-135M has 9 query heads over 3 key/value heads, head dimension
64; at batch 4 against a 74-key cache a call is `4 x 9 x 74 x 64 x 2 = 341` thousand
multiply-adds, which is a few microseconds of arithmetic on one core. 220 us per call is
overhead.

## 1. Where the 220 us go

`aten.rs::sdpa_flash_cpu`'s default path (the reference kernel `crate::flash` is opt-in and
twenty times slower) spells the attention out in candle ops. Per call:

| step | ops | what it allocates at batch 4, 74 keys |
|---|---|---|
| widen and `contiguous` the query, key and value | 3 | the query always (the model passes the transposed `(batch, len, heads, dim)` view) |
| `repeat_kv_heads` for the 3 to 9 grouped-query heads | `unsqueeze`, `expand`, `reshape`, `contiguous`, twice | two copies of the cache |
| `transpose(2, 3)` and `transposed_contiguous` of the key | 2 | a third copy of the key |
| `q.matmul(kt)` | one batched matmul: 36 `gemm` entries of (1 x 64) by (64 x 74) | the scores |
| `affine(scale, 0.0)` | 1 | the scores again |
| `broadcast_add(mask)` | 1 | the scores again |
| `amax`, `broadcast_sub`, `exp`, `sum_keepdim`, `broadcast_div`, `contiguous` | 6 | four of them |
| `p.matmul(v)` | a second batched matmul, 36 `gemm` entries | the output |
| `log`, `broadcast_add`, `squeeze` | 3 | the logsumexp |

About fifteen allocations and seventy-two `gemm` entries, each entry packing its operands for a
product that is one vector against a 74 x 64 block. The arithmetic is a few microseconds; the
rest is the op sequence.

What is **not** the cost, checked rather than assumed: the dispatch of the one aten call (about
8 us on that runner, DISPATCH2.md's number scaled by the Xeon's per-op cost of `view`); the
mask (one add); the reference kernel (off).

## 2. What was changed

`torchnative/rust/torch_c/src/sdpa_decode.rs` (new), reached from `sdpa_flash_cpu` before the
widening and repeating copies, so that avoiding them is the point, and after every argument
check, so that what the op accepts and refuses is unchanged.
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/sdpa_decode.rs try_attend present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/aten.rs sdpa_flash_cpu present -->

| what | how |
|---|---|
| claimed operands | query, key, value rank 4, `float32`, CPU, head dimension contiguous (any strides on the batch, head and row axes: the transposed query and a cache slice are read in place), 1 to 8 query rows, key length and head dimension at least 1, query heads a multiple of key/value heads, `is_causal` false, no dropout, a mask that is `None` or rank 4 `float32` CPU with every axis 1 or the full extent and its last axis contiguous. Everything else: the general path, unchanged |
| routed where | candle reports neither `accelerate` nor `mkl` (every non-Apple build). An Accelerate build keeps the general path: `tests/_support/test_shim.py::test_sdpa_wide_float_is_close_but_not_promised_exact` holds that path to a measured 1e-6 of upstream's Accelerate-backed answer on the same machine, and this kernel was not measured against that bound. The reference kernel, when switched on, keeps its path |
| the row | for each `(batch, head, query row)`: `score[j] = dot(q, k[j]) * scale + 0.0 + mask[b, h, i, j]`, the dot product in 8 partial sums (product `t` into sum `t mod 8`, fused multiply-add where the target has one, lanes added in order); the row maximum (NaN if any score is NaN, as `amax` answers); `w[j] = exp(score[j] - max)` with the standard library's `exp`; `sum` in key order; `out[d] += (w[j] / sum) * v[j][d]` in key order; `logsumexp = ln(sum) + max`. Grouped-query heads are `head / n_rep`, an index, never a copy |
| threads | candle's persistent `BarrierPool` (`utils::barrier_pool`, the pool the quantised matmul, `sort` and the skinny matmul of MM_DECODE.md split over), rows split into one contiguous range per participant when the call is at least `TORCHNATIVE_SDPA_DECODE_THREAD_MADDS` multiply-adds (default 65536: the batch-4 decode call above is 341 thousand, one sequence against a 20-key cache 23 thousand). A reading of the wake cost against the work, **not a measurement**; the bench sweeps it |
| counters | `_C._sdpa_decode_counters()`: `routed`, `enabled`, `calls`, `threaded_calls`, `threads`, `thread_madds`, `simd`. A faster number is not evidence the kernel ran (AGENTS.md §16); the counts are |
| off switch | `TORCHNATIVE_SDPA_DECODE=0`, read once at import. Same binary, general path, for an A/B |

### 2.1 What this cannot change, and what it changes

Stated so a test can pin it (`tests/ops/test_sdpa_decode.py`):

* a row's result depends on its own query, its head's cache and its mask row only: a batch-b
  call equals the b batch-1 calls **bit for bit**, a q_len-l call equals the l one-row calls,
  and grouped-query heads equal the same call with the cache repeated (tested);
* the row split decides which thread computes a row, never how, so **the thread count cannot
  move a bit** (subprocesses at 1 thread, the default, and a forced split of every call);
* a fully masked row answers NaN in the output and the logsumexp, exactly as the general path
  does (`exp(-inf - -inf)`), because that row is what a causal mask plus padding produces and
  `_softmax`'s documented answer is NaN;
* it is **not the general path's summation order** (that path sums the products in `gemm`'s
  blocked order and the weighted sum in a second GEMM, and on a build with Accelerate its `exp`
  is `vvexpf`), so against the general path and against upstream the agreement is a stated
  float32 bound: `8 u (sqrt(d) |score|max + 2 sqrt(s) + 4) |v|max`, the dot product's
  accumulation amplified by the softmax, the exponentials and their sum, the division and the
  weighted sum. The golden harness holds the general path to 1e-5 of upstream at `float32`
  (`tests/golden/cases.py::_sdpa_pair_check`), and the shapes it compares at 8 query rows or
  fewer now take this kernel on the Linux gate.

## 3. What is predicted, and from what

Not measured. Reasoning only, to be replaced by §6's numbers:

* The fifteen allocations and seventy-two `gemm` entries become one pass per row over
  L1-resident data (the query row is 256 bytes, the head's cache 74 x 64 x 4 = 19 KB) and two
  output buffers. The arithmetic at batch 4 is a few microseconds; the kernel should bring the
  call from about 220 us to the order of the op's dispatch (about 10 us on that runner), so
  the thirty calls from 6.4 ms to well under 1 ms per step, about 7 % of the torchnative
  step. On the EPYC runner (Gemstone run 44877496387, step 59.8 ms) the share is similar.
* At batch 1 the same, from 4.9 ms.
* The split across the pool helps only when a call is far above the floor (long caches, many
  sequences); at 341 thousand multiply-adds on 4 threads the barrier is a measurable fraction
  of the work, which is why the floor is a knob.

## 4. What is left

| # | wall | what it needs |
|---|---|---|
| 1 | **The measurement.** This document has none of its own | §6 in CI, then rewrite §3 as a table |
| 2 | `is_causal` with 2 to 8 query rows (a short prompt's prefill without a mask) takes the general path; the shim's causal rule is upper-left aligned | claim it after #1, with the rule transcribed from `tensor.rs::scale_and_causal_mask` |
| 3 | Prefill (9 or more query rows) stays on the general path with its copies and batched GEMMs | a blocked variant, or the reference kernel's blocking made fast; decide after #1 |
| 4 | Accelerate builds are not routed | measure against the 1e-6 bound of `test_sdpa_wide_float_is_close_but_not_promised_exact` on a Mac before deciding |
| 5 | `bfloat16`/`float16` operands are widened to `float32` per call before this path would see them | unchanged by this round |
| 6 | The matmul line: 50.6 of 75.0 ms at batch 4 on the Xeon | docs/perf/MM_DECODE.md, torchnative #419 |

## 5. Report by category

| category | what |
|---|---|
| feature added | the routed `float32` decode attention (`sdpa_decode.rs`), `_C._sdpa_decode_counters()`, two environment switches |
| defect fixed | none claimed: the general path was correct; it was fifteen ops and seventy-two GEMM entries per call |
| test added | `tests/ops/test_sdpa_decode.py` (7 tests: counters, routing by count, float64 agreement with a stated bound, the fully masked row, bit-for-bit row independence, grouped heads against the repeated cache, thread-count and off-switch subprocess checks); `tests/bench/bench_sdpa_decode.py` (bench, not in the gate) |
| documentation corrected | docs/SPEC.md S7.6; this file |
| deleted | nothing |

## 6. How to measure it (CI, x86_64 Linux, alone)

```sh
# six decode shapes, upstream vs shim vs shim with the kernel off, interleaved
# rounds, counters per shape; JSON for the artefact
PYTHONPATH= python tests/bench/bench_sdpa_decode.py --python "$(which python)" --rounds 3 \
    --out "$RUNNER_TEMP/sdpa_decode.json"

# the split floor, without a rebuild
PYTHONPATH= python tests/bench/bench_sdpa_decode.py --python "$(which python)" \
    --sides shim --thread-madds 0,65536,1048576

# the decode step itself (SmolLM2-135M from the HF cache), before and after
python tests/bench/ab_upstream.py --suite model
TORCHNATIVE_SDPA_DECODE=0 python tests/bench/ab_upstream.py --suite model
```

Read `taken(shim)` before the times: a shape whose count is 0 was measured on the general path.
Record `uptime` and the runner's CPU model with the numbers.
