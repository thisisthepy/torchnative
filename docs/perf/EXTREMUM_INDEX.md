# The CPU index reductions in one pass

`torchnative/rust/torch_c/src/extremum_index.rs`, SPEC S7.8, `tests/ops/test_extremum_index.py`.
Agent proposal of 2026-11-22 on the LLM performance line (Gemstone #74); the owner has not
confirmed the choice.

## 0. The evidence

Gemstone's `batching_breakdown` on the Xeon 8573C runner (run 44874740472, SmolLM2-135M,
float32, batch 4, torchnative 0.1.0b11; the transformers continuous-batching manager with
`sdpa_paged`, round 2, medians): one `aten.argmax` per step, over the `[4, 49152]` logits.

| op | torchnative | upstream 2.15 |
|---|---:|---:|
| decode step | 82.8 ms | 39.3 ms |
| `aten.argmax.default`, 1 call | 1.24 ms | below the table's tenth row (0.48 ms) |

1.24 ms for 196,608 `float32` elements is 6.3 ns per element: not one read of the slice.

## 1. Where the 1.24 ms went

`first_extremum_index` (aten.rs) answers every index reduction (`argmax`, `argmin`, the
whole-tensor `max` and `min`, the index half of `max.dim` and `min.dim`) by composition from
candle ops, so that Metal, whose reduction seeds its fold with the finite lowest value and
breaks ties by the lower index, answers upstream's index (issue #29):

| pass | op | what it reads and writes |
|---|---|---|
| 1 | `max_keepdim` | the slice; one value per row |
| 2 | `broadcast_eq` | the slice; a byte per element |
| 3 | `argmax_keepdim` over the bytes | the bytes; one `u32` per row |
| 4 | `to_dtype(i64)` | the indices |
| 5 | `ne` (`x != x`) | the slice; a byte per element |
| 6 | `max_keepdim` over the NaN bytes | the bytes |
| 7 | `argmax_keepdim` over the NaN bytes | the bytes |
| 8 | `to_dtype(i64)`, `where_cond` | the indices |

Each reduction is a scalar loop in candle's `ReduceIndex` (`cpu_backend/mod.rs`), the
elementwise ops allocate, and the slice is read three times.

## 2. What was changed

`extremum_index::try_index` claims a `float32` or `float64` CPU tensor whose reduced axis is
the innermost contiguous one (every later axis of extent 1, stride 1 on the axis, a
contiguous layout) and reads each slice once, in place: the first NaN's index if the slice
holds one, else the first index no later element beats (`>` for the maximum, `<` for the
minimum: ties keep the lower index, an all-`-inf` slice answers 0, `-0.0` and `0.0` compare
equal so the first of them wins). The decision is made on layout and dtype only, so a call is
whole on one path or the other. Every other call keeps the composition, which is the
reference the test holds the one-pass path to, index for index, on the rule rows (ties, all
equal, all `-inf`, `+inf`, NaN first, in the middle, last and twice, signed zeros in both
orders) and on a batch of logits rows with injected ties, for all six ops and both dtypes.
`_C._extremum_index_counters()` counts both paths.

The answer does not change; what changes is that a `[4, 49152]` `argmax` is one read of
786 KB instead of eight passes and five allocations.

## 3. What is predicted, and from what

One scalar pass with two compares per element over 196,608 elements, on one thread: about
0.1 to 0.2 ms on the Xeon runner (the composition's 1.24 ms divided by its eight passes gives
0.15 ms for one). That is a reading, not a measurement; the next Gemstone
`batching_breakdown` run's `aten.argmax` row is the measurement, and the step is 82.8 ms, so
the whole gain is about 1.3 % of it. It is here because it was the one row of that table
outside `mm` (#419) and the flash SDPA (#424) where torchnative paid more than twice
upstream for one call.

## 4. What is left

* An integer slice (`int64` logits never occur on this line) keeps the composition.
* A reduced axis that is not the innermost (`argmax(dim=0)` of a matrix) keeps the
  composition; a kernel over a strided axis would read the slab once per column group.
* Threads: one `[4, 49152]` slice set is below any split threshold worth its wake.

## 5. Report by category

| category | item |
|---|---|
| feature added | `extremum_index.rs`: the one-pass CPU path, hooked in `first_extremum_index`; `_C._extremum_index_counters()` |
| defect fixed | none |
| test added | `tests/ops/test_extremum_index.py` (5 tests, registered BOTH) |
| documentation corrected | docs/SPEC.md S7.8; this file |
| deleted | nothing |
