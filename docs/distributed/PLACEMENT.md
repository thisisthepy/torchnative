# PLACEMENT: capability-weighted layer ranges for pipeline ranks

Status: **builds**. The code and its suite exist; the suite has not been run (AGENTS.md rule 10 sends test
runs to CI), so nothing here is graded *agrees* or *reaches*. Issue #296, BACKEND.md §9 step 17, placement half.

## Provenance

- Leader decision D11 (BACKEND.md §3): uneven devices are balanced with pipeline parallelism first; uneven tensor
  parallelism is listed, not scheduled.
- BACKEND.md §5.4: "give each stage as many layers as its memory and speed justify"; weights per rank (e.g. 5:3:2)
  chosen from a capability record (free memory, measured matmul throughput) that the caller can also pass by hand.
- Everything below that is not that sentence, including the API shape, the objective's exact form and the
  tie-breaking, is an **agent proposal, not confirmed**.

## What it is

`torchnative/python/torchnative/distributed/placement.py`, pure Python, deterministic, no torch import in the
module. `place_layers(layer_costs, capabilities, *, layer_bytes=None, fixed_bytes=None)` returns one half-open
`(start, stop)` per rank, in rank order. `Capability(rank, memory_bytes=None, throughput=None, weight=None)`
describes a rank. `stage_times` and `bottleneck` expose the quantity being minimised so a test can check it.

Importing `torchnative.distributed.placement` runs `torchnative/distributed/__init__.py`, which imports torch.
The test therefore loads the file by path. Whether the package init should stop importing torch is a separate
question and is not decided here.

## Objective

Minimise the maximum stage time, where a stage's time is the sum of its layer costs divided by the rank's speed
(`weight` if given by hand, `throughput` if measured; one kind per call, never mixed). Ranges are contiguous in
rank order, every stage is non-empty, and each stage's bytes (layer bytes plus that rank's `fixed_bytes`, for
example the embedding on rank 0 and the lm_head on the last rank) stay within that rank's `memory_bytes`.

## Algorithm and why it is exact

`g[i][j]` is the least bottleneck when ranks `i..R-1` must cover layers `j..L-1`; `g[R][L] = 0` and
`g[i][j] = min over k of max(time(i, j, k), g[i+1][k])` over stops `k` whose stage fits rank `i`'s memory. The
bottleneck of a split is the max of its first stage and the bottleneck of the rest, and swapping the rest for a
better split of the same suffix cannot raise that max, so the recurrence is the optimum by induction on `R - i`.
Arithmetic is `fractions.Fraction` built from each input's exact binary value: no rounding, no tolerance. Cost is
O(R * L^2). Memory limits are part of the stage-fit test, so feasibility and optimality are decided together.

Tie-break: among optimal splits, the smallest stop for the earliest rank, giving the lexicographically smallest
stops. The result does not depend on the order the capabilities were passed in.

## What is refused (`PlacementRefused`, a `ValueError`)

More ranks than layers; no split fits the memory limits (the message names the rank, its limit and the bytes
needed); non-positive or non-finite costs, weights or throughputs; a capability with neither throughput nor
weight, or with both, or ranks mixing the two kinds; mismatched lengths (`layer_bytes` against `layer_costs`);
memory given without `layer_bytes`; `fixed_bytes` naming a rank that does not exist; duplicate or missing ranks;
empty layers or capabilities.

## What is not promised

Raising one rank's speed never raises the bottleneck (tested), but a rank's range is not promised to grow
monotonically: with ties and a bottleneck set by another stage it can move. The test checks the bottleneck property
and a hand case, not a range-monotonicity theorem.

## Tests and nullification

`tests/distributed/test_placement.py`: hand-worked splits (even, 5:3:2, a slow rank, memory forcing a different split
than speed alone, fixed bytes), a brute-force enumeration of every contiguous partition on 600 seeded random inputs
(optimality, feasibility, and refusal exactly when none is feasible), determinism, and every refusal by name. Planned
nullifications, not yet observed: an even split must turn the brute-force and hand-case tests red; `min` replaced by
`max` in the recurrence must turn the brute-force test red; dropping the memory check must turn the memory cases and
the feasibility assertion red.

## Not part of this item

The measured latency and bytes table that BACKEND.md §9 step 17 says replaces §7's arithmetic. That is a measurement
record and needs devices; this module only decides the ranges.
