"""Capability-weighted placement of contiguous layer ranges on pipeline ranks.

BACKEND.md §5.4 and D11: pipeline parallelism is the uneven-device tool, "give
each stage as many layers as its memory and speed justify". This module is the
pure function that decides how many. It moves no tensor, opens no socket and
imports no ``torch``: the input is numbers, the output is half-open ranges.
(The sibling ``torchnative.distributed`` package imports torch when it is
imported; load this file by path where torch must stay out, as the test does.)

Status: agent proposal, not confirmed (the API shape below), graded *builds*
until CI runs ``tests/distributed/test_placement.py``. The measured
latency/bytes table of BACKEND.md §9 step 17 is a different item and is not
part of this module.

API::

    Capability(rank, memory_bytes=None, throughput=None, weight=None)
    place_layers(layer_costs, capabilities, *, layer_bytes=None,
                 fixed_bytes=None) -> [(start, stop), ...]   # rank order
    stage_times(ranges, layer_costs, capabilities) -> [Fraction, ...]
    bottleneck(ranges, layer_costs, capabilities) -> Fraction

Model. Layer ``i`` costs ``layer_costs[i]`` (any positive unit, e.g. FLOPs or
measured milliseconds on a reference device). Rank ``r`` runs at speed
``weight`` (a hand-given ratio such as 5:3:2) or ``throughput`` (measured, in
cost units per second); one capability must give exactly one of the two, and
all ranks must use the same kind, because a weight and a throughput are not
comparable. A stage's time is ``sum(costs in its range) / speed``.

Objective. Contiguous ranges in rank order (rank 0 holds the first layers),
every stage non-empty, each stage's bytes (``sum(layer_bytes in range)`` plus
that rank's ``fixed_bytes``, e.g. the embedding on rank 0 and the lm_head on the
last rank) at most that rank's ``memory_bytes`` when given. Among all such
splits, minimise the **maximum stage time** (the pipeline's steady-state
bottleneck).

Algorithm (exact). ``g[i][j]`` is the least achievable bottleneck when ranks
``i..R-1`` must cover layers ``j..L-1``. ``g[R][L] = 0``; otherwise
``g[i][j] = min over k of max(time(i, j, k), g[i+1][k])`` for every ``k`` where
the stage ``[j, k)`` fits rank ``i``'s memory. This is the standard optimal
substructure of min-max contiguous partition: the bottleneck of a split is the
max of its first stage and the bottleneck of the rest, and replacing the rest
by a better split of the same suffix cannot raise that max. So the table holds
the true optimum, by induction on ``R - i``. All arithmetic is on
``fractions.Fraction`` built from the exact binary value of each input, so
there is no rounding and no tolerance: two splits tie only when they are equal
as rational numbers. Cost is O(R * L^2) time, O(R * L) memory.

Tie-breaking (deterministic). Among splits with the optimal bottleneck, the
reconstruction walks ranks in order and picks the smallest stop ``k`` that
still reaches the optimum, so the result is the lexicographically smallest
sequence of stops: earlier ranks get the shorter range when the bottleneck
does not care. The output does not depend on the order ``capabilities`` was
passed in (they are sorted by rank).

Not promised: that raising one rank's speed never shrinks its range. The
bottleneck never increases, but with ties and a bottleneck set by another
stage a range can move; the tests check the bottleneck property and hand
cases, not a range-monotonicity theorem.

Refusals. Everything unsupported raises ``PlacementRefused`` (a ``ValueError``)
naming the cause (AGENTS.md §18). It never returns a best-effort split.
"""

import math
from fractions import Fraction
from numbers import Real

__all__ = ["Capability", "PlacementRefused", "place_layers", "stage_times", "bottleneck"]


class PlacementRefused(ValueError):
    """The request cannot be placed; the message names why."""


class Capability:
    """What one pipeline rank can do. ``rank`` is its index in the pipeline.

    ``memory_bytes`` None means unlimited. Exactly one of ``throughput`` and
    ``weight`` must be given (both positive and finite).
    """

    __slots__ = ("rank", "memory_bytes", "throughput", "weight")

    def __init__(self, rank, memory_bytes=None, throughput=None, weight=None):
        self.rank = rank
        self.memory_bytes = memory_bytes
        self.throughput = throughput
        self.weight = weight

    def __repr__(self):
        return (f"Capability(rank={self.rank!r}, memory_bytes={self.memory_bytes!r}, "
                f"throughput={self.throughput!r}, weight={self.weight!r})")


def _positive(value, what):
    if isinstance(value, bool) or not isinstance(value, (Real, Fraction)):
        raise PlacementRefused(f"{what} must be a real number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise PlacementRefused(f"{what} must be finite, got {value!r}")
    f = Fraction(value)
    if f <= 0:
        raise PlacementRefused(f"{what} must be positive, got {value!r}")
    return f


def _bytes(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlacementRefused(f"{what} must be a non-negative int, got {value!r}")
    return value


def _normalise(layer_costs, capabilities, layer_bytes, fixed_bytes):
    """Validate, and return (costs, speeds, memory, layer_bytes, fixed) by rank."""
    costs = [_positive(c, f"layer_costs[{i}]") for i, c in enumerate(layer_costs)]
    n_layers = len(costs)
    if n_layers == 0:
        raise PlacementRefused("layer_costs is empty: there is nothing to place")
    caps = list(capabilities)
    if not caps:
        raise PlacementRefused("capabilities is empty: there is no rank to place on")
    ranks = [c.rank for c in caps]
    for r in ranks:
        if isinstance(r, bool) or not isinstance(r, int) or r < 0:
            raise PlacementRefused(f"rank must be a non-negative int, got {r!r}")
    if len(set(ranks)) != len(ranks):
        dup = sorted({r for r in ranks if ranks.count(r) > 1})
        raise PlacementRefused(f"duplicate rank(s) {dup} in capabilities")
    n_ranks = len(caps)
    if sorted(ranks) != list(range(n_ranks)):
        missing = sorted(set(range(n_ranks)) - set(ranks))
        raise PlacementRefused(
            f"ranks must be exactly 0..{n_ranks - 1}; missing {missing}, got {sorted(ranks)}")
    if n_ranks > n_layers:
        raise PlacementRefused(
            f"{n_ranks} ranks but only {n_layers} layers: every stage must be non-empty")
    caps.sort(key=lambda c: c.rank)

    kinds = set()
    speeds = []
    for c in caps:
        if c.weight is None and c.throughput is None:
            raise PlacementRefused(
                f"rank {c.rank} gives neither throughput nor weight, so its speed is unknown")
        if c.weight is not None and c.throughput is not None:
            raise PlacementRefused(
                f"rank {c.rank} gives both throughput and weight; give exactly one")
        kind, val = ("weight", c.weight) if c.weight is not None else ("throughput", c.throughput)
        kinds.add(kind)
        speeds.append(_positive(val, f"rank {c.rank} {kind}"))
    if len(kinds) > 1:
        raise PlacementRefused(
            "some ranks give weight and others throughput; the two are not comparable, use one kind")

    memory = []
    for c in caps:
        memory.append(None if c.memory_bytes is None
                      else _bytes(c.memory_bytes, f"rank {c.rank} memory_bytes"))

    if layer_bytes is None:
        limited = [c.rank for c, m in zip(caps, memory) if m is not None]
        if limited:
            raise PlacementRefused(
                f"rank(s) {limited} give memory_bytes but layer_bytes is None: "
                "the limit cannot be checked")
        lbytes = [0] * n_layers
    else:
        lbytes = [_bytes(b, f"layer_bytes[{i}]") for i, b in enumerate(layer_bytes)]
        if len(lbytes) != n_layers:
            raise PlacementRefused(
                f"layer_bytes has {len(lbytes)} entries but layer_costs has {n_layers}")

    fixed = [0] * n_ranks
    if fixed_bytes is not None:
        for r, b in dict(fixed_bytes).items():
            if isinstance(r, bool) or not isinstance(r, int) or not 0 <= r < n_ranks:
                raise PlacementRefused(f"fixed_bytes names rank {r!r}, which is not in 0..{n_ranks - 1}")
            fixed[r] = _bytes(b, f"fixed_bytes[{r}]")
    return costs, speeds, memory, lbytes, fixed


def place_layers(layer_costs, capabilities, *, layer_bytes=None, fixed_bytes=None):
    """Assign contiguous half-open ``(start, stop)`` layer ranges, one per rank.

    Returns the ranges in rank order. ``fixed_bytes`` maps rank to bytes that
    rank holds regardless of its layers (embedding, lm_head). The split
    minimises the maximum stage time exactly; ties go to the lexicographically
    smallest stops (see the module docstring). Raises ``PlacementRefused``.
    """
    costs, speeds, memory, lbytes, fixed = _normalise(
        layer_costs, capabilities, layer_bytes, fixed_bytes)
    n_layers, n_ranks = len(costs), len(speeds)
    pc = [Fraction(0)]
    pb = [0]
    for c, b in zip(costs, lbytes):
        pc.append(pc[-1] + c)
        pb.append(pb[-1] + b)

    def fits(i, j, k):
        return memory[i] is None or pb[k] - pb[j] + fixed[i] <= memory[i]

    def time(i, j, k):
        return (pc[k] - pc[j]) / speeds[i]

    # g[i][j]: least bottleneck for ranks i.. covering layers j.. ; None = impossible.
    g = [[None] * (n_layers + 1) for _ in range(n_ranks + 1)]
    g[n_ranks][n_layers] = Fraction(0)
    for i in range(n_ranks - 1, -1, -1):
        rest = n_ranks - 1 - i                       # ranks after i, each needs a layer
        for j in range(i, n_layers - rest):
            best = None
            for k in range(j + 1, n_layers - rest + 1):
                tail = g[i + 1][k]
                if tail is None or not fits(i, j, k):
                    continue
                v = max(time(i, j, k), tail)
                if best is None or v < best:
                    best = v
            g[i][j] = best

    if g[0][0] is None:
        raise PlacementRefused(_explain_memory(memory, lbytes, fixed, n_layers, fits))

    ranges = []
    j = 0
    for i in range(n_ranks):
        rest = n_ranks - 1 - i
        for k in range(j + 1, n_layers - rest + 1):
            tail = g[i + 1][k]
            if tail is None or not fits(i, j, k):
                continue
            if max(time(i, j, k), tail) == g[i][j]:
                ranges.append((j, k))
                j = k
                break
        else:  # unreachable if g is right; refuse rather than invent a split
            raise PlacementRefused(f"internal error: no stop reaches the optimum at rank {i}")
    return ranges


def _explain_memory(memory, lbytes, fixed, n_layers, fits):
    """Name the rank and the bytes that make the memory limits infeasible."""
    n_ranks = len(memory)
    reach = {0}                                       # prefixes ranks 0..i-1 can cover
    for i in range(n_ranks):
        rest = n_ranks - 1 - i
        nxt = {k for j in reach for k in range(j + 1, n_layers - rest + 1) if fits(i, j, k)}
        if not nxt:
            need = min(lbytes[j] + fixed[i] for j in reach)
            return (f"no split fits the memory limits: rank {i} (limit {memory[i]} bytes) "
                    f"needs at least {need} bytes for one layer plus its fixed bytes "
                    f"({fixed[i]}) after ranks 0..{i - 1} took what they can")
        reach = nxt
    # every rank could place something, but never all of the layers
    last = n_ranks - 1
    start = max(j for j in _reach_before(memory, fits, n_layers, last))
    need = sum(lbytes[start:]) + fixed[last]
    return (f"no split fits the memory limits: the layers left for the last rank "
            f"(rank {last}, limit {memory[last]} bytes) need {need} bytes even after "
            f"the earlier ranks take as many layers as they can")


def _reach_before(memory, fits, n_layers, upto):
    """Prefix lengths ranks 0..upto-1 can cover (non-empty stages, room left for the rest)."""
    n_ranks = len(memory)
    reach = {0}
    for i in range(upto):
        rest = n_ranks - 1 - i
        reach = {k for j in reach for k in range(j + 1, n_layers - rest + 1) if fits(i, j, k)}
    return reach


def stage_times(ranges, layer_costs, capabilities):
    """Per-rank stage time (``Fraction``) of ``ranges``, in rank order."""
    costs = [_positive(c, f"layer_costs[{i}]") for i, c in enumerate(layer_costs)]
    caps = sorted(capabilities, key=lambda c: c.rank)
    if len(ranges) != len(caps):
        raise PlacementRefused(f"{len(ranges)} ranges for {len(caps)} ranks")
    out = []
    for (a, b), c in zip(ranges, caps):
        speed = c.weight if c.weight is not None else c.throughput
        if speed is None:
            raise PlacementRefused(f"rank {c.rank} gives neither throughput nor weight")
        out.append(sum(costs[a:b], Fraction(0)) / _positive(speed, f"rank {c.rank} speed"))
    return out


def bottleneck(ranges, layer_costs, capabilities):
    """The largest stage time of ``ranges`` (the quantity ``place_layers`` minimises)."""
    return max(stage_times(ranges, layer_costs, capabilities))
