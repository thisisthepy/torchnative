"""Capability-weighted layer placement (BACKEND.md §5.4, D11, plan step 17).

`place_layers` must return the split that minimises the slowest stage, not a
split that merely looks sensible. Hand-worked cases pin exact answers; a
brute-force enumeration of every contiguous partition (fixed seed, many random
inputs, memory limits included) pins optimality and feasibility; the refusals
are checked by name.

Nullification (AGENTS.md §17.5), planned, to be run by CI or the coordinating
session and reported either way: replace the body of `place_layers` with an
even split (`[(i * L // R, (i + 1) * L // R) for i in range(R)]`) and
`test_matches_brute_force_optimum`, `test_five_three_two_weights`,
`test_a_slow_rank_gets_few_layers` and `test_memory_forces_a_different_split`
must all go red. Replace the DP's `min` with `max` and the brute-force test
must go red. Drop the memory check in `fits` and the memory cases and the
brute-force feasibility assertion must go red. This file has not been run
locally (rule 10), so none of that is observed yet.

The module is loaded by file path: `torchnative.distributed` imports torch when
its package is imported, and this item is pure Python.
"""

import importlib.util
import itertools
import random
from fractions import Fraction
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
_spec = importlib.util.spec_from_file_location(
    "placement_under_test",
    ROOT / "torchnative" / "python" / "torchnative" / "distributed" / "placement.py")
placement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placement)

Capability = placement.Capability
PlacementRefused = placement.PlacementRefused
place_layers = placement.place_layers
bottleneck = placement.bottleneck


def _weights(*ws, memory=None):
    memory = memory or [None] * len(ws)
    return [Capability(r, memory_bytes=m, weight=w) for r, (w, m) in enumerate(zip(ws, memory))]


def _refused(fn, *needles):
    try:
        fn()
    except PlacementRefused as e:
        for n in needles:
            assert n in str(e), f"refusal {str(e)!r} does not mention {n!r}"
        return
    raise AssertionError("expected PlacementRefused, nothing was raised")


def test_refusal_is_a_value_error():
    assert issubclass(PlacementRefused, ValueError)


def test_even_weights_split_evenly():
    assert place_layers([1] * 6, _weights(1, 1, 1)) == [(0, 2), (2, 4), (4, 6)]


def test_ties_go_to_the_smallest_stops():
    # 3 unit layers on 2 equal ranks: (1,2) and (2,1) both give bottleneck 2.
    assert place_layers([1] * 3, _weights(1, 1)) == [(0, 1), (1, 3)]


def test_five_three_two_weights():
    assert place_layers([1] * 10, _weights(5, 3, 2)) == [(0, 5), (5, 8), (8, 10)]


def test_a_slow_rank_gets_few_layers():
    caps = [Capability(0, throughput=4), Capability(1, throughput=1)]
    assert place_layers([1] * 4, caps) == [(0, 3), (3, 4)]


def test_memory_forces_a_different_split():
    # Speed alone (3:1 over 6 unit layers) gives (0,5),(5,6).
    assert place_layers([1] * 6, _weights(3, 1)) == [(0, 5), (5, 6)]
    # Rank 0 can hold only 3 layers of 10 bytes.
    caps = _weights(3, 1, memory=[30, None])
    assert place_layers([1] * 6, caps, layer_bytes=[10] * 6) == [(0, 3), (3, 6)]


def test_fixed_bytes_count_against_memory():
    caps = _weights(1, 1, memory=[40, 40])
    got = place_layers([1] * 4, caps, layer_bytes=[10] * 4, fixed_bytes={0: 20, 1: 20})
    assert got == [(0, 2), (2, 4)]
    # Without the fixed bytes rank 0 could take 3 layers; with weight 3:1 it would.
    caps = _weights(3, 1, memory=[40, 40])
    assert place_layers([1] * 4, caps, layer_bytes=[10] * 4) == [(0, 3), (3, 4)]
    assert place_layers([1] * 4, caps, layer_bytes=[10] * 4,
                        fixed_bytes={0: 20}) == [(0, 2), (2, 4)]


def test_costs_not_just_counts():
    # One heavy layer: the heavy layer alone on a rank is the optimum.
    assert place_layers([1, 1, 1, 9], _weights(1, 1)) == [(0, 3), (3, 4)]


def test_single_rank_takes_everything():
    assert place_layers([2, 3, 4], _weights(1)) == [(0, 3)]


def test_input_order_of_capabilities_does_not_matter():
    caps = _weights(5, 3, 2)
    a = place_layers([1] * 10, caps)
    b = place_layers([1] * 10, list(reversed(caps)))
    assert a == b


def test_determinism():
    caps = _weights(2, 7, 3, 5)
    costs = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    first = place_layers(costs, caps)
    for _ in range(5):
        assert place_layers(costs, caps) == first


def _brute(costs, caps, lbytes, fixed):
    """Best bottleneck over every contiguous partition that fits memory, or None."""
    n, r = len(costs), len(caps)
    best = None
    for cuts in itertools.combinations(range(1, n), r - 1):
        stops = (0,) + cuts + (n,)
        ranges = list(zip(stops, stops[1:]))
        ok = True
        for (a, b), c in zip(ranges, caps):
            if c.memory_bytes is not None and sum(lbytes[a:b]) + fixed.get(c.rank, 0) > c.memory_bytes:
                ok = False
        if not ok:
            continue
        v = bottleneck(ranges, costs, caps)
        if best is None or v < best:
            best = v
    return best


def test_matches_brute_force_optimum():
    rng = random.Random(20261113)
    feasible = refused = 0
    for _ in range(600):
        r = rng.randint(1, 4)
        n = rng.randint(r, 8)
        costs = [rng.randint(1, 9) if rng.random() < 0.7 else rng.random() + 0.1 for _ in range(n)]
        lbytes = [rng.randint(1, 9) for _ in range(n)]
        use_mem = rng.random() < 0.6
        caps = [Capability(i, memory_bytes=rng.randint(5, 30) if use_mem else None,
                           weight=rng.randint(1, 5)) for i in range(r)]
        fixed = {i: rng.randint(0, 4) for i in range(r) if rng.random() < 0.3}
        want = _brute(costs, caps, lbytes, fixed)
        shuffled = caps[:]
        rng.shuffle(shuffled)
        if want is None:
            _refused(lambda: place_layers(costs, shuffled, layer_bytes=lbytes, fixed_bytes=fixed),
                     "no split fits")
            refused += 1
            continue
        got = place_layers(costs, shuffled, layer_bytes=lbytes, fixed_bytes=fixed)
        assert len(got) == r and got[0][0] == 0 and got[-1][1] == n
        assert all(a < b for a, b in got) and all(x[1] == y[0] for x, y in zip(got, got[1:]))
        for (a, b), c in zip(got, caps):
            if c.memory_bytes is not None:
                assert sum(lbytes[a:b]) + fixed.get(c.rank, 0) <= c.memory_bytes, (got, c)
        assert bottleneck(got, costs, caps) == want, (costs, got, want)
        feasible += 1
    # The sweep must exercise both arms or it proves nothing.
    assert feasible > 100 and refused > 10, (feasible, refused)


def test_raising_a_weight_never_raises_the_bottleneck():
    rng = random.Random(7)
    for _ in range(200):
        r = rng.randint(2, 4)
        n = rng.randint(r, 9)
        costs = [rng.randint(1, 9) for _ in range(n)]
        ws = [rng.randint(1, 5) for _ in range(r)]
        i = rng.randrange(r)
        before = bottleneck(place_layers(costs, _weights(*ws)), costs, _weights(*ws))
        ws2 = ws[:]
        ws2[i] += rng.randint(1, 4)
        after = bottleneck(place_layers(costs, _weights(*ws2)), costs, _weights(*ws2))
        assert after <= before, (costs, ws, ws2, before, after)


def test_raising_a_weight_grows_its_range_on_a_hand_case():
    before = place_layers([1] * 8, _weights(1, 1, 1))
    after = place_layers([1] * 8, _weights(1, 2, 1))
    assert before == [(0, 2), (2, 5), (5, 8)]
    assert after == [(0, 2), (2, 6), (6, 8)]
    assert after[1][1] - after[1][0] > before[1][1] - before[1][0]


def test_float_inputs_are_exact_not_toleranced():
    costs = [0.1] * 3
    assert place_layers(costs, _weights(1, 1)) == [(0, 1), (1, 3)]
    assert bottleneck([(0, 1), (1, 3)], costs, _weights(1, 1)) == Fraction(0.1) * 2


def test_refuses_more_ranks_than_layers():
    _refused(lambda: place_layers([1, 1], _weights(1, 1, 1)), "3 ranks", "2 layers")


def test_refuses_infeasible_memory_naming_rank_and_bytes():
    caps = _weights(1, 1, memory=[5, 100])
    _refused(lambda: place_layers([1, 1, 1], caps, layer_bytes=[10, 10, 10]),
             "no split fits", "rank 0", "limit 5", "10 bytes")
    caps = _weights(1, 1, memory=[40, 40])
    _refused(lambda: place_layers([1] * 5, caps, layer_bytes=[10] * 5, fixed_bytes={0: 20, 1: 20}),
             "no split fits", "rank 1", "limit 40")


def test_refuses_non_positive_and_non_finite_numbers():
    _refused(lambda: place_layers([1, 0], _weights(1, 1)), "layer_costs[1]", "positive")
    _refused(lambda: place_layers([1, -2], _weights(1, 1)), "layer_costs[1]", "positive")
    _refused(lambda: place_layers([1, float("nan")], _weights(1, 1)), "finite")
    _refused(lambda: place_layers([1, 1], _weights(1, 0)), "rank 1 weight", "positive")
    _refused(lambda: place_layers([1, 1], [Capability(0, throughput=1), Capability(1, throughput=-1)]),
             "rank 1 throughput", "positive")
    _refused(lambda: place_layers([1, 1], _weights(1, float("inf"))), "finite")
    _refused(lambda: place_layers([], _weights(1)), "empty")


def test_refuses_a_capability_without_speed_or_with_two_kinds():
    _refused(lambda: place_layers([1, 1], [Capability(0, weight=1), Capability(1)]),
             "rank 1", "neither throughput nor weight")
    _refused(lambda: place_layers([1, 1], [Capability(0, weight=1),
                                            Capability(1, weight=1, throughput=1)]),
             "rank 1", "both")
    _refused(lambda: place_layers([1, 1], [Capability(0, weight=1), Capability(1, throughput=1)]),
             "not comparable")


def test_refuses_duplicate_and_missing_ranks():
    _refused(lambda: place_layers([1, 1, 1], [Capability(0, weight=1), Capability(0, weight=1)]),
             "duplicate", "[0]")
    _refused(lambda: place_layers([1, 1, 1], [Capability(0, weight=1), Capability(2, weight=1)]),
             "missing [1]")
    _refused(lambda: place_layers([1, 1, 1], []), "no rank")


def test_refuses_mismatched_lengths_and_unchecked_memory():
    _refused(lambda: place_layers([1, 1, 1], _weights(1, 1), layer_bytes=[1, 1]),
             "2 entries", "3")
    _refused(lambda: place_layers([1, 1, 1], _weights(1, 1, memory=[10, 10])),
             "layer_bytes is None")
    _refused(lambda: place_layers([1, 1, 1], _weights(1, 1), fixed_bytes={5: 1}), "rank 5")
    _refused(lambda: place_layers([1, 1, 1], _weights(1, 1), layer_bytes=[1, -1, 1]), "layer_bytes[1]")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
