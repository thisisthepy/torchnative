"""The decode-shaped `aten.mm` takes the skinny kernel, and the kernel computes
what `skinny_gemm.rs` says it computes (docs/perf/MM_DECODE.md).

`F.linear` at decode time is `(rows, k) @ t(weight)` with `rows` of 1 to 4
(`bootstrap.py::linear` lowers it to `t, view, mm, _unsafe_view`). On a
`gemm`-backed build that `mm` ran on one thread for every projection below the
GEMM threading threshold, and in a 16-row microkernel tile with 4 rows of
data. `torchnative/rust/torch_c/src/skinny_gemm.rs` streams the weight once,
split across candle's barrier pool by columns, and this file pins:

  * *routing*: the kernel is taken exactly for the operands it claims (rows
    1..8, depth >= 32, contiguous lhs rows, contiguous rhs columns, float32,
    CPU) and for nothing else, read from `_C._skinny_gemm_counters()` -- a
    counter, because a right answer is no evidence of which kernel gave it
    (AGENTS.md §16);
  * *agreement*: within the golden harness's scale-aware GEMM bound of a
    float64 reference (`tests/golden/cases.py`, `_gemm_scale_check`), the same
    grade upstream meets at these depths;
  * *the stated arithmetic*: an element depends on its row and column only, so
    an m-row call equals the m one-row calls bit for bit, and the thread count
    cannot move a bit (measured in subprocesses with `CANDLE_NUM_THREADS` and
    `TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES`);
  * *the off switch*: `TORCHNATIVE_SKINNY_GEMM=0` routes nothing and the
    general path's answer is within the same bound of the kernel's.

On a BLAS build (Apple with Accelerate) the kernel is built but not routed;
the routing test then asserts that nothing was taken, and the tests that
need the kernel skip by name.

What this file cannot find: speed. Whether the kernel is faster is the
bench's claim (`tests/bench/bench_skinny_mm.py`), measured in CI, alone.
"""

import hashlib
import json
import math
import os
import struct
import subprocess
import sys

import _C
import _skip

# Golden's float32 GEMM bound (`tests/golden/cases.py`): C * u_acc * sqrt(k)
# for the accumulation plus 2 * u_out for the final rounding, scaled by the
# largest reference magnitude.
_GEMM_ERROR_CONSTANT = 4.0
_U32 = 2.0 ** -24

_DECODE_K = 576


def _dense(values, shape):
    return _C._tensor_from_flat(list(values), list(shape), dtype=_C.float32)


def _flat(t):
    out, stack = [], [t.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _bits(values):
    return struct.pack(f"<{len(values)}f", *values)


def _lcg(n, seed):
    """White noise in [-1, 1), already representable in float32, the same on
    every run so a failure can be bisected (golden's `_gemm_lcg` shape)."""
    out = []
    x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) % (1 << 31)
        v = (x >> 7) / float(1 << 23) - 1.0  # 24 significant bits: exact in f32
        out.append(v)
    return out


def _t(w):
    return _C._aten_dispatch("aten.t.default", w)


def _mm(a, b):
    return _C._aten_dispatch("aten.mm.default", a, b)


def _counters():
    return _C._skinny_gemm_counters()


def _reference(x, w, m, k, n):
    """`x` is m x k row-major, `w` is n x k row-major (the weight), both lists
    of floats; the product `x @ w.T` in float64, row-major."""
    out = []
    for i in range(m):
        xi = x[i * k:(i + 1) * k]
        for j in range(n):
            wj = w[j * k:(j + 1) * k]
            s = 0.0
            for a, b in zip(xi, wj):
                s += a * b
            out.append(s)
    return out


def _assert_within_gemm_bound(got, ref, k, what, sides=1):
    """`sides=1` compares a float32 result with the float64 reference; `sides=2`
    compares two float32 results, each within the bound of that reference, so
    their difference may reach twice it."""
    assert len(got) == len(ref), (what, len(got), len(ref))
    scale = max((abs(v) for v in ref), default=0.0)
    bound = sides * (_GEMM_ERROR_CONSTANT * _U32 * math.sqrt(k) + 2.0 * _U32) * scale
    worst = 0.0
    worst_at = -1
    for idx, (g, r) in enumerate(zip(got, ref)):
        d = abs(g - r)
        if d > worst:
            worst, worst_at = d, idx
    assert worst <= bound, (
        f"{what}: max|d|={worst:.4g} at element {worst_at} exceeds the GEMM bound "
        f"{bound:.4g} (k={k}, |ref|max={scale:.4g}); an accumulation difference, "
        f"not rounding"
    )
    return worst, bound


def _case(m, k, n, seed=1):
    x = _lcg(m * k, seed)
    w = _lcg(n * k, seed + 1)
    return x, w, _dense(x, [m, k]), _dense(w, [n, k])


# --- routing ------------------------------------------------------------------


def test_counters_name_the_build():
    c = _counters()
    for key in ("built", "routed", "enabled", "calls", "threaded_calls", "threads",
                "thread_bytes", "max_rows", "min_depth", "lanes", "simd"):
        assert key in c, (key, c)
    assert c["built"] is True
    assert isinstance(c["routed"], bool) and isinstance(c["enabled"], bool)
    assert isinstance(c["calls"], int) and c["calls"] >= 0
    assert c["threaded_calls"] <= c["calls"]
    assert c["threads"] >= 1
    assert c["max_rows"] == 8 and c["min_depth"] == 32 and c["lanes"] == 8
    assert isinstance(c["simd"], str) and c["simd"]
    print(f"    skinny_gemm: routed={c['routed']} enabled={c['enabled']} "
          f"threads={c['threads']} thread_bytes={c['thread_bytes']} simd={c['simd']}")


def _taken(fn):
    before = _counters()["calls"]
    out = fn()
    return _counters()["calls"] - before, out


def test_the_decode_shape_takes_the_kernel_exactly_when_claimed():
    c = _counters()
    expect = 1 if (c["routed"] and c["enabled"]) else 0
    n = 192
    for m in range(1, 9):
        x, w, xt, wt = _case(m, _DECODE_K, n, seed=10 + m)
        taken, out = _taken(lambda: _mm(xt, _t(wt)))
        assert taken == expect, (f"rows={m}: the kernel was taken {taken} times, "
                                 f"expected {expect} (routed={c['routed']})")
        _assert_within_gemm_bound(_flat(out), _reference(x, w, m, _DECODE_K, n),
                                  _DECODE_K, f"rows={m}")

    # Not claimed: nine rows, a depth below MIN_DEPTH, a plain row-major rhs
    # (its columns are not contiguous), and an lhs whose rows are not
    # contiguous. Each still has to be right, through the general path.
    x, w, xt, wt = _case(9, _DECODE_K, n, seed=30)
    taken, out = _taken(lambda: _mm(xt, _t(wt)))
    assert taken == 0, f"nine rows took the kernel {taken} times"
    _assert_within_gemm_bound(_flat(out), _reference(x, w, 9, _DECODE_K, n), _DECODE_K, "rows=9")

    x, w, xt, wt = _case(4, 31, n, seed=31)
    taken, out = _taken(lambda: _mm(xt, _t(wt)))
    assert taken == 0, f"depth 31 took the kernel {taken} times"
    _assert_within_gemm_bound(_flat(out), _reference(x, w, 4, 31, n), 31, "k=31")

    x, w, xt, _ = _case(4, _DECODE_K, n, seed=32)
    # The same product with the weight stored (k, n) row-major: rhs_rs == n.
    w_kn = [0.0] * (_DECODE_K * n)
    for j in range(n):
        for t in range(_DECODE_K):
            w_kn[t * n + j] = w[j * _DECODE_K + t]
    taken, out = _taken(lambda: _mm(xt, _dense(w_kn, [_DECODE_K, n])))
    assert taken == 0, f"a row-major rhs took the kernel {taken} times"
    _assert_within_gemm_bound(_flat(out), _reference(x, w, 4, _DECODE_K, n), _DECODE_K, "rhs (k, n)")

    x, w, _, wt = _case(4, _DECODE_K, n, seed=33)
    # The lhs as a transposed view of its own transpose: rows have stride 1.
    x_km = [0.0] * (_DECODE_K * 4)
    for i in range(4):
        for t in range(_DECODE_K):
            x_km[t * 4 + i] = x[i * _DECODE_K + t]
    taken, out = _taken(lambda: _mm(_t(_dense(x_km, [_DECODE_K, 4])), _t(wt)))
    assert taken == 0, f"a column-strided lhs took the kernel {taken} times"
    _assert_within_gemm_bound(_flat(out), _reference(x, w, 4, _DECODE_K, n), _DECODE_K, "lhs t()")


def test_the_fold_path_of_linear_reaches_the_kernel():
    """`F.linear` does not call `mm` directly: `bootstrap.py::linear` emits
    `matmul(input, t(weight))` and `_cia_matmul` folds `(batch, 1, k)` into
    `(batch, k)` before `mm`. The counter has to move through that door too,
    or the decode path is not the one this kernel sits on."""
    c = _counters()
    expect = 1 if (c["routed"] and c["enabled"]) else 0
    batch, n = 4, 1536
    x, w, _, wt = _case(batch, _DECODE_K, n, seed=40)
    x3 = _dense(x, [batch, 1, _DECODE_K])
    taken, out = _taken(lambda: _C._aten_dispatch("aten.matmul.default", x3, _t(wt)))
    assert taken == expect, (f"matmul (batch, 1, k) x t(weight) took the kernel {taken} "
                             f"times, expected {expect}")
    assert list(out.shape) == [batch, 1, n], out.shape
    _assert_within_gemm_bound(_flat(out), _reference(x, w, batch, _DECODE_K, n),
                              _DECODE_K, "matmul fold")


# --- agreement ----------------------------------------------------------------


def test_every_claimed_shape_agrees_with_float64_within_the_gemm_bound():
    """Depths that are and are not multiples of the 8-lane block, one column,
    a column count that leaves a tail after the thread split, and SmolLM2's
    real projections. This test passes on the general path too; it is the
    agreement guard, not the red test."""
    seed = 100
    worst_ratio = 0.0
    for m in (1, 4, 8):
        for k in (32, 33, 40, 577):
            for n in (1, 3, 11):
                seed += 1
                x, w, xt, wt = _case(m, k, n, seed)
                got = _flat(_mm(xt, _t(wt)))
                worst, bound = _assert_within_gemm_bound(got, _reference(x, w, m, k, n), k,
                                                         f"m={m} k={k} n={n}")
                worst_ratio = max(worst_ratio, worst / bound if bound else 0.0)
    for m, k, n, name in ((1, 576, 192, "k/v_proj"), (4, 576, 576, "q/o_proj"),
                          (4, 576, 1536, "gate/up_proj"), (4, 1536, 576, "down_proj")):
        seed += 1
        x, w, xt, wt = _case(m, k, n, seed)
        got = _flat(_mm(xt, _t(wt)))
        worst, bound = _assert_within_gemm_bound(got, _reference(x, w, m, k, n), k, name)
        worst_ratio = max(worst_ratio, worst / bound if bound else 0.0)
    print(f"    skinny_gemm: worst error / bound = {worst_ratio:.3f}")


def test_rows_are_independent_bit_for_bit():
    """The stated arithmetic: an element is one lane-split dot product of its
    row and column, whatever the other rows are. The general path does not
    promise this (`gemm` sends one row to a gemv kernel and more to the
    blocked GEMM), so this is asserted only where the kernel is routed."""
    c = _counters()
    if not (c["routed"] and c["enabled"]):
        _skip.skip("the skinny kernel is not routed on this build (BLAS matmul)")
        return
    checked = 0
    for k in (_DECODE_K, 577):
        for n in (192, 11):
            for m in (2, 3, 4, 8):
                x, w, xt, wt = _case(m, k, n, seed=200 + m + k + n)
                batched = _flat(_mm(xt, _t(wt)))
                rows = []
                for i in range(m):
                    rows += _flat(_mm(_dense(x[i * k:(i + 1) * k], [1, k]), _t(wt)))
                if _bits(batched) != _bits(rows):
                    bad = [j for j in range(m * n) if _bits([batched[j]]) != _bits([rows[j]])]
                    raise AssertionError(
                        f"m={m} k={k} n={n}: {len(bad)} of {m * n} elements differ between "
                        f"the m-row call and one-row calls; first at row {bad[0] // n}, "
                        f"col {bad[0] % n}: {batched[bad[0]]!r} vs {rows[bad[0]]!r}")
                checked += m * n
    print(f"    skinny_gemm: m-row call == one-row calls bit for bit, {checked} elements")


# --- subprocess measurements: thread count and the off switch -----------------

_WORKER_SHAPES = ((1, 576, 192), (4, 576, 576), (4, 576, 1536), (3, 1536, 576), (8, 577, 11))


def _worker():
    """Compute the worker shapes, print their bits' digest and the counters."""
    digests = {}
    values = {}
    before = _counters()
    for m, k, n in _WORKER_SHAPES:
        x, w, xt, wt = _case(m, k, n, seed=300 + m + k)
        got = _flat(_mm(xt, _t(wt)))
        digests[f"{m}x{k}x{n}"] = hashlib.sha256(_bits(got)).hexdigest()
        values[f"{m}x{k}x{n}"] = got
    after = _counters()
    print("@@JSON@@" + json.dumps({
        "digests": digests,
        "values": values,
        "calls": after["calls"] - before["calls"],
        "threaded_calls": after["threaded_calls"] - before["threaded_calls"],
        "counters": after,
    }))


def _run_worker(extra_env):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")])
    env.update(extra_env)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"],
                       env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(f"worker {extra_env} failed (exit {p.returncode}):\n"
                             f"{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    line = [l for l in p.stdout.splitlines() if l.startswith("@@JSON@@")]
    assert line, f"worker {extra_env} printed no result:\n{p.stdout[-2000:]}"
    return json.loads(line[-1][len("@@JSON@@"):])


def test_thread_count_cannot_move_a_bit_and_the_switch_falls_back():
    c = _counters()
    if not c["routed"]:
        base = _run_worker({})
        assert base["calls"] == 0, (
            "a BLAS build routed the skinny kernel", base["counters"])
        _skip.skip("the skinny kernel is not routed on this build (BLAS matmul)")
        return
    default = _run_worker({})
    one_thread = _run_worker({"CANDLE_NUM_THREADS": "1"})
    split_all = _run_worker({"TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES": "0"})
    switched_off = _run_worker({"TORCHNATIVE_SKINNY_GEMM": "0"})

    n_shapes = len(_WORKER_SHAPES)
    assert default["calls"] == n_shapes, default["counters"]
    assert one_thread["calls"] == n_shapes, one_thread["counters"]
    assert one_thread["threaded_calls"] == 0, one_thread["counters"]
    assert one_thread["counters"]["threads"] == 1, one_thread["counters"]
    assert split_all["calls"] == n_shapes, split_all["counters"]
    if split_all["counters"]["threads"] > 1:
        # Every worker shape has at least `threads` columns on a hosted runner
        # (the fewest is 11), so with the byte floor at 0 every call splits.
        assert split_all["threaded_calls"] == n_shapes, split_all["counters"]
    assert switched_off["calls"] == 0, switched_off["counters"]
    assert switched_off["counters"]["enabled"] is False, switched_off["counters"]

    for key in default["digests"]:
        assert default["digests"][key] == one_thread["digests"][key], (
            f"{key}: one thread and {default['counters']['threads']} threads gave "
            f"different bits")
        assert default["digests"][key] == split_all["digests"][key], (
            f"{key}: splitting every call changed the bits")
    for m, k, n in _WORKER_SHAPES:
        key = f"{m}x{k}x{n}"
        _assert_within_gemm_bound(default["values"][key], switched_off["values"][key], k,
                                  f"{key}: kernel vs general path", sides=2)
    print(f"    skinny_gemm: {n_shapes} shapes, bits identical across 1/"
          f"{default['counters']['threads']} threads and a forced split; "
          f"general path within the bound; threaded by default: "
          f"{default['threaded_calls']}/{n_shapes}")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    return 1 if _skip.run_tests(items, suite="test_skinny_gemm") else 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        raise SystemExit(_main())
