"""The decode-shaped CPU attention kernel (`sdpa_decode.rs`, docs/perf/SDPA_DECODE.md).

`aten._scaled_dot_product_flash_attention_for_cpu` at decode is a few query
rows per head against the whole key/value cache, thirty times per step. The
general path spells that out in about fifteen candle ops and two batched
GEMMs per call (about 220 us per call on the Xeon of Gemstone run
44874740472 for 340 thousand multiply-adds of work). The kernel reads the
operands in place and fuses the row, and this file pins:

  * *routing*: `_C._sdpa_decode_counters()` says whether the path is routed on
    this build, and the decode shapes take it exactly when claimed (`float32`,
    CPU, 1 to 8 query rows, no causal rule, a mask that is None or rank 4 with
    every axis 1 or full) and not otherwise, counted, never inferred from speed;
  * *agreement*: every claimed shape, tails and grouped-query heads and
    broadcast masks included, is within a stated float32 bound of a float64
    reference, output and logsumexp both;
  * *the arithmetic*: a row depends on its own query, its head's cache and its
    mask row only, so a batch-b call equals the b batch-1 calls bit for bit, a
    q_len-l call equals the l one-row calls, grouped-query heads equal the
    repeated cache, and the thread count cannot move a bit (subprocesses with
    `CANDLE_NUM_THREADS` and `TORCHNATIVE_SDPA_DECODE_THREAD_MADDS`);
  * *the off switch*: `TORCHNATIVE_SDPA_DECODE=0` routes nothing and the
    general path's answer is within the same bound of the kernel's;
  * *a fully masked row* answers NaN, as the general path does.

On a build where the kernel is not routed (Accelerate, MKL), the routing test
asserts that nothing took it and the kernel-only tests skip by name.

What this file cannot find: speed. Whether the kernel is faster is the
bench's claim (`tests/bench/bench_sdpa_decode.py`), measured in CI, alone.
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

_U32 = 2.0 ** -24
_OP = "aten._scaled_dot_product_flash_attention_for_cpu.default"


def _dense(values, shape, dtype=None):
    return _C._tensor_from_flat(list(values), list(shape), dtype=dtype or _C.float32)


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
    """White noise in [-1, 1), exactly representable in float32, the same on
    every run so a failure can be bisected."""
    out = []
    x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) % (1 << 31)
        out.append((x >> 7) / float(1 << 23) - 1.0)
    return out


def _counters():
    return _C._sdpa_decode_counters()


def _sdpa(q, k, v, mask=None, scale=None, causal=False):
    """The op, positionally: (query, key, value, dropout_p, is_causal, attn_mask, scale)."""
    return _C._aten_dispatch(_OP, q, k, v, 0.0, causal, mask, scale)


def _transpose12(t):
    return _C._aten_dispatch("aten.transpose.int", t, 1, 2)


# --- a float64 reference of the stated arithmetic -----------------------------


def _mask_index(mshape, b, h, i, j):
    idx = 0
    for axis, (extent, at) in enumerate(zip(mshape, (b, h, i, j))):
        idx = idx * extent + (at if extent != 1 else 0)
    return idx


def _reference(qf, kf, vf, shape, mask=None, mshape=None, scale=None):
    """`qf`, `kf`, `vf` are flat row-major lists for shapes (b, h, l, d),
    (b, hk, s, d), (b, hk, s, d). Returns (out flat, lse flat, |score| max)."""
    b, h, hk, l, s, d = shape
    n_rep = h // hk
    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    out, lse, smax = [], [], 0.0
    for bi in range(b):
        for hi in range(h):
            kh = hi // n_rep
            for i in range(l):
                qo = ((bi * h + hi) * l + i) * d
                scores = []
                for j in range(s):
                    ko = ((bi * hk + kh) * s + j) * d
                    sc = sum(qf[qo + t] * kf[ko + t] for t in range(d)) * scale
                    if mask is not None:
                        sc += mask[_mask_index(mshape, bi, hi, i, j)]
                    scores.append(sc)
                finite = [x for x in scores if not math.isinf(x) and not math.isnan(x)]
                if finite:
                    smax = max(smax, max(abs(x) for x in finite))
                m = max(scores)
                w = [math.exp(x - m) for x in scores]
                total = sum(w)
                row = [0.0] * d
                for j in range(s):
                    p = w[j] / total
                    vo = ((bi * hk + kh) * s + j) * d
                    for t in range(d):
                        row[t] += p * vf[vo + t]
                out.extend(row)
                lse.append(math.log(total) + m)
    return out, lse, smax


def _bound(smax, d, s, vmax, sides=1):
    """float32 error budget of the stated arithmetic against float64: the dot
    product's accumulation (sqrt(d) u, scaled by the score magnitude the
    softmax then amplifies), the exponentials and their sum (sqrt(s) u), the
    division and the weighted sum (sqrt(s) u), with a constant of 8. `sides=2`
    compares two float32 results, each within the budget of the reference."""
    return sides * 8.0 * _U32 * (math.sqrt(d) * max(1.0, smax) + 2.0 * math.sqrt(s) + 4.0) * max(1.0, vmax)


def _assert_close(got, ref, bound, what):
    assert len(got) == len(ref), (what, len(got), len(ref))
    worst, at = 0.0, -1
    for idx, (g, r) in enumerate(zip(got, ref)):
        if math.isnan(r) or math.isinf(r):
            assert (math.isnan(g) and math.isnan(r)) or g == r, (what, idx, g, r)
            continue
        diff = abs(g - r)
        if diff > worst:
            worst, at = diff, idx
    assert worst <= bound, (
        f"{what}: max|d|={worst:.4g} at element {at} exceeds the bound {bound:.4g}; "
        f"an arithmetic difference, not rounding")
    return worst


def _case(shape, seed=1, mask_kind=None, scale=None):
    """Tensors and flat lists for `shape` = (b, h, hk, l, s, d). `mask_kind`:
    None, 'bhls', 'b1ls', '11ls' (every seventh entry -inf), or 'row_masked'
    ((b,1,l,s) with the first row of every batch entirely -inf)."""
    b, h, hk, l, s, d = shape
    qf = _lcg(b * h * l * d, seed)
    kf = _lcg(b * hk * s * d, seed + 1)
    vf = _lcg(b * hk * s * d, seed + 2)
    mask, mshape, mt = None, None, None
    if mask_kind is not None:
        mshape = {"bhls": (b, h, l, s), "b1ls": (b, 1, l, s), "11ls": (1, 1, l, s),
                  "row_masked": (b, 1, l, s)}[mask_kind]
        n = 1
        for e in mshape:
            n *= e
        mask = _lcg(n, seed + 3)
        if mask_kind == "row_masked":
            for bi in range(b):
                for j in range(s):
                    mask[_mask_index(mshape, bi, 0, 0, j)] = float("-inf")
        else:
            for idx in range(0, n, 7):
                mask[idx] = float("-inf")
        mt = _dense(mask, mshape)
    return dict(shape=shape, qf=qf, kf=kf, vf=vf, mask=mask, mshape=mshape, scale=scale,
                q=_dense(qf, (b, h, l, d)), k=_dense(kf, (b, hk, s, d)), v=_dense(vf, (b, hk, s, d)),
                mt=mt)


def _run_case(c):
    return _sdpa(c["q"], c["k"], c["v"], c["mt"], c["scale"])


def _check_case(c, sides=1, got=None):
    out_t, lse_t = got if got is not None else _run_case(c)
    b, h, hk, l, s, d = c["shape"]
    assert tuple(out_t.shape) == (b, h, l, d), out_t.shape
    assert tuple(lse_t.shape) == (b, h, l), lse_t.shape
    ref_out, ref_lse, smax = _reference(c["qf"], c["kf"], c["vf"], c["shape"], c["mask"], c["mshape"], c["scale"])
    vmax = max(abs(x) for x in c["vf"])
    bound = _bound(smax, d, s, vmax, sides)
    w1 = _assert_close(_flat(out_t), ref_out, bound, f"{c['shape']} output")
    lmax = max((abs(x) for x in ref_lse if not math.isinf(x)), default=1.0)
    w2 = _assert_close(_flat(lse_t), ref_lse, bound * max(1.0, lmax), f"{c['shape']} logsumexp")
    return max(w1 / bound, w2 / (bound * max(1.0, lmax)))


# (b, h, hk, l, s, d): decode of SmolLM2-135M at batch 4 and 1 (9 query heads,
# 3 key/value heads, 64), the packed continuous batch, and tails on every axis.
_SHAPES = (
    (4, 9, 3, 1, 74, 64),
    (1, 9, 3, 1, 20, 64),
    (1, 9, 3, 4, 300, 64),
    (2, 4, 2, 8, 33, 80),
    (1, 1, 1, 1, 1, 5),
    (1, 2, 2, 3, 7, 13),
    (2, 3, 1, 2, 9, 8),
)


# --- routing ------------------------------------------------------------------


def test_counters_name_the_build():
    c = _counters()
    for key in ("built", "routed", "enabled", "calls", "threaded_calls", "threads",
                "thread_madds", "max_query_rows", "lanes", "simd"):
        assert key in c, (key, c)
    assert c["built"] is True
    assert isinstance(c["routed"], bool) and isinstance(c["enabled"], bool)
    assert c["calls"] >= c["threaded_calls"] >= 0
    assert c["threads"] >= 1
    assert c["max_query_rows"] == 8 and c["lanes"] == 8
    assert isinstance(c["simd"], str) and c["simd"]
    print(f"    sdpa_decode: routed={c['routed']} enabled={c['enabled']} threads={c['threads']} "
          f"thread_madds={c['thread_madds']} simd={c['simd']}")


def _taken(fn):
    before = _counters()["calls"]
    result = fn()
    return _counters()["calls"] - before, result


def test_the_decode_shape_takes_the_kernel_exactly_when_claimed():
    c = _counters()
    claimed = [
        ("batch 4 decode, no mask", lambda: _run_case(_case((4, 9, 3, 1, 74, 64)))),
        ("packed batch, (1,1,l,s) mask", lambda: _run_case(_case((1, 9, 3, 4, 300, 64), mask_kind="11ls"))),
        ("eight rows, full mask", lambda: _run_case(_case((2, 4, 4, 8, 16, 32), mask_kind="bhls"))),
        ("transposed query view", lambda: (lambda cc: _sdpa(
            _transpose12(_dense(cc["qf"], (1, 1, 9, 64))), cc["k"], cc["v"]))(_case((1, 9, 3, 1, 30, 64)))),
        ("explicit scale", lambda: _run_case(_case((1, 2, 2, 2, 11, 16), scale=0.3))),
    ]
    not_claimed = [
        ("nine query rows", lambda: _run_case(_case((1, 2, 2, 9, 16, 32)))),
        ("is_causal", lambda: (lambda cc: _sdpa(cc["q"], cc["k"], cc["v"], None, None, True))(
            _case((1, 2, 2, 4, 4, 16)))),
        ("float64", lambda: (lambda cc: _sdpa(_dense(cc["qf"], (1, 2, 1, 8), _C.float64),
                                              _dense(cc["kf"], (1, 2, 5, 8), _C.float64),
                                              _dense(cc["vf"], (1, 2, 5, 8), _C.float64)))(
            _case((1, 2, 2, 1, 5, 8)))),
        ("rank-2 mask", lambda: (lambda cc: _sdpa(cc["q"], cc["k"], cc["v"], _dense(_lcg(2 * 5, 9), (2, 5))))(
            _case((1, 2, 2, 2, 5, 8)))),
    ]
    for what, fn in claimed:
        n, _ = _taken(fn)
        want = 1 if c["routed"] else 0
        assert n == want, f"{what}: {n} kernel calls, wanted {want} (routed={c['routed']})"
    for what, fn in not_claimed:
        n, _ = _taken(fn)
        assert n == 0, f"{what}: took the kernel ({n} calls); it is not a claimed shape"
    print(f"    sdpa_decode: {len(claimed)} claimed shapes took the kernel "
          f"{'once each' if c['routed'] else 'never (not routed)'}, {len(not_claimed)} others never")


# --- agreement ----------------------------------------------------------------


def test_every_claimed_shape_agrees_with_float64_within_the_bound():
    """Passes on the general path too: it is the agreement guard, and the
    bound is what the kernel promises, not what it changes."""
    worst = 0.0
    n = 0
    for shape in _SHAPES:
        for mask_kind in (None, "b1ls", "bhls", "11ls"):
            worst = max(worst, _check_case(_case(shape, seed=10 + n, mask_kind=mask_kind)))
            n += 1
    print(f"    sdpa_decode: {n} cases, worst error / bound = {worst:.3f}")


def test_a_fully_masked_row_answers_nan_as_the_general_path_does():
    c = _case((2, 2, 2, 2, 6, 8), mask_kind="row_masked")
    out_t, lse_t = _run_case(c)
    out, lse = _flat(out_t), _flat(lse_t)
    b, h, hk, l, s, d = c["shape"]
    for bi in range(b):
        for hi in range(h):
            for i in range(l):
                r = (bi * h + hi) * l + i
                row = out[r * d:(r + 1) * d]
                if i == 0:
                    assert all(math.isnan(x) for x in row), (bi, hi, i, row)
                    assert math.isnan(lse[r]), (bi, hi, i, lse[r])
                else:
                    assert all(math.isfinite(x) for x in row), (bi, hi, i, row)
                    assert math.isfinite(lse[r]), (bi, hi, i, lse[r])
    _check_case(c)


# --- the arithmetic -----------------------------------------------------------


def test_rows_are_independent_bit_for_bit():
    c = _counters()
    if not c["routed"]:
        _skip.skip("the decode kernel is not routed on this build (BLAS matmul)")
        return
    shape = (3, 4, 2, 3, 23, 16)
    b, h, hk, l, s, d = shape
    whole = _case(shape, seed=40, mask_kind="b1ls")
    n, (out_t, lse_t) = _taken(lambda: _run_case(whole))
    assert n == 1
    out, lse = _flat(out_t), _flat(lse_t)

    # batch by batch
    for bi in range(b):
        q1 = _dense(whole["qf"][bi * h * l * d:(bi + 1) * h * l * d], (1, h, l, d))
        k1 = _dense(whole["kf"][bi * hk * s * d:(bi + 1) * hk * s * d], (1, hk, s, d))
        v1 = _dense(whole["vf"][bi * hk * s * d:(bi + 1) * hk * s * d], (1, hk, s, d))
        m1 = _dense(whole["mask"][bi * l * s:(bi + 1) * l * s], (1, 1, l, s))
        o1, l1 = _sdpa(q1, k1, v1, m1)
        assert _bits(_flat(o1)) == _bits(out[bi * h * l * d:(bi + 1) * h * l * d]), f"batch {bi} output"
        assert _bits(_flat(l1)) == _bits(lse[bi * h * l:(bi + 1) * h * l]), f"batch {bi} logsumexp"

    # row by row, one batch
    bi = 1
    for i in range(l):
        qrows = []
        for hi in range(h):
            base = ((bi * h + hi) * l + i) * d
            qrows.extend(whole["qf"][base:base + d])
        q1 = _dense(qrows, (1, h, 1, d))
        k1 = _dense(whole["kf"][bi * hk * s * d:(bi + 1) * hk * s * d], (1, hk, s, d))
        v1 = _dense(whole["vf"][bi * hk * s * d:(bi + 1) * hk * s * d], (1, hk, s, d))
        m1 = _dense(whole["mask"][(bi * l + i) * s:(bi * l + i + 1) * s], (1, 1, 1, s))
        o1, l1 = _sdpa(q1, k1, v1, m1)
        o1f, l1f = _flat(o1), _flat(l1)
        for hi in range(h):
            r = (bi * h + hi) * l + i
            assert _bits(o1f[hi * d:(hi + 1) * d]) == _bits(out[r * d:(r + 1) * d]), f"row {i} head {hi}"
            assert _bits([l1f[hi]]) == _bits([lse[r]]), f"row {i} head {hi} logsumexp"
    print(f"    sdpa_decode: batch-{b} call == {b} batch-1 calls, {l}-row call == {l} one-row calls, bit for bit")


def test_grouped_query_heads_equal_the_repeated_cache_bit_for_bit():
    c = _counters()
    if not c["routed"]:
        _skip.skip("the decode kernel is not routed on this build (BLAS matmul)")
        return
    shape = (2, 6, 2, 2, 17, 32)
    b, h, hk, l, s, d = shape
    grouped = _case(shape, seed=50, mask_kind="bhls")
    n_rep = h // hk
    kr, vr = [], []
    for bi in range(b):
        for hi in range(h):
            kh = hi // n_rep
            base = (bi * hk + kh) * s * d
            kr.extend(grouped["kf"][base:base + s * d])
            vr.extend(grouped["vf"][base:base + s * d])
    o_g, l_g = _run_case(grouped)
    o_r, l_r = _sdpa(grouped["q"], _dense(kr, (b, h, s, d)), _dense(vr, (b, h, s, d)), grouped["mt"])
    assert _bits(_flat(o_g)) == _bits(_flat(o_r)), "grouped heads differ from the repeated cache"
    assert _bits(_flat(l_g)) == _bits(_flat(l_r)), "grouped heads differ from the repeated cache (logsumexp)"
    print(f"    sdpa_decode: {h} query heads over {hk} key/value heads == the repeated cache, bit for bit")


# --- subprocess measurements: thread count and the off switch -----------------

_WORKER_SHAPES = ((4, 9, 3, 1, 74, 64), (1, 9, 3, 4, 300, 64), (2, 4, 2, 8, 33, 80), (1, 2, 2, 3, 7, 13))


def _worker():
    digests, values, lses, layouts = {}, {}, {}, {}
    before = _counters()
    for n, shape in enumerate(_WORKER_SHAPES):
        c = _case(shape, seed=300 + n, mask_kind="b1ls")
        out_t, lse_t = _run_case(c)
        key = "x".join(str(x) for x in shape)
        got = _flat(out_t)
        digests[key] = hashlib.sha256(_bits(got) + _bits(_flat(lse_t))).hexdigest()
        values[key] = got
        lses[key] = _flat(lse_t)
        # The pair's layout, which upstream lays out by the query's layout
        # (`tests/ops/test_elemstride.py` holds that against upstream; here
        # the kernel is held to the general path).
        layouts[key] = [list(out_t.shape), list(out_t.stride()), list(lse_t.shape), list(lse_t.stride())]
    after = _counters()
    print("@@JSON@@" + json.dumps({
        "digests": digests, "values": values, "lses": lses, "layouts": layouts,
        "calls": after["calls"] - before["calls"],
        "threaded_calls": after["threaded_calls"] - before["threaded_calls"],
        "counters": after,
    }))


def _run_worker(extra_env):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")])
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
        assert base["calls"] == 0, ("a BLAS build routed the decode kernel", base["counters"])
        _skip.skip("the decode kernel is not routed on this build (BLAS matmul)")
        return
    default = _run_worker({})
    one_thread = _run_worker({"CANDLE_NUM_THREADS": "1"})
    split_all = _run_worker({"TORCHNATIVE_SDPA_DECODE_THREAD_MADDS": "0"})
    switched_off = _run_worker({"TORCHNATIVE_SDPA_DECODE": "0"})

    n_shapes = len(_WORKER_SHAPES)
    assert default["calls"] == n_shapes, default["counters"]
    assert one_thread["calls"] == n_shapes, one_thread["counters"]
    assert one_thread["counters"]["threads"] == 1, one_thread["counters"]
    assert one_thread["threaded_calls"] == 0, one_thread["counters"]
    assert split_all["calls"] == n_shapes, split_all["counters"]
    if split_all["counters"]["threads"] > 1:
        # Every worker shape has at least two rows, so with the floor at 0 every call splits.
        assert split_all["threaded_calls"] == n_shapes, split_all["counters"]
    assert switched_off["calls"] == 0, switched_off["counters"]
    assert switched_off["counters"]["enabled"] is False, switched_off["counters"]

    for key in default["digests"]:
        assert default["digests"][key] == one_thread["digests"][key], f"{key}: one thread moved a bit"
        assert default["digests"][key] == split_all["digests"][key], f"{key}: a forced split moved a bit"
        assert default["layouts"][key] == switched_off["layouts"][key], (
            f"{key}: the kernel's pair is laid out differently from the general path's: "
            f"kernel {default['layouts'][key]}, general path {switched_off['layouts'][key]}")
    for n, shape in enumerate(_WORKER_SHAPES):
        key = "x".join(str(x) for x in shape)
        c2 = _case(shape, seed=300 + n, mask_kind="b1ls")
        b, h, hk, l, s, d = shape
        ref_out, ref_lse, smax = _reference(c2["qf"], c2["kf"], c2["vf"], shape, c2["mask"], c2["mshape"])
        vmax = max(abs(x) for x in c2["vf"])
        bound = _bound(smax, d, s, vmax, sides=2)
        _assert_close(switched_off["values"][key], default["values"][key], bound, f"{key}: general path vs kernel")
        lmax = max((abs(x) for x in ref_lse if not math.isinf(x)), default=1.0)
        _assert_close(switched_off["lses"][key], default["lses"][key], bound * max(1.0, lmax),
                      f"{key}: general path vs kernel (logsumexp)")
    print(f"    sdpa_decode: {n_shapes} shapes, bits identical across 1/{default['counters']['threads']} "
          f"threads and a forced split; general path within the bound and laid out the same; threaded by default: "
          f"{default['threaded_calls']}/{n_shapes}")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    return _skip.run_tests(items, "ops/test_sdpa_decode")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        raise SystemExit(1 if _main() else 0)
