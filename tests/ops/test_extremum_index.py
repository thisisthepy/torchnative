"""The CPU index reductions in one pass (`extremum_index.rs`, docs/perf/EXTREMUM_INDEX.md).

`argmax`, `argmin`, the whole-tensor `max` and `min`, and the index half of
`max.dim` and `min.dim` all answer through one rule: the first NaN's index if
the slice holds one, else the first index no later element beats. The shim
composed that rule from candle ops (about seven passes over the slice), which
cost one `argmax` over the `[4, 49152]` logits 1.24 ms of an 82.8 ms decode
step on the Xeon of Gemstone run 44874740472. A `float32` or `float64` CPU
tensor reduced along its innermost contiguous axis now takes one pass over
the slice; everything else keeps the composition, which stays the reference.

This file pins:

  * *routing*: `_C._extremum_index_counters()` counts both paths, and a call
    takes the one-pass path exactly when claimed (CPU, `float32`/`float64`,
    the reduced axis innermost and contiguous), counted, never inferred from
    speed;
  * *agreement*: on every rule row (ties, all equal, all `-inf`, `+inf`,
    NaN first, in the middle and last, `-0.0` before and after `0.0`, and a
    pseudo-random batch of logits rows with injected ties) the one-pass index
    equals a Python statement of the rule and equals the composition's index
    on the same values in an unclaimed layout, for all six ops, both dtypes;
    the value half of `max.dim`/`min.dim`/`max`/`min` is the element at that
    index;
  * *shapes*: `keepdim` and the flatten form keep upstream's shapes.

What this file cannot find: speed. The next Gemstone `batching_breakdown`
run's `aten.argmax` row measures it.
"""

import json
import math
import struct

import _C
import _skip

_DTYPES = {"float32": _C.float32, "float64": _C.float64}


def _counters():
    return _C._extremum_index_counters()


def _dense(values, shape, dtype):
    return _C._tensor_from_flat(list(values), list(shape), dtype=dtype)


def _flat(t):
    out, stack = [], [t.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _held(value, dname):
    """The element as a tensor of `dname` holds it: a row's Python float is
    rounded to float32 for a float32 tensor (`1e-30` is stored as
    `1.0000000031710769e-30`), so the value half is compared with what the
    tensor holds rather than with the float64 literal."""
    if dname == "float32" and isinstance(value, float):
        return struct.unpack("f", struct.pack("f", value))[0]
    return value


def _same(a, b):
    """Equal as values, NaN equal to NaN, signed zeros told apart."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return a == b


def _ref_index(row, want_max):
    """The rule in Python: the first NaN's index, else the first index no
    later element beats."""
    for i, x in enumerate(row):
        if math.isnan(x):
            return i
    best, at = row[0], 0
    for i, x in enumerate(row[1:], 1):
        if (x > best) if want_max else (x < best):
            best, at = x, i
    return at


# --- the rule rows --------------------------------------------------------------

_INF = float("inf")
_NAN = float("nan")
_ROWS = {
    "ties": [1.0, 3.0, 3.0, 2.0, 3.0, 1.0],
    "all_equal": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    "all_neg_inf": [-_INF, -_INF, -_INF, -_INF, -_INF, -_INF],
    "pos_inf_inside": [1.0, _INF, 2.0, _INF, -_INF, 0.0],
    "nan_first": [_NAN, 5.0, 1.0, 7.0, 2.0, 7.0],
    "nan_middle": [5.0, 1.0, 7.0, _NAN, 2.0, 9.0],
    "nan_last": [5.0, 1.0, 7.0, 2.0, 9.0, _NAN],
    "two_nans": [5.0, _NAN, 7.0, _NAN, 2.0, 9.0],
    "neg_zero_first": [-0.0, 0.0, -0.0, 0.0, -1.0, 1e-30],
    "pos_zero_first": [0.0, -0.0, 0.0, -0.0, -1.0, 1e-30],
    "descending": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    "ascending": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
_N = 6


def _lcg(n, seed):
    x = seed & 0xFFFFFFFF
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        yield (x / 0x7FFFFFFF) * 20.0 - 10.0


def _logits_rows(rows, n, seed):
    """Pseudo-random rows with a tie for the maximum and for the minimum
    injected into each, at positions that differ per row."""
    out = []
    for r in range(rows):
        row = list(_lcg(n, seed + 97 * r))
        hi, lo = max(row), min(row)
        row[(7 * r + 3) % n] = hi
        row[(11 * r + 5) % n] = hi
        row[(13 * r + 1) % n] = lo
        row[(17 * r + 2) % n] = lo
        out.append(row)
    return out


# --- the two layouts: claimed (one pass) and unclaimed (the composition) ------


def _claimed(rows, dtype):
    """`[len(rows), n]` contiguous: the innermost axis is the slice."""
    flat = [x for row in rows for x in row]
    return _dense(flat, [len(rows), len(rows[0])], dtype), 1


def _unclaimed(rows, dtype):
    """The same values as the transpose of an `[n, len(rows)]` tensor: the
    reduced axis has stride 1 but the layout is not contiguous (two or more
    rows), so the composition answers."""
    assert len(rows) >= 2
    n = len(rows[0])
    flat = [rows[r][i] for i in range(n) for r in range(len(rows))]
    base = _dense(flat, [n, len(rows)], dtype)
    return _C._aten_dispatch("aten.t.default", base), 1


def _indices(op, t, dim):
    if op in ("argmax", "argmin"):
        got = _C._aten_dispatch(f"aten.{op}.default", t, dim, False)
        return _flat(got), None
    pair = _C._aten_dispatch(f"aten.{op}.dim", t, dim, False)
    return _flat(pair[1]), _flat(pair[0])


_OPS = (("argmax", True), ("argmin", False), ("max", True), ("min", False))


# --- tests ---------------------------------------------------------------------


def test_counters_count_both_paths_exactly():
    before = _counters()
    assert set(before) == {"one_pass_calls", "composed_calls"}, before
    rows = [_ROWS["ties"], _ROWS["descending"]]
    t32, dim = _claimed(rows, _C.float32)
    _C._aten_dispatch("aten.argmax.default", t32, dim, False)
    a = _counters()
    assert (a["one_pass_calls"] - before["one_pass_calls"], a["composed_calls"] - before["composed_calls"]) == (1, 0), (before, a)
    # The transpose: stride 1 on the reduced axis, not contiguous: composed.
    tt, _ = _unclaimed(rows, _C.float32)
    _C._aten_dispatch("aten.argmax.default", tt, 1, False)
    b = _counters()
    assert (b["one_pass_calls"] - a["one_pass_calls"], b["composed_calls"] - a["composed_calls"]) == (0, 1), (a, b)
    # Reducing axis 0 of the contiguous tensor: not the innermost axis: composed.
    _C._aten_dispatch("aten.argmax.default", t32, 0, False)
    c = _counters()
    assert (c["one_pass_calls"] - b["one_pass_calls"], c["composed_calls"] - b["composed_calls"]) == (0, 1), (b, c)
    # An integer tensor: composed (the one-pass path claims floats only).
    ti = _dense([3, 1, 4, 1, 5, 9], [2, 3], _C.int64)
    _C._aten_dispatch("aten.argmax.default", ti, 1, False)
    d = _counters()
    assert (d["one_pass_calls"] - c["one_pass_calls"], d["composed_calls"] - c["composed_calls"]) == (0, 1), (c, d)
    # The flatten form (`dim=None`) reduces the flattened, contiguous tensor: one pass.
    _C._aten_dispatch("aten.argmax.default", t32)
    e = _counters()
    assert (e["one_pass_calls"] - d["one_pass_calls"], e["composed_calls"] - d["composed_calls"]) == (1, 0), (d, e)
    # `max.dim`, `min.dim`, the whole-tensor `max` and `min`, `argmin`: all through the one rule.
    for op, args in (("aten.max.dim", (dim, False)), ("aten.min.dim", (dim, False)), ("aten.max.default", ()),
                     ("aten.min.default", ()), ("aten.argmin.default", (dim, False))):
        f0 = _counters()
        _C._aten_dispatch(op, t32, *args)
        f1 = _counters()
        assert f1["one_pass_calls"] - f0["one_pass_calls"] == 1, (op, f0, f1)
    print(f"    extremum_index: one pass {e['one_pass_calls']} calls, composed {e['composed_calls']} calls so far")


def test_every_rule_row_answers_the_rule_and_the_composition():
    names = list(_ROWS)
    rows = [_ROWS[n] for n in names]
    problems = []
    for dname, dtype in _DTYPES.items():
        claimed, cdim = _claimed(rows, dtype)
        unclaimed, udim = _unclaimed(rows, dtype)
        for op, want_max in _OPS:
            before = _counters()
            got, values = _indices(op, claimed, cdim)
            mid = _counters()
            ref, ref_values = _indices(op, unclaimed, udim)
            after = _counters()
            assert mid["one_pass_calls"] - before["one_pass_calls"] == 1, (op, dname, before, mid)
            assert after["composed_calls"] - mid["composed_calls"] == 1, (op, dname, mid, after)
            for r, name in enumerate(names):
                want = _ref_index(rows[r], want_max)
                if got[r] != want:
                    problems.append(f"{op} {dname} {name}: one pass {got[r]}, the rule {want}")
                if got[r] != ref[r]:
                    problems.append(f"{op} {dname} {name}: one pass {got[r]}, the composition {ref[r]}")
                if values is not None and not _same(values[r], _held(rows[r][want], dname)):
                    problems.append(f"{op} {dname} {name}: value {values[r]!r}, the element at {want} is "
                                    f"{_held(rows[r][want], dname)!r}")
    assert not problems, "\n".join(problems)
    print(f"    extremum_index: {len(names)} rule rows x {len(_OPS)} ops x {len(_DTYPES)} dtypes agree with the rule and the composition")


def test_a_batch_of_logits_rows_takes_one_pass_and_matches():
    rows = _logits_rows(4, 4096, seed=11)
    for dname, dtype in _DTYPES.items():
        claimed, dim = _claimed(rows, dtype)
        unclaimed, udim = _unclaimed(rows, dtype)
        for op, want_max in _OPS:
            before = _counters()
            got, values = _indices(op, claimed, dim)
            assert _counters()["one_pass_calls"] - before["one_pass_calls"] == 1, (op, dname)
            ref, _ = _indices(op, unclaimed, udim)
            for r in range(4):
                want = _ref_index(rows[r], want_max)
                assert got[r] == want == ref[r], (op, dname, r, got[r], want, ref[r])
                if values is not None:
                    assert _same(values[r], _held(rows[r][want], dname)), (op, dname, r, values[r], rows[r][want])
    print("    extremum_index: 4 x 4096 logits rows with injected ties: one pass == the rule == the composition")


def test_the_whole_tensor_max_and_min_read_the_element_at_the_index():
    for dname, dtype in _DTYPES.items():
        for name, row in _ROWS.items():
            t = _dense(row, [2, 3], dtype)
            for op, want_max in (("max", True), ("min", False)):
                before = _counters()
                got = _C._aten_dispatch(f"aten.{op}.default", t)
                assert _counters()["one_pass_calls"] - before["one_pass_calls"] == 1, (op, dname, name)
                want = _held(row[_ref_index(row, want_max)], dname)
                assert _same(got.item(), want), (op, dname, name, got.item(), want)
    print("    extremum_index: whole-tensor max and min: the element at the one-pass index")


def test_keepdim_and_the_flatten_form_keep_their_shapes():
    rows = [_ROWS["ties"], _ROWS["nan_middle"], _ROWS["descending"]]
    t, dim = _claimed(rows, _C.float32)
    assert list(_C._aten_dispatch("aten.argmax.default", t, dim, True).shape) == [3, 1]
    assert list(_C._aten_dispatch("aten.argmax.default", t, dim, False).shape) == [3]
    whole = _C._aten_dispatch("aten.argmax.default", t)
    assert list(whole.shape) == [], list(whole.shape)
    assert whole.item() == _ref_index([x for r in rows for x in r], True)
    pair = _C._aten_dispatch("aten.max.dim", t, dim, True)
    assert list(pair[0].shape) == [3, 1] and list(pair[1].shape) == [3, 1]
    one = _dense([4.0], [1, 1], _C.float32)
    assert _C._aten_dispatch("aten.argmax.default", one, 1, False).tolist() == [0]
    print("    extremum_index: keepdim, the flatten form and a one-element slice keep upstream's shapes")


_SPELL = r'''
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented") == (sys.argv[1] == "shim"), "the wrong torch"
nan = float("nan")
x = torch.tensor([[3.0, 1.0, 2.0], [0.5, nan, -1.0]])
i = torch.tensor([[4, -2, 7], [1, 1, 0]])
out = {
    "torch.argmin(x, 1)": torch.argmin(x, 1).tolist(),
    "x.argmin(0)": x.argmin(0).tolist(),
    "x.argmin(dim=1, keepdim=True)": x.argmin(dim=1, keepdim=True).tolist(),
    "torch.argmin(x)": torch.argmin(x).tolist(),
    "i.argmin(1)": i.argmin(1).tolist(),
    "i.argmin() dtype": str(i.argmin().dtype),
}
print(json.dumps(out))
'''


def test_torch_argmin_and_tensor_argmin_answer_as_upstream():
    """`torch.argmin` and `Tensor.argmin` (`overloads.json`, `methods.json`)
    reach the kernel and answer upstream's indices: per dim, keepdim, the
    flatten form, a NaN row (the first NaN wins), int64 ties (the first
    index). The vendored shim and upstream torch, each in its own process."""
    import os
    import pathlib
    import subprocess
    import sys

    repo = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
    vendor = repo / "torchnative" / "python"
    if not (vendor / "torch" / "__init__.py").is_file():
        _skip.skip("vendored tree not built")
        return
    got = {}
    for side in ("shim", "upstream"):
        env = dict(os.environ)
        if side == "shim":
            env["PYTHONPATH"] = str(vendor)
            env["TORCH_USE_RTLD_GLOBAL"] = "1"
        else:
            env.pop("PYTHONPATH", None)
            env.pop("TORCH_USE_RTLD_GLOBAL", None)
        proc = subprocess.run([sys.executable, "-c", _SPELL, side], capture_output=True, text=True,
                              env=env, timeout=600)
        if side == "upstream" and proc.returncode != 0 and "the wrong torch" in proc.stderr:
            _skip.skip("upstream torch not importable without the vendored tree")
            return
        assert proc.returncode == 0, (side, proc.stderr[-2000:])
        got[side] = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["upstream"]["torch.argmin(x, 1)"] == [1, 1], got["upstream"]
    assert got["shim"] == got["upstream"], got
    print(f"    extremum_index: {len(got['shim'])} torch.argmin / Tensor.argmin calls answer as upstream")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    return _skip.run_tests(items, "ops/test_extremum_index")


if __name__ == "__main__":
    raise SystemExit(1 if _main() else 0)
