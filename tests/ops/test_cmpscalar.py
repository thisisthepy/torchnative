"""A Python float scalar is narrowed to the comparison dtype (issue #93).

Upstream casts a Python scalar to the comparison dtype before comparing: the
tensor's own float dtype, or the default float (float32) for an integral
tensor. So `float32([0.1]) == 0.1` is True, `float16([inf]) == 65520.0` is
True (65520 rounds to inf in float16), and `int64([16777217]) == 16777216.0`
is True (both become float32 16777216). Measured on torch 2.13.0
(`.caches/spike-venv`, cpu). The cpu path here widened both sides to f64 and
answered False for every one of them; mps already narrowed (PR #83).

A 0-dim float64 *tensor* (not a wrapped Python number) follows ordinary
promotion: it does not promote a dimensioned float tensor (narrowed, so
`float32([0.1]) == tensor(0.1, float64)` is True) but it does promote an
integral one, to float64 (so `int64([16777217]) == tensor(16777216.0,
float64)` is False).

Nullification: in `compare_scalar` (aten.rs), compare the unnarrowed
`other.as_f64()` instead of `float_narrower(ctag)(...)` and
`test_scalar_rows_match_upstream` goes red on every float16/bfloat16/float32
row; drop the default-float cast of an integral operand in `compare_common`
and the int64 row goes red.
"""

import os

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _C
import _skip

try:
    import torch as _upstream_torch
except ImportError:  # pragma: no cover - most interpreters running this file
    _upstream_torch = None


_DTYPES = {"float32": _C.float32, "float64": _C.float64, "float16": _C.float16,
           "bfloat16": _C.bfloat16, "int64": _C.int64}

_INF = float("inf")
_NAN = float("nan")

# (dtype, tensor values, scalar, eq, lt, gt): upstream 2.13.0 cpu answers.
# ne = not eq, le = eq or lt, ge = eq or gt (holds for NaN too).
_ROWS = [
    ("float32", [0.1, 0.2], 0.1, [True, False], [False, False], [False, True]),
    ("float32", [1 / 3, 0.0], 1 / 3, [True, False], [False, True], [False, False]),
    ("float64", [0.30000001192092896, 0.3], 0.3, [False, True], [False, False], [True, False]),
    ("float16", [0.1, 0.0], 0.1, [True, False], [False, True], [False, False]),
    ("float16", [1.0009765625, 1.001953125], 1.0009765625 + 2 ** -12,
     [True, False], [False, False], [False, True]),
    ("float16", [_INF, 65504.0], 65520.0, [True, False], [False, True], [False, False]),
    ("bfloat16", [0.1, 0.0], 0.1, [True, False], [False, True], [False, False]),
    ("bfloat16", [1.0078125, 1.015625], 1.0078125 + 2 ** -9,
     [True, False], [False, False], [False, True]),
    ("bfloat16", [_INF, 3.3895313892515355e38], 2.0 ** 128 * (1 - 2 ** -9),
     [True, False], [False, True], [False, False]),
    ("int64", [16777217, 16777215], 16777216.0, [True, False], [False, True], [False, False]),
    ("float16", [_NAN, 1.0], _NAN, [False, False], [False, False], [False, False]),
    ("float32", [_INF, 1.0], _INF, [True, False], [False, True], [False, False]),
]

# (dtype, values, 0-dim float64 value, eq, lt, gt) for the `.Tensor` overloads.
_ZERO_DIM = [
    ("float32", [0.1, 0.2], 0.1, [True, False], [False, False], [False, True]),
    ("float16", [0.1, 0.0], 0.1, [True, False], [False, True], [False, False]),
    ("int64", [16777217, 16777215], 16777216.0, [False, False], [False, True], [True, False]),
]


def _expected(eq, lt, gt):
    return {
        "eq": eq, "lt": lt, "gt": gt,
        "ne": [not e for e in eq],
        "le": [e or l for e, l in zip(eq, lt)],
        "ge": [e or g for e, g in zip(eq, gt)],
    }


def _shim(values, dtype, shape=None):
    return _C._tensor_from_flat(values, [len(values)] if shape is None else shape,
                                _DTYPES[dtype])


def test_scalar_rows_match_upstream():
    problems = []
    for dtype, values, scalar, eq, lt, gt in _ROWS:
        for name, want in _expected(eq, lt, gt).items():
            got = _C._aten_dispatch("aten.%s.Scalar" % name, _shim(values, dtype), scalar).tolist()
            if got != want:
                problems.append("%s.Scalar(%s %s, %r) = %r, upstream %r"
                                % (name, dtype, values, scalar, got, want))
    assert not problems, "\n".join(problems)


def test_zero_dim_float64_tensor_operand():
    problems = []
    for dtype, values, other, eq, lt, gt in _ZERO_DIM:
        for name, want in _expected(eq, lt, gt).items():
            got = _C._aten_dispatch("aten.%s.Tensor" % name, _shim(values, dtype),
                                    _shim([other], "float64", [])).tolist()
            if got != want:
                problems.append("%s.Tensor(%s %s, 0-dim float64 %r) = %r, upstream %r"
                                % (name, dtype, values, other, got, want))
    assert not problems, "\n".join(problems)


def test_the_pinned_answers_are_upstreams():
    """Re-asks upstream in this process, so the tables above are not values
    written down once. Skipped by name when upstream is not importable, or
    when `torch` here is the shim."""
    if _upstream_torch is None or hasattr(_upstream_torch._C, "_aten_implemented"):
        _skip.skip("   (skipped test_the_pinned_answers_are_upstreams: no upstream torch "
                   "in this interpreter)")
        return
    T = _upstream_torch
    problems = []
    for rows, zero_dim in ((_ROWS, False), (_ZERO_DIM, True)):
        for dtype, values, other, eq, lt, gt in rows:
            t = T.tensor(values, dtype=getattr(T, dtype))
            o = T.tensor(other, dtype=T.float64) if zero_dim else other
            for name, want in _expected(eq, lt, gt).items():
                got = getattr(T, name)(t, o).tolist()
                if got != want:
                    problems.append("upstream %s(%s %s, %r%s) = %r, table says %r"
                                    % (name, dtype, values, other,
                                       " 0-dim" if zero_dim else "", got, want))
    assert not problems, "\n".join(problems)


# add(dimensioned, 0-dim) result dtypes, upstream 2.13.0 cpu (bool+bool via
# `|`, since add refuses it). c10's ResultTypeState: within a category the
# dimensioned operand wins; a higher category on the 0-dim side still
# promotes. issue #93 found the shim promoting float32[2] + tensor(0.1,
# float64) to float64. Complex rows are left out: `aten.ones` cannot build a
# complex tensor in this shim, so combine_categories' complex branch is not
# exercised here.
_PROMOTION = [
    ('bool', 'bool', 'bool'),
    ('bool', 'uint8', 'uint8'),
    ('bool', 'int32', 'int32'),
    ('bool', 'int64', 'int64'),
    ('bool', 'float16', 'float16'),
    ('bool', 'bfloat16', 'bfloat16'),
    ('bool', 'float32', 'float32'),
    ('bool', 'float64', 'float64'),
    ('uint8', 'bool', 'uint8'),
    ('uint8', 'uint8', 'uint8'),
    ('uint8', 'int32', 'uint8'),
    ('uint8', 'int64', 'uint8'),
    ('uint8', 'float16', 'float16'),
    ('uint8', 'bfloat16', 'bfloat16'),
    ('uint8', 'float32', 'float32'),
    ('uint8', 'float64', 'float64'),
    ('int32', 'bool', 'int32'),
    ('int32', 'uint8', 'int32'),
    ('int32', 'int32', 'int32'),
    ('int32', 'int64', 'int32'),
    ('int32', 'float16', 'float16'),
    ('int32', 'bfloat16', 'bfloat16'),
    ('int32', 'float32', 'float32'),
    ('int32', 'float64', 'float64'),
    ('int64', 'bool', 'int64'),
    ('int64', 'uint8', 'int64'),
    ('int64', 'int32', 'int64'),
    ('int64', 'int64', 'int64'),
    ('int64', 'float16', 'float16'),
    ('int64', 'bfloat16', 'bfloat16'),
    ('int64', 'float32', 'float32'),
    ('int64', 'float64', 'float64'),
    ('float16', 'bool', 'float16'),
    ('float16', 'uint8', 'float16'),
    ('float16', 'int32', 'float16'),
    ('float16', 'int64', 'float16'),
    ('float16', 'float16', 'float16'),
    ('float16', 'bfloat16', 'float16'),
    ('float16', 'float32', 'float16'),
    ('float16', 'float64', 'float16'),
    ('bfloat16', 'bool', 'bfloat16'),
    ('bfloat16', 'uint8', 'bfloat16'),
    ('bfloat16', 'int32', 'bfloat16'),
    ('bfloat16', 'int64', 'bfloat16'),
    ('bfloat16', 'float16', 'bfloat16'),
    ('bfloat16', 'bfloat16', 'bfloat16'),
    ('bfloat16', 'float32', 'bfloat16'),
    ('bfloat16', 'float64', 'bfloat16'),
    ('float32', 'bool', 'float32'),
    ('float32', 'uint8', 'float32'),
    ('float32', 'int32', 'float32'),
    ('float32', 'int64', 'float32'),
    ('float32', 'float16', 'float32'),
    ('float32', 'bfloat16', 'float32'),
    ('float32', 'float32', 'float32'),
    ('float32', 'float64', 'float32'),
    ('float64', 'bool', 'float64'),
    ('float64', 'uint8', 'float64'),
    ('float64', 'int32', 'float64'),
    ('float64', 'int64', 'float64'),
    ('float64', 'float16', 'float64'),
    ('float64', 'bfloat16', 'float64'),
    ('float64', 'float32', 'float64'),
    ('float64', 'float64', 'float64'),
]


def _shim_result_dtype(a, b):
    t = _C._aten_dispatch("aten.ones.default", [2], dtype=getattr(_C, a))
    z = _C._aten_dispatch("aten.ones.default", [], dtype=getattr(_C, b))
    op = "aten.bitwise_or.Tensor" if a == b == "bool" else "aten.add.Tensor"
    return str(_C._aten_dispatch(op, t, z).dtype).split(".")[-1]


def test_a_zero_dim_operand_does_not_outrank_a_dimensioned_one():
    problems = []
    for a, b, want in _PROMOTION:
        try:
            got = _shim_result_dtype(a, b)
        except Exception as e:  # noqa: BLE001
            got = "%s: %s" % (type(e).__name__, str(e)[:80])
        if got != want:
            problems.append("add(%s[2], %s 0-dim) -> %s, upstream %s" % (a, b, got, want))
    assert not problems, "%d of %d differ:\n%s" % (len(problems), len(_PROMOTION), "\n".join(problems))


def test_the_promotion_table_is_upstreams():
    if _upstream_torch is None or hasattr(_upstream_torch._C, "_aten_implemented"):
        _skip.skip("   (skipped test_the_promotion_table_is_upstreams: no upstream torch "
                   "in this interpreter)")
        return
    T = _upstream_torch
    problems = []
    for a, b, want in _PROMOTION:
        t = T.ones(2, dtype=getattr(T, a))
        z = T.tensor(1, dtype=getattr(T, b))
        got = \
            str(((t | z) if a == b == "bool" else (t + z)).dtype).split(".")[-1]
        if got != want:
            problems.append("upstream add(%s[2], %s 0-dim) = %s, table says %s" % (a, b, got, want))
    assert not problems, "\n".join(problems)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_cmpscalar")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
