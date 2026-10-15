"""The sign of a zero answer from `min()` / `max()` (issue #49).

Upstream's whole-tensor `min()`/`max()` order `-0.0 < +0.0`: `min` answers
`-0.0` whenever a `-0.0` is present and `max` answers `+0.0` whenever a `+0.0`
is, wherever it sits. Measured on torch 2.13.0
(`.caches/spike-venv`) for every position of lengths 1..69 in float32, float16
and bfloat16, zero exceptions. The `dim=` overloads follow a different rule
(the first equal element), so `min([0., -0.], dim=0)` is `0.0` upstream and
that is pinned here too, so a fix cannot overreach into them.

This build answered the first equal element for both, so `min([0., -0., 0.,
-0.])` was `0.0`. The golden comparator could not see it (`isclose(0.0, -0.0)`
is True); compare.py now compares the sign for cases marked `signed_zero`, and
`min_default_cases`/`max_default_cases` carry them. This file pins the same
answers without the harness, and re-asks upstream live when the interpreter
has it.

Nullification: `extremum_default` without its `signed_zero_extremum` call
turns `test_min_max_default_order_negative_zero_first` red (the `min` rows
answer `0.0`, the `max` rows `-0.0`).
"""

import math
import os

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _C
import _skip

try:
    import torch as _upstream_torch
except ImportError:  # pragma: no cover - most interpreters running this file
    _upstream_torch = None


_DTYPES = {"float32": _C.float32, "float64": _C.float64,
           "float16": _C.float16, "bfloat16": _C.bfloat16}

# (values, op, expected sign of the zero answer) -- upstream's answers,
# measured as stated in the module docstring.
_CASES = [
    ([0.0, -0.0, 0.0, -0.0], "aten.min.default", -1.0),
    ([-0.0, 0.0, -0.0, 0.0], "aten.min.default", -1.0),
    ([0.0, 0.0, 0.0, -0.0], "aten.min.default", -1.0),
    ([0.0, 0.0], "aten.min.default", 1.0),
    ([0.0, -0.0, 0.0, -0.0], "aten.max.default", 1.0),
    ([-0.0, -0.0, -0.0, 0.0], "aten.max.default", 1.0),
    ([-0.0, -0.0], "aten.max.default", -1.0),
]


def _sign(x):
    return math.copysign(1.0, x)


def _shim(values, dtype):
    return _C._tensor_from_flat(values, [len(values)], _DTYPES[dtype])


def test_min_max_default_order_negative_zero_first():
    problems = []
    for dtype in _DTYPES:
        for values, op, want in _CASES:
            got = _C._aten_dispatch(op, _shim(values, dtype)).tolist()
            if got != 0.0 or _sign(got) != want:
                problems.append("%s(%s) %s = %r, upstream %r"
                                % (op, values, dtype, got, math.copysign(0.0, want)))
    assert not problems, "\n".join(problems)


def test_min_max_dim_keep_the_first_equal_element():
    # `max.dim`/`min.dim` answer the element at the first equal index:
    # `min([0., -0.], dim=0)` is `0.0` and `max([-0., 0.], dim=0)` is `-0.0`
    # upstream (measured). The whole-tensor rule must not leak into them.
    problems = []
    for dtype in _DTYPES:
        for values, op in (([0.0, -0.0], "aten.min.dim"), ([-0.0, 0.0], "aten.max.dim")):
            got = _C._aten_dispatch(op, _shim(values, dtype), 0)
            value, index = got[0].tolist(), got[1].tolist()
            if index != 0 or _sign(value) != _sign(values[0]):
                problems.append("%s(%s) %s = (%r, %r), upstream (%r, 0)"
                                % (op, values, dtype, value, index, values[0]))
    assert not problems, "\n".join(problems)


def test_the_pinned_answers_are_upstreams():
    """Re-asks upstream in this process, so the table above is not a value
    written down once. Skipped by name when upstream is not importable (a
    shim-only interpreter), or when `torch` here is the shim."""
    if _upstream_torch is None or hasattr(_upstream_torch._C, "_aten_implemented"):
        _skip.skip("   (skipped test_the_pinned_answers_are_upstreams: no upstream torch "
                   "in this interpreter)")
        return
    problems = []
    for dtype in _DTYPES:
        for values, op, want in _CASES:
            t = _upstream_torch.tensor(values, dtype=getattr(_upstream_torch, dtype))
            got = (t.min() if op == "aten.min.default" else t.max()).item()
            if _sign(got) != want:
                problems.append("upstream %s(%s) %s = %r, table says sign %r"
                                % (op, values, dtype, got, want))
    assert not problems, "\n".join(problems)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_signedzero")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
