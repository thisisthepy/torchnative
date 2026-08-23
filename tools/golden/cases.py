"""Per-op golden test cases.

The set of *ops* to check is never hardcoded here or anywhere in this
harness -- `compare.py` gets that list from `_C._aten_implemented()` at run
time, so it grows automatically as `rust/torch_c` implements more. What
*is* necessarily hand-written, per op, is which inputs to feed both sides:
`aten.full.default`, `aten.add.Tensor` and `aten.mm.default` take
different arguments and there is no generic way to invent valid calls for
an arbitrary aten op.

To keep that hand-written part from silently going stale as coverage grows,
`compare.py` treats "op is in `_aten_implemented()` but has no entry in
`CASE_BUILDERS`" as a hard failure rather than a skip -- see its
`_missing_case_builder` handling. Adding a new op to the shim without also
adding a case builder here will fail the harness, on purpose.

Every case's `expect` documents what the harness should see:

  "match"       -- both sides must succeed and agree on dtype, shape, and
                   value (within tolerance). This is the default and covers
                   the "do they compute the same answer" question.
  "both_error"  -- both sides are expected to refuse (e.g. mm on non-2D
                   input). Either exception is accepted; the point is that
                   neither one silently computes something.
  "c_error"     -- a documented capability gap: torch computes a result,
                   `_C` refuses. If this ever flips to both succeeding, the
                   case should be promoted to "match" and its real values
                   diffed. If `_C` starts succeeding with the *wrong*
                   answer instead, this will fail loudly.
  "torch_error" -- the mirror image: `_C` computes a result in a case where
                   even upstream torch's own CPU kernel refuses (e.g.
                   uint32 add). Not a shim bug -- torch simply has no
                   kernel -- but worth tracking explicitly rather than
                   silently excluding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import dtypes as dt
from build import pair_from_flat


@dataclass
class Case:
    name: str
    op: str
    run_torch: Callable[[], Any]
    run_c: Callable[[], Any]
    expect: str = "match"  # "match" | "both_error" | "c_error" | "torch_error"
    note: str = ""


# --- aten.full.default -------------------------------------------------

# (fill_value, expect, note) per dtype. Two entries are deliberately live
# regression traps for real bugs found while building this harness (see the
# final report): torch refuses on overflow, `_C` silently wraps/saturates
# instead of refusing. They are left as `expect="match"` -- NOT "c_error"
# -- specifically so the harness keeps failing on them until someone fixes
# rust/torch_c (out of scope for this change) rather than the gap being
# quietly filed away as "known and accepted".
_FULL_FILLS: dict[str, list[tuple[Any, str, str]]] = {
    "float64": [
        (0.0, "match", ""),
        (-12345.6789, "match", ""),
        (3.14159265358979, "match", ""),
        (1e10, "match", ""),
        (-1e10, "match", ""),
    ],
    "float32": [
        (0.0, "match", ""),
        (-12345.678, "match", ""),
        (3.1415927, "match", ""),
        (1e6, "match", ""),
        (-1e6, "match", ""),
    ],
    "float16": [
        (0.0, "match", ""),
        (-100.5, "match", ""),
        (65504.0, "match", "max finite float16"),
        (
            1e6,
            "match",
            "BUG (found by this harness): torch.ops.aten.full.default "
            "refuses fill values that overflow float16 "
            "(RuntimeError: value cannot be converted to type c10::Half "
            "without overflow); _C silently returns inf instead of "
            "refusing. Left as expect=match so the harness keeps failing "
            "until rust/torch_c is fixed.",
        ),
    ],
    "bfloat16": [
        (0.0, "match", ""),
        (-1000.0, "match", ""),
        (1e5, "match", ""),
        (-1e5, "match", ""),
    ],
    "int64": [
        (0, "match", ""),
        (-1234567890123, "match", ""),
        (10**12, "match", ""),
        (-(10**12), "match", ""),
    ],
    "int32": [
        (0, "match", ""),
        (-2147483648, "match", "int32 min"),
        (2147483647, "match", "int32 max"),
        (
            2147483648,
            "match",
            "BUG (found by this harness): torch.ops.aten.full.default "
            "refuses a fill value one past int32 max (RuntimeError: value "
            "cannot be converted to type int without overflow); _C "
            "silently wraps around to int32 min (two's-complement "
            "overflow) instead of refusing. Left as expect=match so the "
            "harness keeps failing until rust/torch_c is fixed.",
        ),
    ],
    "int16": [
        (0, "match", ""),
        (-32768, "match", ""),
        (32767, "match", ""),
    ],
    "uint8": [
        (0, "match", ""),
        (255, "match", "uint8 max"),
        (128, "match", ""),
        (-1, "match", "both sides wrap to 255 -- consistent modular wraparound"),
    ],
    "uint32": [
        (0, "match", ""),
        (4294967295, "match", "uint32 max"),
        (12345, "match", ""),
    ],
}

_FULL_SHAPES_SMALL: list[tuple[int, ...]] = [(), (3,), (2, 3)]
_FULL_SHAPES_EXTRA: list[tuple[int, ...]] = [(0, 3), (1, 4, 1), (50,)]


def _full_case(torch_module, c_module, torch_call, shape, fill, dtype_name, expect, note):
    shape_list = list(shape)

    if dtype_name is None:
        run_torch = lambda: torch_call(shape_list, fill)
        run_c = lambda: c_module._aten_dispatch("aten.full.default", shape_list, fill)
    else:
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        c_dt = dt.c_dtype(c_module, dtype_name)
        run_torch = lambda: torch_call(shape_list, fill, dtype=t_dt)
        run_c = lambda: c_module._aten_dispatch(
            "aten.full.default", shape_list, fill, dtype=c_dt
        )

    name = f"full(shape={shape_list}, fill={fill!r}, dtype={dtype_name or 'inferred'})"
    return Case(
        name=name,
        op="aten.full.default",
        run_torch=run_torch,
        run_c=run_c,
        expect=expect,
        note=note,
    )


def full_cases(torch_module, c_module, torch_call) -> list[Case]:
    cases: list[Case] = []

    # Default dtype inference: torch.full infers int64 from a python int
    # fill value and float32 from a python float one (docs/TORCH_C.md §2).
    for fill, note in [
        (0, "default-dtype int fill -> expect int64"),
        (-7, "default-dtype int fill -> expect int64"),
        (3.5, "default-dtype float fill -> expect float32"),
        (-125000.25, "default-dtype float fill -> expect float32"),
    ]:
        for shape in _FULL_SHAPES_SMALL:
            cases.append(
                _full_case(torch_module, c_module, torch_call, shape, fill, None, "match", note)
            )

    for dtype_name in dt.DEFAULT_DTYPES:
        for fill, expect, note in _FULL_FILLS[dtype_name]:
            for shape in _FULL_SHAPES_SMALL:
                cases.append(
                    _full_case(
                        torch_module, c_module, torch_call, shape, fill, dtype_name, expect, note
                    )
                )

    for shape in _FULL_SHAPES_EXTRA:
        cases.append(
            _full_case(
                torch_module, c_module, torch_call, shape, 3.5, "float32", "match", "extra shape coverage"
            )
        )
        cases.append(
            _full_case(
                torch_module, c_module, torch_call, shape, 7, "int64", "match", "extra shape coverage"
            )
        )

    return cases


# --- aten.add.Tensor -----------------------------------------------------

_FLOAT_ADD_DTYPES = ["float64", "float32", "float16", "bfloat16"]


def _float_add_scenarios(big: float) -> list[dict]:
    return [
        dict(a_flat=[1.0, 2.0, 3.0, 4.0], a_shape=(2, 2), b_flat=[10.0, 20.0], b_shape=(1, 2), alpha=None, note="row-broadcast"),
        dict(a_flat=[1.0, 2.0, 3.0, 4.0], a_shape=(2, 2), b_flat=[10.0, 20.0], b_shape=(1, 2), alpha=2.0, note="row-broadcast, alpha=2"),
        dict(a_flat=[0.0, -5.0, big, -big], a_shape=(2, 2), b_flat=[0.0, 5.0, -big, big], b_shape=(2, 2), alpha=-1.5, note="boundary values (0/neg/large), alpha=-1.5"),
        dict(a_flat=[5.0], a_shape=(), b_flat=[1.0, 2.0, 3.0, 4.0], b_shape=(2, 2), alpha=None, note="scalar (0-d) broadcast"),
        dict(a_flat=[1.0] * 24, a_shape=(2, 3, 4), b_flat=[1.0, 2.0, 3.0, 4.0], b_shape=(4,), alpha=0.0, note="3D broadcast, alpha=0"),
        dict(a_flat=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], a_shape=(2, 3), b_flat=[10.0, 20.0], b_shape=(2, 1), alpha=None, note="column-broadcast"),
    ]


_FLOAT_ADD_MAGNITUDE = {"float64": 1e5, "float32": 1e5, "float16": 100.0, "bfloat16": 1e4}


def _int_add_scenarios_signed(big: int) -> list[dict]:
    return [
        dict(a_flat=[1, 2, 3, 4], a_shape=(2, 2), b_flat=[10, 20, 30, 40], b_shape=(2, 2), alpha=None, note="elementwise"),
        dict(a_flat=[1, 2, 3, 4], a_shape=(2, 2), b_flat=[10, 20, 30, 40], b_shape=(2, 2), alpha=3, note="elementwise, alpha=3"),
        dict(a_flat=[0, -5, big, -big], a_shape=(2, 2), b_flat=[0, 5, -big, big], b_shape=(2, 2), alpha=-1, note="boundary values (0/neg/large), alpha=-1"),
        dict(a_flat=[7], a_shape=(), b_flat=[1, 2, 3, 4], b_shape=(2, 2), alpha=None, note="scalar (0-d) broadcast"),
    ]


def _uint_add_scenarios(big: int) -> list[dict]:
    return [
        dict(a_flat=[1, 2, 3, 4], a_shape=(2, 2), b_flat=[10, 20, 30, 40], b_shape=(2, 2), alpha=None, note="elementwise"),
        dict(a_flat=[0, 1, big, big - 1], a_shape=(2, 2), b_flat=[0, 1, 0, 1], b_shape=(2, 2), alpha=1, note="boundary values (0/large), alpha=1"),
        dict(a_flat=[7], a_shape=(), b_flat=[1, 2, 3, 4], b_shape=(2, 2), alpha=None, note="scalar (0-d) broadcast"),
    ]


def _add_case(torch_module, c_module, torch_call, dtype_name, scenario) -> Case:
    a_t, a_c = pair_from_flat(torch_module, c_module, scenario["a_flat"], scenario["a_shape"], dtype_name)
    b_t, b_c = pair_from_flat(torch_module, c_module, scenario["b_flat"], scenario["b_shape"], dtype_name)
    alpha = scenario["alpha"]

    if alpha is None:
        run_torch = lambda: torch_call(a_t, b_t)
        run_c = lambda: c_module._aten_dispatch("aten.add.Tensor", a_c, b_c)
    else:
        run_torch = lambda: torch_call(a_t, b_t, alpha=alpha)
        run_c = lambda: c_module._aten_dispatch("aten.add.Tensor", a_c, b_c, alpha=alpha)

    name = f"add(dtype={dtype_name}, a_shape={scenario['a_shape']}, b_shape={scenario['b_shape']}, alpha={alpha}) [{scenario['note']}]"
    return Case(name=name, op="aten.add.Tensor", run_torch=run_torch, run_c=run_c)


def add_cases(torch_module, c_module, torch_call) -> list[Case]:
    cases: list[Case] = []

    for dtype_name in _FLOAT_ADD_DTYPES:
        for scenario in _float_add_scenarios(_FLOAT_ADD_MAGNITUDE[dtype_name]):
            cases.append(_add_case(torch_module, c_module, torch_call, dtype_name, scenario))

    for dtype_name, big in [("int64", 10**9), ("int32", 10**6), ("int16", 1000)]:
        for scenario in _int_add_scenarios_signed(big):
            cases.append(_add_case(torch_module, c_module, torch_call, dtype_name, scenario))

    for scenario in _uint_add_scenarios(200):
        cases.append(_add_case(torch_module, c_module, torch_call, "uint8", scenario))

    # Known gap: torch's own CPU kernel has no `add_stub` for UInt32, so
    # torch refuses while `_C` (via candle) computes fine. Not a shim bug.
    for scenario in _uint_add_scenarios(4_000_000_000):
        a_t, a_c = pair_from_flat(torch_module, c_module, scenario["a_flat"], scenario["a_shape"], "uint32")
        b_t, b_c = pair_from_flat(torch_module, c_module, scenario["b_flat"], scenario["b_shape"], "uint32")
        run_torch = lambda a_t=a_t, b_t=b_t: torch_call(a_t, b_t)
        run_c = lambda a_c=a_c, b_c=b_c: c_module._aten_dispatch("aten.add.Tensor", a_c, b_c)
        cases.append(
            Case(
                name=f"add(dtype=uint32, a_shape={scenario['a_shape']}, b_shape={scenario['b_shape']}) [{scenario['note']}]",
                op="aten.add.Tensor",
                run_torch=run_torch,
                run_c=run_c,
                expect="torch_error",
                note="torch has no CPU add kernel for uint32 (NotImplementedError: \"add_stub\" not implemented for 'UInt32'); _C computes it via candle. Not a shim bug -- torch is the one missing the kernel.",
            )
        )

    # Known gap, explicitly called out in docs/TORCH_C.md §2: dtype
    # promotion is refused rather than guessed.
    af_t, af_c = pair_from_flat(torch_module, c_module, [1.0], [1], "float32")
    bf_t, bf_c = pair_from_flat(torch_module, c_module, [1.0], [1], "float64")
    cases.append(
        Case(
            name="add(dtype-mismatch float32 vs float64)",
            op="aten.add.Tensor",
            run_torch=lambda: torch_call(af_t, bf_t),
            run_c=lambda: c_module._aten_dispatch("aten.add.Tensor", af_c, bf_c),
            expect="c_error",
            note="dtype promotion is deliberately unimplemented (docs/TORCH_C.md §2); torch promotes to float64, _C refuses rather than guess a promotion table.",
        )
    )

    return cases


# --- aten.mm.default -------------------------------------------------------

_MM_MATCH_DTYPES = ["float32", "float64", "float16"]
_MM_C_ERROR_DTYPES = ["int64", "int32", "int16", "uint8", "bfloat16"]


def _mm_case(torch_module, c_module, torch_call, dtype_name, a_flat, a_shape, b_flat, b_shape, expect="match", note=""):
    a_t, a_c = pair_from_flat(torch_module, c_module, a_flat, a_shape, dtype_name)
    b_t, b_c = pair_from_flat(torch_module, c_module, b_flat, b_shape, dtype_name)
    name = f"mm(dtype={dtype_name}, a_shape={a_shape}, b_shape={b_shape}) [{note or 'plain'}]"
    return Case(
        name=name,
        op="aten.mm.default",
        run_torch=lambda: torch_call(a_t, b_t),
        run_c=lambda: c_module._aten_dispatch("aten.mm.default", a_c, b_c),
        expect=expect,
        note=note,
    )


def mm_cases(torch_module, c_module, torch_call) -> list[Case]:
    cases: list[Case] = []

    for dtype_name in _MM_MATCH_DTYPES:
        big = 50.0 if dtype_name == "float16" else 1000.0
        square = [1.0, 2.0, 3.0, 4.0]
        square_b = [5.0, 6.0, 7.0, 8.0]  # torch.mm([[1,2],[3,4]],[[5,6],[7,8]]) == [[19,22],[43,50]]
        cases.append(_mm_case(torch_module, c_module, torch_call, dtype_name, square, (2, 2), square_b, (2, 2), note="known 2x2 answer"))

        rect_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        rect_b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        cases.append(_mm_case(torch_module, c_module, torch_call, dtype_name, rect_a, (3, 2), rect_b, (2, 4), note="rectangular"))

        dot_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        dot_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        cases.append(_mm_case(torch_module, c_module, torch_call, dtype_name, dot_a, (1, 5), dot_b, (5, 1), note="row-vector . column-vector (dot product)"))

        boundary_a = [0.0, -1.0, big, -big, 2.0, -2.0, 0.0, 5.0, 1.0, 1.0, 1.0, 1.0, -5.0, 5.0, -5.0, 5.0]
        identity_4x4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        cases.append(
            _mm_case(
                torch_module,
                c_module,
                torch_call,
                dtype_name,
                boundary_a,
                (4, 4),
                identity_4x4,
                (4, 4),
                note="boundary values (0/neg/large) x identity",
            )
        )

    # Known gaps: candle's matmul kernel does not support these dtypes at
    # all (RuntimeError: "candle: unsupported dtype <X> for op matmul"),
    # while torch's CPU addmm does.
    for dtype_name in _MM_C_ERROR_DTYPES:
        a_flat = [1, 2, 3, 4] if dtype_name != "uint8" else [1, 2, 3, 4]
        b_flat = [1, 0, 0, 1]
        cases.append(
            _mm_case(
                torch_module,
                c_module,
                torch_call,
                dtype_name,
                a_flat,
                (2, 2),
                b_flat,
                (2, 2),
                expect="c_error",
                note=f"candle's matmul has no kernel for {dtype_name}; torch's CPU addmm does. See docs/TORCH_C.md §2 for int64 specifically -- int32/int16/uint8/bfloat16 have the same gap, found while building this harness.",
            )
        )

    # Both refuse: torch's own addmm has no UInt32 kernel either, and
    # candle refuses it too (for an unrelated reason). Either way, neither
    # side should silently produce a value.
    cases.append(
        _mm_case(
            torch_module,
            c_module,
            torch_call,
            "uint32",
            [1, 2, 3, 4],
            (2, 2),
            [1, 0, 0, 1],
            (2, 2),
            expect="both_error",
            note="torch: NotImplementedError (\"addmm_impl_cpu_\" not implemented for 'UInt32'); _C: candle has no matmul kernel for U32 either.",
        )
    )

    # mm must reject non-2D input on both sides -- candle's matmul batches,
    # torch.mm does not, and the shim explicitly guards against silently
    # standing in for bmm/matmul (docs/TORCH_C.md §2).
    a3_t, a3_c = pair_from_flat(torch_module, c_module, [1.0] * 8, (2, 2, 2), "float32")
    cases.append(
        Case(
            name="mm(3D input rejected on both sides)",
            op="aten.mm.default",
            run_torch=lambda: torch_call(a3_t, a3_t),
            run_c=lambda: c_module._aten_dispatch("aten.mm.default", a3_c, a3_c),
            expect="both_error",
            note="mm must not silently fall back to a batched matmul.",
        )
    )

    return cases


CASE_BUILDERS: dict[str, Callable[[Any, Any, Callable], list[Case]]] = {
    "aten.full.default": full_cases,
    "aten.add.Tensor": add_cases,
    "aten.mm.default": mm_cases,
}
