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

Below the three ops this harness started with (full/add/mm) are 16 more
CASE_BUILDERS entries pre-seeded for ops rust/torch_c is actively
implementing but has not landed yet -- see the longer note right before
`arange_default_cases` for which ops, why, and how `compare.py` keeps them
from failing the harness before they exist.
"""

from __future__ import annotations

import math
import platform
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
    # Only consulted when expect == "match" and both sides succeeded.
    # None means "use the default dtype/shape/value pipeline in compare.py".
    # Set this instead for results that pipeline can't handle: a plain
    # Python bool (is_floating_point), uninitialized memory whose bytes are
    # meaningless (empty), or a random draw whose *sequence* can't be
    # matched across two independent RNG implementations even with a
    # matched seed (randint). Signature: (torch_result, c_result) ->
    # (ok, detail_message).
    value_check: Callable[[Any, Any], tuple[bool, str]] | None = None


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


# --- pre-seeded case builders for ops rust/torch_c does not implement yet --
#
# docs/C_SURFACE.md traced a small Llama forward+generate() against real
# upstream torch and found exactly 13 torch.<op> names actually get called
# (not just referenced -- called): arange, argmax, cat, embedding, empty,
# full, is_floating_point, isin, ones, pow, randint, rsqrt, tensor. `full`
# already has a builder above (aten.full.default, one of the three ops this
# harness started with). The other 12 are written here ahead of the
# implementation that is landing them in rust/torch_c, so the moment an op
# shows up in `_C._aten_implemented()` the golden comparison for it is
# already in place -- no gap between "implemented" and "checked".
#
# Until an op is actually implemented, `compare.py`'s `run()` simply never
# calls its builder (it only iterates `_aten_implemented()`), so having a
# builder registered here for an unimplemented op is inert: it costs
# nothing and cannot fail the run. `run()` prints how many are waiting
# under "PENDING", separately from the pass/fail count.
#
# Three of these ops don't produce a plain value-comparable tensor, so they
# use `Case.value_check` instead of relying on compare.py's default
# dtype/shape/value pipeline:
#   - `is_floating_point` returns a Python bool, not a 0-d Tensor.
#   - `empty` returns uninitialized memory -- there is no "correct" value
#     to diff, only dtype and shape are meaningful.
#   - `randint` returns a random draw. Two independent RNG implementations
#     (torch's CPU generator vs whatever rust/torch_c's backend uses)
#     produce different sequences even given "the same" seed -- a seed only
#     pins a *specific generator's* stream, it says nothing about another
#     generator's algorithm. There is no seed value that makes their
#     outputs equal, so this harness does not attempt to synchronize seeds
#     and instead checks dtype, shape, and that every value falls inside
#     the requested [low, high) range.


def _dtype_shape_only_check(t_res, c_res) -> tuple[bool, str]:
    """dtype + shape must agree; the values themselves are not meaningful
    (see `empty_cases` -- uninitialized memory has no "correct" value)."""
    t_dtype, c_dtype = dt.dtype_name(t_res.dtype), dt.dtype_name(c_res.dtype)
    if t_dtype != c_dtype:
        return False, f"dtype mismatch: torch={t_dtype} c={c_dtype}"
    t_shape = tuple(int(x) for x in t_res.shape)
    c_shape = tuple(int(x) for x in c_res.shape)
    if t_shape != c_shape:
        return False, f"shape mismatch: torch={t_shape} c={c_shape}"
    return True, f"dtype={t_dtype} shape={t_shape} (values unchecked -- uninitialized memory)"


def _flatten_values(x) -> list:
    if isinstance(x, list):
        out: list = []
        for v in x:
            out.extend(_flatten_values(v))
        return out
    return [x]


def _range_check(lo, hi):
    """dtype + shape must agree, and every value on both sides must land in
    [lo, hi); the *sequence* is deliberately not compared -- see the module
    note above on why two RNG implementations can't be seeded to match."""

    def check(t_res, c_res) -> tuple[bool, str]:
        t_dtype, c_dtype = dt.dtype_name(t_res.dtype), dt.dtype_name(c_res.dtype)
        if t_dtype != c_dtype:
            return False, f"dtype mismatch: torch={t_dtype} c={c_dtype}"
        t_shape = tuple(int(x) for x in t_res.shape)
        c_shape = tuple(int(x) for x in c_res.shape)
        if t_shape != c_shape:
            return False, f"shape mismatch: torch={t_shape} c={c_shape}"
        for label, res in (("torch", t_res), ("c", c_res)):
            for v in _flatten_values(res.tolist()):
                if not (lo <= v < hi):
                    return False, f"{label} produced {v!r}, outside requested range [{lo}, {hi})"
        return True, f"dtype={t_dtype} shape={t_shape}, all values within [{lo}, {hi}) (sequence unchecked -- see note above)"

    return check


def _scalar_match_check(t_res, c_res) -> tuple[bool, str]:
    """For ops returning a plain Python scalar rather than a Tensor (e.g.
    is_floating_point -> bool)."""
    if type(t_res) is not type(c_res):
        return (
            False,
            f"type mismatch: torch={type(t_res).__name__}({t_res!r}) c={type(c_res).__name__}({c_res!r})",
        )
    if t_res != c_res:
        return False, f"value mismatch: torch={t_res!r} c={c_res!r}"
    return True, f"scalar match: {t_res!r}"


# --- aten.arange.default / .start / .start_step -----------------------------

_ARANGE_INT_DTYPES = ["int64", "int32", "int16", "uint8", "uint32"]
_ARANGE_FLOAT_DTYPES = ["float64", "float32", "float16", "bfloat16"]


def arange_default_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.arange.default"
    cases: list[Case] = []

    for end, note in [(5, "dtype inferred from int end -> int64"), (0, "zero end -> empty result"), (1, "single element")]:
        cases.append(
            Case(
                name=f"arange(end={end}, dtype=inferred)",
                op=op,
                run_torch=lambda end=end: torch_call(end),
                run_c=lambda end=end: c_module._aten_dispatch(op, end),
                note=note,
            )
        )
    cases.append(
        Case(
            name="arange(end=5.0, dtype=inferred)",
            op=op,
            run_torch=lambda: torch_call(5.0),
            run_c=lambda: c_module._aten_dispatch(op, 5.0),
            note="dtype inferred from float end -> float32",
        )
    )

    for dtype_name in _ARANGE_INT_DTYPES + _ARANGE_FLOAT_DTYPES:
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        c_dt = dt.c_dtype(c_module, dtype_name)
        for end in [0, 1, 20]:
            cases.append(
                Case(
                    name=f"arange(end={end}, dtype={dtype_name})",
                    op=op,
                    run_torch=lambda end=end, t_dt=t_dt: torch_call(end, dtype=t_dt),
                    run_c=lambda end=end, c_dt=c_dt: c_module._aten_dispatch(op, end, dtype=c_dt),
                    note="explicit dtype",
                )
            )

    cases.append(
        Case(
            name="arange(end=-1, negative end rejected)",
            op=op,
            run_torch=lambda: torch_call(-1),
            run_c=lambda: c_module._aten_dispatch(op, -1),
            expect="both_error",
            note="upper/lower bound sign inconsistent with (implicit) step -- torch refuses.",
        )
    )

    return cases


def arange_start_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.arange.start"
    cases: list[Case] = []

    for start, end, note in [
        (2, 5, "dtype inferred -> int64"),
        (-5, 5, "negative start, dtype inferred"),
        (3, 3, "start == end -> empty"),
    ]:
        cases.append(
            Case(
                name=f"arange(start={start}, end={end}, dtype=inferred)",
                op=op,
                run_torch=lambda start=start, end=end: torch_call(start, end),
                run_c=lambda start=start, end=end: c_module._aten_dispatch(op, start, end),
                note=note,
            )
        )

    for dtype_name in _ARANGE_INT_DTYPES + _ARANGE_FLOAT_DTYPES:
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        c_dt = dt.c_dtype(c_module, dtype_name)
        start = 0 if dtype_name in ("uint8", "uint32") else -5
        cases.append(
            Case(
                name=f"arange(start={start}, end=10, dtype={dtype_name})",
                op=op,
                run_torch=lambda start=start, t_dt=t_dt: torch_call(start, 10, dtype=t_dt),
                run_c=lambda start=start, c_dt=c_dt: c_module._aten_dispatch(op, start, 10, dtype=c_dt),
                note="explicit dtype",
            )
        )

    cases.append(
        Case(
            name="arange(start=5, end=0, rejected -- wrong direction for implicit step=1)",
            op=op,
            run_torch=lambda: torch_call(5, 0),
            run_c=lambda: c_module._aten_dispatch(op, 5, 0),
            expect="both_error",
            note="start > end with an implicit positive step -- torch refuses rather than returning empty.",
        )
    )

    return cases


def arange_start_step_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.arange.start_step"
    cases: list[Case] = []

    for start, end, step, note in [
        (0, 10, 2, "positive step, dtype inferred -> int64"),
        (10, 0, -2, "negative step, dtype inferred"),
        (0, 1, 0.3, "fractional step -> dtype inferred float32"),
    ]:
        cases.append(
            Case(
                name=f"arange(start={start}, end={end}, step={step}, dtype=inferred)",
                op=op,
                run_torch=lambda start=start, end=end, step=step: torch_call(start, end, step),
                run_c=lambda start=start, end=end, step=step: c_module._aten_dispatch(op, start, end, step),
                note=note,
            )
        )

    for dtype_name in _ARANGE_FLOAT_DTYPES:
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        c_dt = dt.c_dtype(c_module, dtype_name)
        cases.append(
            Case(
                name=f"arange(start=0, end=2, step=0.25, dtype={dtype_name})",
                op=op,
                run_torch=lambda t_dt=t_dt: torch_call(0, 2, 0.25, dtype=t_dt),
                run_c=lambda c_dt=c_dt: c_module._aten_dispatch(op, 0, 2, 0.25, dtype=c_dt),
                note="explicit float dtype, fractional step",
            )
        )

    for dtype_name in ["int64", "int32", "int16"]:
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        c_dt = dt.c_dtype(c_module, dtype_name)
        cases.append(
            Case(
                name=f"arange(start=20, end=0, step=-3, dtype={dtype_name})",
                op=op,
                run_torch=lambda t_dt=t_dt: torch_call(20, 0, -3, dtype=t_dt),
                run_c=lambda c_dt=c_dt: c_module._aten_dispatch(op, 20, 0, -3, dtype=c_dt),
                note="explicit int dtype, negative step",
            )
        )

    cases.append(
        Case(
            name="arange(start=0, end=5, step=0, rejected)",
            op=op,
            run_torch=lambda: torch_call(0, 5, 0),
            run_c=lambda: c_module._aten_dispatch(op, 0, 5, 0),
            expect="both_error",
            note="step must be nonzero -- torch refuses.",
        )
    )
    cases.append(
        Case(
            name="arange(start=0, end=5, step=-1, sign-mismatched, rejected)",
            op=op,
            run_torch=lambda: torch_call(0, 5, -1),
            run_c=lambda: c_module._aten_dispatch(op, 0, 5, -1),
            expect="both_error",
            note="upper/lower bound sign inconsistent with step -- torch refuses.",
        )
    )

    return cases


# --- aten.argmax.default -----------------------------------------------------

_ARGMAX_DTYPES = ["float64", "float32", "float16", "bfloat16", "int64", "int32"]


def argmax_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.argmax.default"
    cases: list[Case] = []

    # Flat values chosen so the maximum is unique in every reduced slice --
    # ties are broken by implementation-defined rules that torch and candle
    # need not agree on, and that disagreement would not be a real bug.
    scenarios = [
        dict(flat=[1, 5, 2, 9, 0, 3], shape=(2, 3), dim=None, keepdim=False, note="global argmax, flattened"),
        dict(flat=[1, 5, 2, 9, 0, 3], shape=(2, 3), dim=1, keepdim=False, note="argmax along last dim"),
        dict(flat=[1, 5, 2, 9, 0, 3], shape=(2, 3), dim=1, keepdim=True, note="argmax along last dim, keepdim"),
        dict(flat=[1, 5, 2, 9, 0, 3], shape=(2, 3), dim=0, keepdim=False, note="argmax along first dim"),
        dict(flat=[1, 5, 2, 9, 0, 3], shape=(2, 3), dim=-1, keepdim=False, note="argmax along dim=-1"),
        dict(flat=[7], shape=(1,), dim=None, keepdim=False, note="single element"),
        dict(
            flat=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            shape=(2, 3, 2),
            dim=2,
            keepdim=False,
            note="3D, argmax along last dim",
        ),
        dict(flat=[-5, -1, -9, -3], shape=(2, 2), dim=None, keepdim=False, note="all-negative values"),
    ]

    for dtype_name in _ARGMAX_DTYPES:
        for sc in scenarios:
            a_t, a_c = pair_from_flat(torch_module, c_module, sc["flat"], sc["shape"], dtype_name)
            dim, keepdim = sc["dim"], sc["keepdim"]
            cases.append(
                Case(
                    name=f"argmax(dtype={dtype_name}, shape={sc['shape']}, dim={dim}, keepdim={keepdim}) [{sc['note']}]",
                    op=op,
                    run_torch=lambda a_t=a_t, dim=dim, keepdim=keepdim: torch_call(a_t, dim, keepdim),
                    run_c=lambda a_c=a_c, dim=dim, keepdim=keepdim: c_module._aten_dispatch(op, a_c, dim, keepdim),
                )
            )

    return cases


# --- aten.cat.default ---------------------------------------------------------

_CAT_DTYPES = ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]


def cat_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.cat.default"
    cases: list[Case] = []

    for dtype_name in _CAT_DTYPES:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), dtype_name)
        b_t, b_c = pair_from_flat(torch_module, c_module, [5, 6, 7, 8], (2, 2), dtype_name)
        cases.append(
            Case(
                name=f"cat(dtype={dtype_name}, dim=0, 2 tensors)",
                op=op,
                run_torch=lambda a_t=a_t, b_t=b_t: torch_call([a_t, b_t], 0),
                run_c=lambda a_c=a_c, b_c=b_c: c_module._aten_dispatch(op, [a_c, b_c], 0),
            )
        )
        cases.append(
            Case(
                name=f"cat(dtype={dtype_name}, dim=1, 2 tensors)",
                op=op,
                run_torch=lambda a_t=a_t, b_t=b_t: torch_call([a_t, b_t], 1),
                run_c=lambda a_c=a_c, b_c=b_c: c_module._aten_dispatch(op, [a_c, b_c], 1),
            )
        )
        cases.append(
            Case(
                name=f"cat(dtype={dtype_name}, dim=-1, 2 tensors)",
                op=op,
                run_torch=lambda a_t=a_t, b_t=b_t: torch_call([a_t, b_t], -1),
                run_c=lambda a_c=a_c, b_c=b_c: c_module._aten_dispatch(op, [a_c, b_c], -1),
            )
        )

    # 1D unequal-length concat along the only dim -- common for KV-cache-style growth.
    a1_t, a1_c = pair_from_flat(torch_module, c_module, [1.0, 2.0], (2,), "float32")
    b1_t, b1_c = pair_from_flat(torch_module, c_module, [3.0, 4.0, 5.0], (3,), "float32")
    cases.append(
        Case(
            name="cat(dtype=float32, 1D unequal lengths, dim=0)",
            op=op,
            run_torch=lambda: torch_call([a1_t, b1_t], 0),
            run_c=lambda: c_module._aten_dispatch(op, [a1_c, b1_c], 0),
            note="unequal length along the concat dim, as in incremental KV-cache growth",
        )
    )

    # Three tensors at once.
    c1_t, c1_c = pair_from_flat(torch_module, c_module, [1.0, 2.0], (1, 2), "float32")
    c2_t, c2_c = pair_from_flat(torch_module, c_module, [3.0, 4.0], (1, 2), "float32")
    c3_t, c3_c = pair_from_flat(torch_module, c_module, [5.0, 6.0], (1, 2), "float32")
    cases.append(
        Case(
            name="cat(dtype=float32, dim=0, 3 tensors)",
            op=op,
            run_torch=lambda: torch_call([c1_t, c2_t, c3_t], 0),
            run_c=lambda: c_module._aten_dispatch(op, [c1_c, c2_c, c3_c], 0),
        )
    )

    # Mismatched non-cat-dim shapes -- both sides must refuse.
    m1_t, m1_c = pair_from_flat(torch_module, c_module, [0.0] * 6, (2, 3), "float32")
    m2_t, m2_c = pair_from_flat(torch_module, c_module, [0.0] * 8, (2, 4), "float32")
    cases.append(
        Case(
            name="cat(mismatched non-cat-dim shapes rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call([m1_t, m2_t], 0),
            run_c=lambda: c_module._aten_dispatch(op, [m1_c, m2_c], 0),
            expect="both_error",
            note="shapes disagree outside the concat dim.",
        )
    )

    return cases


# --- aten.embedding.default ---------------------------------------------------

_EMBEDDING_WEIGHT_DTYPES = ["float64", "float32", "float16", "bfloat16"]


def embedding_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.embedding.default"
    cases: list[Case] = []

    vocab, dim = 8, 4
    weight_flat = [float(i) * 0.1 for i in range(vocab * dim)]

    for dtype_name in _EMBEDDING_WEIGHT_DTYPES:
        w_t, w_c = pair_from_flat(torch_module, c_module, weight_flat, (vocab, dim), dtype_name)

        idx_1d_t, idx_1d_c = pair_from_flat(torch_module, c_module, [0, 3, 7, 2], (4,), "int64")
        cases.append(
            Case(
                name=f"embedding(weight_dtype={dtype_name}, indices=1D)",
                op=op,
                run_torch=lambda w_t=w_t, idx_1d_t=idx_1d_t: torch_call(w_t, idx_1d_t),
                run_c=lambda w_c=w_c, idx_1d_c=idx_1d_c: c_module._aten_dispatch(op, w_c, idx_1d_c),
                note="token-embedding-lookup shape",
            )
        )

        idx_2d_t, idx_2d_c = pair_from_flat(torch_module, c_module, [0, 1, 2, 7, 6, 5], (2, 3), "int64")
        cases.append(
            Case(
                name=f"embedding(weight_dtype={dtype_name}, indices=2D batch)",
                op=op,
                run_torch=lambda w_t=w_t, idx_2d_t=idx_2d_t: torch_call(w_t, idx_2d_t),
                run_c=lambda w_c=w_c, idx_2d_c=idx_2d_c: c_module._aten_dispatch(op, w_c, idx_2d_c),
                note="batch x sequence indices shape, as in a real forward pass",
            )
        )

        idx_edge_t, idx_edge_c = pair_from_flat(torch_module, c_module, [0, 0, vocab - 1, vocab - 1], (4,), "int64")
        cases.append(
            Case(
                name=f"embedding(weight_dtype={dtype_name}, indices=boundary 0/vocab-1)",
                op=op,
                run_torch=lambda w_t=w_t, idx_edge_t=idx_edge_t: torch_call(w_t, idx_edge_t),
                run_c=lambda w_c=w_c, idx_edge_c=idx_edge_c: c_module._aten_dispatch(op, w_c, idx_edge_c),
                note="first and last valid row indices",
            )
        )

    return cases


# --- aten.empty.memory_format --------------------------------------------------

_EMPTY_SHAPES: list[tuple[int, ...]] = [(), (3,), (2, 3), (0, 3), (1, 4, 1)]


def empty_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.empty.memory_format"
    cases: list[Case] = []

    for shape in _EMPTY_SHAPES:
        shape_list = list(shape)
        cases.append(
            Case(
                name=f"empty(shape={shape_list}, dtype=inferred)",
                op=op,
                run_torch=lambda shape_list=shape_list: torch_call(shape_list),
                run_c=lambda shape_list=shape_list: c_module._aten_dispatch(op, shape_list),
                value_check=_dtype_shape_only_check,
                note="uninitialized memory -- only dtype/shape are meaningful, see module note above",
            )
        )

    for dtype_name in dt.DEFAULT_DTYPES:
        c_dt = dt.c_dtype(c_module, dtype_name)
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        cases.append(
            Case(
                name=f"empty(shape=[2, 3], dtype={dtype_name})",
                op=op,
                run_torch=lambda t_dt=t_dt: torch_call([2, 3], dtype=t_dt),
                run_c=lambda c_dt=c_dt: c_module._aten_dispatch(op, [2, 3], dtype=c_dt),
                value_check=_dtype_shape_only_check,
                note="uninitialized memory -- only dtype/shape are meaningful, see module note above",
            )
        )

    return cases


# --- aten.is_floating_point.default -------------------------------------------

_IS_FLOATING_POINT_DTYPES = ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]


def is_floating_point_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.is_floating_point.default"
    cases: list[Case] = []

    for dtype_name in _IS_FLOATING_POINT_DTYPES:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3], (3,), dtype_name)
        cases.append(
            Case(
                name=f"is_floating_point(dtype={dtype_name})",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                value_check=_scalar_match_check,
                note="returns a plain Python bool, not a Tensor",
            )
        )

    for shape, dtype_name, note in [((), "float32", "0-d float"), ((2, 2), "int32", "2D int")]:
        flat = [1.0] if shape == () else [1, 2, 3, 4]
        a_t, a_c = pair_from_flat(torch_module, c_module, flat, shape, dtype_name)
        cases.append(
            Case(
                name=f"is_floating_point(dtype={dtype_name}, shape={shape})",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                value_check=_scalar_match_check,
                note=f"returns a plain Python bool, not a Tensor -- {note} shape coverage",
            )
        )

    return cases


# --- aten.isin.Tensor_Tensor ---------------------------------------------------

_ISIN_DTYPES = ["float64", "float32", "int64", "int32", "uint8"]


def isin_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.isin.Tensor_Tensor"
    cases: list[Case] = []

    for dtype_name in _ISIN_DTYPES:
        elements_t, elements_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5], (5,), dtype_name)
        test_t, test_c = pair_from_flat(torch_module, c_module, [2, 4], (2,), dtype_name)
        cases.append(
            Case(
                name=f"isin(dtype={dtype_name}, some matches)",
                op=op,
                run_torch=lambda e=elements_t, tt=test_t: torch_call(e, tt),
                run_c=lambda e=elements_c, tt=test_c: c_module._aten_dispatch(op, e, tt),
            )
        )

        empty_test_t, empty_test_c = pair_from_flat(torch_module, c_module, [], (0,), dtype_name)
        cases.append(
            Case(
                name=f"isin(dtype={dtype_name}, empty test_elements -> all False)",
                op=op,
                run_torch=lambda e=elements_t, tt=empty_test_t: torch_call(e, tt),
                run_c=lambda e=elements_c, tt=empty_test_c: c_module._aten_dispatch(op, e, tt),
            )
        )

        full_test_t, full_test_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5], (5,), dtype_name)
        cases.append(
            Case(
                name=f"isin(dtype={dtype_name}, test_elements == elements -> all True)",
                op=op,
                run_torch=lambda e=elements_t, tt=full_test_t: torch_call(e, tt),
                run_c=lambda e=elements_c, tt=full_test_c: c_module._aten_dispatch(op, e, tt),
            )
        )

        cases.append(
            Case(
                name=f"isin(dtype={dtype_name}, invert=True)",
                op=op,
                run_torch=lambda e=elements_t, tt=test_t: torch_call(e, tt, invert=True),
                run_c=lambda e=elements_c, tt=test_c: c_module._aten_dispatch(op, e, tt, invert=True),
            )
        )

    return cases


# --- aten.ones.default ---------------------------------------------------------

_ONES_SHAPES: list[tuple[int, ...]] = [(), (3,), (2, 3), (0, 3), (1, 4, 1)]


def ones_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.ones.default"
    cases: list[Case] = []

    for shape in _ONES_SHAPES:
        shape_list = list(shape)
        cases.append(
            Case(
                name=f"ones(shape={shape_list}, dtype=inferred)",
                op=op,
                run_torch=lambda shape_list=shape_list: torch_call(shape_list),
                run_c=lambda shape_list=shape_list: c_module._aten_dispatch(op, shape_list),
                note="dtype inferred -> torch.ones defaults to float32",
            )
        )

    for dtype_name in dt.DEFAULT_DTYPES:
        c_dt = dt.c_dtype(c_module, dtype_name)
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        for shape in [(3,), (2, 3)]:
            shape_list = list(shape)
            cases.append(
                Case(
                    name=f"ones(shape={shape_list}, dtype={dtype_name})",
                    op=op,
                    run_torch=lambda shape_list=shape_list, t_dt=t_dt: torch_call(shape_list, dtype=t_dt),
                    run_c=lambda shape_list=shape_list, c_dt=c_dt: c_module._aten_dispatch(op, shape_list, dtype=c_dt),
                )
            )

    return cases


# --- aten.pow.Tensor_Scalar / .Tensor_Tensor / .Scalar ------------------------

_POW_FLOAT_DTYPES = ["float64", "float32", "float16", "bfloat16"]
_POW_INT_DTYPES = ["int64", "int32", "int16"]


def pow_tensor_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.pow.Tensor_Scalar"
    cases: list[Case] = []

    for dtype_name in _POW_FLOAT_DTYPES:
        base_t, base_c = pair_from_flat(torch_module, c_module, [0.0, 1.0, 2.0, -2.0, 4.0], (5,), dtype_name)
        for exponent, note in [(2, "square, int exponent -- RMSNorm-style"), (0, "x**0 -> 1"), (0.5, "sqrt via float exponent"), (-1, "negative exponent")]:
            cases.append(
                Case(
                    name=f"pow(dtype={dtype_name}, exponent={exponent}) [{note}]",
                    op=op,
                    run_torch=lambda base_t=base_t, exponent=exponent: torch_call(base_t, exponent),
                    run_c=lambda base_c=base_c, exponent=exponent: c_module._aten_dispatch(op, base_c, exponent),
                    note=note,
                )
            )

    for dtype_name in _POW_INT_DTYPES:
        base_t, base_c = pair_from_flat(torch_module, c_module, [0, 1, 2, -2, 4], (5,), dtype_name)
        cases.append(
            Case(
                name=f"pow(dtype={dtype_name}, exponent=2)",
                op=op,
                run_torch=lambda base_t=base_t: torch_call(base_t, 2),
                run_c=lambda base_c=base_c: c_module._aten_dispatch(op, base_c, 2),
                note="square, int exponent",
            )
        )
        cases.append(
            Case(
                name=f"pow(dtype={dtype_name}, exponent=-1, rejected)",
                op=op,
                run_torch=lambda base_t=base_t: torch_call(base_t, -1),
                run_c=lambda base_c=base_c: c_module._aten_dispatch(op, base_c, -1),
                expect="both_error",
                note="integers to negative integer powers are not allowed -- torch refuses.",
            )
        )

    return cases


def pow_tensor_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.pow.Tensor_Tensor"
    cases: list[Case] = []

    for dtype_name in _POW_FLOAT_DTYPES:
        base_t, base_c = pair_from_flat(torch_module, c_module, [2.0, 3.0, 4.0, 0.0], (2, 2), dtype_name)
        exp_t, exp_c = pair_from_flat(torch_module, c_module, [2.0, 0.5, 0.0, 3.0], (2, 2), dtype_name)
        cases.append(
            Case(
                name=f"pow(dtype={dtype_name}, elementwise tensor exponent)",
                op=op,
                run_torch=lambda base_t=base_t, exp_t=exp_t: torch_call(base_t, exp_t),
                run_c=lambda base_c=base_c, exp_c=exp_c: c_module._aten_dispatch(op, base_c, exp_c),
            )
        )
        exp_scalar_t, exp_scalar_c = pair_from_flat(torch_module, c_module, [2.0], (), dtype_name)
        cases.append(
            Case(
                name=f"pow(dtype={dtype_name}, 0-d exponent broadcast)",
                op=op,
                run_torch=lambda base_t=base_t, exp_scalar_t=exp_scalar_t: torch_call(base_t, exp_scalar_t),
                run_c=lambda base_c=base_c, exp_scalar_c=exp_scalar_c: c_module._aten_dispatch(op, base_c, exp_scalar_c),
                note="scalar (0-d) broadcast",
            )
        )

    return cases


def pow_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.pow.Scalar"
    cases: list[Case] = []

    for dtype_name in _POW_FLOAT_DTYPES:
        exp_t, exp_c = pair_from_flat(torch_module, c_module, [1.0, 2.0, 3.0, 0.0], (2, 2), dtype_name)
        for base, note in [(2.0, "float base"), (0.0, "0**x"), (-1.0, "negative base")]:
            cases.append(
                Case(
                    name=f"pow(base={base}, dtype={dtype_name}, tensor exponent) [{note}]",
                    op=op,
                    run_torch=lambda base=base, exp_t=exp_t: torch_call(base, exp_t),
                    run_c=lambda base=base, exp_c=exp_c: c_module._aten_dispatch(op, base, exp_c),
                    note=note,
                )
            )

    return cases


# --- aten.randint.low -----------------------------------------------------------
#
# See the module note above on random draws: dtype/shape/range are checked
# via Case.value_check, the sequence itself is not.

_RANDINT_DTYPES = ["int64", "int32", "int16"]


def randint_low_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.randint.low"
    cases: list[Case] = []

    scenarios = [
        dict(low=0, high=10, size=[5], note="1D, small positive range"),
        dict(low=-5, high=5, size=[4], note="range straddling zero"),
        dict(low=0, high=100, size=[2, 3], note="2D shape"),
        dict(low=0, high=1, size=[3], note="degenerate range -- only low itself is valid"),
    ]
    for sc in scenarios:
        low, high, size = sc["low"], sc["high"], sc["size"]
        for dtype_name in _RANDINT_DTYPES:
            t_dt = dt.torch_dtype(torch_module, dtype_name)
            c_dt = dt.c_dtype(c_module, dtype_name)
            cases.append(
                Case(
                    name=f"randint(low={low}, high={high}, size={size}, dtype={dtype_name}) [{sc['note']}]",
                    op=op,
                    run_torch=lambda low=low, high=high, size=size, t_dt=t_dt: torch_call(low, high, size, dtype=t_dt),
                    run_c=lambda low=low, high=high, size=size, c_dt=c_dt: c_module._aten_dispatch(
                        op, low, high, size, dtype=c_dt
                    ),
                    value_check=_range_check(low, high),
                    note=sc["note"] + " -- random draw, sequence unchecked (see module note above)",
                )
            )

    return cases


# --- aten.rsqrt.default ----------------------------------------------------------

_RSQRT_DTYPES = ["float64", "float32", "float16", "bfloat16"]


def rsqrt_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.rsqrt.default"
    cases: list[Case] = []

    for dtype_name in _RSQRT_DTYPES:
        for flat, shape, note in [
            ([1.0, 4.0, 9.0, 16.0], (2, 2), "perfect squares"),
            ([0.5, 2.0, 100.0, 0.01], (2, 2), "assorted magnitudes"),
            ([0.0], (1,), "zero -> +inf, RMSNorm epsilon should avoid this in practice"),
            ([-1.0, -4.0], (2,), "negative -> NaN"),
            ([1.0], (), "0-d"),
        ]:
            a_t, a_c = pair_from_flat(torch_module, c_module, flat, shape, dtype_name)
            cases.append(
                Case(
                    name=f"rsqrt(dtype={dtype_name}, shape={shape}) [{note}]",
                    op=op,
                    run_torch=lambda a_t=a_t: torch_call(a_t),
                    run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                    note=note,
                )
            )

    return cases


# --- aten.lift_fresh.default (backs `torch.tensor(...)` construction) --------
#
# `torch.tensor([...])` -- the 13th name docs/C_SURFACE.md traced -- does
# not dispatch through a single clean "construct from Python data" aten op;
# tracing it with a TorchDispatchMode shows it going through
# `aten.lift_fresh.default(Tensor self) -> Tensor`, an identity-shaped op
# that marks an already-built tensor as a fresh autograd leaf. This matches
# build.py's own docstring on `_tensor_from_flat`: "there is no aten op yet
# that takes a Python list of numbers" -- the list-to-tensor step itself is
# scaffolding outside the aten dispatch surface this harness compares.
# `aten.lift_fresh.default` is the op name most likely to show up in
# `_aten_implemented()` for the "tensor" entry point; this builder checks
# it is a true identity (same dtype/shape/values as its input) across the
# dtypes/shapes this harness already exercises elsewhere. If rust/torch_c
# ends up exposing tensor construction under a different name, this
# builder simply stays pending -- see the module note above.

_LIFT_FRESH_DTYPES = ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]


def lift_fresh_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.lift_fresh.default"
    cases: list[Case] = []

    for dtype_name in _LIFT_FRESH_DTYPES:
        for flat, shape in [([0], ()), ([1, 2, 3], (3,)), ([1, 2, 3, 4], (2, 2))]:
            a_t, a_c = pair_from_flat(torch_module, c_module, flat, shape, dtype_name)
            cases.append(
                Case(
                    name=f"lift_fresh(dtype={dtype_name}, shape={shape}) [identity]",
                    op=op,
                    run_torch=lambda a_t=a_t: torch_call(a_t),
                    run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                    note="backs torch.tensor(...) construction -- see note above",
                )
            )

    return cases


# --- pre-seeded case builders for the ops backing TensorBase's 50 --------
# actually-used members (docs/C_SURFACE.md §4)
#
# docs/C_SURFACE.md traced a small Llama forward+generate() against real
# upstream torch (torch 2.13.0) and found 50 `TensorBase` members actually
# get accessed via a `torch.Tensor` instance -- 49 real API names plus the
# `__class__` bookkeeping dunder (not a real API, not covered here). Another
# change is implementing the rust/torch_c side of these; the builders below
# are written ahead of that landing, exactly like the 16 pre-seeded above,
# so coverage activates the moment each op appears in `_aten_implemented()`
# -- see the module note above `arange_default_cases` for how `compare.py`
# keeps an unimplemented op's builder inert (PENDING, not FAIL) until then.
#
# **Methods are not functions -- overloads were measured, not guessed.**
# `x * 2`, `x + 2`, `x - 2`, `x / 2` (dunder-operator paths) were checked
# with a real `TorchDispatchMode` probe against torch 2.13.0 before writing
# any of this, because the difference between an op's overloads takes
# different arguments and there is no way to guess it correctly from the
# method name alone (see the `full`/`add`/`mm` docstring above). Two results
# were surprising enough to call out explicitly:
#
#   - `x * 2`, `x + 2`, `x - 2`, `x / 2` all dispatch through the *.Tensor*
#     overload (`mul.Tensor`, `add.Tensor`, `sub.Tensor`, `div.Tensor`) --
#     the Python scalar is silently wrapped into a 0-d tensor by the dunder
#     before the dispatcher ever sees it. There is no reachable `.Scalar`
#     overload for these dunders.
#   - `x & True`, `x | True`, `x == 2`, `x < 2`, `x.ne(2)` do the opposite:
#     the Python scalar stays a Scalar and dispatches to the `.Scalar`
#     overload (`bitwise_and.Scalar`, `bitwise_or.Scalar`, `eq.Scalar`,
#     `lt.Scalar`, `ne.Scalar`). Same "dunder/method given a Python number"
#     shape as the arithmetic ops above, opposite dispatch behaviour --
#     each family was probed independently rather than assumed to match.
#
# `__matmul__` needs no new builder here: the probe showed it dispatches to
# `aten.mm.default`, already covered by `mm_cases` above.
#
# **Nine of the 50 names never reach the ATen dispatcher at all.** `device`,
# `dim`, `dtype`, `grad_fn`, `ndim`, `numel`, `requires_grad`,
# `requires_grad_`, `shape`, `size` were probed the same way (called inside
# a `TorchDispatchMode`) and produced zero recorded dispatcher calls --
# they read (or, for `requires_grad_`, mutate) metadata already sitting on
# the TensorImpl without going through the aten op dispatch this harness
# compares. There is no aten overload for `_aten_dispatch` to be given for
# these, so they intentionally have no case builder; this is a property of
# the harness's aten-op-granularity design, not an oversight.
#
# **In-place methods compare differently.** `fill_`, `copy_`, `normal_`,
# `uniform_` mutate their operand rather than returning a fresh value.
# Structurally this needs no different harness machinery than any other
# case -- `torch.ops.aten.<op>_.<overload>` returns the same (now mutated)
# object, so the existing run_torch/run_c/compare pipeline already compares
# the *result* of an in-place call correctly -- but it does mean every
# in-place case here builds a **fresh** operand pair per case (never shares
# one tensor across two cases), or an earlier case's mutation would leak
# into a later one that expects a clean starting value. `normal_` and
# `uniform_` still use `Case.value_check` instead of the default pipeline,
# but no longer because they are unmatchable -- see their builders below, and
# `_rng_stream_check`, for what they compare now.
#
# `contiguous()` needs no separate builder: on a non-contiguous input it
# dispatches to `aten.clone.default`, the same op `clone()` uses (probed);
# on an already-contiguous input it is a no-op that never reaches the
# dispatcher, so there is nothing there to compare either.
#
# `reshape()` needs no separate builder for the same reason: on a
# contiguous input it dispatches to `aten.view.default` (the same op
# `view()` uses, see `view_cases`); on a non-contiguous input it instead
# dispatches to `aten.clone.default` + `aten._unsafe_view.default` -- a
# second op this harness's single-op-per-case design does not attempt to
# chain, matching the same granularity limitation already documented for
# `contiguous()`.


def _binary_tensor_case(
    torch_module, c_module, op, torch_call, dtype_name, a_flat, a_shape, b_flat, b_shape, note, kwargs=None
) -> Case:
    kwargs = kwargs or {}
    a_t, a_c = pair_from_flat(torch_module, c_module, a_flat, a_shape, dtype_name)
    b_t, b_c = pair_from_flat(torch_module, c_module, b_flat, b_shape, dtype_name)
    short = op.split(".", 2)[1]
    name = f"{short}(dtype={dtype_name}, a_shape={a_shape}, b_shape={b_shape}) [{note}]"
    return Case(
        name=name,
        op=op,
        run_torch=lambda: torch_call(a_t, b_t, **kwargs),
        run_c=lambda: c_module._aten_dispatch(op, a_c, b_c, **kwargs),
        note=note,
    )


def _binary_scalar_case(torch_module, c_module, op, torch_call, dtype_name, a_flat, a_shape, scalar, note) -> Case:
    a_t, a_c = pair_from_flat(torch_module, c_module, a_flat, a_shape, dtype_name)
    short = op.split(".", 2)[1]
    name = f"{short}(dtype={dtype_name}, a_shape={a_shape}, scalar={scalar!r}) [{note}]"
    return Case(
        name=name,
        op=op,
        run_torch=lambda: torch_call(a_t, scalar),
        run_c=lambda: c_module._aten_dispatch(op, a_c, scalar),
        note=note,
    )


def _unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note, kwargs=None) -> Case:
    kwargs = kwargs or {}
    a_t, a_c = pair_from_flat(torch_module, c_module, flat, shape, dtype_name)
    short = op.split(".", 2)[1]
    name = f"{short}(dtype={dtype_name}, shape={shape}) [{note}]"
    return Case(
        name=name,
        op=op,
        run_torch=lambda: torch_call(a_t, **kwargs),
        run_c=lambda: c_module._aten_dispatch(op, a_c, **kwargs),
        note=note,
    )


_ELEMENTWISE_SCENARIOS: list[dict] = [
    dict(a_flat=[1, 2, 3, 4], a_shape=(2, 2), b_flat=[5, 6, 7, 8], b_shape=(2, 2), note="elementwise"),
    dict(a_flat=[7], a_shape=(), b_flat=[1, 2, 3, 4], b_shape=(2, 2), note="scalar (0-d) broadcast"),
    dict(a_flat=[1, 2, 3, 4, 5, 6], a_shape=(2, 3), b_flat=[10, 20, 30], b_shape=(3,), note="row broadcast"),
]


def _elementwise_boundary_scenario(big) -> dict:
    return dict(
        a_flat=[0, -5, big, -big],
        a_shape=(2, 2),
        b_flat=[0, 5, -big, big],
        b_shape=(2, 2),
        note="boundary values (0/neg/large)",
    )


# --- aten.sub.Tensor -------------------------------------------------------
# `__sub__` -- probe confirmed the `.Tensor` overload even for `x - 2`
# (scalar wrapped to a 0-d tensor), same shape as `add.Tensor` above, so
# this reuses `add_cases`' own scenario generators verbatim.


def sub_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.sub.Tensor"
    cases: list[Case] = []

    for dtype_name in _FLOAT_ADD_DTYPES:
        for scenario in _float_add_scenarios(_FLOAT_ADD_MAGNITUDE[dtype_name]):
            a_t, a_c = pair_from_flat(torch_module, c_module, scenario["a_flat"], scenario["a_shape"], dtype_name)
            b_t, b_c = pair_from_flat(torch_module, c_module, scenario["b_flat"], scenario["b_shape"], dtype_name)
            alpha = scenario["alpha"]
            kwargs = {} if alpha is None else {"alpha": alpha}
            name = (
                f"sub(dtype={dtype_name}, a_shape={scenario['a_shape']}, b_shape={scenario['b_shape']}, "
                f"alpha={alpha}) [{scenario['note']}]"
            )
            cases.append(
                Case(
                    name=name,
                    op=op,
                    run_torch=lambda a_t=a_t, b_t=b_t, kwargs=kwargs: torch_call(a_t, b_t, **kwargs),
                    run_c=lambda a_c=a_c, b_c=b_c, kwargs=kwargs: c_module._aten_dispatch(op, a_c, b_c, **kwargs),
                )
            )

    for dtype_name, big in [("int64", 10**9), ("int32", 10**6), ("int16", 1000)]:
        for scenario in _int_add_scenarios_signed(big):
            a_t, a_c = pair_from_flat(torch_module, c_module, scenario["a_flat"], scenario["a_shape"], dtype_name)
            b_t, b_c = pair_from_flat(torch_module, c_module, scenario["b_flat"], scenario["b_shape"], dtype_name)
            alpha = scenario["alpha"]
            kwargs = {} if alpha is None else {"alpha": alpha}
            name = (
                f"sub(dtype={dtype_name}, a_shape={scenario['a_shape']}, b_shape={scenario['b_shape']}, "
                f"alpha={alpha}) [{scenario['note']}]"
            )
            cases.append(
                Case(
                    name=name,
                    op=op,
                    run_torch=lambda a_t=a_t, b_t=b_t, kwargs=kwargs: torch_call(a_t, b_t, **kwargs),
                    run_c=lambda a_c=a_c, b_c=b_c, kwargs=kwargs: c_module._aten_dispatch(op, a_c, b_c, **kwargs),
                )
            )

    return cases


# --- aten.mul.Tensor / aten.div.Tensor -------------------------------------
# `__mul__`/`__truediv__` -- same probe result as sub: `.Tensor` overload
# even for a Python-scalar RHS, no `.Scalar` overload reachable from these
# dunders.

_MUL_DIV_FLOAT_DTYPES = ["float64", "float32", "float16", "bfloat16"]
_MUL_DIV_INT_DTYPES = ["int64", "int32", "int16"]
_MUL_MAGNITUDE = {
    "float64": 1e3,
    "float32": 1e3,
    "float16": 10.0,
    "bfloat16": 100.0,
    "int64": 1000,
    "int32": 100,
    "int16": 50,
}


def mul_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.mul.Tensor"
    cases: list[Case] = []
    for dtype_name in _MUL_DIV_FLOAT_DTYPES + _MUL_DIV_INT_DTYPES:
        big = _MUL_MAGNITUDE[dtype_name]
        for sc in _ELEMENTWISE_SCENARIOS + [_elementwise_boundary_scenario(big)]:
            cases.append(
                _binary_tensor_case(
                    torch_module, c_module, op, torch_call, dtype_name,
                    sc["a_flat"], sc["a_shape"], sc["b_flat"], sc["b_shape"], sc["note"],
                )
            )
    # uint8 wraparound -- both sides should wrap identically (modular
    # arithmetic), same precedent as add_cases' uint8 wraparound case.
    cases.append(
        _binary_tensor_case(
            torch_module, c_module, op, torch_call, "uint8",
            [200, 200], (2,), [100, 2], (2,), "uint8 overflow wraps (both sides modular)",
        )
    )
    return cases


def div_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.div.Tensor"
    cases: list[Case] = []
    for dtype_name in _MUL_DIV_FLOAT_DTYPES + _MUL_DIV_INT_DTYPES:
        for sc in _ELEMENTWISE_SCENARIOS:
            cases.append(
                _binary_tensor_case(
                    torch_module, c_module, op, torch_call, dtype_name,
                    sc["a_flat"], sc["a_shape"], sc["b_flat"], sc["b_shape"], sc["note"],
                )
            )
    # Division by zero: `div.Tensor` is always true division (confirmed by
    # probe -- it never does integer floor division, even for integer
    # inputs, and promotes to a floating dtype), and IEEE-754 makes 0
    # division well-defined (inf/-inf/nan) rather than an error. Both sides
    # are expected to *compute* the same pattern, not refuse.
    for dtype_name in _MUL_DIV_FLOAT_DTYPES:
        cases.append(
            _binary_tensor_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [1.0, -1.0, 0.0, 5.0], (2, 2), [0.0, 0.0, 0.0, 2.0], (2, 2),
                "division by zero -> inf/-inf/nan, not an error",
            )
        )
    return cases


# --- aten.bitwise_and.{Tensor,Scalar} / aten.bitwise_or.{Tensor,Scalar} ----
# `__and__`/`__or__` -- probe confirmed these *do* keep a Python scalar as a
# Scalar (unlike the arithmetic dunders above), so both overloads are
# reachable and both get builders.
#
# **No `dtype=bool` scenario here (or in bitwise_not/select/slice/clone/
# local_scalar_dense/any below).** `_C`'s own `_tensor_from_flat` refuses to
# construct a bool tensor directly (found while smoke-testing these
# builders against real torch): `NotImplementedError: _tensor_from_flat:
# torch.bool is not accepted here -- a bool tensor must come from an op
# that guarantees 0/1 bytes (BOOL.md §6.3)`. That refusal happens at
# case-*list*-construction time (`cases = builder(...)` in compare.py's
# `run()`), which is not wrapped in a try/except the way a single case's
# run_torch/run_c is -- so an eager `pair_from_flat(..., "bool")` call in a
# module-level scope here would crash the *entire* harness run the moment
# this op lands in `_aten_implemented()`, not just fail one case. Real,
# attention-mask-flavoured bool coverage is still int-dtype-representative
# via `_BITWISE_INT_DTYPES` (bitwise ops are defined identically on bool
# and integer types), so nothing is silently lost by leaving bool out here.
# `masked_fill_cases` and `index_tensor_cases` below need an actual bool
# mask (schema-mandated), so those defer construction into the run_torch/
# run_c lambdas instead of dropping it -- see their module notes.

_BITWISE_INT_DTYPES = ["int64", "int32", "int16", "uint8"]


def bitwise_and_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.bitwise_and.Tensor"
    cases: list[Case] = []
    for dtype_name in _BITWISE_INT_DTYPES:
        cases.append(
            _binary_tensor_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [0b1100, 0b1010, 0b1111, 0], (2, 2), [0b1010, 0b0110, 0b0000, 0b1111], (2, 2), "elementwise AND",
            )
        )
    return cases


def bitwise_and_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.bitwise_and.Scalar"
    cases: list[Case] = []
    for dtype_name in _BITWISE_INT_DTYPES:
        cases.append(
            _binary_scalar_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [0b1100, 0b1010, 0b1111, 0], (2, 2), 0b1010, "x & 0b1010",
            )
        )
    return cases


def bitwise_or_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.bitwise_or.Tensor"
    cases: list[Case] = []
    for dtype_name in _BITWISE_INT_DTYPES:
        cases.append(
            _binary_tensor_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [0b1100, 0b1010, 0b0000, 0], (2, 2), [0b0010, 0b0100, 0b1111, 0b0001], (2, 2), "elementwise OR",
            )
        )
    return cases


def bitwise_or_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.bitwise_or.Scalar"
    cases: list[Case] = []
    for dtype_name in _BITWISE_INT_DTYPES:
        cases.append(
            _binary_scalar_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [0b1100, 0b1010, 0b0000, 0], (2, 2), 0b0001, "x | 0b0001",
            )
        )
    return cases


# --- aten.bitwise_not.default -----------------------------------------------
# `__invert__`. See the bool-construction note above bitwise_and_tensor_cases.

_BITWISE_NOT_SIGNED = [0, -1, 5, -5]
_BITWISE_NOT_UNSIGNED = [0, 1, 5, 250]


def bitwise_not_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.bitwise_not.default"
    cases: list[Case] = []
    for dtype_name in ["int64", "int32", "int16"]:
        cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, _BITWISE_NOT_SIGNED, (2, 2), "bitwise not, signed"))
    cases.append(_unary_case(torch_module, c_module, op, torch_call, "uint8", _BITWISE_NOT_UNSIGNED, (2, 2), "bitwise not, unsigned"))
    return cases


# --- aten.eq.{Tensor,Scalar} / aten.lt.{Tensor,Scalar} / aten.ne.{Tensor,Scalar} --
# `__eq__`, `__lt__`, `ne` -- probe confirmed Scalar-overload comparisons
# keep the Python scalar as a Scalar (same family as bitwise_and/or above).

_CMP_DTYPES = ["float64", "float32", "float16", "bfloat16", "int64", "int32", "int16", "uint8"]
_CMP_SCENARIOS: list[dict] = [
    dict(a_flat=[1, 2, 3, 4], a_shape=(2, 2), b_flat=[1, 5, 3, 0], b_shape=(2, 2), note="mixed equal/unequal"),
    dict(a_flat=[7], a_shape=(), b_flat=[1, 7, 3, 4], b_shape=(2, 2), note="scalar (0-d) broadcast"),
]


def eq_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.eq.Tensor"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        for sc in _CMP_SCENARIOS:
            cases.append(
                _binary_tensor_case(
                    torch_module, c_module, op, torch_call, dtype_name,
                    sc["a_flat"], sc["a_shape"], sc["b_flat"], sc["b_shape"], sc["note"],
                )
            )
    return cases


def eq_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.eq.Scalar"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        cases.append(
            _binary_scalar_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [1, 2, 3, 4], (2, 2), 3, "x == 3, as reached from __eq__ with a python scalar",
            )
        )
    return cases


def lt_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.lt.Tensor"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        for sc in _CMP_SCENARIOS:
            cases.append(
                _binary_tensor_case(
                    torch_module, c_module, op, torch_call, dtype_name,
                    sc["a_flat"], sc["a_shape"], sc["b_flat"], sc["b_shape"], sc["note"],
                )
            )
    return cases


def lt_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.lt.Scalar"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        cases.append(
            _binary_scalar_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [1, 2, 3, 4], (2, 2), 3, "x < 3, as reached from __lt__ with a python scalar",
            )
        )
    return cases


def ne_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.ne.Tensor"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        for sc in _CMP_SCENARIOS:
            cases.append(
                _binary_tensor_case(
                    torch_module, c_module, op, torch_call, dtype_name,
                    sc["a_flat"], sc["a_shape"], sc["b_flat"], sc["b_shape"], sc["note"],
                )
            )
    return cases


def ne_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.ne.Scalar"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        cases.append(
            _binary_scalar_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [1, 2, 3, 4], (2, 2), 3, "x.ne(3) -- attention-mask padding check style",
            )
        )
    return cases


# --- aten._local_scalar_dense.default ---------------------------------------
# `__bool__` -- probe: `bool(tensor)` dispatches straight to
# `_local_scalar_dense` for an already-0-d tensor (no `lift_fresh` involved
# once the tensor exists). Returns a plain Python scalar, not a Tensor, so
# this reuses `_scalar_match_check` like `is_floating_point_cases` above.

_LOCAL_SCALAR_DENSE_DTYPES = ["float64", "float32", "float16", "bfloat16", "int64", "int32", "int16", "uint8"]


def local_scalar_dense_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten._local_scalar_dense.default"
    cases: list[Case] = []
    for dtype_name in _LOCAL_SCALAR_DENSE_DTYPES:
        for value, note in [(0, "falsy scalar -- backs bool(x)"), (1, "truthy scalar -- backs bool(x)")]:
            a_t, a_c = pair_from_flat(torch_module, c_module, [value], (), dtype_name)
            cases.append(
                Case(
                    name=f"_local_scalar_dense(dtype={dtype_name}, value={value})",
                    op=op,
                    run_torch=lambda a_t=a_t: torch_call(a_t),
                    run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                    value_check=_scalar_match_check,
                    note=note + " -- returns a plain Python scalar, not a Tensor; backs __bool__ extraction",
                )
            )
    return cases


# --- aten.select.int / aten.slice.Tensor / aten.index.Tensor ---------------
# `__getitem__` -- probe found three distinct overloads depending on the
# index expression's shape: an int index -> select.int, a slice -> slice.
# Tensor, a tensor/bool-mask index -> index.Tensor (via a Tensor?[] list of
# indices). All three are given builders since real code hits all three
# forms.

_SELECT_DTYPES = ["float64", "float32", "int64", "int32", "uint8"]


def select_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.select.int"
    cases: list[Case] = []
    for dtype_name in _SELECT_DTYPES:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        cases.append(
            Case(
                name=f"select(dtype={dtype_name}, dim=0, index=1)",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, 0, 1),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, 0, 1),
                note="x[1] -- first-axis row selection",
            )
        )
        cases.append(
            Case(
                name=f"select(dtype={dtype_name}, dim=-1, index=-1)",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, -1, -1),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, -1, -1),
                note="negative dim and index",
            )
        )
    return cases


def slice_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.slice.Tensor"
    cases: list[Case] = []
    for dtype_name in _SELECT_DTYPES:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6, 7, 8], (2, 4), dtype_name)
        cases.append(
            Case(
                name=f"slice(dtype={dtype_name}, dim=1, start=1, end=3, step=1)",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, 1, 1, 3, 1),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, 1, 1, 3, 1),
                note="x[:, 1:3] -- last-axis slicing",
            )
        )
        cases.append(
            Case(
                name=f"slice(dtype={dtype_name}, dim=1, start=None, end=None, step=1) [identity]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, 1, None, None, 1),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, 1, None, None, 1),
                note="x[:, :] -- identity slice",
            )
        )
    return cases


def index_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.index.Tensor"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "int64", "int32"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        idx_t, idx_c = pair_from_flat(torch_module, c_module, [0, 1], (2,), "int64")
        cases.append(
            Case(
                name=f"index(dtype={dtype_name}, integer index tensor)",
                op=op,
                run_torch=lambda a_t=a_t, idx_t=idx_t: torch_call(a_t, [idx_t]),
                run_c=lambda a_c=a_c, idx_c=idx_c: c_module._aten_dispatch(op, a_c, [idx_c]),
                note="x[idx_tensor] -- advanced (fancy) indexing",
            )
        )
        # Boolean-mask indexing needs an actual bool tensor. `_C`'s
        # `_tensor_from_flat` refuses to build one directly (BOOL.md §6.3,
        # see the note above bitwise_and_tensor_cases) and no other
        # bool-producing op is implemented yet either, so construction is
        # deferred into each lambda -- run_c's failure is then caught by
        # compare.py's normal per-case try/except instead of crashing the
        # whole harness run at case-list-build time.
        a_flat, a_shape = [1, 2, 3, 4, 5, 6], (2, 3)
        mask_flat = [True, False, True, False, True, False]
        cases.append(
            Case(
                name=f"index(dtype={dtype_name}, boolean mask)",
                op=op,
                run_torch=lambda dtype_name=dtype_name, a_flat=a_flat, a_shape=a_shape, mask_flat=mask_flat: torch_call(
                    torch_module.tensor(a_flat, dtype=dt.torch_dtype(torch_module, dtype_name)).reshape(list(a_shape)),
                    [torch_module.tensor(mask_flat).reshape(list(a_shape))],
                ),
                run_c=lambda dtype_name=dtype_name, a_flat=a_flat, a_shape=a_shape, mask_flat=mask_flat: c_module._aten_dispatch(
                    op,
                    c_module._tensor_from_flat(a_flat, list(a_shape), dtype=dt.c_dtype(c_module, dtype_name)),
                    [c_module._tensor_from_flat([int(v) for v in mask_flat], list(a_shape), dtype=c_module.bool)],
                ),
                note="x[bool_mask] -- boolean mask indexing, as in attention masking",
            )
        )
    return cases


# --- aten.any.default / aten.any.dim ----------------------------------------
# `any` -- probed on int input (`torch.any` treats any nonzero element as
# true regardless of dtype), avoiding the bool-construction constraint
# noted above bitwise_and_tensor_cases entirely rather than working around
# it, since int input exercises the real op just as well.

def any_default_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.any.default"
    cases: list[Case] = []
    for flat, shape, note in [
        ([1, 0, 1, 0], (2, 2), "some true"),
        ([0, 0, 0, 0], (2, 2), "all false"),
        ([1, 1, 1, 1], (2, 2), "all true"),
    ]:
        cases.append(_unary_case(torch_module, c_module, op, torch_call, "int64", flat, shape, note))
    return cases


def any_dim_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.any.dim"
    cases: list[Case] = []
    scenarios = [
        dict(flat=[1, 0, 0, 0, 1, 1], shape=(2, 3), dim=1, keepdim=False, note="along last dim"),
        dict(flat=[1, 0, 0, 0, 1, 1], shape=(2, 3), dim=1, keepdim=True, note="along last dim, keepdim"),
        dict(flat=[1, 0, 0, 0, 1, 1], shape=(2, 3), dim=0, keepdim=False, note="along first dim"),
    ]
    for sc in scenarios:
        a_t, a_c = pair_from_flat(torch_module, c_module, sc["flat"], sc["shape"], "int64")
        dim, keepdim = sc["dim"], sc["keepdim"]
        cases.append(
            Case(
                name=f"any(dtype=int64, dim={dim}, keepdim={keepdim}) [{sc['note']}]",
                op=op,
                run_torch=lambda a_t=a_t, dim=dim, keepdim=keepdim: torch_call(a_t, dim, keepdim),
                run_c=lambda a_c=a_c, dim=dim, keepdim=keepdim: c_module._aten_dispatch(op, a_c, dim, keepdim),
                note=sc["note"],
            )
        )
    return cases


# --- aten.clone.default / aten.detach.default -------------------------------
# `clone` (also backs `contiguous()` on a non-contiguous input, see the
# module note above) and `detach` (identity view, drops autograd tracking).


def clone_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.clone.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]:
        for flat, shape in [([0], ()), ([1, 2, 3], (3,)), ([1, 2, 3, 4], (2, 2))]:
            cases.append(
                _unary_case(
                    torch_module, c_module, op, torch_call, dtype_name, flat, shape,
                    "identity copy -- also backs contiguous() on a non-contiguous input",
                )
            )
    return cases


def detach_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.detach.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "int64", "uint8"]:
        for flat, shape in [([0], ()), ([1, 2, 3, 4], (2, 2))]:
            cases.append(
                _unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, "identity view, drops autograd tracking")
            )
    return cases


# --- aten.cos.default / aten.sin.default / aten.reciprocal.default ---------
# RoPE (cos/sin) and RMSNorm (reciprocal, alongside pow/mean above) inputs.

_TRIG_DTYPES = ["float64", "float32", "float16", "bfloat16"]
_TRIG_SCENARIOS = [
    ([0.0, 1.5707963267948966, 3.141592653589793, -1.5707963267948966], (2, 2), "0/pi-over-2/pi/-pi-over-2 -- RoPE angle boundary values"),
    ([0.1, 0.5, 1.0, 2.0], (2, 2), "assorted angles"),
    ([0.0], (), "0-d"),
]


def cos_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.cos.default"
    cases: list[Case] = []
    for dtype_name in _TRIG_DTYPES:
        for flat, shape, note in _TRIG_SCENARIOS:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))
    return cases


def sin_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.sin.default"
    cases: list[Case] = []
    for dtype_name in _TRIG_DTYPES:
        for flat, shape, note in _TRIG_SCENARIOS:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))
    return cases


def reciprocal_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.reciprocal.default"
    cases: list[Case] = []
    for dtype_name in _TRIG_DTYPES:
        for flat, shape, note in [
            ([1.0, 2.0, 4.0, 0.5], (2, 2), "assorted magnitudes"),
            ([0.0], (1,), "zero -> +inf"),
            ([-1.0, -4.0], (2,), "negative values"),
            ([2.0], (), "0-d"),
        ]:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))
    return cases


# --- aten.cumsum.default -----------------------------------------------------

def cumsum_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.cumsum.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        cases.append(
            Case(
                name=f"cumsum(dtype={dtype_name}, dim=0)",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, 0),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, 0),
                note="running total along first dim",
            )
        )
        cases.append(
            Case(
                name=f"cumsum(dtype={dtype_name}, dim=-1)",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, -1),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, -1),
                note="running total along last dim, negative dim",
            )
        )
    return cases


# --- aten.expand.default -----------------------------------------------------

def expand_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.expand.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "int64", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2], (2, 1), dtype_name)
        cases.append(
            Case(
                name=f"expand(dtype={dtype_name}, (2,1)->(2,3))",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, [2, 3]),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, [2, 3]),
                note="broadcast a singleton dim, as in expanding a KV head across query heads",
            )
        )
        cases.append(
            Case(
                name=f"expand(dtype={dtype_name}, (2,1)->(2,-1)) [keep-dim sentinel]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, [2, -1]),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, [2, -1]),
                note="-1 means 'keep this dim's existing size'",
            )
        )
    a3_t, a3_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), "float32")
    cases.append(
        Case(
            name="expand(non-singleton dim rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(a3_t, [2, 5]),
            run_c=lambda: c_module._aten_dispatch(op, a3_c, [2, 5]),
            expect="both_error",
            note="expand can only broadcast size-1 dims.",
        )
    )
    return cases


# --- aten.masked_fill.Scalar --------------------------------------------------

def masked_fill_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.masked_fill.Scalar"
    cases: list[Case] = []
    # `mask` must be an actual bool tensor -- torch's own masked_fill_
    # refuses a non-bool mask (probed: "masked_fill_ only supports boolean
    # masks"). `_C`'s `_tensor_from_flat` refuses to build a bool tensor
    # directly (BOOL.md §6.3, see the note above bitwise_and_tensor_cases),
    # and there is no other sanctioned way to get a bool tensor into `_C`
    # either, so -- unlike the ops above that could just drop bool coverage
    # -- masked_fill genuinely cannot be exercised yet. Mask construction is
    # deferred into the run_torch/run_c lambdas (built fresh each call,
    # rather than shared like every other case here) so that, until BOOL.md's
    # gap is closed, this surfaces as an ordinary per-case failure the
    # moment masked_fill lands in `_aten_implemented()` -- not a hard crash
    # of the whole harness run at case-list-construction time.
    a_flat, a_shape = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)
    mask_flat = [True, False, True, False, True, False]
    for dtype_name in ["float64", "float32", "float16", "bfloat16"]:
        big = _FLOAT_ADD_MAGNITUDE[dtype_name]
        for value, note in [(0.0, "zero fill"), (-big, "large negative -- attention masking style")]:
            cases.append(
                Case(
                    name=f"masked_fill(dtype={dtype_name}, value={value}) [{note}]",
                    op=op,
                    run_torch=lambda dtype_name=dtype_name, value=value: torch_call(
                        torch_module.tensor(a_flat, dtype=dt.torch_dtype(torch_module, dtype_name)).reshape(list(a_shape)),
                        torch_module.tensor(mask_flat).reshape(list(a_shape)),
                        value,
                    ),
                    run_c=lambda dtype_name=dtype_name, value=value: c_module._aten_dispatch(
                        op,
                        c_module._tensor_from_flat(a_flat, list(a_shape), dtype=dt.c_dtype(c_module, dtype_name)),
                        c_module._tensor_from_flat([int(v) for v in mask_flat], list(a_shape), dtype=c_module.bool),
                        value,
                    ),
                    note=note,
                )
            )
    return cases


# --- aten.max.default / aten.max.dim -----------------------------------------
# `max.dim` returns a (values, indices) pair (torch: a `torch.return_types.
# max` namedtuple, 2-tuple-indexable) -- see `_pair_result_check` below.

_REDUCE_DTYPES = ["float64", "float32", "float16", "bfloat16", "int64", "int32"]


def max_default_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.max.default"
    cases: list[Case] = []
    for dtype_name in _REDUCE_DTYPES:
        for flat, shape, note in [
            ([1, 5, 2, 9, 0, 3], (2, 3), "global max, flattened"),
            ([-5, -1, -9, -3], (2, 2), "all-negative values"),
            ([7], (1,), "single element"),
        ]:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))
    return cases


def _pair_result_check(t_res, c_res) -> tuple[bool, str]:
    """For ops returning a (values, indices) pair, like `max.dim` -- torch
    returns a `torch.return_types.max` namedtuple, indexable like a plain
    2-tuple. `values` is compared like any tensor result (dtype/shape/value
    within tolerance); `indices` must match exactly (integer positions)."""
    try:
        t_values, t_indices = t_res[0], t_res[1]
        c_values, c_indices = c_res[0], c_res[1]
    except (TypeError, IndexError, KeyError) as e:
        return False, f"expected a 2-element (values, indices) result on both sides: {e!r}"

    t_dtype, c_dtype = dt.dtype_name(t_values.dtype), dt.dtype_name(c_values.dtype)
    if t_dtype != c_dtype:
        return False, f"values dtype mismatch: torch={t_dtype} c={c_dtype}"
    t_shape = tuple(int(x) for x in t_values.shape)
    c_shape = tuple(int(x) for x in c_values.shape)
    if t_shape != c_shape:
        return False, f"values shape mismatch: torch={t_shape} c={c_shape}"

    tol = dt.tolerance_for(t_dtype)
    t_flat, c_flat = _flatten_values(t_values.tolist()), _flatten_values(c_values.tolist())
    if len(t_flat) != len(c_flat):
        return False, f"values length differs: torch={len(t_flat)} c={len(c_flat)}"
    for i, (x, y) in enumerate(zip(t_flat, c_flat)):
        xf, yf = float(x), float(y)
        # NaN is a *result* for these ops, not a failure: torch orders NaN as
        # the greatest element, so `sort` and `topk` hand it back and it
        # arrives here. `math.isclose(nan, nan)` is False, so without this the
        # two sides agreeing on NaN reads as a mismatch -- which is exactly
        # what happened the first time a NaN case was added.
        if math.isnan(xf) or math.isnan(yf):
            if math.isnan(xf) and math.isnan(yf):
                continue
            return False, f"values[{i}] mismatch: torch={x!r} c={y!r} (NaN on one side only)"
        if not math.isclose(xf, yf, rel_tol=tol.rtol, abs_tol=tol.atol):
            return False, f"values[{i}] mismatch: torch={x!r} c={y!r}"

    t_idx_shape = tuple(int(x) for x in t_indices.shape)
    c_idx_shape = tuple(int(x) for x in c_indices.shape)
    if t_idx_shape != c_idx_shape:
        return False, f"indices shape mismatch: torch={t_idx_shape} c={c_idx_shape}"
    t_idx_flat, c_idx_flat = _flatten_values(t_indices.tolist()), _flatten_values(c_indices.tolist())
    if t_idx_flat != c_idx_flat:
        return False, f"indices mismatch: torch={t_idx_flat!r} c={c_idx_flat!r}"
    return True, f"values dtype={t_dtype} shape={t_shape}, indices matched exactly"


def max_dim_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.max.dim"
    cases: list[Case] = []
    # Flat values chosen so the maximum is unique in every reduced slice --
    # ties are implementation-defined, same reasoning as argmax_cases above.
    scenarios = [
        dict(flat=[1, 5, 2, 9, 0, 3], shape=(2, 3), dim=1, keepdim=False, note="along last dim"),
        dict(flat=[1, 5, 2, 9, 0, 3], shape=(2, 3), dim=1, keepdim=True, note="along last dim, keepdim"),
        dict(flat=[1, 5, 2, 9, 0, 3], shape=(2, 3), dim=0, keepdim=False, note="along first dim"),
        dict(flat=[-5, -1, -9, -3], shape=(2, 2), dim=1, keepdim=False, note="all-negative values"),
    ]
    for dtype_name in _REDUCE_DTYPES:
        for sc in scenarios:
            a_t, a_c = pair_from_flat(torch_module, c_module, sc["flat"], sc["shape"], dtype_name)
            dim, keepdim = sc["dim"], sc["keepdim"]
            cases.append(
                Case(
                    name=f"max(dtype={dtype_name}, dim={dim}, keepdim={keepdim}) [{sc['note']}]",
                    op=op,
                    run_torch=lambda a_t=a_t, dim=dim, keepdim=keepdim: torch_call(a_t, dim, keepdim),
                    run_c=lambda a_c=a_c, dim=dim, keepdim=keepdim: c_module._aten_dispatch(op, a_c, dim, keepdim),
                    value_check=_pair_result_check,
                    note=sc["note"] + " -- returns (values, indices), see _pair_result_check",
                )
            )
    return cases


# --- aten.mean.default / aten.mean.dim / aten.sum.default / aten.sum.dim_IntList --

def mean_default_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.mean.default"
    cases: list[Case] = []
    for dtype_name in _MUL_DIV_FLOAT_DTYPES:
        for flat, shape, note in [
            ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), "global mean, flattened"),
            ([-2.0, 2.0], (2,), "symmetric around zero -> 0"),
            ([5.0], (), "0-d"),
        ]:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))
    return cases


def mean_dim_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.mean.dim"
    cases: list[Case] = []
    for dtype_name in _MUL_DIV_FLOAT_DTYPES:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), dtype_name)
        for dim, keepdim, note in [
            ([-1], True, "RMSNorm-style: reduce last dim, keepdim"),
            ([0], False, "reduce first dim"),
            (None, False, "dim=None -- reduce all"),
        ]:
            cases.append(
                Case(
                    name=f"mean(dtype={dtype_name}, dim={dim}, keepdim={keepdim}) [{note}]",
                    op=op,
                    run_torch=lambda a_t=a_t, dim=dim, keepdim=keepdim: torch_call(a_t, dim, keepdim),
                    run_c=lambda a_c=a_c, dim=dim, keepdim=keepdim: c_module._aten_dispatch(op, a_c, dim, keepdim),
                    note=note,
                )
            )
    return cases


def sum_default_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.sum.default"
    cases: list[Case] = []
    for dtype_name in _REDUCE_DTYPES:
        for flat, shape, note in [
            ([1, 2, 3, 4, 5, 6], (2, 3), "global sum, flattened"),
            ([-2, 2], (2,), "cancelling values -> 0"),
            ([7], (), "0-d"),
        ]:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))
    return cases


def sum_dim_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.sum.dim_IntList"
    cases: list[Case] = []
    for dtype_name in _REDUCE_DTYPES:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        for dim, keepdim, note in [
            ([-1], True, "reduce last dim, keepdim"),
            ([0], False, "reduce first dim"),
            (None, False, "dim=None -- reduce all"),
        ]:
            cases.append(
                Case(
                    name=f"sum(dtype={dtype_name}, dim={dim}, keepdim={keepdim}) [{note}]",
                    op=op,
                    run_torch=lambda a_t=a_t, dim=dim, keepdim=keepdim: torch_call(a_t, dim, keepdim),
                    run_c=lambda a_c=a_c, dim=dim, keepdim=keepdim: c_module._aten_dispatch(op, a_c, dim, keepdim),
                    note=note,
                )
            )
    return cases


# --- aten.new_ones.default ----------------------------------------------------

def new_ones_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.new_ones.default"
    cases: list[Case] = []
    for self_dtype in ["float64", "float32", "int64", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), self_dtype)
        cases.append(
            Case(
                name=f"new_ones(self_dtype={self_dtype}, shape=[2,3]) [dtype inherited from self]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, [2, 3]),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, [2, 3]),
                note="dtype inherited from the self tensor, not a default",
            )
        )
    for dtype_name in dt.DEFAULT_DTYPES:
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        c_dt = dt.c_dtype(c_module, dtype_name)
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), "float32")
        cases.append(
            Case(
                name=f"new_ones(self_dtype=float32, shape=[2,2], dtype_override={dtype_name})",
                op=op,
                run_torch=lambda a_t=a_t, t_dt=t_dt: torch_call(a_t, [2, 2], dtype=t_dt),
                run_c=lambda a_c=a_c, c_dt=c_dt: c_module._aten_dispatch(op, a_c, [2, 2], dtype=c_dt),
                note="explicit dtype override beats the self tensor's dtype",
            )
        )
    return cases


# --- aten.transpose.int / aten.unsqueeze.default / aten.view.default -------

def transpose_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.transpose.int"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "int64", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        cases.append(
            Case(
                name=f"transpose(dtype={dtype_name}, dim0=0, dim1=1)",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, 0, 1),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, 0, 1),
                note="swap the two axes of a 2D tensor",
            )
        )
        cases.append(
            Case(
                name=f"transpose(dtype={dtype_name}, dim0=-2, dim1=-1)",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, -2, -1),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, -2, -1),
                note="negative dims, as in attention's q @ k.transpose(-2, -1)",
            )
        )
    return cases


def unsqueeze_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.unsqueeze.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "int64", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (4,), dtype_name)
        for dim, note in [(0, "prepend a new axis"), (-1, "append a new axis (negative dim)"), (1, "insert in the middle")]:
            cases.append(
                Case(
                    name=f"unsqueeze(dtype={dtype_name}, dim={dim}) [{note}]",
                    op=op,
                    run_torch=lambda a_t=a_t, dim=dim: torch_call(a_t, dim),
                    run_c=lambda a_c=a_c, dim=dim: c_module._aten_dispatch(op, a_c, dim),
                    note=note,
                )
            )
    return cases


def view_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.view.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        cases.append(
            Case(
                name=f"view(dtype={dtype_name}, (2,3)->(6,)) [also backs reshape() on a contiguous input]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, [6]),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, [6]),
                note=(
                    "flatten -- reshape()'s non-contiguous fallback (clone + "
                    "_unsafe_view) is a different op pair, outside this harness's "
                    "per-op comparison granularity; see the module note above."
                ),
            )
        )
        cases.append(
            Case(
                name=f"view(dtype={dtype_name}, (2,3)->(3,2))",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, [3, 2]),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, [3, 2]),
                note="reshape to a different rank-2 shape",
            )
        )
        cases.append(
            Case(
                name=f"view(dtype={dtype_name}, (2,3)->(-1,)) [inferred dim]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, [-1]),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, [-1]),
                note="-1 means 'infer this dim's size'",
            )
        )
    return cases


# --- aten._to_copy.default ----------------------------------------------------
# `float()`, `long()`, `to(dtype)` all dispatch to the same cast op.

def to_copy_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten._to_copy.default"
    cases: list[Case] = []
    conversions = [
        ("int64", "float32", "backs .float()"),
        ("float32", "int64", "backs .long()"),
        ("float32", "float64", "backs .to(dtype) upcast"),
        ("float64", "float32", "backs .to(dtype) downcast"),
        ("float32", "float16", "backs .to(dtype), precision loss"),
        ("int32", "int64", "backs .to(dtype), int widening"),
        ("uint8", "float32", "backs .to(dtype), unsigned to float"),
    ]
    for src_dtype, dst_dtype, note in conversions:
        a_t, a_c = pair_from_flat(torch_module, c_module, [0, 1, 2, 3], (2, 2), src_dtype)
        t_dt = dt.torch_dtype(torch_module, dst_dtype)
        c_dt = dt.c_dtype(c_module, dst_dtype)
        cases.append(
            Case(
                name=f"_to_copy({src_dtype} -> {dst_dtype}) [{note}]",
                op=op,
                run_torch=lambda a_t=a_t, t_dt=t_dt: torch_call(a_t, dtype=t_dt),
                run_c=lambda a_c=a_c, c_dt=c_dt: c_module._aten_dispatch(op, a_c, dtype=c_dt),
                note=note,
            )
        )
    return cases


# --- in-place methods: aten.fill_.Scalar / aten.copy_.default / -------------
# aten.normal_.default / aten.uniform_.default
#
# See the module note above on why in-place cases build a fresh operand
# pair per case rather than sharing one across cases.

def fill__cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.fill_.Scalar"
    cases: list[Case] = []
    # Reuses the fill-value/boundary table from full_cases above for
    # dtype/boundary coverage parity between "construct filled" (full) and
    # "fill in place" (fill_) -- including its two live regression traps
    # (the float16-overflow and int32-overflow entries). Those are still
    # `expect="match"` in the table itself (never `"c_error"`, per
    # full_cases' own docstring), and a probe against real torch confirmed
    # `fill_.Scalar` refuses the same overflowing values `full.default`
    # does (`RuntimeError: value cannot be converted to type ... without
    # overflow`) -- so the same trap applies here unchanged: this stays
    # `expect="match"` and will only fail if `_C`'s eventual fill_ silently
    # computes instead of refusing, exactly like the original trap's intent.
    # `expect != "match"` guards against `_FULL_FILLS` growing a non-match
    # entry later; it is a no-op today since every current entry is "match".
    for dtype_name in dt.DEFAULT_DTYPES:
        for fill, expect, note in _FULL_FILLS[dtype_name]:
            if expect != "match":
                continue
            a_t, a_c = pair_from_flat(torch_module, c_module, [0, 0, 0, 0], (2, 2), dtype_name)
            cases.append(
                Case(
                    name=f"fill_(dtype={dtype_name}, value={fill!r})",
                    op=op,
                    run_torch=lambda a_t=a_t, fill=fill: torch_call(a_t, fill),
                    run_c=lambda a_c=a_c, fill=fill: c_module._aten_dispatch(op, a_c, fill),
                    note=(note or "in-place fill") + " -- compares the mutated operand fill_ returns",
                )
            )
    return cases


def copy__cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.copy_.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]:
        dst_t, dst_c = pair_from_flat(torch_module, c_module, [0, 0, 0, 0], (2, 2), dtype_name)
        src_t, src_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), dtype_name)
        cases.append(
            Case(
                name=f"copy_(dtype={dtype_name}, same shape)",
                op=op,
                run_torch=lambda dst_t=dst_t, src_t=src_t: torch_call(dst_t, src_t),
                run_c=lambda dst_c=dst_c, src_c=src_c: c_module._aten_dispatch(op, dst_c, src_c),
                note="in-place: compares the mutated dst operand copy_ returns",
            )
        )
    dst2_t, dst2_c = pair_from_flat(torch_module, c_module, [0, 0, 0, 0], (2, 2), "float32")
    src2_t, src2_c = pair_from_flat(torch_module, c_module, [9, 8], (1, 2), "float32")
    cases.append(
        Case(
            name="copy_(dtype=float32, broadcast src)",
            op=op,
            run_torch=lambda: torch_call(dst2_t, src2_t),
            run_c=lambda: c_module._aten_dispatch(op, dst2_c, src2_c),
            note="src (1,2) broadcasts to fill dst (2,2) in place",
        )
    )
    return cases


# The RNG ops' streams are the SAME stream, and these two builders say so.
#
# When they were written, `_C` drew from candle and the module note above was
# right: two independent generators cannot be lined up by a seed. docs/RNG.md
# then established that candle's CPU backend *refuses* to be seeded at all, so
# there was no version of that plan, and torch's own CPU generator was ported
# into `rust/torch_c/src/rng.rs` instead. That makes the seed mean the same
# thing on both sides, and these cases were promoted off `_range_check` to say
# so: seed both generators to the same value inside the run lambdas, then
# compare the draws **element by element**.
#
# What each op may be held to is not the same, and docs/RNG.md §3.3 and §5
# item 3 are the authority for the split:
#
#   * `uniform_` is a masked integer times a power-of-two divisor and an
#     affine step. No libm, no vector specialisation, so **bit-for-bit on
#     every platform**.
#   * `normal_` is Box-Muller through `log`/`cos`/`sin`. Bit-for-bit was
#     *measured* on aarch64, where `NormalFill16`'s AVX2/VSX specialisations
#     do not compile and upstream runs the same scalar libm calls this shim
#     does. On a host where the vector path is live, nobody has measured
#     whether upstream's `sincos256_ps` agrees with libm -- RNG.md §6 records
#     it as unmeasured, and demanding bit equality there would be asserting
#     something no one knows. So: exact on aarch64, dtype tolerance elsewhere.
#
# The tolerance fallback is not a weaker version of the same claim; it is a
# different claim, and `_rng_stream_check` prints which one it made.

_BITWISE_NORMAL_FILL = platform.machine().lower() in ("arm64", "aarch64")


def _same_float(x, y) -> bool:
    """Exact equality, with NaN equal to NaN.

    `tolist()` widens float16/bfloat16/float32 to Python floats losslessly, so
    `==` here really is a comparison of the stored bits -- with the two
    exceptions Python's `==` has: NaN (handled) and -0.0 == 0.0 (left alone;
    neither `uniform_` nor `normal_` can produce a signed zero from a nonzero
    std or a nondegenerate range)."""
    xf, yf = float(x), float(y)
    if math.isnan(xf) or math.isnan(yf):
        return math.isnan(xf) and math.isnan(yf)
    return xf == yf


def _rng_stream_check(*, bitwise: bool, bounds=None):
    """dtype, shape, optional [lo, hi) bound, and then the draws themselves.

    `bitwise=True` demands exact agreement -- the strongest statement this
    harness can make about a random op, and the whole point of porting the
    generator. `bitwise=False` falls back to the dtype's tolerance and says so
    in its message, so a run on an unmeasured platform cannot be read as
    having proved bit equality.
    """

    def check(t_res, c_res) -> tuple[bool, str]:
        t_dtype, c_dtype = dt.dtype_name(t_res.dtype), dt.dtype_name(c_res.dtype)
        if t_dtype != c_dtype:
            return False, f"dtype mismatch: torch={t_dtype} c={c_dtype}"
        t_shape = tuple(int(x) for x in t_res.shape)
        c_shape = tuple(int(x) for x in c_res.shape)
        if t_shape != c_shape:
            return False, f"shape mismatch: torch={t_shape} c={c_shape}"
        t_flat = _flatten_values(t_res.tolist())
        c_flat = _flatten_values(c_res.tolist())
        if len(t_flat) != len(c_flat):
            return False, f"length differs: torch={len(t_flat)} c={len(c_flat)}"
        if bounds is not None:
            lo, hi = bounds
            for label, flat in (("torch", t_flat), ("c", c_flat)):
                for v in flat:
                    if not (lo <= v < hi):
                        return False, f"{label} produced {v!r}, outside requested range [{lo}, {hi})"
        if bitwise:
            for i, (x, y) in enumerate(zip(t_flat, c_flat)):
                if not _same_float(x, y):
                    return (
                        False,
                        f"stream divergence at index {i}: torch={x!r} c={y!r} "
                        "-- same seed, same generator, different value",
                    )
            return True, f"dtype={t_dtype} shape={t_shape}, {len(t_flat)} draws identical bit-for-bit"
        tol = dt.tolerance_for(t_dtype)
        ok, detail = _values_close_local(t_flat, c_flat, tol.atol, tol.rtol)
        if not ok:
            return False, detail
        return True, (
            f"dtype={t_dtype} shape={t_shape}, {len(t_flat)} draws within "
            f"atol={tol.atol}/rtol={tol.rtol} (bit equality NOT asserted: "
            f"machine={platform.machine()!r} is outside what docs/RNG.md §3.3 measured)"
        )

    return check


def _values_close_local(t_flat, c_flat, atol, rtol) -> tuple[bool, str]:
    for i, (x, y) in enumerate(zip(t_flat, c_flat)):
        xf, yf = float(x), float(y)
        if math.isnan(xf) or math.isnan(yf):
            if not (math.isnan(xf) and math.isnan(yf)):
                return False, f"index {i}: torch={x!r} c={y!r} (NaN mismatch)"
            continue
        if not math.isclose(xf, yf, rel_tol=rtol, abs_tol=atol):
            return False, f"index {i}: torch={x!r} c={y!r} (|diff|={abs(xf - yf):.6g})"
    return True, ""


def _seeded_inplace(torch_module, c_module, torch_call, op, dtype_name, shape, seed, args):
    """Both sides seeded to the same value, then the same in-place call.

    The seeding lives inside the lambdas rather than beside them because
    `compare.py` runs `run_torch` and `run_c` one after the other, and each has
    to start from the beginning of the stream rather than from wherever the
    other left it."""
    numel = 1
    for d in shape:
        numel *= d

    def run_torch():
        torch_module.manual_seed(seed)
        target = pair_from_flat(torch_module, c_module, [0.0] * numel, shape, dtype_name)[0]
        return torch_call(target, *args)

    def run_c():
        c_module._shim_manual_seed(seed)
        target = pair_from_flat(torch_module, c_module, [0.0] * numel, shape, dtype_name)[1]
        return c_module._aten_dispatch(op, target, *args)

    return run_torch, run_c


def normal__cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.normal_.default"
    cases: list[Case] = []
    # The sizes are not arbitrary. `normal_kernel` branches on
    # `size >= 16 && is_contiguous()`, and docs/RNG.md §5 item 2 names exactly
    # these five as the ones that cover both sides of it plus the tail
    # redraw: 15 is path B, 16 is path A with no tail, 17 and 20 are path A
    # *rewriting sixteen elements it already computed*, and 32 is path A with
    # the tail case skipped again. From one seed, n=15 and n=16 share not a
    # single value; n=17 differs from n=16 in its first element too. A shim
    # that reproduced the stream and missed the blocking would pass a
    # distribution test and fail every one of these.
    sizes = [
        (6, "path B (size < 16), the size this case used before the port"),
        (15, "path B, one below the boundary"),
        (16, "path A, exactly one block, no tail redraw"),
        (17, "path A + tail: the last 16 elements are drawn again over the top"),
        (20, "path A + tail, mid-block"),
        (32, "path A, two whole blocks, no tail"),
    ]
    for dtype_name in _MUL_DIV_FLOAT_DTYPES:
        for mean, std, why in [
            (0.0, 1.0, "standard normal -- typical init"),
            (0.0, 0.02, "small std, transformer-init style"),
        ]:
            for n, path in sizes:
                for seed in (0, 42):
                    run_torch, run_c = _seeded_inplace(
                        torch_module, c_module, torch_call, op, dtype_name, (n,), seed, (mean, std)
                    )
                    cases.append(
                        Case(
                            name=(
                                f"normal_(dtype={dtype_name}, n={n}, mean={mean}, std={std}, "
                                f"seed={seed}) [{path}]"
                            ),
                            op=op,
                            run_torch=run_torch,
                            run_c=run_c,
                            value_check=_rng_stream_check(bitwise=_BITWISE_NORMAL_FILL),
                            note=why + " -- " + path,
                        )
                    )
    cases.append(
        Case(
            name="normal_(float32, std < 0 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(
                pair_from_flat(torch_module, c_module, [0.0] * 4, (4,), "float32")[0], 0.0, -1.0
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, pair_from_flat(torch_module, c_module, [0.0] * 4, (4,), "float32")[1], 0.0, -1.0
            ),
            expect="both_error",
            note="torch: 'normal expects std >= 0.0, but found std -1'",
        )
    )
    return cases


def uniform__cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.uniform_.default"
    cases: list[Case] = []
    # `(2.0, 7.5)` is here for a specific reason: every range anyone tries
    # first -- (0,1), (-1,1), (-0.5,0.5) -- has a power-of-two width, which
    # makes the affine step's multiply exact and hides whether it was written
    # as `x*(to-from)+from` or as the fused multiply-add clang actually
    # compiles. rng.rs' comment records ~9.5% of draws coming out 1 ulp low
    # before that was `mul_add`. Only a non-power-of-two width can see it, and
    # only a bit-exact comparison can report it.
    ranges = [
        (0.0, 1.0, "default range"),
        (-1.0, 1.0, "range straddling zero"),
        (2.0, 7.5, "non-power-of-two width -- the only range that sees the fused multiply-add"),
    ]
    for dtype_name in _MUL_DIV_FLOAT_DTYPES:
        for lo, hi, why in ranges:
            for n in (6, 17, 1000):
                for seed in (0, 42):
                    run_torch, run_c = _seeded_inplace(
                        torch_module, c_module, torch_call, op, dtype_name, (n,), seed, (lo, hi)
                    )
                    cases.append(
                        Case(
                            name=f"uniform_(dtype={dtype_name}, n={n}, from={lo}, to={hi}, seed={seed}) [{why}]",
                            op=op,
                            run_torch=run_torch,
                            run_c=run_c,
                            # The bound is kept alongside the exact comparison,
                            # not replaced by it: the half-open guarantee is
                            # enforced by a clamp applied *after* the narrowing
                            # cast, and on float16 with to=1.0 that clamp fires
                            # about one draw in 4096. A shim missing it agrees
                            # with the stream everywhere else.
                            value_check=_rng_stream_check(bitwise=True, bounds=(lo, hi)),
                            note=why + " -- bit-for-bit; docs/RNG.md §5 item 3 allows this on every platform",
                        )
                    )
    cases.append(
        Case(
            name="uniform_(float32, from > to rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(
                pair_from_flat(torch_module, c_module, [0.0] * 4, (4,), "float32")[0], 1.0, 0.0
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, pair_from_flat(torch_module, c_module, [0.0] * 4, (4,), "float32")[1], 1.0, 0.0
            ),
            expect="both_error",
            note="torch: 'uniform_ expects to return a [from, to) range, but found from=1 > to=0'",
        )
    )
    return cases


# --- the eight ops a greedy 2-layer Llama forward stopped on (docs/GAP.md §3) --
#
# These are not "more coverage". Before them `_aten_dispatch` refused eight
# names the forward pass reaches, so the model could not run at all; docs/OPS8.md
# records the run that follows from landing them.
#
# Two of the eight are not shaped like anything else in this file and get their
# own treatment below:
#
#   * `bmm` is the batched matmul, and it inherits `mm`'s dtype gap exactly --
#     candle has no matmul kernel for the integer dtypes or bfloat16, so the
#     same split into match / c_error / both_error that `mm_cases` uses applies
#     here verbatim. It also has three *refusals* worth pinning: 2-D input,
#     mismatched dtypes, and -- the one that matters -- a batch of 1 against a
#     batch of 2. `matmul` broadcasts that; `bmm` must not, and a case is the
#     only thing that keeps a later "simplification" from routing one at the
#     other.
#   * `_scaled_dot_product_flash_attention_for_cpu` answers with a *pair*
#     (output, logsumexp) and the two halves do not even share a dtype: for a
#     `float16` input the output is `float16` and the logsumexp is `float32`.
#     `_sdpa_pair_check` below compares both halves including that asymmetry,
#     because it is the observable evidence that the accumulation happens in
#     float, and a shim that returned `float16` there would be wrong in a way
#     no value comparison would catch.


def _deterministic(n: int, seed: int = 1) -> list[float]:
    """`n` reproducible values in roughly [-1, 1].

    Not `random` and not `torch.randn`: both sides of every case have to be
    fed *the same* numbers, and the only way to guarantee that across two
    tensor libraries is to compute them in plain Python and hand the same
    list to each. Values stay near unit magnitude so that `bfloat16`'s
    tolerance (6e-2, dtypes.py) is still a real check rather than a rubber
    stamp.
    """
    out, state = [], seed
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append(round((state / 2147483648.0) * 2.0 - 1.0, 4))
    return out


def bmm_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.bmm.default"
    cases: list[Case] = []

    # (2, 3, 4) @ (2, 4, 5) -- attention's shape, and a batch big enough that
    # a kernel that quietly dropped the batch dimension would show up.
    a_flat, a_shape = list(range(24)), (2, 3, 4)
    b_flat, b_shape = list(range(40)), (2, 4, 5)
    # (1, 2, 2) @ (1, 2, 2) -- a batch of one, where the answer is `mm`'s and
    # can be checked by hand: [[1,2],[3,4]] @ [[5,6],[7,8]] == [[19,22],[43,50]].
    unit_a, unit_b = [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]

    for dtype_name in _MM_MATCH_DTYPES:
        for af, ash, bf, bsh, note in [
            (a_flat, a_shape, b_flat, b_shape, "batched (2,3,4)x(2,4,5) -- attention's QK^T shape"),
            (unit_a, (1, 2, 2), unit_b, (1, 2, 2), "batch of one, hand-checkable 2x2"),
        ]:
            at, ac = pair_from_flat(torch_module, c_module, af, ash, dtype_name)
            bt, bc = pair_from_flat(torch_module, c_module, bf, bsh, dtype_name)
            cases.append(
                Case(
                    name=f"bmm(dtype={dtype_name}, {ash}x{bsh}) [{note}]",
                    op=op,
                    run_torch=lambda at=at, bt=bt: torch_call(at, bt),
                    run_c=lambda ac=ac, bc=bc: c_module._aten_dispatch(op, ac, bc),
                    note=note,
                )
            )

    # The same candle gap `mm_cases` records, re-checked through `bmm`: if it
    # ever closes for one op it should close for both, and these cases say so.
    for dtype_name in _MM_C_ERROR_DTYPES:
        at, ac = pair_from_flat(torch_module, c_module, a_flat, a_shape, dtype_name)
        bt, bc = pair_from_flat(torch_module, c_module, b_flat, b_shape, dtype_name)
        cases.append(
            Case(
                name=f"bmm(dtype={dtype_name}, batched)",
                op=op,
                run_torch=lambda at=at, bt=bt: torch_call(at, bt),
                run_c=lambda ac=ac, bc=bc: c_module._aten_dispatch(op, ac, bc),
                expect="c_error",
                note=(
                    f"candle's matmul has no kernel for {dtype_name}; torch's CPU baddbmm "
                    "does. Same gap mm_cases records for mm -- tracked separately so "
                    "closing it for one op cannot silently look like closing it for both."
                ),
            )
        )

    at, ac = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (1, 2, 2), "uint32")
    cases.append(
        Case(
            name="bmm(dtype=uint32, batched)",
            op=op,
            run_torch=lambda: torch_call(at, at),
            run_c=lambda: c_module._aten_dispatch(op, ac, ac),
            expect="both_error",
            note="neither torch nor candle has a uint32 matmul; neither should invent one.",
        )
    )

    # The three refusals. The batch-broadcast one is the load-bearing case:
    # `matmul.default`'s kernel (candle's `broadcast_matmul`) computes here,
    # and `bmm` must not.
    two_t, two_c = pair_from_flat(torch_module, c_module, [1.0] * 12, (3, 4), "float32")
    three_t, three_c = pair_from_flat(torch_module, c_module, [1.0] * 20, (4, 5), "float32")
    cases.append(
        Case(
            name="bmm(2D input rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(two_t, three_t),
            run_c=lambda: c_module._aten_dispatch(op, two_c, three_c),
            expect="both_error",
            note="torch: 'batch1 must be a 3D tensor'. bmm must not stand in for mm.",
        )
    )

    one_t, one_c = pair_from_flat(torch_module, c_module, [1.0] * 12, (1, 3, 4), "float32")
    many_t, many_c = pair_from_flat(torch_module, c_module, [1.0] * 40, (2, 4, 5), "float32")
    cases.append(
        Case(
            name="bmm(batch 1 x batch 2 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(one_t, many_t),
            run_c=lambda: c_module._aten_dispatch(op, one_c, many_c),
            expect="both_error",
            note=(
                "bmm does not broadcast its batch dimension -- torch: 'Expected size for "
                "first two dimensions of batch2 tensor to be: [1, 4] but got: [2, 4].' "
                "This is exactly the case a one-line route into matmul_default would get "
                "wrong, because candle's broadcast_matmul computes it."
            ),
        )
    )

    f32_t, f32_c = pair_from_flat(torch_module, c_module, [1.0] * 24, (2, 3, 4), "float32")
    f64_t, f64_c = pair_from_flat(torch_module, c_module, [1.0] * 40, (2, 4, 5), "float64")
    cases.append(
        Case(
            name="bmm(float32 x float64 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(f32_t, f64_t),
            run_c=lambda: c_module._aten_dispatch(op, f32_c, f64_c),
            expect="both_error",
            note="torch: 'expected scalar type Float but found Double'; _C: same_dtype refuses.",
        )
    )
    return cases


def unsafe_view_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten._unsafe_view.default"
    cases: list[Case] = []
    # Deliberately the same shape battery `view_cases` uses. `_unsafe_view` is
    # `view`'s value with a different promise to autograd, and there is no
    # autograd here -- so what these cases pin is that the two keys keep
    # answering the same thing, which is the claim the shared kernel makes.
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        for size, note in [
            ([6], "flatten -- the spelling reshape() emits on a non-contiguous input"),
            ([3, 2], "reshape to a different rank-2 shape"),
            ([-1], "-1 means 'infer this dim's size'"),
            ([1, 2, 3], "add a leading axis"),
        ]:
            cases.append(
                Case(
                    name=f"_unsafe_view(dtype={dtype_name}, (2,3)->{size})",
                    op=op,
                    run_torch=lambda a_t=a_t, size=size: torch_call(a_t, size),
                    run_c=lambda a_c=a_c, size=size: c_module._aten_dispatch(op, a_c, size),
                    note=note,
                )
            )
    return cases


def alias_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.alias.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]:
        for flat, shape in [([0], ()), ([1, 2, 3], (3,)), ([1, 2, 3, 4], (2, 2))]:
            cases.append(
                _unary_case(
                    torch_module, c_module, op, torch_call, dtype_name, flat, shape,
                    "identity view -- the storage sharing is not reproduced, only the value",
                )
            )
    return cases


_NEG_INT_DTYPES = ["int64", "int32", "int16", "uint8"]


def neg_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.neg.default"
    cases: list[Case] = []
    for dtype_name in _TRIG_DTYPES:
        for flat, shape, note in [
            ([1.0, -2.0, 0.0, 0.5], (2, 2), "assorted signs, including -0.0 from 0.0"),
            ([_FLOAT_ADD_MAGNITUDE[dtype_name], -_FLOAT_ADD_MAGNITUDE[dtype_name]], (2,), "large magnitudes"),
            ([1.5], (), "0-d"),
        ]:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))

    for dtype_name in _NEG_INT_DTYPES:
        # `neg` keeps the input dtype -- it does *not* promote an integral
        # input to float the way cos/sin/reciprocal do -- so the integer
        # dtypes are the cases that would catch a wrong helper being reused.
        signed = dtype_name != "uint8"
        flat = [1, -2, 0, 7] if signed else [0, 1, 2, 255]
        note = (
            "signed integers keep their dtype (not promoted to float)"
            if signed
            else "unsigned wraps: neg(uint8 1) == 255, matching torch's modular answer"
        )
        cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, (2, 2), note))

    ut, uc = pair_from_flat(torch_module, c_module, [1, 2], (2,), "uint32")
    cases.append(
        Case(
            name="neg(dtype=uint32) [no neg_cpu kernel upstream]",
            op=op,
            run_torch=lambda: torch_call(ut),
            run_c=lambda: c_module._aten_dispatch(op, uc),
            expect="both_error",
            note="torch: NotImplementedError(\"neg_cpu\" not implemented for 'UInt32'); _C copies the refusal.",
        )
    )

    # Bool is built inside the lambdas rather than shared, the same deferral
    # masked_fill_cases uses -- `_tensor_from_flat` is the only route to a bool
    # tensor in `_C` and it is worth keeping this case's construction failure,
    # if it ever comes back, inside one case instead of the whole harness run.
    cases.append(
        Case(
            name="neg(dtype=bool) [torch points at ~ instead]",
            op=op,
            run_torch=lambda: torch_call(torch_module.tensor([True, False])),
            run_c=lambda: c_module._aten_dispatch(
                op, c_module._tensor_from_flat([1, 0], [2], dtype=c_module.bool)
            ),
            expect="both_error",
            note=(
                "torch: 'Negation, the `-` operator, on a bool tensor is not supported.' "
                "A shim that computed here would turn a mask negation into arithmetic."
            ),
        )
    )
    return cases


def rsub_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.rsub.Scalar"
    cases: list[Case] = []
    for dtype_name in _TRIG_DTYPES:
        for scalar, alpha, note in [
            (1.0, None, "1 - x, the mask-building shape"),
            (0.0, None, "0 - x, i.e. negation by another spelling"),
            (2.5, 2.0, "alpha scales *self*, not the scalar"),
            (-1.0, -0.5, "negative scalar and negative alpha"),
        ]:
            a_t, a_c = pair_from_flat(
                torch_module, c_module, [1.0, -2.0, 0.0, 0.5], (2, 2), dtype_name
            )
            args = (scalar,) if alpha is None else (scalar, alpha)
            cases.append(
                Case(
                    name=f"rsub(dtype={dtype_name}, other={scalar}, alpha={alpha}) [{note}]",
                    op=op,
                    run_torch=lambda a_t=a_t, args=args: torch_call(a_t, *args),
                    run_c=lambda a_c=a_c, args=args: c_module._aten_dispatch(op, a_c, *args),
                    note=note,
                )
            )

    # The dtype rule, which is `sub.Scalar`'s: an integral tensor stays
    # integral under an int scalar and floats under a float one (torch's
    # "wrapped number" rule), and `uint8` wraps rather than clamping.
    for dtype_name, flat, scalar, alpha, note in [
        ("int64", [1, 2, 3, 4], 5, None, "int tensor, int scalar -> int64"),
        ("int64", [1, 2, 3, 4], 5.0, None, "int tensor, FLOAT scalar -> float32"),
        ("int32", [1, 2, 3, 4], 3, 2, "int32 with alpha=2 -> 3 - 2*x"),
        ("int16", [1, 2, 3, 4], 0, None, "0 - x on int16"),
        ("uint8", [1, 2, 3, 4], 1, None, "unsigned wraps: 1 - 2 == 255"),
    ]:
        a_t, a_c = pair_from_flat(torch_module, c_module, flat, (2, 2), dtype_name)
        args = (scalar,) if alpha is None else (scalar, alpha)
        cases.append(
            Case(
                name=f"rsub(dtype={dtype_name}, other={scalar!r}, alpha={alpha}) [{note}]",
                op=op,
                run_torch=lambda a_t=a_t, args=args: torch_call(a_t, *args),
                run_c=lambda a_c=a_c, args=args: c_module._aten_dispatch(op, a_c, *args),
                note=note,
            )
        )

    cases.append(
        Case(
            name="rsub(dtype=bool) [torch refuses subtraction on masks]",
            op=op,
            run_torch=lambda: torch_call(torch_module.tensor([True, False]), 1),
            run_c=lambda: c_module._aten_dispatch(
                op, c_module._tensor_from_flat([1, 0], [2], dtype=c_module.bool), 1
            ),
            expect="both_error",
            note="torch: 'Subtraction, the `-` operator, with a bool tensor is not supported.'",
        )
    )
    return cases


def silu_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.silu.default"
    cases: list[Case] = []
    for dtype_name in _TRIG_DTYPES:
        for flat, shape, note in [
            ([1.0, -1.0, 0.0, 3.5], (2, 2), "assorted signs -- SwiGLU's gate input"),
            ([-8.0, -4.0, 4.0, 8.0], (2, 2), "saturating tails, where sigmoid rounds to 0/1"),
            ([0.0], (), "0-d"),
        ]:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))

    # The refusal, and it is the reason silu is not another `Unary` variant:
    # cos/sin/reciprocal promote an integral input to the default float,
    # silu has no integral kernel upstream at all.
    for dtype_name in ["int64", "int32"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2], (2,), dtype_name)
        cases.append(
            Case(
                name=f"silu(dtype={dtype_name}) [no silu_cpu kernel upstream]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                expect="both_error",
                note=(
                    "torch: NotImplementedError(\"silu_cpu\" not implemented for 'Long'). "
                    "Unlike cos/sin, silu does NOT promote an integral input -- a shim "
                    "that reused the unary-float helper here would compute where torch refuses."
                ),
            )
        )
    return cases


def t_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.t.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]:
        for flat, shape, note in [
            ([5], (), "0-d comes back unchanged"),
            ([1, 2, 3], (3,), "1-d comes back unchanged -- NOT transposed"),
            ([1, 2, 3, 4, 5, 6], (2, 3), "2-d swaps, as nn.Linear's x @ w.t() needs"),
            ([1, 2, 3, 4], (1, 4), "row vector -> column vector"),
        ]:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))

    a_t, a_c = pair_from_flat(torch_module, c_module, [1.0] * 24, (2, 3, 4), "float32")
    cases.append(
        Case(
            name="t(3D input rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(a_t),
            run_c=lambda: c_module._aten_dispatch(op, a_c),
            expect="both_error",
            note=(
                "torch: 't() expects a tensor with <= 2 dimensions, but self is 3D'. "
                "t() is not transpose(-2, -1) -- reading it that way would compute on a "
                "batched input where upstream raises."
            ),
        )
    )
    return cases


def _sdpa_pair_check(t_res, c_res) -> tuple[bool, str]:
    """For `_scaled_dot_product_flash_attention_for_cpu`, which returns
    `(output, logsumexp)`.

    Both halves are checked for dtype, shape and value, and the dtypes are
    checked *independently*: for a `float16` input torch answers
    `(float16, float32)`, and that asymmetry is the only externally visible
    evidence that the kernel accumulates in float. Comparing the pair as if
    it shared one dtype would let a shim that computed the whole thing in
    `float16` pass.
    """
    try:
        halves = [(t_res[0], c_res[0], "output"), (t_res[1], c_res[1], "logsumexp")]
    except (TypeError, IndexError, KeyError) as e:
        return False, f"expected a 2-element (output, logsumexp) result on both sides: {e!r}"

    seen = []
    for t_half, c_half, label in halves:
        t_dtype, c_dtype = dt.dtype_name(t_half.dtype), dt.dtype_name(c_half.dtype)
        if t_dtype != c_dtype:
            return False, f"{label} dtype mismatch: torch={t_dtype} c={c_dtype}"
        t_shape = tuple(int(x) for x in t_half.shape)
        c_shape = tuple(int(x) for x in c_half.shape)
        if t_shape != c_shape:
            return False, f"{label} shape mismatch: torch={t_shape} c={c_shape}"
        tol = dt.tolerance_for(t_dtype)
        t_flat, c_flat = _flatten_values(t_half.tolist()), _flatten_values(c_half.tolist())
        if len(t_flat) != len(c_flat):
            return False, f"{label} length differs: torch={len(t_flat)} c={len(c_flat)}"
        for i, (x, y) in enumerate(zip(t_flat, c_flat)):
            xf, yf = float(x), float(y)
            if math.isinf(xf) or math.isinf(yf):
                if xf != yf:
                    return False, f"{label}[{i}] inf mismatch: torch={x!r} c={y!r}"
                continue
            if not math.isclose(xf, yf, rel_tol=tol.rtol, abs_tol=tol.atol):
                return False, f"{label}[{i}] mismatch: torch={x!r} c={y!r}"
        seen.append(f"{label} dtype={t_dtype} shape={t_shape}")
    return True, ", ".join(seen)


_SDPA_DTYPES = ["float64", "float32", "float16", "bfloat16"]


def sdpa_flash_cpu_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten._scaled_dot_product_flash_attention_for_cpu.default"
    cases: list[Case] = []

    b, h, t, e = 1, 2, 3, 4
    n = b * h * t * e
    q_flat, k_flat, v_flat = _deterministic(n, 1), _deterministic(n, 2), _deterministic(n, 3)
    shape = (b, h, t, e)

    for dtype_name in _SDPA_DTYPES:
        for extra_args, kwargs, note in [
            ((0.0, False), {}, "plain attention, default scale = 1/sqrt(head_dim)"),
            ((0.0, True), {}, "is_causal -- upper-left aligned, measured not assumed"),
            ((0.0, False), {"scale": 0.25}, "explicit scale overrides 1/sqrt(head_dim)"),
        ]:
            q_t, q_c = pair_from_flat(torch_module, c_module, q_flat, shape, dtype_name)
            k_t, k_c = pair_from_flat(torch_module, c_module, k_flat, shape, dtype_name)
            v_t, v_c = pair_from_flat(torch_module, c_module, v_flat, shape, dtype_name)
            cases.append(
                Case(
                    name=f"sdpa_flash_cpu(dtype={dtype_name}, args={extra_args}, kwargs={kwargs}) [{note}]",
                    op=op,
                    run_torch=lambda q_t=q_t, k_t=k_t, v_t=v_t, extra_args=extra_args, kwargs=kwargs: torch_call(
                        q_t, k_t, v_t, *extra_args, **kwargs
                    ),
                    run_c=lambda q_c=q_c, k_c=k_c, v_c=v_c, extra_args=extra_args, kwargs=kwargs: c_module._aten_dispatch(
                        op, q_c, k_c, v_c, *extra_args, **kwargs
                    ),
                    value_check=_sdpa_pair_check,
                    note=note + " -- (output, logsumexp), see _sdpa_pair_check",
                )
            )

    # An additive mask, including a `-inf` column: this is how a padding or
    # causal mask actually arrives from transformers, and it is the case that
    # proves the softmax subtracts the row maximum first -- without that,
    # `exp(-inf)` and `exp(large)` both land on NaN.
    mask_shape = (1, 1, t, t)
    mask_flat = [0.0, 0.0, float("-inf")] * t
    for dtype_name in ["float64", "float32"]:
        for is_causal, note in [
            (False, "additive mask with a -inf column"),
            (True, "is_causal AND attn_mask together -- upstream composes them, measured"),
        ]:
            q_t, q_c = pair_from_flat(torch_module, c_module, q_flat, shape, dtype_name)
            k_t, k_c = pair_from_flat(torch_module, c_module, k_flat, shape, dtype_name)
            v_t, v_c = pair_from_flat(torch_module, c_module, v_flat, shape, dtype_name)
            m_t, m_c = pair_from_flat(torch_module, c_module, mask_flat, mask_shape, dtype_name)
            cases.append(
                Case(
                    name=f"sdpa_flash_cpu(dtype={dtype_name}, attn_mask, is_causal={is_causal}) [{note}]",
                    op=op,
                    run_torch=lambda q_t=q_t, k_t=k_t, v_t=v_t, m_t=m_t, is_causal=is_causal: torch_call(
                        q_t, k_t, v_t, 0.0, is_causal, attn_mask=m_t
                    ),
                    run_c=lambda q_c=q_c, k_c=k_c, v_c=v_c, m_c=m_c, is_causal=is_causal: c_module._aten_dispatch(
                        op, q_c, k_c, v_c, 0.0, is_causal, attn_mask=m_c
                    ),
                    value_check=_sdpa_pair_check,
                    note=note,
                )
            )

    # Key/value longer than query, causal. This is the shape a decode step with
    # a KV cache has, and it is where the two possible readings of `is_causal`
    # disagree: upper-left (row i attends keys 0..i) vs bottom-right (row i
    # attends keys 0..i+kv-q). Every element differs between them.
    kv = 5
    q2 = _deterministic(1 * 1 * 2 * e, 4)
    k2 = _deterministic(1 * 1 * kv * e, 5)
    v2 = _deterministic(1 * 1 * kv * e, 6)
    for dtype_name in ["float64", "float32"]:
        q_t, q_c = pair_from_flat(torch_module, c_module, q2, (1, 1, 2, e), dtype_name)
        k_t, k_c = pair_from_flat(torch_module, c_module, k2, (1, 1, kv, e), dtype_name)
        v_t, v_c = pair_from_flat(torch_module, c_module, v2, (1, 1, kv, e), dtype_name)
        cases.append(
            Case(
                name=f"sdpa_flash_cpu(dtype={dtype_name}, q_len=2, kv_len=5, is_causal=True)",
                op=op,
                run_torch=lambda q_t=q_t, k_t=k_t, v_t=v_t: torch_call(q_t, k_t, v_t, 0.0, True),
                run_c=lambda q_c=q_c, k_c=k_c, v_c=v_c: c_module._aten_dispatch(
                    op, q_c, k_c, v_c, 0.0, True
                ),
                value_check=_sdpa_pair_check,
                note="pins the causal alignment: upper-left, not bottom-right",
            )
        )

    # The refusals, all four measured on upstream first.
    q_t, q_c = pair_from_flat(torch_module, c_module, q_flat, shape, "float32")
    k_t, k_c = pair_from_flat(torch_module, c_module, k_flat, shape, "float32")
    v_t, v_c = pair_from_flat(torch_module, c_module, v_flat, shape, "float32")
    cases.append(
        Case(
            name="sdpa_flash_cpu(dropout_p=0.5 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(q_t, k_t, v_t, 0.5, False),
            run_c=lambda: c_module._aten_dispatch(op, q_c, k_c, v_c, 0.5, False),
            expect="both_error",
            note=(
                "torch: 'Currently do not support dropout > 0'. The shim refuses for the "
                "same reason upstream does, not for want of an RNG."
            ),
        )
    )

    q3_t, q3_c = pair_from_flat(torch_module, c_module, _deterministic(h * t * e, 7), (h, t, e), "float32")
    cases.append(
        Case(
            name="sdpa_flash_cpu(3D input rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(q3_t, q3_t, q3_t, 0.0, False),
            run_c=lambda: c_module._aten_dispatch(op, q3_c, q3_c, q3_c, 0.0, False),
            expect="both_error",
            note="torch: 'Accept only 4 dims inputs shape of {B, H, T, K}'.",
        )
    )

    qi_t, qi_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4] * (n // 4), shape, "int64")
    cases.append(
        Case(
            name="sdpa_flash_cpu(int64 input rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(qi_t, qi_t, qi_t, 0.0, False),
            run_c=lambda: c_module._aten_dispatch(op, qi_c, qi_c, qi_c, 0.0, False),
            expect="both_error",
            note="torch: 'Expected data type in FP32, FP64, BF16, FP16, but got Long instead.'",
        )
    )

    m64_t, m64_c = pair_from_flat(torch_module, c_module, [0.0] * (t * t), mask_shape, "float64")
    cases.append(
        Case(
            name="sdpa_flash_cpu(float32 query with float64 mask rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(q_t, k_t, v_t, 0.0, False, attn_mask=m64_t),
            run_c=lambda: c_module._aten_dispatch(op, q_c, k_c, v_c, 0.0, False, attn_mask=m64_c),
            expect="both_error",
            note="torch: 'Attention mask is the same data type as query'.",
        )
    )
    return cases


# --- the eight ops `do_sample=True` stops on (docs/SAMPLING.md) --------------
#
# docs/GAP.md §4 predicted ten; the coordinating session re-measured a real
# transformers Llama with `TorchDispatchMode` and found eight `_aten_dispatch`
# still refused. These are their cases.
#
# Every builder below defers tensor construction into the `run_torch`/`run_c`
# lambdas, the way `masked_fill_cases` does. For the in-place `fill_.Tensor`
# that is load-bearing (a shared operand would carry one case's mutation into
# the next); for the rest it is uniformity, and it keeps a builder from being
# able to crash the whole harness at case-list time.


def _pair(torch_module, c_module, flat, shape, dtype_name):
    """The two operands, built fresh. Same numbers on both sides by
    construction -- see build.pair_from_flat, which this is the deferred form
    of."""
    return pair_from_flat(torch_module, c_module, flat, shape, dtype_name)


# --- aten._softmax.default ---------------------------------------------------
#
# `_softmax` is the op behind `nn.functional.softmax`, and in the sampling loop
# it is what turns logits into the probability vector `multinomial` draws from.
#
# Both of its refusals are reproduced and both are pinned here, because each
# would otherwise be a silent success:
#
#   * `half_to_float=True` raises on CPU for *every* dtype -- float16,
#     bfloat16 and float32 alike (measured). It is a CUDA-only fusion. A shim
#     that honoured it would answer float32 where upstream raises.
#   * an integral input raises `NotImplementedError` naming the kernel, not a
#     `RuntimeError`.
#
# The `-inf` cases are the ones that matter numerically: a masked attention
# row and an all-masked row are both real inputs, and the max-subtraction is
# the only thing standing between the first one and a NaN.

_SOFTMAX_DTYPES = ["float64", "float32", "float16", "bfloat16"]


def softmax_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten._softmax.default"
    cases: list[Case] = []
    scenarios = [
        ([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], (2, 3), -1, "last dim -- the vectorised path"),
        ([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], (2, 3), 0, "first dim -- the strided path"),
        ([1.0, 2.0, 3.0], (3,), 0, "1-D"),
        ([3.0], (), -1, "0-d: the single element is the whole distribution, so 1.0"),
    ]
    for dtype_name in _SOFTMAX_DTYPES:
        for flat, shape, dim, note in scenarios:
            cases.append(
                Case(
                    name=f"_softmax(dtype={dtype_name}, shape={shape}, dim={dim}) [{note}]",
                    op=op,
                    run_torch=lambda flat=flat, shape=shape, dim=dim, dtype_name=dtype_name: torch_call(
                        _pair(torch_module, c_module, flat, shape, dtype_name)[0], dim, False
                    ),
                    run_c=lambda flat=flat, shape=shape, dim=dim, dtype_name=dtype_name: c_module._aten_dispatch(
                        op, _pair(torch_module, c_module, flat, shape, dtype_name)[1], dim, False
                    ),
                    note=note,
                )
            )

    edge = [
        ([1.0, float("-inf"), 2.0], (3,), "one masked position",
         "exp(-inf - max) is a clean zero only because the max is subtracted first"),
        ([float("-inf"), float("-inf")], (2,), "a fully masked row",
         "NaN on both sides, and that agreement is the point"),
        ([1000.0, 1001.0, 999.0], (3,), "large logits",
         "without the max subtraction every exp overflows to inf"),
        ([], (0,), "empty", "no lane to reduce"),
    ]
    for flat, shape, label, note in edge:
        cases.append(
            Case(
                name=f"_softmax(float32, {label})",
                op=op,
                run_torch=lambda flat=flat, shape=shape: torch_call(
                    _pair(torch_module, c_module, flat, shape, "float32")[0], -1, False
                ),
                run_c=lambda flat=flat, shape=shape: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, flat, shape, "float32")[1], -1, False
                ),
                note=note,
            )
        )

    for dtype_name, why in [
        ("float32", "torch: 'softmax with half to float conversion is not supported on CPU'"),
        ("float16", "the same refusal, for the dtype whose name the flag comes from"),
    ]:
        cases.append(
            Case(
                name=f"_softmax(dtype={dtype_name}, half_to_float=True rejected on both sides)",
                op=op,
                run_torch=lambda dtype_name=dtype_name: torch_call(
                    _pair(torch_module, c_module, [1.0, 2.0, 3.0], (3,), dtype_name)[0], -1, True
                ),
                run_c=lambda dtype_name=dtype_name: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, [1.0, 2.0, 3.0], (3,), dtype_name)[1], -1, True
                ),
                expect="both_error",
                note=why,
            )
        )
    cases.append(
        Case(
            name="_softmax(int64 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [1, 2, 3], (3,), "int64")[0], -1, False
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [1, 2, 3], (3,), "int64")[1], -1, False
            ),
            expect="both_error",
            note='torch: NotImplementedError, "softmax_lastdim_kernel_impl" not implemented for \'Long\'',
        )
    )
    cases.append(
        Case(
            name="_softmax(dim out of range rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [1.0, 2.0, 3.0], (3,), "float32")[0], 5, False
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [1.0, 2.0, 3.0], (3,), "float32")[1], 5, False
            ),
            expect="both_error",
            note="IndexError on both sides",
        )
    )
    return cases


# --- aten.le.Scalar ----------------------------------------------------------
# The same family as lt/eq/ne above; `x <= v` keeps the Python number as a
# Scalar rather than lifting it to a tensor. Reached by the repetition-penalty
# and min-length warpers.


def le_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.le.Scalar"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        cases.append(
            _binary_scalar_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [1, 2, 3, 4], (2, 2), 3,
                "x <= 3, as reached from __le__ with a python scalar -- note 3 itself is "
                "included, which is the whole difference from lt.Scalar",
            )
        )
    cases.append(
        Case(
            name="le(float32, nan <= 1.0) [every comparison against NaN is false]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [float("nan"), 1.0], (2,), "float32")[0], 1.0
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [float("nan"), 1.0], (2,), "float32")[1], 1.0
            ),
            note="NaN is not <= anything, including itself",
        )
    )
    return cases


# --- aten.squeeze.dim --------------------------------------------------------
#
# The generation loop's `next_tokens.squeeze(1)`. The rule that has to be right
# is the one that looks like a bug: **a dimension whose size is not 1 is a
# no-op, not an error** (measured). Refusing there would break the loop the
# moment a batch had a single row.


def squeeze_dim_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.squeeze.dim"
    cases: list[Case] = []
    flat, shape = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], (1, 3, 1, 2)
    for dim, note in [
        (0, "leading size-1 dim removed"),
        (2, "interior size-1 dim removed"),
        (1, "size 3 -- NO-OP, not an error"),
        (-1, "size 2 via a negative dim -- also a no-op"),
        (-4, "the leading dim again, addressed from the end"),
    ]:
        cases.append(
            Case(
                name=f"squeeze({shape}, dim={dim}) [{note}]",
                op=op,
                run_torch=lambda dim=dim: torch_call(
                    _pair(torch_module, c_module, flat, shape, "float32")[0], dim
                ),
                run_c=lambda dim=dim: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, flat, shape, "float32")[1], dim
                ),
                note=note,
            )
        )
    for dtype_name in ["float64", "float32", "int64", "uint8"]:
        cases.append(
            Case(
                name=f"squeeze(dtype={dtype_name}, (2,1) dim=1)",
                op=op,
                run_torch=lambda dtype_name=dtype_name: torch_call(
                    _pair(torch_module, c_module, [1, 2], (2, 1), dtype_name)[0], 1
                ),
                run_c=lambda dtype_name=dtype_name: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, [1, 2], (2, 1), dtype_name)[1], 1
                ),
                note="the shape the sampling loop actually squeezes: (batch, 1) -> (batch,)",
            )
        )
    cases.append(
        Case(
            name="squeeze(0-d, dim=0) [torch accepts dim 0 and -1 on a 0-d tensor]",
            op=op,
            run_torch=lambda: torch_call(_pair(torch_module, c_module, [5.0], (), "float32")[0], 0),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [5.0], (), "float32")[1], 0
            ),
            note="nothing to remove, so the 0-d tensor comes back unchanged",
        )
    )
    cases.append(
        Case(
            name="squeeze(dim out of range rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(_pair(torch_module, c_module, flat, shape, "float32")[0], 9),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, flat, shape, "float32")[1], 9
            ),
            expect="both_error",
            note="IndexError on both sides -- a no-op dim is not the same as a nonexistent one",
        )
    )
    return cases


# --- aten.sort.default / aten.topk.default -----------------------------------
#
# Both answer a (values, indices) pair, so both reuse `_pair_result_check`.
# What they do *not* share is how far the agreement goes, and that difference
# is measured, not assumed:
#
#   * **`sort` is stable, in both directions.** `[3,1,3,1,2,3]` descending
#     answers indices `[0,2,5,4,1,3]` -- the three 3.0s in increasing index
#     order, not reversed. An 80-element all-ties tensor comes back as
#     `0..79`. So ties can be compared exactly and are.
#   * **`topk` is a partial selection and its tie order is not stable.** On
#     that same input `k=3` agrees with a stable sort (`[0,2,5]`) but `k=6`
#     does not: upstream answers `[0,2,5,4,3,1]`, reversing the two 1.0s.
#     Upstream promises nothing there, so the tied `topk` case below compares
#     values only, via `_topk_multiset_check`, and every case that compares
#     indices uses tie-free input. docs/SAMPLING.md §4.
#
# `sorted=False` is the same situation one step further: upstream returns a
# partition artefact (`k=3` of an 8-element tensor gives `[7,6,0]` where
# `sorted=True` gives `[6,7,0]`), so those cases are multiset-compared too.

_ORDER_DTYPES = ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]

_TIED = [3.0, 1.0, 3.0, 1.0, 2.0, 3.0]
_DISTINCT = [5.0, 1.0, 4.0, 2.0, 3.0, 0.0]


def _topk_multiset_check(t_res, c_res) -> tuple[bool, str]:
    """dtype and shape exactly; values as a sorted multiset; indices only
    through the (value, index) pairing.

    For the cases where upstream's own order is an artefact of its partition
    rather than a promise -- ties, and `sorted=False`. Pinning the order there
    would be pinning an implementation detail; dropping the check entirely
    would let a shim return the wrong *elements*. This checks the elements.
    """
    try:
        t_values, t_indices = t_res[0], t_res[1]
        c_values, c_indices = c_res[0], c_res[1]
    except (TypeError, IndexError, KeyError) as e:
        return False, f"expected a 2-element (values, indices) result on both sides: {e!r}"
    t_dtype, c_dtype = dt.dtype_name(t_values.dtype), dt.dtype_name(c_values.dtype)
    if t_dtype != c_dtype:
        return False, f"values dtype mismatch: torch={t_dtype} c={c_dtype}"
    t_shape = tuple(int(x) for x in t_values.shape)
    c_shape = tuple(int(x) for x in c_values.shape)
    if t_shape != c_shape:
        return False, f"values shape mismatch: torch={t_shape} c={c_shape}"
    t_pairs = sorted(zip(_flatten_values(t_values.tolist()), _flatten_values(t_indices.tolist())))
    c_pairs = sorted(zip(_flatten_values(c_values.tolist()), _flatten_values(c_indices.tolist())))
    if t_pairs != c_pairs:
        return False, f"selected elements differ: torch={t_pairs!r} c={c_pairs!r}"
    return True, (
        f"values dtype={t_dtype} shape={t_shape}, same {len(t_pairs)} (value, index) pairs "
        "-- order deliberately unchecked, see the note above"
    )


def sort_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.sort.default"
    cases: list[Case] = []
    for dtype_name in _ORDER_DTYPES:
        for descending in (False, True):
            cases.append(
                Case(
                    name=f"sort(dtype={dtype_name}, descending={descending}) [ties at 3.0 and 1.0]",
                    op=op,
                    run_torch=lambda dtype_name=dtype_name, d=descending: torch_call(
                        _pair(torch_module, c_module, _TIED, (6,), dtype_name)[0], -1, d
                    ),
                    run_c=lambda dtype_name=dtype_name, d=descending: c_module._aten_dispatch(
                        op, _pair(torch_module, c_module, _TIED, (6,), dtype_name)[1], -1, d
                    ),
                    value_check=_pair_result_check,
                    note="ties compared exactly -- upstream's CPU sort is stable in both directions",
                )
            )
    for dim in (0, 1, -1):
        cases.append(
            Case(
                name=f"sort(float32, (2,3), dim={dim})",
                op=op,
                run_torch=lambda dim=dim: torch_call(
                    _pair(torch_module, c_module, [3, 1, 3, 2, 2, 1], (2, 3), "float32")[0], dim, False
                ),
                run_c=lambda dim=dim: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, [3, 1, 3, 2, 2, 1], (2, 3), "float32")[1], dim, False
                ),
                value_check=_pair_result_check,
                note="lane extraction along a non-last dim",
            )
        )
    nan_flat = [1.0, float("nan"), 0.0, float("inf"), float("-inf")]
    for descending in (False, True):
        cases.append(
            Case(
                name=f"sort(float32, NaN/inf, descending={descending})",
                op=op,
                run_torch=lambda d=descending: torch_call(
                    _pair(torch_module, c_module, nan_flat, (5,), "float32")[0], -1, d
                ),
                run_c=lambda d=descending: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, nan_flat, (5,), "float32")[1], -1, d
                ),
                value_check=_pair_result_check,
                note=(
                    "torch orders NaN as GREATEST -- last ascending, first descending. "
                    "IEEE says every comparison against it is false, so this is a choice "
                    "torch made and a shim has to copy rather than inherit."
                ),
            )
        )
    cases.append(
        Case(
            name="sort(float32, 80 elements all tied two ways, descending) [stability at scale]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [1.0] * 40 + [0.0] * 40, (80,), "float32")[0], -1, True
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [1.0] * 40 + [0.0] * 40, (80,), "float32")[1], -1, True
            ),
            value_check=_pair_result_check,
            note="indices must come back as 0..79 -- an unstable sort would pass a value check and fail this",
        )
    )
    for flat, shape, note in [([5.0], (), "0-d: value and index both 0-d"), ([], (0,), "empty")]:
        cases.append(
            Case(
                name=f"sort(float32, {note})",
                op=op,
                run_torch=lambda flat=flat, shape=shape: torch_call(
                    _pair(torch_module, c_module, flat, shape, "float32")[0], -1, False
                ),
                run_c=lambda flat=flat, shape=shape: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, flat, shape, "float32")[1], -1, False
                ),
                value_check=_pair_result_check,
                note=note,
            )
        )
    cases.append(
        Case(
            name="sort(dim out of range rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, _TIED, (6,), "float32")[0], 3, False
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, _TIED, (6,), "float32")[1], 3, False
            ),
            expect="both_error",
            note="IndexError on both sides",
        )
    )
    return cases


def topk_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.topk.default"
    cases: list[Case] = []
    for dtype_name in _ORDER_DTYPES:
        for k in (1, 3, 6):
            for largest in (True, False):
                cases.append(
                    Case(
                        name=f"topk(dtype={dtype_name}, k={k}, largest={largest}) [distinct values]",
                        op=op,
                        run_torch=lambda dtype_name=dtype_name, k=k, lg=largest: torch_call(
                            _pair(torch_module, c_module, _DISTINCT, (6,), dtype_name)[0], k, -1, lg, True
                        ),
                        run_c=lambda dtype_name=dtype_name, k=k, lg=largest: c_module._aten_dispatch(
                            op, _pair(torch_module, c_module, _DISTINCT, (6,), dtype_name)[1], k, -1, lg, True
                        ),
                        value_check=_pair_result_check,
                        note="no ties, so the indices are determined and compared exactly",
                    )
                )
    cases.append(
        Case(
            name="topk(float32, k=3, largest=True) [ties -- values and selection only]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, _TIED, (6,), "float32")[0], 3, -1, True, True
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, _TIED, (6,), "float32")[1], 3, -1, True, True
            ),
            value_check=_topk_multiset_check,
            note="upstream's tie order here is a partition artefact -- see the note above",
        )
    )
    cases.append(
        Case(
            name="topk(float32, k=6, largest=True) [ties, k == n -- upstream reverses the tied pair]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, _TIED, (6,), "float32")[0], 6, -1, True, True
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, _TIED, (6,), "float32")[1], 6, -1, True, True
            ),
            value_check=_topk_multiset_check,
            note=(
                "MEASURED DIVERGENCE, deliberately not chased: upstream answers indices "
                "[0,2,5,4,3,1] and this shim answers [0,2,5,4,1,3]. Same six elements, "
                "different order among equal values. torch.topk documents no order for "
                "ties; the values -- which is all TopKLogitsWarper reads -- are identical."
            ),
        )
    )
    for k in (1, 3):
        cases.append(
            Case(
                name=f"topk(float32, k={k}, sorted=False) [order unspecified upstream]",
                op=op,
                run_torch=lambda k=k: torch_call(
                    _pair(torch_module, c_module, [5.0, 1.0, 4.0, 2.0, 3.0, 0.0, 9.0, 7.0], (8,), "float32")[0],
                    k, -1, True, False,
                ),
                run_c=lambda k=k: c_module._aten_dispatch(
                    op,
                    _pair(torch_module, c_module, [5.0, 1.0, 4.0, 2.0, 3.0, 0.0, 9.0, 7.0], (8,), "float32")[1],
                    k, -1, True, False,
                ),
                value_check=_topk_multiset_check,
                note=(
                    "sorted=False licenses any order and upstream uses it: k=3 answers "
                    "[7,6,0] where sorted=True answers [6,7,0]. This shim always sorts, "
                    "which is within the licence."
                ),
            )
        )
    cases.append(
        Case(
            name="topk(float32, (2,3), k=2, dim=0)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [3, 1, 3, 2, 2, 1], (2, 3), "float32")[0], 2, 0, True, True
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [3, 1, 3, 2, 2, 1], (2, 3), "float32")[1], 2, 0, True, True
            ),
            value_check=_pair_result_check,
            note="selection along a non-last dim; no ties within a lane",
        )
    )
    cases.append(
        Case(
            name="topk(float32, 0-d, k=1)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [5.0], (), "float32")[0], 1, -1, True, True
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [5.0], (), "float32")[1], 1, -1, True, True
            ),
            value_check=_pair_result_check,
            note="torch answers a 0-d value and a 0-d index of 0 rather than refusing",
        )
    )
    cases.append(
        Case(
            name="topk(float32, k=0)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, _DISTINCT, (6,), "float32")[0], 0, -1, True, True
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, _DISTINCT, (6,), "float32")[1], 0, -1, True, True
            ),
            value_check=_pair_result_check,
            note="k=0 is legal and gives empty results, unlike k<0",
        )
    )
    for k, why in [(7, "k > n"), (-1, "k < 0 -- torch gives the same 'out of range' message")]:
        cases.append(
            Case(
                name=f"topk(k={k} rejected on both sides) [{why}]",
                op=op,
                run_torch=lambda k=k: torch_call(
                    _pair(torch_module, c_module, _DISTINCT, (6,), "float32")[0], k, -1, True, True
                ),
                run_c=lambda k=k: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, _DISTINCT, (6,), "float32")[1], k, -1, True, True
                ),
                expect="both_error",
                note="torch: RuntimeError 'selected index k out of range'",
            )
        )
    return cases


# --- aten.scatter.src --------------------------------------------------------
#
# Written against torch's shape rule, which is looser than candle's: torch asks
# only that `index` be **no larger** than `self` (off the scatter axis) and than
# `src`, while candle demands equality. The generation loop uses exactly the
# shape candle rejects -- a `(batch, k)` index scattered into a `(batch, vocab)`
# row -- so the two `index smaller than src/self` cases below are the ones that
# would break if someone "simplified" this onto `Tensor::scatter`.


def scatter_src_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.scatter.src"
    cases: list[Case] = []

    def case(name, self_arg, dim, idx_arg, src_arg, note, expect="match"):
        cases.append(
            Case(
                name=name,
                op=op,
                run_torch=lambda: torch_call(
                    _pair(torch_module, c_module, *self_arg)[0],
                    dim,
                    _pair(torch_module, c_module, *idx_arg)[0],
                    _pair(torch_module, c_module, *src_arg)[0],
                ),
                run_c=lambda: c_module._aten_dispatch(
                    op,
                    _pair(torch_module, c_module, *self_arg)[1],
                    dim,
                    _pair(torch_module, c_module, *idx_arg)[1],
                    _pair(torch_module, c_module, *src_arg)[1],
                ),
                expect=expect,
                note=note,
            )
        )

    src_35 = (list(range(1, 16)), (3, 5), "float32")
    zeros_35 = ([0.0] * 15, (3, 5), "float32")
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32"]:
        case(
            f"scatter(dtype={dtype_name}, dim=1, index (3,3) into self (3,5))",
            ([0] * 15, (3, 5), dtype_name),
            1,
            ([0, 1, 2] * 3, (3, 3), "int64"),
            (list(range(1, 16)), (3, 5), dtype_name),
            "index smaller than both self and src along the scatter axis -- the shape "
            "candle's own scatter refuses",
        )
    case(
        "scatter(float32, dim=0, index (1,5) into self (3,5))",
        zeros_35, 0, ([0, 1, 2, 0, 0], (1, 5), "int64"), src_35,
        "index smaller than self along a non-scatter axis",
    )
    case(
        "scatter(float32, duplicate indices -- last write wins)",
        ([0.0] * 5, (1, 5), "float32"), 1, ([0, 0, 0], (1, 3), "int64"), ([1.0, 2.0, 3.0, 0.0, 0.0], (1, 5), "float32"),
        "three writes to column 0; torch leaves the last (3.0), measured",
    )
    case(
        "scatter(float32, 1-D)",
        ([0.0] * 5, (5,), "float32"), 0, ([4, 0], (2,), "int64"), ([9.0, 8.0], (2,), "float32"),
        "rank 1 -- the scatter axis is the only axis",
    )
    case(
        "scatter(float32, int32 index accepted)",
        ([0.0] * 3, (3,), "float32"), 0, ([1], (1,), "int32"), ([1.0], (1,), "float32"),
        "torch accepts int32 here -- unlike masked_fill's mask, which must be exactly bool",
    )
    case(
        "scatter(float32, index out of bounds rejected on both sides)",
        ([0.0] * 3, (3,), "float32"), 0, ([5], (1,), "int64"), ([1.0], (1,), "float32"),
        "torch: 'index 5 is out of bounds for dimension 0 with size 3'",
        expect="both_error",
    )
    case(
        "scatter(float32, negative index rejected on both sides)",
        ([0.0] * 3, (3,), "float32"), 0, ([-1], (1,), "int64"), ([1.0], (1,), "float32"),
        "unlike a dim, a scatter index is not wrapped -- torch refuses -1",
        expect="both_error",
    )
    case(
        "scatter(self/src dtype mismatch rejected on both sides)",
        ([0.0] * 3, (3,), "float32"), 0, ([1], (1,), "int64"), ([1], (1,), "int64"),
        "torch: 'scatter(): Expected self.dtype to be equal to src.dtype' -- no promotion",
        expect="both_error",
    )
    case(
        "scatter(index rank mismatch rejected on both sides)",
        zeros_35, 1, ([0, 1, 2], (3,), "int64"), src_35,
        "torch: 'Index tensor must have the same number of dimensions as self tensor'",
        expect="both_error",
    )
    case(
        "scatter(index larger than src rejected on both sides)",
        zeros_35, 1, ([0, 1, 2], (1, 3), "int64"), ([1.0, 2.0], (1, 2), "float32"),
        "'no larger than self ... and no larger size than src' -- this violates the src half",
        expect="both_error",
    )
    return cases


# --- aten.fill_.Tensor -------------------------------------------------------
#
# `fill_` with a 0-d tensor rather than a Python number. It was implemented
# alongside `fill_.Scalar` and parked in `IMPLEMENTED_AWAITING_GOLDEN` for want
# of exactly this builder; these cases are what moved it across.
#
# The difference from `fill_.Scalar` is not cosmetic: the `c10::checked_convert`
# overflow refusal that `fill_.Scalar` reproduces does **not** apply to a tensor
# value. `fill_(float16_tensor, tensor(1e6))` gives `inf` on upstream too
# (measured), where `fill_(float16_tensor, 1e6)` raises. So the float16 case
# below is the mirror image of `full_cases`' live regression trap, and it is
# `expect="match"` for the opposite reason: here `inf` is the right answer.


def fill__tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.fill_.Tensor"
    cases: list[Case] = []
    for dtype_name in dt.DEFAULT_DTYPES:
        cases.append(
            Case(
                name=f"fill_.Tensor(dtype={dtype_name}, 0-d value of the same dtype)",
                op=op,
                run_torch=lambda dtype_name=dtype_name: torch_call(
                    _pair(torch_module, c_module, [0, 0, 0, 0], (2, 2), dtype_name)[0],
                    _pair(torch_module, c_module, [3], (), dtype_name)[0],
                ),
                run_c=lambda dtype_name=dtype_name: c_module._aten_dispatch(
                    op,
                    _pair(torch_module, c_module, [0, 0, 0, 0], (2, 2), dtype_name)[1],
                    _pair(torch_module, c_module, [3], (), dtype_name)[1],
                ),
                note="in-place: compares the mutated operand fill_ returns",
            )
        )
    cases.append(
        Case(
            name="fill_.Tensor(int64 <- 0-d float32 2.7) [the value is cast to self's dtype, truncating]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [0, 0], (2,), "int64")[0],
                _pair(torch_module, c_module, [2.7], (), "float32")[0],
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module, [0, 0], (2,), "int64")[1],
                _pair(torch_module, c_module, [2.7], (), "float32")[1],
            ),
            note="self keeps its dtype; the value does not widen it",
        )
    )
    cases.append(
        Case(
            name="fill_.Tensor(float16 <- 0-d 1e6) [overflows to inf on BOTH sides, unlike fill_.Scalar]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [0.0] * 3, (3,), "float16")[0],
                _pair(torch_module, c_module, [1e6], (), "float32")[0],
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module, [0.0] * 3, (3,), "float16")[1],
                _pair(torch_module, c_module, [1e6], (), "float32")[1],
            ),
            note=(
                "the mirror of full_cases' float16 trap: c10::checked_convert guards the "
                "*Scalar* overload, not this one, so inf is upstream's own answer here"
            ),
        )
    )
    cases.append(
        Case(
            name="fill_.Tensor(1-D value rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [0.0] * 3, (3,), "float32")[0],
                _pair(torch_module, c_module, [1.0], (1,), "float32")[0],
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module, [0.0] * 3, (3,), "float32")[1],
                _pair(torch_module, c_module, [1.0], (1,), "float32")[1],
            ),
            expect="both_error",
            note=(
                "torch: 'fill_ only supports 0-dimension value tensor but got tensor with "
                "1 dimensions.' The shim raises TypeError rather than RuntimeError; "
                "both_error accepts either, and the refusal is the point."
            ),
        )
    )
    return cases


# --- aten.multinomial.default ------------------------------------------------
#
# The only op in this file that draws, and therefore the only one whose cases
# have to say *when* each side draws. Both sides are seeded inside the
# `run_torch`/`run_c` lambdas -- `torch.manual_seed(s)` and
# `_C._shim_manual_seed(s)` -- so each case starts both generators at the same
# point in the same stream and the sampled indices can be compared **exactly**.
#
# That is only meaningful because docs/RNG.md's port makes the two streams the
# same object. The module note above still says a seed cannot synchronise two
# independent RNGs; it is right about *independent* ones and no longer
# describes this codebase (see the uniform_/normal_ builders, which were
# promoted off `_range_check` for the same reason).
#
# The cases have to cross the branch that decides which algorithm runs, because
# the branch is not where the argument names suggest: upstream takes the
# Gumbel-style fast path when `!replacement` **or `num_samples == 1`**, so
# `multinomial(probs, 1)` -- the call `GenerationMixin._sample` makes -- takes
# it even with `replacement` left False. `num_samples=3, replacement=True` is
# the only combination below that reaches the cumulative-sum kernel.

_MULTINOMIAL_DTYPES = ["float64", "float32", "float16", "bfloat16"]
_MULTINOMIAL_SEEDS = [0, 1, 42, 1234]


def _seeded_multinomial(torch_module, c_module, torch_call, op, flat, shape, dtype_name, n_sample, replacement, seed):
    def run_torch():
        torch_module.manual_seed(seed)
        return torch_call(
            _pair(torch_module, c_module, flat, shape, dtype_name)[0], n_sample, replacement
        )

    def run_c():
        c_module._shim_manual_seed(seed)
        return c_module._aten_dispatch(
            op, _pair(torch_module, c_module, flat, shape, dtype_name)[1], n_sample, replacement
        )

    return run_torch, run_c


def multinomial_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.multinomial.default"
    cases: list[Case] = []
    # Deliberately unnormalised and deliberately not uniform: upstream
    # normalises by the row sum itself, and a uniform row would hide any
    # ordering mistake.
    row = [(i % 7) + 1 for i in range(11)]
    grid = [
        (row, (11,), 1, False, "1-D, one sample -- the fast path, and the call the sampler makes"),
        (row, (11,), 1, True, "replacement=True with one sample still takes the FAST path"),
        (row, (11,), 3, False, "no replacement, three samples -- fast path via topk"),
        (row, (11,), 3, True, "the cumulative-sum kernel, the only combination that reaches it"),
        (row * 3, (3, 11), 1, False, "2-D: one sample per row"),
        (row * 3, (3, 11), 5, True, "2-D cumulative-sum kernel, five samples per row"),
    ]
    for dtype_name in _MULTINOMIAL_DTYPES:
        for flat, shape, n_sample, replacement, note in grid:
            for seed in _MULTINOMIAL_SEEDS:
                run_torch, run_c = _seeded_multinomial(
                    torch_module, c_module, torch_call, op, flat, shape, dtype_name,
                    n_sample, replacement, seed,
                )
                cases.append(
                    Case(
                        name=(
                            f"multinomial(dtype={dtype_name}, shape={shape}, num_samples={n_sample}, "
                            f"replacement={replacement}, seed={seed})"
                        ),
                        op=op,
                        run_torch=run_torch,
                        run_c=run_c,
                        note=note + " -- both generators seeded to the same value; indices compared exactly",
                    )
                )

    def refusal(name, flat, shape, dtype_name, n_sample, replacement, note):
        cases.append(
            Case(
                name=name,
                op=op,
                run_torch=lambda: torch_call(
                    _pair(torch_module, c_module, flat, shape, dtype_name)[0], n_sample, replacement
                ),
                run_c=lambda: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, flat, shape, dtype_name)[1], n_sample, replacement
                ),
                expect="both_error",
                note=note,
            )
        )

    refusal("multinomial(num_samples=0 rejected on both sides)", row, (11,), "float32", 0, False,
            "torch: 'cannot sample n_sample <= 0 samples'")
    refusal("multinomial(num_samples > categories without replacement rejected on both sides)",
            row, (11,), "float32", 12, False,
            "torch: 'cannot sample n_sample > prob_dist.size(-1) samples without replacement'")
    refusal("multinomial(int64 input rejected on both sides)", [1, 2, 3], (3,), "int64", 1, False,
            "torch: 'multinomial only supports floating-point dtypes for input, got: Long'")
    refusal("multinomial(3-D input rejected on both sides)", [1.0] * 8, (2, 2, 2), "float32", 1, False,
            "torch: 'prob_dist must be 1 or 2 dim'")
    refusal("multinomial(negative probability rejected on both sides)", [-1.0, 2.0], (2,), "float32", 1, False,
            "torch: 'probability tensor contains either `inf`, `nan` or element < 0'")
    refusal("multinomial(inf probability rejected on both sides)", [float("inf"), 2.0], (2,), "float32", 1, False,
            "same check -- max() < INFINITY is false")
    refusal("multinomial(NaN probability rejected on both sides)", [float("nan"), 2.0], (2,), "float32", 1, False,
            "same check -- every comparison against NaN is false, so both halves of it fail")
    refusal("multinomial(all-zero row rejected on both sides)", [0.0, 0.0], (2,), "float32", 1, False,
            "torch: 'invalid multinomial distribution (sum of probabilities <= 0)'")
    refusal("multinomial(all-zero row, with replacement, rejected on both sides)", [0.0, 0.0], (2,), "float32", 2, True,
            "the same refusal from the other kernel -- reached by a different check upstream")
    return cases


CASE_BUILDERS: dict[str, Callable[[Any, Any, Callable], list[Case]]] = {
    "aten.full.default": full_cases,
    "aten.add.Tensor": add_cases,
    "aten.mm.default": mm_cases,
    # Pre-seeded ahead of implementation -- see the module note above.
    "aten.arange.default": arange_default_cases,
    "aten.arange.start": arange_start_cases,
    "aten.arange.start_step": arange_start_step_cases,
    "aten.argmax.default": argmax_cases,
    "aten.cat.default": cat_cases,
    "aten.embedding.default": embedding_cases,
    "aten.empty.memory_format": empty_cases,
    "aten.is_floating_point.default": is_floating_point_cases,
    "aten.isin.Tensor_Tensor": isin_cases,
    "aten.ones.default": ones_cases,
    "aten.pow.Tensor_Scalar": pow_tensor_scalar_cases,
    "aten.pow.Tensor_Tensor": pow_tensor_tensor_cases,
    "aten.pow.Scalar": pow_scalar_cases,
    "aten.randint.low": randint_low_cases,
    "aten.rsqrt.default": rsqrt_cases,
    "aten.lift_fresh.default": lift_fresh_cases,
    # Pre-seeded ahead of implementation for TensorBase's 50 actually-used
    # members (docs/C_SURFACE.md §4) -- see the longer module note above.
    "aten.sub.Tensor": sub_cases,
    "aten.mul.Tensor": mul_cases,
    "aten.div.Tensor": div_cases,
    "aten.bitwise_and.Tensor": bitwise_and_tensor_cases,
    "aten.bitwise_and.Scalar": bitwise_and_scalar_cases,
    "aten.bitwise_or.Tensor": bitwise_or_tensor_cases,
    "aten.bitwise_or.Scalar": bitwise_or_scalar_cases,
    "aten.bitwise_not.default": bitwise_not_cases,
    "aten.eq.Tensor": eq_tensor_cases,
    "aten.eq.Scalar": eq_scalar_cases,
    "aten.lt.Tensor": lt_tensor_cases,
    "aten.lt.Scalar": lt_scalar_cases,
    "aten.ne.Tensor": ne_tensor_cases,
    "aten.ne.Scalar": ne_scalar_cases,
    "aten._local_scalar_dense.default": local_scalar_dense_cases,
    "aten.select.int": select_cases,
    "aten.slice.Tensor": slice_cases,
    "aten.index.Tensor": index_tensor_cases,
    "aten.any.default": any_default_cases,
    "aten.any.dim": any_dim_cases,
    "aten.clone.default": clone_cases,
    "aten.detach.default": detach_cases,
    "aten.cos.default": cos_cases,
    "aten.sin.default": sin_cases,
    "aten.reciprocal.default": reciprocal_cases,
    "aten.cumsum.default": cumsum_cases,
    "aten.expand.default": expand_cases,
    "aten.masked_fill.Scalar": masked_fill_cases,
    "aten.max.default": max_default_cases,
    "aten.max.dim": max_dim_cases,
    "aten.mean.default": mean_default_cases,
    "aten.mean.dim": mean_dim_cases,
    "aten.sum.default": sum_default_cases,
    "aten.sum.dim_IntList": sum_dim_cases,
    "aten.new_ones.default": new_ones_cases,
    "aten.transpose.int": transpose_cases,
    "aten.unsqueeze.default": unsqueeze_cases,
    "aten.view.default": view_cases,
    "aten._to_copy.default": to_copy_cases,
    "aten.fill_.Scalar": fill__cases,
    "aten.copy_.default": copy__cases,
    "aten.normal_.default": normal__cases,
    "aten.uniform_.default": uniform__cases,
    # The eight docs/GAP.md §3 measured a greedy 2-layer Llama stopping on.
    "aten.bmm.default": bmm_cases,
    "aten._unsafe_view.default": unsafe_view_cases,
    "aten.alias.default": alias_cases,
    "aten.neg.default": neg_cases,
    "aten.rsub.Scalar": rsub_scalar_cases,
    "aten.silu.default": silu_cases,
    "aten.t.default": t_cases,
    "aten._scaled_dot_product_flash_attention_for_cpu.default": sdpa_flash_cpu_cases,
    # The eight docs/SAMPLING.md measured `do_sample=True` stopping on.
    "aten._softmax.default": softmax_cases,
    "aten.fill_.Tensor": fill__tensor_cases,
    "aten.le.Scalar": le_scalar_cases,
    "aten.multinomial.default": multinomial_cases,
    "aten.scatter.src": scatter_src_cases,
    "aten.sort.default": sort_cases,
    "aten.squeeze.dim": squeeze_dim_cases,
    "aten.topk.default": topk_cases,
}
