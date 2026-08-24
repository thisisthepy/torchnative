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
}
