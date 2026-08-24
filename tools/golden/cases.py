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

# `bfloat16` moved from the gap list to the match list when `mm`/`bmm`/`addmm`
# started accumulating reduced-precision GEMMs in float32, which is what torch
# does (`at::opmath_type`, measured bitwise -- see `gemm_accumulate_in` in
# rust/torch_c/src/aten.rs). candle still has no BF16 matmul kernel; the point
# is that it is never asked for one, because upstream does not ask for one
# either. The integral dtypes stay: that gap is real, and float32 cannot stand
# in for an int64 product.
_MM_MATCH_DTYPES = ["float32", "float64", "float16", "bfloat16"]
_MM_C_ERROR_DTYPES = ["int64", "int32", "int16", "uint8"]


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

    # Model-scale reduction depths -- the docs/GPT2.md §7 gap. See the long
    # note above `_gemm_scale_check`.
    for dtype_name, m, k, n, note in [
        ("float32", 8, 512, 8, "GPT-2 small's per-head depth, narrow output"),
        ("float32", 8, 1024, 8, "the first depth at which the shim and torch stop agreeing bitwise"),
        ("float32", 64, 1024, 64, "4096 outputs at depth 1024"),
        ("float64", 8, 1024, 8, "float64 at the same depth, for contrast"),
        ("float16", 8, 512, 8, "float16 at depth 512 -- the accumulation-dtype question"),
        ("bfloat16", 8, 512, 8, "bfloat16, which only has a matmul at all because of that"),
    ]:
        cases.append(
            _big_gemm_case(torch_module, c_module, torch_call, "aten.mm.default",
                           dtype_name, m, k, n, note=note)
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
    # The indices' *dtype*, not just their values. Upstream promises `int64`
    # here, and that index goes straight into `index_select`/`gather`/
    # `embedding`; a shim returning value-identical `int32` passes every other
    # check in this function and breaks downstream instead. `--self-test`
    # reported this as `_pair_result_check + dtype-last` (docs/HARNESS.md §6).
    t_idx_dtype, c_idx_dtype = dt.dtype_name(t_indices.dtype), dt.dtype_name(c_indices.dtype)
    if t_idx_dtype != c_idx_dtype:
        return False, f"indices dtype mismatch: torch={t_idx_dtype} c={c_idx_dtype}"
    t_idx_flat, c_idx_flat = _flatten_values(t_indices.tolist()), _flatten_values(c_indices.tolist())
    if t_idx_flat != c_idx_flat:
        return False, f"indices mismatch: torch={t_idx_flat!r} c={c_idx_flat!r}"
    return True, (
        f"values dtype={t_dtype} shape={t_shape}, "
        f"indices matched exactly (dtype={t_idx_dtype})"
    )


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

    # Model-scale, batched. Attention's QK^T is a bmm of depth head_dim and
    # its AV is a bmm of depth seq_len, so this is where a large k actually
    # shows up in a transformer -- see the note above `_gemm_scale_check`.
    for dtype_name, note in [
        ("float32", "batched depth 512 -- attention's AV at seq_len 512"),
        ("float16", "the same, in the dtype a device would actually run"),
    ]:
        cases.append(
            _big_gemm_case(torch_module, c_module, torch_call, "aten.bmm.default",
                           dtype_name, 8, 512, 8, batch=2, note=note)
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
    # The indices' shape and dtype, both of which the multiset below is blind
    # to: it flattens, so a reshaped index tensor survives it, and it compares
    # values, so an `int32` index survives it. The *order* is what this
    # comparator deliberately does not check -- the shape and the dtype are
    # promises, not partition artefacts. `--self-test` reported both as
    # `_topk_multiset_check + shape-last` / `+ dtype-last` (docs/HARNESS.md §6).
    t_idx_shape = tuple(int(x) for x in t_indices.shape)
    c_idx_shape = tuple(int(x) for x in c_indices.shape)
    if t_idx_shape != c_idx_shape:
        return False, f"indices shape mismatch: torch={t_idx_shape} c={c_idx_shape}"
    t_idx_dtype, c_idx_dtype = dt.dtype_name(t_indices.dtype), dt.dtype_name(c_indices.dtype)
    if t_idx_dtype != c_idx_dtype:
        return False, f"indices dtype mismatch: torch={t_idx_dtype} c={c_idx_dtype}"
    t_pairs = sorted(zip(_flatten_values(t_values.tolist()), _flatten_values(t_indices.tolist())))
    c_pairs = sorted(zip(_flatten_values(c_values.tolist()), _flatten_values(c_indices.tolist())))
    if t_pairs != c_pairs:
        return False, f"selected elements differ: torch={t_pairs!r} c={c_pairs!r}"
    return True, (
        f"values dtype={t_dtype} shape={t_shape}, same {len(t_pairs)} (value, index) pairs "
        f"(indices dtype={t_idx_dtype} shape={t_idx_shape}) "
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


# --- the four ops docs/GPT2.md measured a 2-layer GPT-2 stopping on ----------
#
# Re-measured against `_aten_implemented()` at 78 ops, not docs/GAP.md's 60-op
# snapshot: a GPT-2 that Llama-shaped work had already unblocked still stops on
# `addmm`, `native_layer_norm`, `split.Tensor` and `tanh`, and on nothing else.


_TANH_DTYPES = ["float64", "float32", "float16", "bfloat16"]
# The promoting inputs. `uint8` is here with *non-negative* values only: a
# negative literal is where `_C._tensor_from_flat` and `torch.tensor` disagree
# (torch wraps -1 to 255, `_tensor_from_flat` saturates to 0), which is a
# constructor difference and would make this op's cases fail for a reason that
# has nothing to do with `tanh`. See docs/GPT2.md.
_TANH_PROMOTING_DTYPES = ["int64", "int32", "int16", "uint8"]


def tanh_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.tanh.default"
    cases: list[Case] = []
    scenarios = [
        ([0.0, 1.0, -1.0, 2.0], (2, 2), "assorted"),
        # tanh saturates: torch answers exactly +-1.0 well before the input
        # overflows, and a shim that computed exp() naively would give NaN.
        ([-100.0, 100.0, 20.0, -20.0], (2, 2), "saturation -- torch gives exactly +-1.0"),
        ([0.0], (), "0-d"),
        ([float("nan"), float("inf"), float("-inf")], (3,), "NaN/+-inf -- nan, 1.0, -1.0"),
    ]
    for dtype_name in _TANH_DTYPES:
        for flat, shape, note in scenarios:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))
    # The promotion rule, which is `cos`/`sin`'s and not `silu`'s: an integral
    # input gives float32 rather than raising.
    for dtype_name in _TANH_PROMOTING_DTYPES:
        cases.append(
            _unary_case(
                torch_module, c_module, op, torch_call, dtype_name, [0, 1, 2, 3], (2, 2),
                "integral input promotes to the default float",
            )
        )
    return cases


# --- aten.addmm.default ------------------------------------------------------
# The dtype split is `mm`'s, for `mm`'s reason: the multiply is candle's
# `matmul`, which has no integral or bfloat16 kernel.


def _addmm_case(
    torch_module, c_module, torch_call, dtype_name,
    self_flat, self_shape, m1_flat, m1_shape, m2_flat, m2_shape,
    kwargs=None, expect="match", note="",
) -> Case:
    kwargs = kwargs or {}
    op = "aten.addmm.default"
    s_t, s_c = pair_from_flat(torch_module, c_module, self_flat, self_shape, dtype_name)
    a_t, a_c = pair_from_flat(torch_module, c_module, m1_flat, m1_shape, dtype_name)
    b_t, b_c = pair_from_flat(torch_module, c_module, m2_flat, m2_shape, dtype_name)
    return Case(
        name=f"addmm(dtype={dtype_name}, self={self_shape}, {m1_shape}x{m2_shape}, {kwargs}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(s_t, a_t, b_t, **kwargs),
        run_c=lambda: c_module._aten_dispatch(op, s_c, a_c, b_c, **kwargs),
        expect=expect,
        note=note,
    )


# (3x2) @ (2x3) -> (3x3); the product is [[1,2,8],[3,4,18],[5,6,28]].
_ADDMM_M1 = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
_ADDMM_M2 = ([1.0, 0.0, 2.0, 0.0, 1.0, 3.0], (2, 3))
_ADDMM_SELF = ([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], (3, 3))


def addmm_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.addmm.default"
    cases: list[Case] = []

    scenarios = [
        (None, "plain: self + mat1 @ mat2"),
        (dict(beta=2, alpha=3), "integer beta/alpha"),
        (dict(beta=0.5, alpha=0.25), "fractional beta/alpha"),
        # The two quick returns. Upstream skips the branch rather than
        # multiplying by zero, which is only visible with a non-finite operand
        # -- covered separately below.
        (dict(beta=0), "beta=0 -- self dropped"),
        (dict(alpha=0), "alpha=0 -- product dropped"),
        (dict(beta=0, alpha=0), "both zero -- a shaped tensor of zeros, not an error"),
        (dict(beta=True), "beta as a bool, which torch reads as 1"),
        (dict(beta=False), "beta as a bool, which torch reads as 0"),
    ]
    for dtype_name in _MM_MATCH_DTYPES:
        for kwargs, note in scenarios:
            cases.append(
                _addmm_case(
                    torch_module, c_module, torch_call, dtype_name,
                    *_ADDMM_SELF, *_ADDMM_M1, *_ADDMM_M2, kwargs=kwargs, note=note,
                )
            )
        # Every shape `self` is allowed to arrive in. The 1-D one is the whole
        # reason this op exists here: `nn.Linear`'s bias is a 1-D row.
        for self_flat, self_shape, note in [
            ([1.0, 2.0, 3.0], (3,), "1-D bias -- nn.Linear's shape"),
            ([7.0], (), "0-d bias"),
            ([1.0, 2.0, 3.0], (3, 1), "column broadcast"),
            ([1.0, 2.0, 3.0], (1, 3), "row broadcast"),
            ([1.0], (1, 1), "(1,1) broadcast"),
        ]:
            cases.append(
                _addmm_case(
                    torch_module, c_module, torch_call, dtype_name,
                    self_flat, self_shape, *_ADDMM_M1, *_ADDMM_M2, note=note,
                )
            )

    # The quick returns, proven rather than asserted: with a non-finite value in
    # the dropped branch, "skip" and "multiply by zero" give different answers.
    inf_m1 = ([float("inf"), 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
    nan_self = ([float("nan")] * 9, (3, 3))
    cases.append(
        _addmm_case(
            torch_module, c_module, torch_call, "float32",
            *nan_self, *_ADDMM_M1, *_ADDMM_M2, kwargs=dict(beta=0),
            note="beta=0 with a NaN self -- 0*nan would be nan, torch gives the clean product",
        )
    )
    cases.append(
        _addmm_case(
            torch_module, c_module, torch_call, "float32",
            *_ADDMM_SELF, *inf_m1, *_ADDMM_M2, kwargs=dict(alpha=0),
            note="alpha=0 with an inf in mat1 -- 0*inf would be nan, torch gives the clean self",
        )
    )

    # Empty extents: torch answers, it does not refuse.
    cases.append(
        _addmm_case(
            torch_module, c_module, torch_call, "float32",
            [], (0, 3), [], (0, 2), *_ADDMM_M2, note="zero rows",
        )
    )
    cases.append(
        _addmm_case(
            torch_module, c_module, torch_call, "float32",
            [], (3, 0), *_ADDMM_M1, [], (2, 0), note="zero columns",
        )
    )

    # Refusals. Each one is a sentence torch says; the point is that the shim
    # does not compute where torch declines.
    for self_flat, self_shape, m1, m2, note in [
        ([1.0, 2.0], (2,), _ADDMM_M1, _ADDMM_M2, "self not expandable to the product's shape"),
        ([1.0] * 18, (2, 3, 3), _ADDMM_M1, _ADDMM_M2, "self has more dims than the product"),
        (*_ADDMM_SELF, ([1.0] * 6, (1, 3, 2)), _ADDMM_M2, "mat1 is 3-D -- 'mat1 must be a matrix'"),
        (*_ADDMM_SELF, _ADDMM_M1, ([1.0, 0.0, 2.0], (3,)), "mat2 is 1-D -- 'mat2 must be a matrix'"),
        (*_ADDMM_SELF, _ADDMM_M1, ([1.0] * 9, (3, 3)), "k mismatch -- shapes cannot be multiplied"),
    ]:
        cases.append(
            _addmm_case(
                torch_module, c_module, torch_call, "float32",
                self_flat, self_shape, *m1, *m2, expect="both_error", note=note,
            )
        )
    # `self` is validated even when nothing reads it -- measured with
    # beta=0, alpha=0, where neither operand contributes.
    cases.append(
        _addmm_case(
            torch_module, c_module, torch_call, "float32",
            [1.0, 2.0], (2,), *_ADDMM_M1, *_ADDMM_M2,
            kwargs=dict(beta=0, alpha=0), expect="both_error",
            note="self shape is checked even when both factors are zero",
        )
    )
    # bool has no addmm_impl_cpu_ upstream, and the shim says the same thing.
    cases.append(
        Case(
            name="addmm(dtype=bool) [no addmm_impl_cpu_ for Bool on either side]",
            op=op,
            run_torch=lambda: torch_call(
                torch_module.ones(2, 2, dtype=torch_module.bool),
                torch_module.ones(2, 2, dtype=torch_module.bool),
                torch_module.ones(2, 2, dtype=torch_module.bool),
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                c_module._tensor_from_flat([1, 1, 1, 1], [2, 2], dtype=c_module.bool),
                c_module._tensor_from_flat([1, 1, 1, 1], [2, 2], dtype=c_module.bool),
                c_module._tensor_from_flat([1, 1, 1, 1], [2, 2], dtype=c_module.bool),
            ),
            expect="both_error",
            note="torch: '\"addmm_impl_cpu_\" not implemented for Bool'",
        )
    )

    # The inherited gap: candle's matmul has no kernel for these, exactly as
    # `aten.mm.default` already records (docs/TORCH_C.md §2).
    for dtype_name in _MM_C_ERROR_DTYPES:
        cases.append(
            _addmm_case(
                torch_module, c_module, torch_call, dtype_name,
                [1, 1, 1, 1], (2, 2), [1, 2, 3, 4], (2, 2), [1, 0, 0, 1], (2, 2),
                expect="c_error",
                note=f"candle's matmul has no kernel for {dtype_name}; torch's CPU addmm does. "
                     "Same gap aten.mm.default already carries.",
            )
        )
        # ...but with alpha=0 there is no matmul to refuse, and both sides
        # agree. Not a workaround -- it is the quick return, and pinning it
        # keeps the gap above honest about *what* is missing (the multiply,
        # not the op).
        cases.append(
            _addmm_case(
                torch_module, c_module, torch_call, dtype_name,
                [1, 1, 1, 1], (2, 2), [1, 2, 3, 4], (2, 2), [1, 0, 0, 1], (2, 2),
                kwargs=dict(alpha=0),
                note=f"{dtype_name} with alpha=0 -- no matmul happens, so the gap above does not apply",
            )
        )

    # Model-scale, with the bias -- this is `nn.Linear` at a real width, which
    # is the shape docs/GPT2.md §3.3 measured and §7 left uncovered here.
    for dtype_name, m, k, n, note in [
        ("float32", 64, 512, 64, "nn.Linear(512, 64) on a batch of 64"),
        ("float32", 8, 1024, 8, "depth 1024, where mm's agreement with torch ends"),
        ("float16", 8, 512, 8, "float16 at depth 512 -- the accumulation-dtype question"),
    ]:
        cases.append(
            _big_gemm_case(torch_module, c_module, torch_call, "aten.addmm.default",
                           dtype_name, m, k, n, with_bias=True, note=note)
        )
    return cases


# --- aten.split.Tensor -------------------------------------------------------
# The one op here that answers with a *list*, so the default dtype/shape/value
# pipeline in compare.py cannot read it. `_chunk_list_check` below is the
# equivalent for a sequence of tensors.


def _chunk_list_check(t_res, c_res) -> tuple[bool, str]:
    """For ops returning a list of tensors (`split`): compare the *number* of
    chunks first -- a shim that padded the last chunk instead of shortening it
    would get every element right and the count wrong -- then each chunk's
    dtype, shape and values."""
    try:
        t_list, c_list = list(t_res), list(c_res)
    except TypeError as e:
        return False, f"expected a sequence of tensors on both sides: {e!r}"
    if len(t_list) != len(c_list):
        return False, (
            f"chunk count differs: torch={len(t_list)} c={len(c_list)} "
            f"(torch shapes {[tuple(x.shape) for x in t_list]}, "
            f"c shapes {[tuple(x.shape) for x in c_list]})"
        )
    for i, (t_chunk, c_chunk) in enumerate(zip(t_list, c_list)):
        t_dtype, c_dtype = dt.dtype_name(t_chunk.dtype), dt.dtype_name(c_chunk.dtype)
        if t_dtype != c_dtype:
            return False, f"chunk[{i}] dtype mismatch: torch={t_dtype} c={c_dtype}"
        t_shape = tuple(int(x) for x in t_chunk.shape)
        c_shape = tuple(int(x) for x in c_chunk.shape)
        if t_shape != c_shape:
            return False, f"chunk[{i}] shape mismatch: torch={t_shape} c={c_shape}"
        tol = dt.tolerance_for(t_dtype)
        t_flat = _flatten_values(t_chunk.tolist())
        c_flat = _flatten_values(c_chunk.tolist())
        for j, (x, y) in enumerate(zip(t_flat, c_flat)):
            if isinstance(x, bool) or isinstance(y, bool):
                if x != y:
                    return False, f"chunk[{i}][{j}] mismatch: torch={x!r} c={y!r}"
                continue
            if not math.isclose(float(x), float(y), rel_tol=tol.rtol, abs_tol=tol.atol):
                return False, f"chunk[{i}][{j}] mismatch: torch={x!r} c={y!r}"
    return True, f"{len(t_list)} chunks matched, shapes {[tuple(x.shape) for x in t_list]}"


_SPLIT_DTYPES = ["float32", "float64", "float16", "int64", "int32"]


def _split_case(torch_module, c_module, torch_call, dtype_name, flat, shape, args, expect="match", note="") -> Case:
    op = "aten.split.Tensor"
    a_t, a_c = pair_from_flat(torch_module, c_module, flat, shape, dtype_name)
    return Case(
        name=f"split(dtype={dtype_name}, shape={shape}, args={args}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(a_t, *args),
        run_c=lambda: c_module._aten_dispatch(op, a_c, *args),
        expect=expect,
        value_check=_chunk_list_check if expect == "match" else None,
        note=note + " -- returns a list of tensors, see _chunk_list_check",
    )


def split_cases(torch_module, c_module, torch_call) -> list[Case]:
    cases: list[Case] = []
    ten = list(range(10))
    for dtype_name in _SPLIT_DTYPES:
        for args, note in [
            ((3,), "10 into 3 -- the last chunk is short (3,3,3,1), not padded"),
            ((5,), "10 into 5 -- exact division"),
            ((20,), "split_size larger than the dimension gives one whole chunk"),
            ((1,), "split_size 1 -- ten chunks"),
            ((10,), "split_size equal to the dimension"),
        ]:
            cases.append(_split_case(torch_module, c_module, torch_call, dtype_name, ten, (10,), args, note=note))
    # The GPT-2 shape: one projection of width 3*d split three ways on the last
    # dim. This is the call the whole op is here for.
    qkv = list(range(24))
    cases.append(
        _split_case(
            torch_module, c_module, torch_call, "float32", qkv, (2, 2, 6), (2, 2),
            note="GPT-2's c_attn(x).split(d, dim=2) -- q, k, v",
        )
    )
    cases.append(
        _split_case(
            torch_module, c_module, torch_call, "float32", qkv, (2, 2, 6), (2, -1),
            note="same split, addressed with a negative dim",
        )
    )
    cases.append(
        _split_case(
            torch_module, c_module, torch_call, "float32", list(range(12)), (3, 4), (3, 1),
            note="uneven split on dim 1 -- chunks are (3,3) and (3,1)",
        )
    )
    cases.append(
        _split_case(
            torch_module, c_module, torch_call, "float32", list(range(12)), (3, 4), (2, 0),
            note="split on dim 0",
        )
    )
    # An empty dimension: one empty chunk for any split size, including 0.
    for args, note in [((3,), "empty dim, split_size 3"), ((0,), "empty dim, split_size 0 -- the one place 0 is legal")]:
        cases.append(
            _split_case(torch_module, c_module, torch_call, "float32", [], (0,), args, note=note)
        )
    # Refusals.
    for flat, shape, args, note in [
        (ten, (10,), (0,), "split_size 0 on a non-empty dim"),
        (ten, (10,), (-1,), "negative split_size"),
        (ten, (10,), (1, 2), "dim out of range"),
        ([1.0], (), (1,), "0-d tensor -- 'split expects at least a 1-dimensional tensor'"),
    ]:
        cases.append(
            _split_case(
                torch_module, c_module, torch_call, "float32", flat, shape, args,
                expect="both_error", note=note,
            )
        )
    return cases


# --- aten.native_layer_norm.default -----------------------------------------
# Three results, not one: (output, mean, rstd). `_triple_result_check` is
# `_pair_result_check`'s shape for a schema with one more tensor in it -- and
# unlike that one, every member is a float tensor compared within tolerance,
# because none of them is an index.


def _triple_result_check(t_res, c_res) -> tuple[bool, str]:
    try:
        t_parts = (t_res[0], t_res[1], t_res[2])
        c_parts = (c_res[0], c_res[1], c_res[2])
    except (TypeError, IndexError, KeyError) as e:
        return False, f"expected a 3-element (out, mean, rstd) result on both sides: {e!r}"

    for label, t_part, c_part in zip(("out", "mean", "rstd"), t_parts, c_parts):
        t_dtype, c_dtype = dt.dtype_name(t_part.dtype), dt.dtype_name(c_part.dtype)
        if t_dtype != c_dtype:
            return False, f"{label} dtype mismatch: torch={t_dtype} c={c_dtype}"
        t_shape = tuple(int(x) for x in t_part.shape)
        c_shape = tuple(int(x) for x in c_part.shape)
        if t_shape != c_shape:
            return False, f"{label} shape mismatch: torch={t_shape} c={c_shape}"
        tol = dt.tolerance_for(t_dtype)
        t_flat = _flatten_values(t_part.tolist())
        c_flat = _flatten_values(c_part.tolist())
        if len(t_flat) != len(c_flat):
            return False, f"{label} length differs: torch={len(t_flat)} c={len(c_flat)}"
        for i, (x, y) in enumerate(zip(t_flat, c_flat)):
            xf, yf = float(x), float(y)
            # A negative eps is not refused upstream -- it gives NaN, and the
            # two sides agreeing on NaN is the result being checked.
            if math.isnan(xf) or math.isnan(yf):
                if math.isnan(xf) and math.isnan(yf):
                    continue
                return False, f"{label}[{i}] mismatch: torch={x!r} c={y!r} (NaN on one side only)"
            if not math.isclose(xf, yf, rel_tol=tol.rtol, abs_tol=tol.atol):
                return False, f"{label}[{i}] mismatch: torch={x!r} c={y!r}"
    return True, (
        "out/mean/rstd matched; shapes "
        f"{[tuple(int(v) for v in p.shape) for p in t_parts]}"
    )


_LAYER_NORM_DTYPES = ["float64", "float32", "float16", "bfloat16"]
# 24 values that are not a nice round arithmetic progression in the normalised
# dimension, so a wrong reduction axis shows up as a wrong number rather than
# as the same number by symmetry.
_LN_INPUT = [round(i / 7.0, 6) for i in range(24)]
_LN_WEIGHT = [1.0, 2.0, 0.5, -1.0]
_LN_BIAS = [0.1, 0.2, 0.3, 0.4]


def _layer_norm_case(
    torch_module, c_module, torch_call, dtype_name, flat, shape, normalized_shape,
    weight=None, bias=None, eps=1e-5, param_dtype=None, expect="match", note="",
) -> Case:
    op = "aten.native_layer_norm.default"
    param_dtype = param_dtype or dtype_name
    x_t, x_c = pair_from_flat(torch_module, c_module, flat, shape, dtype_name)
    if weight is None:
        w_t = w_c = None
    else:
        w_t, w_c = pair_from_flat(torch_module, c_module, weight, (len(weight),), param_dtype)
    if bias is None:
        b_t = b_c = None
    else:
        b_t, b_c = pair_from_flat(torch_module, c_module, bias, (len(bias),), param_dtype)
    return Case(
        name=(
            f"native_layer_norm(dtype={dtype_name}, shape={shape}, "
            f"normalized_shape={normalized_shape}, weight={weight is not None}, "
            f"bias={bias is not None}, eps={eps}, param_dtype={param_dtype}) [{note}]"
        ),
        op=op,
        run_torch=lambda: torch_call(x_t, normalized_shape, w_t, b_t, eps),
        run_c=lambda: c_module._aten_dispatch(op, x_c, normalized_shape, w_c, b_c, eps),
        expect=expect,
        value_check=_triple_result_check if expect == "match" else None,
        note=note + " -- returns (out, mean, rstd), see _triple_result_check",
    )


def native_layer_norm_cases(torch_module, c_module, torch_call) -> list[Case]:
    cases: list[Case] = []
    for dtype_name in _LAYER_NORM_DTYPES:
        for weight, bias, note in [
            (_LN_WEIGHT, _LN_BIAS, "weight and bias -- nn.LayerNorm's default"),
            (None, None, "elementwise_affine=False"),
            (_LN_WEIGHT, None, "weight only"),
            (None, _LN_BIAS, "bias only"),
        ]:
            cases.append(
                _layer_norm_case(
                    torch_module, c_module, torch_call, dtype_name,
                    _LN_INPUT, (2, 3, 4), [4], weight=weight, bias=bias, note=note,
                )
            )
        # mean/rstd keep the input's rank with the normalised dims set to 1;
        # normalising over more than one trailing dim is where a flat
        # (M,)-shaped statistic would be caught.
        cases.append(
            _layer_norm_case(
                torch_module, c_module, torch_call, dtype_name,
                _LN_INPUT, (2, 3, 4), [3, 4],
                note="two normalised dims -- mean/rstd are (2,1,1)",
            )
        )
        cases.append(
            _layer_norm_case(
                torch_module, c_module, torch_call, dtype_name,
                _LN_INPUT, (2, 3, 4), [2, 3, 4],
                note="everything normalised -- mean/rstd are (1,1,1)",
            )
        )
        cases.append(
            _layer_norm_case(
                torch_module, c_module, torch_call, dtype_name,
                _LN_INPUT[:8], (2, 4), [4], weight=_LN_WEIGHT, bias=_LN_BIAS,
                note="2-D input -- the shape a flattened GPT-2 block reaches",
            )
        )
        cases.append(
            _layer_norm_case(
                torch_module, c_module, torch_call, dtype_name,
                _LN_INPUT[:4], (4,), [4], note="1-D input -- mean/rstd are (1,)",
            )
        )
    # eps lands on the variance, before the reciprocal square root. These three
    # cases are what pin that: a constant row has zero variance, so its rstd is
    # exactly 1/sqrt(eps) and nothing else.
    for eps, note in [
        (0.0, "eps=0"),
        (1.0, "eps=1 -- large enough that it dominates the variance"),
        (-1.0, "negative eps is not refused upstream; it gives NaN"),
    ]:
        cases.append(
            _layer_norm_case(
                torch_module, c_module, torch_call, "float32",
                _LN_INPUT, (2, 3, 4), [4], eps=eps, note=note,
            )
        )
    cases.append(
        _layer_norm_case(
            torch_module, c_module, torch_call, "float32",
            [1.0] * 8, (2, 4), [4],
            note="constant row -- zero variance, so rstd is exactly 1/sqrt(eps)",
        )
    )
    cases.append(
        _layer_norm_case(
            torch_module, c_module, torch_call, "float32",
            [], (0, 4), [4], note="zero rows -- every result is empty, not an error",
        )
    )
    # Upstream's autocast pairing: a reduced-precision input with float32
    # parameters is *supported*, and it moves mean/rstd to float32 while the
    # output stays reduced. Getting this wrong is invisible in the values and
    # visible only in the dtype, which is why it has its own cases.
    for dtype_name in ["float16", "bfloat16"]:
        cases.append(
            _layer_norm_case(
                torch_module, c_module, torch_call, dtype_name,
                _LN_INPUT, (2, 3, 4), [4], weight=_LN_WEIGHT, bias=_LN_BIAS,
                param_dtype="float32",
                note="mixed dtype: float32 parameters move mean/rstd to float32, output stays reduced",
            )
        )
        cases.append(
            _layer_norm_case(
                torch_module, c_module, torch_call, dtype_name,
                _LN_INPUT, (2, 3, 4), [4], weight=_LN_WEIGHT, param_dtype="float32",
                note="mixed dtype, weight only",
            )
        )
    # Refusals.
    refusals = [
        dict(dtype_name="float32", normalized_shape=[5], note="normalized_shape is not the input's suffix"),
        dict(dtype_name="float32", normalized_shape=[], note="empty normalized_shape"),
        dict(dtype_name="float32", normalized_shape=[2, 3, 4, 5], note="normalized_shape longer than the input"),
        dict(dtype_name="int64", normalized_shape=[4], note="integral input -- no LayerNormKernelImpl"),
    ]
    for spec in refusals:
        flat = _LN_INPUT if spec["dtype_name"] != "int64" else [i % 2 for i in range(24)]
        cases.append(
            _layer_norm_case(
                torch_module, c_module, torch_call, spec["dtype_name"],
                flat, (2, 3, 4), spec["normalized_shape"],
                expect="both_error", note=spec["note"],
            )
        )
    cases.append(
        _layer_norm_case(
            torch_module, c_module, torch_call, "float32",
            _LN_INPUT, (2, 3, 4), [4], weight=[1.0, 2.0, 3.0],
            expect="both_error", note="weight is not normalized_shape's shape",
        )
    )
    cases.append(
        _layer_norm_case(
            torch_module, c_module, torch_call, "float32",
            _LN_INPUT, (2, 3, 4), [4], bias=[1.0, 2.0, 3.0],
            expect="both_error", note="bias is not normalized_shape's shape",
        )
    )
    cases.append(
        _layer_norm_case(
            torch_module, c_module, torch_call, "float32",
            _LN_INPUT, (2, 3, 4), [4], weight=_LN_WEIGHT, param_dtype="float16",
            expect="both_error",
            note="float32 input with float16 parameters -- 'mixed dtype (CPU)' on both sides",
        )
    )
    cases.append(
        _layer_norm_case(
            torch_module, c_module, torch_call, "float64",
            _LN_INPUT, (2, 3, 4), [4], weight=_LN_WEIGHT, param_dtype="float32",
            expect="both_error",
            note="float64 input with float32 parameters -- torch refuses this pairing too",
        )
    )
    # A zero-extent normalised dimension: torch answers with mean=0 and
    # rstd=nan, which do not describe the same reduction. The shim refuses by
    # name rather than reproducing an inconsistency from a single observation.
    cases.append(
        Case(
            name="native_layer_norm(normalized_shape=[0]) [torch answers mean=0 with rstd=nan; the shim refuses]",
            op="aten.native_layer_norm.default",
            run_torch=lambda: torch_call(
                torch_module.zeros(2, 0), [0], None, None, 1e-5
            ),
            run_c=lambda: c_module._aten_dispatch(
                "aten.native_layer_norm.default",
                c_module._tensor_from_flat([], [2, 0], dtype=c_module.float32),
                [0], None, None, 1e-5,
            ),
            expect="c_error",
            note="documented gap: upstream's mean (0) and rstd (nan) disagree about "
                 "what a reduction over no elements is, and one observation is not "
                 "enough to reproduce that. See rust/torch_c/src/aten.rs.",
        )
    )
    return cases


# --- aten.gelu.default -------------------------------------------------------
#
# The op with two functions behind one name. `approximate="none"` is the exact
# erf form, `approximate="tanh"` is Hendrycks' cubic; on `[-3, 3]` in float32
# they differ by up to **4.12e-04**, against this harness's float32 tolerance of
# 1e-5. So the three variants below (default / explicit "none" / explicit
# "tanh") are not redundant coverage -- the default-vs-tanh pair is a live trap
# for a shim that picked the wrong formula for the unqualified call, and it
# would fire with 40x the tolerance, not at the edge of it.
#
# Measured (docs/ARCH.md §2): Gemma and Gemma-2 pass `"tanh"`; BERT, RoBERTa,
# ELECTRA, DistilBERT, DeBERTa-v2, BART, Falcon, GPT-NeoX, GPT-BigCode,
# Starcoder2, MPT and ViT all take the default.

_GELU_DTYPES = ["float64", "float32", "float16", "bfloat16"]
_GELU_APPROX: list[tuple[dict, str]] = [
    ({}, "unqualified -- upstream's schema default is 'none', the erf form"),
    ({"approximate": "none"}, "explicit 'none' -- must equal the unqualified call"),
    ({"approximate": "tanh"}, "explicit 'tanh' -- Gemma's gelu_pytorch_tanh"),
]
_GELU_SCENARIOS: list[tuple[list, tuple, str]] = [
    ([-3.0, -1.0, -0.5, 0.0], (2, 2), "the negative lobe, where the two formulas diverge most"),
    ([0.5, 1.0, 2.0, 3.0], (2, 2), "the positive side"),
    ([-8.0, 8.0, -0.001, 0.001], (2, 2), "saturating tails and near-zero"),
    ([0.5], (), "0-d"),
    ([], (0,), "empty"),
]


def gelu_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.gelu.default"
    cases: list[Case] = []
    for dtype_name in _GELU_DTYPES:
        for kwargs, kw_note in _GELU_APPROX:
            for flat, shape, note in _GELU_SCENARIOS:
                cases.append(
                    _unary_case(
                        torch_module, c_module, op, torch_call, dtype_name, flat, shape,
                        f"{note}; {kw_note}", kwargs=kwargs,
                    )
                )
        # gelu(-inf) is `nan`, not 0: the exact form multiplies -inf by
        # (1 + erf(-inf)) == 0, and the tanh form multiplies it by
        # (1 + tanh(-inf)) == 0. Both give the indeterminate product, on both
        # sides. Kept because a kernel written as "clamp then scale" would give
        # -0.0 here and pass every other case in this file.
        for kwargs, kw_note in _GELU_APPROX:
            cases.append(
                _unary_case(
                    torch_module, c_module, op, torch_call, dtype_name,
                    [float("nan"), float("inf"), float("-inf")], (3,),
                    f"nan/+-inf -- +inf passes through, -inf gives NaN; {kw_note}",
                    kwargs=kwargs,
                )
            )

    # The refusals. This is `silu`'s side of the promotion split, not
    # `tanh`'s: there is no GeluKernelImpl for an integral or boolean input,
    # so a shim built on the promoting unary helper would compute where torch
    # raises. bool is deferred into the lambdas for the usual reason (see
    # masked_fill_cases).
    for dtype_name in ["int64", "int32", "int16", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [0, 1, 2, 3], (2, 2), dtype_name)
        cases.append(
            Case(
                name=f"gelu(dtype={dtype_name}) [no GeluKernelImpl for an integral input]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                expect="both_error",
                note="torch: NotImplementedError(\"GeluKernelImpl\" not implemented for 'Long').",
            )
        )
    cases.append(
        Case(
            name="gelu(dtype=bool) [no GeluKernelImpl for a mask either]",
            op=op,
            run_torch=lambda: torch_call(torch_module.tensor([True, False])),
            run_c=lambda: c_module._aten_dispatch(
                op, c_module._tensor_from_flat([1, 0], [2], dtype=c_module.bool)
            ),
            expect="both_error",
            note="torch: NotImplementedError(\"GeluKernelImpl\" not implemented for 'Bool').",
        )
    )

    # The two ways to get `approximate` wrong. Both are RuntimeErrors upstream
    # and both must stay refusals: a shim that fell back to "none" on an
    # unrecognised string would answer a Gemma typo with a BERT activation.
    x_t, x_c = pair_from_flat(torch_module, c_module, [1.0, -1.0], (2,), "float32")
    for bad in ["TANH", "erf", ""]:
        cases.append(
            Case(
                name=f"gelu(approximate={bad!r}) [only 'none' and 'tanh' exist]",
                op=op,
                run_torch=lambda x_t=x_t, bad=bad: torch_call(x_t, approximate=bad),
                run_c=lambda x_c=x_c, bad=bad: c_module._aten_dispatch(op, x_c, approximate=bad),
                expect="both_error",
                note="torch: 'approximate argument must be either none or tanh.'",
            )
        )
    cases.append(
        Case(
            name="gelu(x, 'tanh') positionally [approximate is keyword-only]",
            op=op,
            run_torch=lambda: torch_call(x_t, "tanh"),
            run_c=lambda: c_module._aten_dispatch(op, x_c, "tanh"),
            expect="both_error",
            note=(
                "The schema is `gelu(Tensor self, *, str approximate=\"none\")`. Accepting "
                "the positional form would let `gelu(x, 'tanh')` mean the tanh formula in "
                "the shim and be an error upstream -- divergence in the loudest direction."
            ),
        )
    )
    cases.append(
        Case(
            name="gelu(approximate=None) [the schema says str, not str?]",
            op=op,
            run_torch=lambda: torch_call(x_t, approximate=None),
            run_c=lambda: c_module._aten_dispatch(op, x_c, approximate=None),
            expect="both_error",
            note="torch: Expected a value of type 'str' ... but instead found type 'NoneType'.",
        )
    )
    return cases


# --- aten.gather.default -----------------------------------------------------
#
# `scatter.src` read backwards. The cases below are the three places candle's
# own `Tensor::gather` and torch disagree (index dtype, out-of-range handling,
# contiguity), plus the rank rule that is easy to get wrong in the safe-looking
# direction -- see the kernel's docstring in rust/torch_c/src/aten.rs.

_GATHER_SELF = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))


def _gather_case(torch_module, c_module, torch_call, dtype_name, self_flat, self_shape,
                 dim, idx_flat, idx_shape, idx_dtype="int64", kwargs=None,
                 expect="match", note="") -> Case:
    op = "aten.gather.default"
    kwargs = kwargs or {}
    s_t, s_c = pair_from_flat(torch_module, c_module, self_flat, self_shape, dtype_name)
    i_t, i_c = pair_from_flat(torch_module, c_module, idx_flat, idx_shape, idx_dtype)
    return Case(
        name=f"gather(dtype={dtype_name}, self={self_shape}, dim={dim}, "
             f"index={idx_shape}/{idx_dtype}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(s_t, dim, i_t, **kwargs),
        run_c=lambda: c_module._aten_dispatch(op, s_c, dim, i_c, **kwargs),
        expect=expect,
        note=note,
    )


def gather_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.gather.default"
    flat, shape = _GATHER_SELF
    cases: list[Case] = []

    # Every storable dtype, including bool -- gather moves bits, it does not
    # compute, so there is no kernel-coverage split here the way there is for
    # mm. bool goes through the deferred-construction form.
    for dtype_name in ["float64", "float32", "float16", "bfloat16",
                       "int64", "int32", "int16", "uint8"]:
        cases.append(
            _gather_case(torch_module, c_module, torch_call, dtype_name, flat, shape,
                         1, [0, 2, 1, 0], (2, 2), note="along the last dim")
        )
    cases.append(
        Case(
            name="gather(dtype=bool, dim=1) [a mask is gathered like anything else]",
            op=op,
            run_torch=lambda: torch_call(
                torch_module.tensor([[True, False, True], [False, True, False]]),
                1,
                torch_module.tensor([[0, 2], [1, 0]]),
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                c_module._tensor_from_flat([1, 0, 1, 0, 1, 0], [2, 3], dtype=c_module.bool),
                1,
                c_module._tensor_from_flat([0, 2, 1, 0], [2, 2], dtype=c_module.int64),
            ),
            note="the bool tag has to survive the round trip through the flat reader",
        )
    )

    # Shape and axis coverage.
    for dim, idx_flat, idx_shape, note in [
        (0, [0, 1, 0, 1, 0, 1], (2, 3), "along dim 0"),
        (-1, [0, 1, 1, 0], (2, 2), "negative dim"),
        (1, [0], (1, 1), "index SMALLER than self off-axis -- the extra row is never read"),
        (1, [0, 1, 2, 0, 1], (1, 5), "index LONGER than self along the axis -- values repeat"),
        (1, [], (2, 0), "empty index gives an empty result, not an error"),
        (1, [2, 2, 2, 2, 2, 2], (2, 3), "every index the same -- a broadcast written as a gather"),
    ]:
        cases.append(
            _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                         dim, idx_flat, idx_shape, note=note)
        )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                     1, [0, 2, 1, 0], (2, 2), idx_dtype="int32",
                     note="int32 index -- torch accepts exactly int32 and int64")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                     1, [0, 1], (1, 2), kwargs={"sparse_grad": True},
                     note="sparse_grad picks an autograd representation; the forward answer "
                          "is identical either way (measured)")
    )
    # 3-d, which is the shape BERT's path actually produces.
    cases.append(
        _gather_case(
            torch_module, c_module, torch_call, "float32",
            [float(i) for i in range(24)], (2, 3, 4), 2,
            [0, 3, 1, 2, 3, 0, 2, 2, 0, 1, 3, 3], (2, 3, 2),
            note="3-d gather along the last axis",
        )
    )

    # The rank rule, which is `max(rank, 1)` on BOTH sides and not "the ranks
    # must be equal". Guessing the stricter rule refuses two calls torch
    # answers; guessing a looser one accepts a call torch refuses.
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", [7.0], (), 0,
                     [0, 0], (2,), note="0-d self with a 1-d index -> shape (2,)")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", [7.0], (), 0,
                     [0], (), note="0-d self with a 0-d index -> 0-d")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", [1.0, 2.0], (2,), 0,
                     [1], (), note="1-d self with a 0-d index -> 0-d")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", [7.0], (), 0,
                     [0], (1, 1), expect="both_error",
                     note="0-d self with a 2-d index IS refused -- max(rank,1) is 1, not 2")
    )

    # The refusals, one per disagreement with candle's own gather.
    for idx_dtype in ["uint8", "int16", "float32"]:
        cases.append(
            _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                         1, [0, 1], (1, 2), idx_dtype=idx_dtype, expect="both_error",
                         note="torch: 'gather(): Expected dtype int32/int64 for index'. candle's "
                              "own gather accepts u8/u32 -- a uint8 mask would be read as positions")
        )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                     1, [0, 3], (1, 2), expect="both_error",
                     note="index past the axis extent")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                     1, [0, -1], (1, 2), expect="both_error",
                     note="gather does NOT take negative indices, unlike select/slice. candle's "
                          "i64 path would reach as_usize() and name a huge number instead of -1")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                     1, [0, 1], (2,), expect="both_error",
                     note="index rank must match self rank")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                     1, [0, 0, 0], (3, 1), expect="both_error",
                     note="index LARGER than self off-axis is refused (the mirror of the "
                          "'smaller' case above, which is allowed)")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", flat, shape,
                     2, [0, 0, 0, 0], (2, 2), expect="both_error",
                     note="dim out of range")
    )
    cases.append(
        _gather_case(torch_module, c_module, torch_call, "float32", [], (2, 0), 1,
                     [0, 0], (2, 1), expect="both_error",
                     note="a zero-extent axis has no valid index at all")
    )

    # Non-contiguous self. candle's gather refuses this outright
    # (RequiresContiguous); torch answers. Built through `transpose.int` so the
    # non-contiguity is real on both sides rather than asserted.
    cases.append(
        Case(
            name="gather(non-contiguous self via transpose) [candle's own gather refuses this]",
            op=op,
            run_torch=lambda: torch_call(
                torch_module.ops.aten.transpose.int(
                    torch_module.tensor([float(i) for i in range(12)]).reshape(3, 4), 0, 1
                ),
                1,
                torch_module.tensor([[0, 2], [1, 0], [2, 1], [0, 0]]),
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                c_module._aten_dispatch(
                    "aten.transpose.int",
                    c_module._tensor_from_flat([float(i) for i in range(12)], [3, 4],
                                               dtype=c_module.float32),
                    0, 1,
                ),
                1,
                c_module._tensor_from_flat([0, 2, 1, 0, 2, 1, 0, 0], [4, 2], dtype=c_module.int64),
            ),
            note="torch gathers from a transposed tensor without comment; candle's kernel "
                 "raises RequiresContiguous, which is why this shim reads the elements itself.",
        )
    )
    return cases


# --- aten.zero_.default ------------------------------------------------------
#
# Not a spelling of `fill_(0)` -- a separate overload upstream, so a separate
# op here and a separate set of cases. Unlike `fill_` there is no overflow
# table to reuse: zero is exact in every dtype, so what is worth checking is
# the dtype tag surviving, the in-place identity, and the degenerate shapes.
#
# Where it fires was measured, and it is not the forward pass:
# `nn.LayerNorm(8)`'s *constructor* calls it once (the bias), the forward calls
# it never. See the kernel docstring.


def zero__cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.zero_.default"
    cases: list[Case] = []
    for dtype_name in dt.DEFAULT_DTYPES:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), dtype_name)
        cases.append(
            Case(
                name=f"zero_(dtype={dtype_name})",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                note="in-place: compares the mutated operand zero_ returns",
            )
        )
    cases.append(
        Case(
            name="zero_(dtype=bool) [False, not 0]",
            op=op,
            run_torch=lambda: torch_call(torch_module.tensor([[True, True], [False, True]])),
            run_c=lambda: c_module._aten_dispatch(
                op, c_module._tensor_from_flat([1, 1, 0, 1], [2, 2], dtype=c_module.bool)
            ),
            note="the bool tag must survive -- a shim that zeroed through uint8 would "
                 "return a uint8 tensor and the dtype check would catch it",
        )
    )
    for label, flat, shape in [
        ("0-d", [3.0], ()),
        ("empty", [], (0,)),
        ("nan/inf overwritten", [float("nan"), float("inf"), float("-inf")], (3,)),
    ]:
        a_t, a_c = pair_from_flat(torch_module, c_module, flat, shape, "float32")
        cases.append(
            Case(
                name=f"zero_({label})",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                note=label,
            )
        )
    return cases


# --- large-size GEMM cases ---------------------------------------------------
#
# docs/GPT2.md §7 left this open: the golden harness only ever multiplied
# matrices small enough that accumulation order could not matter, so "does the
# shim's GEMM agree with torch's at model scale" had never been asked here.
# These cases ask it. What they found is written up in docs/ARCH.md §4; the
# short version is that the answer is dtype-dependent and the *flat* tolerance
# in dtypes.py is the wrong instrument for a reduction of depth k.
#
# Why a flat atol cannot describe a GEMM. dtypes.py sizes its tolerances at
# "roughly one ULP at magnitude ~1" -- by its own docstring. A dot product of
# length k does not produce a magnitude-1 answer from magnitude-1 inputs; it
# produces one of magnitude ~sqrt(k), and the standard forward error bound for
# a floating-point dot product is proportional to k*u*sum|a_i*b_i|, i.e. it
# grows with BOTH the depth and the output's own scale. Comparing that against
# a constant is a category error: it will pass a large tensor whose every
# element is wrong by a relative 1e-3 (if the elements are small) and fail a
# correct one (if they are large).
#
# So these cases carry their own checker, which asserts
#
#     max|torch - c|  <=  C * u(accumulate) * sqrt(k) * max|torch|
#                          + 1 ulp of the STORAGE dtype at that magnitude
#
# with C = 4 for headroom. Three things about that formula are load-bearing:
#
#   * `sqrt(k)`, not `k`. The textbook forward bound for a length-k dot product
#     is linear in k because it assumes every rounding error has the same sign.
#     They do not, and a linear bound is loose enough to pass an actively wrong
#     kernel. sqrt(k) is the statistical growth and still holds with margin at
#     every size measured (docs/ARCH.md §4 tabulates them).
#
#   * `u(accumulate)`, not `u(storage)`. **torch's CPU GEMM accumulates in
#     float32 no matter what the tensors are stored as** -- `at::opmath_type`.
#     That is measured, not assumed: `mm(half a, half b)` is *bitwise* equal to
#     `half(mm(float(a), float(b)))` at k = 4, 64 and 512, and the same for bmm
#     and addmm. Sizing a float16 GEMM's tolerance by float16's own unit
#     roundoff (4.9e-4) would let a kernel that accumulates in float16 pass
#     with 20x room to spare -- which is precisely the bug this checker was
#     rewritten to catch, after it did exactly that on the first attempt.
#
#   * the trailing ulp term. torch rounds once, at the end. A shim that also
#     accumulates in float32 lands within that single rounding, so the term is
#     what makes "match torch's method" the passing condition rather than
#     "be lucky".
#
# It is NOT a widened tolerance. At every size measured it is between 5x and
# 20x tighter than the answer, and it *fails* on an accumulation-dtype
# mismatch by a factor of 4.4 -- see docs/ARCH.md §4.
#
# The checker also re-derives what the default flat-tolerance pipeline would
# have said, so a verbose run prints both verdicts side by side and nobody has
# to take this note's word for it.

_GEMM_UNIT_ROUNDOFF = {
    "float64": 2.0 ** -53,
    "float32": 2.0 ** -24,
    "float16": 2.0 ** -11,
    "bfloat16": 2.0 ** -8,
}
# What torch accumulates a GEMM of this storage dtype in (`at::opmath_type`).
_GEMM_ACCUMULATE_IN = {
    "float64": "float64",
    "float32": "float32",
    "float16": "float32",
    "bfloat16": "float32",
}
_GEMM_ERROR_CONSTANT = 4.0


def _gemm_lcg(n: int, seed: int) -> list[float]:
    """White noise in [-1, 1), identical on both sides because both sides are
    handed this same list. Deliberately not `random` -- a golden case that
    changes its inputs between runs cannot be bisected."""
    out: list[float] = []
    x = seed
    for _ in range(n):
        x = (1103515245 * x + 12345) % (1 << 31)
        out.append((x / (1 << 30)) - 1.0)
    return out


def _gemm_scale_check(dtype_name: str, k: int):
    """A value_check for a GEMM of reduction depth `k`. Replaces the whole
    default pipeline, so it has to check dtype and shape itself."""
    tol = dt.tolerance_for(dtype_name)
    u_acc = _GEMM_UNIT_ROUNDOFF[_GEMM_ACCUMULATE_IN[dtype_name]]
    u_out = _GEMM_UNIT_ROUNDOFF[dtype_name]
    # C*u_acc*sqrt(k) for the accumulation, + 2*u_out for the single final
    # rounding into the storage dtype (2u is one ulp; u is half of one).
    bound_factor = _GEMM_ERROR_CONSTANT * u_acc * math.sqrt(k) + 2.0 * u_out

    def check(t_res, c_res) -> tuple[bool, str]:
        t_name, c_name = dt.dtype_name(t_res.dtype), dt.dtype_name(c_res.dtype)
        if t_name != c_name:
            return False, f"dtype mismatch: torch={t_name} c={c_name}"
        t_shape = tuple(int(x) for x in t_res.shape)
        c_shape = tuple(int(x) for x in c_res.shape)
        if t_shape != c_shape:
            return False, f"shape mismatch: torch={t_shape} c={c_shape}"

        def flatten(v):
            if isinstance(v, list):
                out = []
                for item in v:
                    out.extend(flatten(item))
                return out
            return [v]

        tv, cv = flatten(t_res.tolist()), flatten(c_res.tolist())
        scale = max((abs(v) for v in tv), default=0.0)
        max_abs = 0.0
        max_rel = 0.0
        flat_tol_failures = 0
        for x, y in zip(tv, cv):
            d = abs(x - y)
            if d > max_abs:
                max_abs = d
            if x != 0.0 and d / abs(x) > max_rel:
                max_rel = d / abs(x)
            if not math.isclose(x, y, rel_tol=tol.rtol, abs_tol=tol.atol):
                flat_tol_failures += 1
        bound = bound_factor * scale
        flat_verdict = (
            "would also pass" if flat_tol_failures == 0
            else f"would FAIL on {flat_tol_failures}/{len(tv)} elements"
        )
        detail = (
            f"k={k} n={len(tv)} max|d|={max_abs:.4g} max_rel={max_rel:.4g} "
            f"|out|max={scale:.4g} bound={bound:.4g} "
            f"(={_GEMM_ERROR_CONSTANT:g}*u[{_GEMM_ACCUMULATE_IN[dtype_name]}]"
            f"*sqrt(k)+2*u[{dtype_name}], scaled by |out|max); "
            f"flat atol={tol.atol:g}/rtol={tol.rtol:g} {flat_verdict}"
        )
        if max_abs > bound:
            return False, (
                "GEMM error exceeds the scale-aware bound -- this is an accumulation "
                f"difference, not rounding: {detail}"
            )
        return True, detail

    return check


def _big_gemm_case(torch_module, c_module, torch_call, op, dtype_name, m, k, n,
                   with_bias=False, batch=None, note="") -> Case:
    """One large mm / bmm / addmm case, checked against the scale-aware bound."""
    b = batch or 1
    a_flat = _gemm_lcg(b * m * k, 1)
    w_flat = _gemm_lcg(b * k * n, 2)
    a_shape = (b, m, k) if batch else (m, k)
    w_shape = (b, k, n) if batch else (k, n)
    a_t, a_c = pair_from_flat(torch_module, c_module, a_flat, a_shape, dtype_name)
    w_t, w_c = pair_from_flat(torch_module, c_module, w_flat, w_shape, dtype_name)
    if with_bias:
        bias_flat = _gemm_lcg(n, 3)
        bias_t, bias_c = pair_from_flat(torch_module, c_module, bias_flat, (n,), dtype_name)
        run_torch = lambda: torch_call(bias_t, a_t, w_t)  # noqa: E731
        run_c = lambda: c_module._aten_dispatch(op, bias_c, a_c, w_c)  # noqa: E731
    else:
        run_torch = lambda: torch_call(a_t, w_t)  # noqa: E731
        run_c = lambda: c_module._aten_dispatch(op, a_c, w_c)  # noqa: E731
    short = op.split(".", 2)[1]
    return Case(
        name=f"{short}(dtype={dtype_name}, {a_shape}x{w_shape}) [model-scale, k={k}: {note}]",
        op=op,
        run_torch=run_torch,
        run_c=run_c,
        note=note,
        value_check=_gemm_scale_check(dtype_name, k),
    )


# --- the four ops falcon/gptj/bloom/mpt all ask for, plus stack and relu -----
#
# docs/ARCH.md measured 32 architectures and found the largest single cluster in
# the tail: `falcon`, `gptj`, `bloom` and `mpt` are missing *exactly* the same
# four ops -- `le.Tensor`, `scalar_tensor.default`, `where.self` and
# `permute.default`. All four are the same idiom, and re-tracing the four models
# shows it end to end (docs/OPS4.md §1):
#
#     mask   = arange(...) <= arange(...)          aten.le.Tensor      (int64)
#     fill   = scalar_tensor(finfo(dtype).min)     aten.scalar_tensor  (0-d f32)
#     scores = where(mask, fill, scores)           aten.where.self
#     q/k/v  = permute(x, [0, 2, 1, 3])            aten.permute.default
#
# `stack.default` (gptj, cohere, helium, mamba) and `relu.default` (opt,
# nemotron, persimmon) ride along as the next two by architecture count.
#
# Every case below builds its tensors *inside* the lambdas, for the reason the
# note above `_pair` gives: a builder must not be able to crash the whole run at
# case-list time.


def _scalar_tensor_case(torch_module, c_module, torch_call, value, dtype_name=None,
                        kwargs=None, expect="match", note="") -> Case:
    op = "aten.scalar_tensor.default"
    kwargs = kwargs or {}

    def t_kw():
        out = dict(kwargs)
        if dtype_name is not None:
            out["dtype"] = dt.torch_dtype(torch_module, dtype_name)
        return out

    def c_kw():
        out = dict(kwargs)
        if dtype_name is not None:
            out["dtype"] = dt.c_dtype(c_module, dtype_name)
        return out

    extra = "".join(f", {k}={v!r}" for k, v in kwargs.items())
    return Case(
        name=f"scalar_tensor({value!r}, dtype={dtype_name or 'inferred'}{extra}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(value, **t_kw()),
        run_c=lambda: c_module._aten_dispatch(op, value, **c_kw()),
        expect=expect,
        note=note,
    )


def scalar_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    cases: list[Case] = []

    # The rule that would have been got wrong by copying `full`: the fill
    # value's category is *ignored*. `full([], 3)` is int64 and `full([], True)`
    # is bool, but `scalar_tensor` of either is float32 (measured). These four
    # cases are the whole difference and they are the reason the op has its own
    # kernel rather than a route into `full_default`.
    for value, note in [
        (1.5, "python float -> the default float, as expected"),
        (3, "python INT -> float32, NOT int64 -- `full([], 3)` would be int64"),
        (True, "python BOOL -> float32, NOT bool -- `full([], True)` would be bool"),
        (-2, "negative int, same rule"),
        (0.0, "zero is not a special case"),
        (float("nan"), "nan survives into the default float"),
        (float("inf"), "inf survives into the default float"),
    ]:
        cases.append(_scalar_tensor_case(torch_module, c_module, torch_call, value, note=note))

    # Every storable dtype, from each of the three scalar categories, since the
    # inference rule above means the category can only show up through the
    # conversion and never through the result dtype.
    for dtype_name in ["float64", "float32", "float16", "bfloat16",
                       "int64", "int32", "int16", "uint8", "bool"]:
        for value in [3, 1.5, True]:
            cases.append(
                _scalar_tensor_case(
                    torch_module, c_module, torch_call, value, dtype_name,
                    note="explicit dtype -- the conversion, not the inference",
                )
            )

    # `checked_convert` at numel == 1, which is where upstream's own fast-path
    # hole lives. The float16/bfloat16 rows must NOT raise and the float32 one
    # must -- getting that backwards is the failure this pins.
    for value, dtype_name, note in [
        (1e6, "float16", "OVERFLOWS float16 and is NOT refused: numel==1 takes upstream's "
                         "unchecked fast path and saturates to inf (`full([3], 1e6, float16)` raises)"),
        (1e300, "bfloat16", "same hole, bfloat16"),
        (1e300, "float32", "float32 is NOT in the hole -- upstream checks it even at one element"),
        (2 ** 40, "int32", "integer overflow is refused at any numel"),
        (-1, "uint8", "negative into unsigned WRAPS to 255 -- allowed, magnitude fits"),
        (300, "uint8", "magnitude does not fit -- refused"),
        (-1.5, "int64", "float into int TRUNCATES toward zero, giving -1 (not -2)"),
        (float("nan"), "int64", "int64 has no nan -- refused"),
        (float("inf"), "int64", "int64 has no inf -- refused"),
        (float("inf"), "float16", "float16 does have inf -- not an overflow"),
        (2, "bool", "any nonzero is True; bool never overflows"),
        (0.0, "bool", "zero is False"),
    ]:
        cases.append(
            _scalar_tensor_case(torch_module, c_module, torch_call, value, dtype_name, note=note)
        )

    # The factory keyword arguments. `layout=torch.strided` is passed
    # *explicitly* by bloom, mpt and gptj at this call site (measured), so it
    # has to be accepted rather than refused -- see `reject_layout`.
    cases.append(
        _scalar_tensor_case(
            torch_module, c_module, torch_call, 2.5, "float32",
            kwargs={"device": "cpu"}, note="device given as a string",
        )
    )
    cases.append(
        Case(
            name="scalar_tensor(2.5, layout=torch.strided) [the real call sites pass this]",
            op="aten.scalar_tensor.default",
            run_torch=lambda: torch_call(2.5, layout=torch_module.strided),
            run_c=lambda: c_module._aten_dispatch(
                "aten.scalar_tensor.default", 2.5, layout=torch_module.strided
            ),
            note="bloom/mpt/gptj send layout=torch.strided; it names the only layout "
                 "the shim has, so refusing it would block them on nothing",
        )
    )
    cases.append(
        Case(
            name="scalar_tensor(2.5, layout=torch.sparse_coo) [any other layout still refuses]",
            op="aten.scalar_tensor.default",
            run_torch=lambda: torch_call(2.5, layout=torch_module.sparse_coo),
            run_c=lambda: c_module._aten_dispatch(
                "aten.scalar_tensor.default", 2.5, layout=torch_module.sparse_coo
            ),
            expect="c_error",
            note="torch answers a dense tensor and ignores the request at this size; the "
                 "shim has no sparse layout and says so rather than silently answering dense",
        )
    )
    # The exact call the four architectures make.
    cases.append(
        Case(
            name="scalar_tensor(finfo(float32).min, dtype=float32) [the mask fill value, verbatim]",
            op="aten.scalar_tensor.default",
            run_torch=lambda: torch_call(
                torch_module.finfo(torch_module.float32).min, dtype=torch_module.float32
            ),
            run_c=lambda: c_module._aten_dispatch(
                "aten.scalar_tensor.default",
                torch_module.finfo(torch_module.float32).min,
                dtype=c_module.float32,
            ),
            note="falcon/gptj/bloom/mpt all build their attention mask fill this way",
        )
    )
    return cases


# --- aten.where.self ---------------------------------------------------------


def _where_case(torch_module, c_module, torch_call, cond, lhs, rhs,
                expect="match", note="") -> Case:
    """`cond`/`lhs`/`rhs` are `(flat, shape, dtype_name)` triples, built inside
    the lambdas."""
    op = "aten.where.self"

    def build(index):
        return tuple(
            _pair(torch_module, c_module, flat, shape, dtype_name)[index]
            for flat, shape, dtype_name in (cond, lhs, rhs)
        )

    return Case(
        name=f"where(cond={cond[1]}/{cond[2]}, self={lhs[1]}/{lhs[2]}, "
             f"other={rhs[1]}/{rhs[2]}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(*build(0)),
        run_c=lambda: c_module._aten_dispatch(op, *build(1)),
        expect=expect,
        note=note,
    )


def where_self_cases(torch_module, c_module, torch_call) -> list[Case]:
    cases: list[Case] = []
    mask4 = ([1, 0, 1, 0], (4,), "bool")

    # Both branches in every storable dtype. `where` moves elements rather than
    # computing on them, so unlike `mm` there is no kernel-coverage split.
    for dtype_name in ["float64", "float32", "float16", "bfloat16",
                       "int64", "int32", "int16", "uint8", "bool"]:
        cases.append(
            _where_case(
                torch_module, c_module, torch_call, mask4,
                ([1, 2, 3, 4], (4,), dtype_name), ([9, 8, 7, 6], (4,), dtype_name),
                note="elementwise select, matched dtypes",
            )
        )

    # The shape rules. The third is the one a condition-shaped implementation
    # gets wrong: the result is the join of all THREE, not the condition's.
    for cond, lhs, rhs, note in [
        (([1, 0, 0, 1], (2, 2), "bool"), ([1.0], (), "float32"), ([-1.0], (), "float32"),
         "0-D BRANCHES broadcast up to the mask -- exactly what the four architectures do"),
        (([1], (), "bool"), ([float(i) for i in range(6)], (2, 3), "float32"),
         ([0.0, 0.0, 0.0], (3,), "float32"),
         "0-D CONDITION: result is (2,3), the join of all three, not the condition's ()"),
        (([1, 0], (2, 1), "bool"), ([1.0, 2.0, 3.0], (1, 3), "float32"), ([0.0], (1, 1), "float32"),
         "all three broadcast against each other"),
        (([1], (), "bool"), ([1.0], (), "float32"), ([2.0], (), "float32"),
         "everything 0-D -> a 0-D result"),
        (([], (0,), "bool"), ([], (0,), "float32"), ([], (0,), "float32"),
         "empty in, empty out -- not an error"),
        (([1, 0, 1], (3,), "bool"), ([1.0, 2.0], (2,), "float32"), ([0.0, 0.0], (2,), "float32"),
         "shapes that do not broadcast are refused by both"),
    ]:
        cases.append(
            _where_case(torch_module, c_module, torch_call, cond, lhs, rhs,
                        expect="both_error" if "refused" in note else "match", note=note)
        )

    # The condition's dtype. `uint8` is ACCEPTED here (deprecated upstream but
    # answered), which is the opposite of `masked_fill`'s rule -- copying that
    # refusal over would have refused a call torch answers.
    cases.append(
        _where_case(
            torch_module, c_module, torch_call, ([2, 0, 5, 0], (4,), "uint8"),
            ([1.0, 2.0, 3.0, 4.0], (4,), "float32"), ([-1.0, -2.0, -3.0, -4.0], (4,), "float32"),
            note="uint8 condition is ACCEPTED (deprecated upstream, still answered) and read "
                 "as truthiness -- 2 and 5 both select the first branch",
        )
    )
    for cond_dtype in ["int64", "int32", "float32"]:
        cases.append(
            _where_case(
                torch_module, c_module, torch_call, ([1, 0, 1, 0], (4,), cond_dtype),
                ([1.0, 2.0, 3.0, 4.0], (4,), "float32"), ([0.0, 0.0, 0.0, 0.0], (4,), "float32"),
                expect="both_error",
                note=f"a {cond_dtype} condition is refused -- only bool and (deprecated) uint8",
            )
        )

    # Values that a blend rather than a select would get wrong.
    cases.append(
        _where_case(
            torch_module, c_module, torch_call, ([1, 0], (2,), "bool"),
            ([float("nan"), 1.0], (2,), "float32"), ([2.0, float("inf")], (2,), "float32"),
            note="nan and inf are selected, not arithmetic -- the result is [nan, inf]",
        )
    )
    cases.append(
        _where_case(
            torch_module, c_module, torch_call, ([1], (1,), "bool"),
            ([1.0], (1,), "float32"), ([float("nan")], (1,), "float32"),
            note="the UNSELECTED branch is never read for its value: nan there is not "
                 "contagious (measured upstream)",
        )
    )

    # The gap: upstream promotes, this shim refuses. The full 9x9 table is in
    # docs/OPS4.md §2; these are the four rows that would be got wrong by
    # assuming "the wider one wins" (an integral branch never widens a floating
    # one, and float16 with bfloat16 escapes to float32).
    for lhs_dtype, rhs_dtype, upstream, note in [
        ("float32", "float64", "float64", "the ordinary widening"),
        ("float16", "int64", "float16", "an INTEGRAL branch does not widen a floating one"),
        ("float16", "bfloat16", "float32", "two reduced floats promote OUT to float32"),
        ("bool", "int64", "int64", "bool is the bottom of the lattice"),
    ]:
        cases.append(
            _where_case(
                torch_module, c_module, torch_call, mask4,
                ([1, 2, 3, 4], (4,), lhs_dtype), ([9, 8, 7, 6], (4,), rhs_dtype),
                expect="c_error",
                note=f"upstream promotes to {upstream} ({note}); the shim refuses rather than "
                     "guessing a promotion -- see same_dtype in rust/torch_c/src/aten.rs",
            )
        )

    # The idiom itself, at the shape falcon/gptj/bloom/mpt actually produce.
    cases.append(
        _where_case(
            torch_module, c_module, torch_call,
            ([1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1], (1, 1, 4, 4), "bool"),
            ([0.0], (), "float32"), ([-3.4028234663852886e38], (), "float32"),
            note="the causal-mask idiom verbatim: a (1,1,S,S) bool mask selecting between "
                 "two 0-D scalar_tensor results",
        )
    )
    return cases


# --- aten.permute.default ----------------------------------------------------


def _permute_case(torch_module, c_module, torch_call, flat, shape, dims,
                  dtype_name="float32", expect="match", note="") -> Case:
    op = "aten.permute.default"
    return Case(
        name=f"permute(dtype={dtype_name}, shape={shape}, dims={dims}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(
            _pair(torch_module, c_module, flat, shape, dtype_name)[0], list(dims)
        ),
        run_c=lambda: c_module._aten_dispatch(
            op, _pair(torch_module, c_module, flat, shape, dtype_name)[1], list(dims)
        ),
        expect=expect,
        note=note,
    )


def permute_cases(torch_module, c_module, torch_call) -> list[Case]:
    six = [float(i) for i in range(6)]
    cases: list[Case] = []

    for dtype_name in ["float64", "float32", "float16", "bfloat16",
                       "int64", "int32", "int16", "uint8", "bool"]:
        values = [1, 0, 0, 1, 1, 0] if dtype_name == "bool" else six
        cases.append(
            _permute_case(torch_module, c_module, torch_call, values, (2, 3), [1, 0],
                          dtype_name, note="the 2-D transpose spelling, every dtype")
        )

    for flat, shape, dims, note in [
        (six, (2, 3), [0, 1], "the identity permutation is a legal no-op"),
        (six, (2, 3), [-1, -2], "negative entries normalise: [-1,-2] == [1,0]"),
        ([float(i) for i in range(24)], (2, 3, 4), [2, 0, 1], "3-D rotate -> (4,2,3)"),
        ([float(i) for i in range(24)], (2, 3, 4), [1, 2, 0], "3-D the other way -> (3,4,2)"),
        ([float(i) for i in range(24)], (1, 2, 3, 4), [0, 2, 1, 3],
         "the attention spelling all four architectures call"),
        ([7.0], (), [], "0-D takes an EMPTY dims list and comes back unchanged"),
        ([1.0, 2.0, 3.0], (3,), [0], "1-D identity"),
        ([], (0, 3), [1, 0], "an empty extent permutes to (3,0)"),
    ]:
        cases.append(
            _permute_case(torch_module, c_module, torch_call, flat, shape, dims, note=note)
        )

    # The refusals. The length rule is exact -- `rank.max(1)` would let the
    # 0-D/[0] case through, and upstream refuses it.
    for flat, shape, dims, note in [
        ([7.0], (), [0], "0-D with a 1-element dims list IS refused (rank is 0, not max(0,1))"),
        (six, (2, 3), [0], "too few dims"),
        (six, (2, 3), [0, 1, 2], "too many dims"),
        (six, (2, 3), [0, 0], "a duplicate dim is refused -- it is not a permutation"),
        (six, (2, 3), [2, 0], "out of range, positive"),
        (six, (2, 3), [-3, 0], "out of range, negative"),
    ]:
        cases.append(
            _permute_case(torch_module, c_module, torch_call, flat, shape, dims,
                          expect="both_error", note=note)
        )
    return cases


# --- aten.stack.default ------------------------------------------------------


def _stack_case(torch_module, c_module, torch_call, entries, dim=None,
                expect="match", note="") -> Case:
    """`entries` is a list of `(flat, shape, dtype_name)` triples."""
    op = "aten.stack.default"
    args = () if dim is None else (dim,)

    def build(index):
        return [_pair(torch_module, c_module, f, s, d)[index] for f, s, d in entries]

    shapes = [e[1] for e in entries]
    dtypes = sorted({e[2] for e in entries})
    return Case(
        name=f"stack(n={len(entries)}, shapes={shapes}, dtypes={dtypes}, "
             f"dim={'default 0' if dim is None else dim}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(build(0), *args),
        run_c=lambda: c_module._aten_dispatch(op, build(1), *args),
        expect=expect,
        note=note,
    )


def stack_cases(torch_module, c_module, torch_call) -> list[Case]:
    cases: list[Case] = []

    for dtype_name in ["float64", "float32", "float16", "bfloat16",
                       "int64", "int32", "int16", "uint8", "bool"]:
        a = ([1, 0] if dtype_name == "bool" else [1, 2], (2,), dtype_name)
        b = ([0, 1] if dtype_name == "bool" else [3, 4], (2,), dtype_name)
        cases.append(
            _stack_case(torch_module, c_module, torch_call, [a, b],
                        note="two 1-D entries on a new leading axis, every dtype")
        )

    a = ([1.0, 2.0], (2,), "float32")
    b = ([3.0, 4.0], (2,), "float32")
    # The dim rule, which is `unsqueeze`'s and NOT `cat`'s: the new axis may go
    # after the last existing one, so dim == rank is legal.
    for dim, note in [
        (0, "explicit dim 0 -> (2,2), rows are the entries"),
        (1, "dim == RANK is legal (cat would refuse it) -> (2,2), columns are the entries"),
        (-1, "-1 is the new last axis, same as dim=1 here"),
        (-2, "-2 is the new first axis, same as dim=0 here"),
    ]:
        cases.append(_stack_case(torch_module, c_module, torch_call, [a, b], dim, note=note))
    for dim in [2, -3]:
        cases.append(
            _stack_case(torch_module, c_module, torch_call, [a, b], dim, expect="both_error",
                        note=f"dim={dim} is outside [-(rank+1), rank]")
        )

    scalar_a = ([1.0], (), "float32")
    scalar_b = ([2.0], (), "float32")
    for entries, dim, expect, note in [
        ([a], None, "match", "a single entry still gets the new axis -> (1,2)"),
        ([a, b, a], 1, "match", "three entries, dim=1 -> (2,3)"),
        ([scalar_a, scalar_b], None, "match", "0-D entries stack into a 1-D result"),
        ([scalar_a, scalar_b], 1, "both_error", "0-D entries have no dim 1"),
        ([([], (0,), "float32"), ([], (0,), "float32")], None, "match",
         "empty entries give shape (2,0), not an error"),
        ([([float(i) for i in range(6)], (2, 3), "float32")] * 2, 1, "match",
         "2-D entries, new axis in the middle -> (2,2,3)"),
        ([([float(i) for i in range(8)], (1, 2, 2, 2), "float32")] * 2, -1, "match",
         "the gptj rotary spelling: two 4-D halves stacked on a new last axis"),
    ]:
        cases.append(
            _stack_case(torch_module, c_module, torch_call, entries, dim, expect=expect, note=note)
        )

    # The size rule, which is `stack`'s alone: `cat` lets the join axis differ,
    # `stack` requires every entry to be identical, rank included.
    cases.append(
        _stack_case(
            torch_module, c_module, torch_call,
            [a, ([1.0, 2.0, 3.0], (3,), "float32")], expect="both_error",
            note="differing extents are refused -- cat on dim 0 would accept these",
        )
    )
    cases.append(
        _stack_case(
            torch_module, c_module, torch_call,
            [a, ([1.0, 2.0], (2, 1), "float32")], expect="both_error",
            note="differing RANK is refused too, even at the same element count",
        )
    )
    cases.append(
        Case(
            name="stack([]) [an empty list has no shape to invent]",
            op="aten.stack.default",
            run_torch=lambda: torch_call([]),
            run_c=lambda: c_module._aten_dispatch("aten.stack.default", []),
            expect="both_error",
            note="upstream: 'stack expects a non-empty TensorList'",
        )
    )

    # The gap: upstream promotes here as well.
    for lhs_dtype, rhs_dtype, upstream in [
        ("float32", "float64", "float64"),
        ("int64", "float32", "float32"),
        ("bool", "int64", "int64"),
    ]:
        cases.append(
            _stack_case(
                torch_module, c_module, torch_call,
                [([1, 2], (2,), lhs_dtype), ([3, 4], (2,), rhs_dtype)],
                expect="c_error",
                note=f"upstream promotes the entries to {upstream}; the shim refuses, the same "
                     "way cat_default does",
            )
        )
    return cases


# --- aten.relu.default -------------------------------------------------------


def relu_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.relu.default"
    cases: list[Case] = []

    # relu has an integral CPU kernel upstream (silu does not), so the integral
    # dtypes belong here as matches rather than refusals.
    for dtype_name in ["float64", "float32", "float16", "bfloat16",
                       "int64", "int32", "int16"]:
        cases.append(
            _unary_case(torch_module, c_module, op, torch_call, dtype_name,
                        [-2, -1, 0, 1, 2], (5,), "negatives clamp, non-negatives pass through")
        )
    # uint8 with non-negative literals only: `torch.tensor(-1, uint8)` wraps to
    # 255 while `_C._tensor_from_flat` saturates to 0, a constructor difference
    # docs/GPT2.md §7 already recorded and which has nothing to do with relu.
    cases.append(
        _unary_case(torch_module, c_module, op, torch_call, "uint8",
                    [0, 1, 2, 255], (4,),
                    "on an unsigned dtype relu is the identity -- no element is negative")
    )
    cases.append(
        Case(
            name="relu(dtype=bool, shape=(2,)) [upstream refuses bool]",
            op=op,
            run_torch=lambda: torch_call(_pair(torch_module, c_module, [1, 0], (2,), "bool")[0]),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [1, 0], (2,), "bool")[1]
            ),
            expect="both_error",
            note="'Boolean inputs not supported for relu' -- a bool relu would be the identity, "
                 "and upstream would rather say no than answer it",
        )
    )

    # The two inputs that separate `x < 0 ? 0 : x` from `max(x, 0)`. A max-shaped
    # implementation passes everything above and fails exactly here.
    cases.append(
        Case(
            name="relu(float32, [nan, inf, -inf, -0.0, 0.0]) [NOT max(x,0)]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module,
                      [float("nan"), float("inf"), float("-inf"), -0.0, 0.0], (5,), "float32")[0]
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module,
                      [float("nan"), float("inf"), float("-inf"), -0.0, 0.0], (5,), "float32")[1],
            ),
            note="measured upstream: [nan, inf, 0.0, -0.0, 0.0]. nan SURVIVES (so this is not a "
                 "comparison-ordered maximum) and -0.0 keeps its sign (so this is not a clamp "
                 "that normalises) -- both fall out of `x < 0 ? 0 : x`",
        )
    )
    cases.append(
        Case(
            name="relu(float64, [nan, -0.0]) [the same two, at double precision]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [float("nan"), -0.0], (2,), "float64")[0]
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [float("nan"), -0.0], (2,), "float64")[1]
            ),
            note="not a float32-only accident",
        )
    )

    for flat, shape, note in [
        ([-3.0], (), "0-D input"),
        ([], (0,), "empty input"),
        ([float(i) - 3 for i in range(12)], (3, 4), "2-D, straddling zero"),
        ([-1e30, 1e30, -1e-30, 1e-30], (4,), "magnitudes far from 1"),
    ]:
        cases.append(
            _unary_case(torch_module, c_module, op, torch_call, "float32", flat, shape, note)
        )
    return cases


# --- aten.relu_.default ------------------------------------------------------
#
# `relu.default`'s in-place sibling -- `F.relu(x, inplace=True)` traces to
# this overload, not `relu.default` (docs/SPELLINGS.md §6.6 measured the
# kernel gap: zero hits before this). The value is `relu.default`'s
# unchanged; what's new here is the in-place contract, so the cases below
# follow `add__tensor_cases`'s shape (mutated-receiver comparisons) rather
# than re-deriving `relu_cases`' value coverage from scratch.


def relu__cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.relu_.default"
    cases: list[Case] = []

    for dtype_name in ["float64", "float32", "float16", "bfloat16",
                       "int64", "int32", "int16"]:
        dst_t, dst_c = pair_from_flat(torch_module, c_module, [-2, -1, 0, 1, 2], (5,), dtype_name)
        cases.append(
            Case(
                name=f"relu_(dtype={dtype_name}) [in-place: compares the mutated receiver]",
                op=op,
                run_torch=lambda dst_t=dst_t: torch_call(dst_t),
                run_c=lambda dst_c=dst_c: c_module._aten_dispatch(op, dst_c),
                note="negatives clamp, non-negatives pass through, same as relu.default",
            )
        )

    # uint8: identity, same reasoning `relu_cases` gives for the out-of-place
    # overload (no unsigned element is negative).
    dst_u8_t, dst_u8_c = pair_from_flat(torch_module, c_module, [0, 1, 2, 255], (4,), "uint8")
    cases.append(
        Case(
            name="relu_(dtype=uint8) [identity, in-place]",
            op=op,
            run_torch=lambda: torch_call(dst_u8_t),
            run_c=lambda: c_module._aten_dispatch(op, dst_u8_c),
            note="on an unsigned dtype relu_ is the identity -- no element is negative",
        )
    )

    # The two values that separate `x < 0 ? 0 : x` from `max(x, 0)`, re-hit
    # in-place -- measured on real torch to make sure the in-place overload
    # answers the same values relu.default does rather than assuming it.
    dst_special_t, dst_special_c = pair_from_flat(
        torch_module, c_module,
        [float("nan"), float("inf"), float("-inf"), -0.0, 0.0], (5,), "float32",
    )
    cases.append(
        Case(
            name="relu_(float32, [nan, inf, -inf, -0.0, 0.0]) [NOT max(x,0), in-place]",
            op=op,
            run_torch=lambda: torch_call(dst_special_t),
            run_c=lambda: c_module._aten_dispatch(op, dst_special_c),
            note="measured upstream: [nan, inf, 0.0, -0.0, 0.0] -- nan survives, -0.0 keeps its "
                 "sign, same as relu.default",
        )
    )

    bool_t, bool_c = pair_from_flat(torch_module, c_module, [1, 0], (2,), "bool")
    cases.append(
        Case(
            name="relu_(dtype=bool) [upstream refuses bool, in-place too]",
            op=op,
            run_torch=lambda: torch_call(bool_t),
            run_c=lambda: c_module._aten_dispatch(op, bool_c),
            expect="both_error",
            note="'Boolean inputs not supported for relu' -- same refusal wording as "
                 "relu.default, measured on the in-place overload too",
        )
    )
    return cases


# --- aten.le.Tensor ----------------------------------------------------------
#
# The Tensor-overload sibling of `le.Scalar`. It exists as its own key for the
# same reason `lt.Tensor`/`lt.Scalar` do -- different schemas -- and it is the
# op the four architectures build their causal mask with.


def le_tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.le.Tensor"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        for sc in _CMP_SCENARIOS:
            cases.append(
                _binary_tensor_case(
                    torch_module, c_module, op, torch_call, dtype_name,
                    sc["a_flat"], sc["a_shape"], sc["b_flat"], sc["b_shape"], sc["note"],
                )
            )
    cases.append(
        Case(
            name="le(float32, nan <= nan) [every comparison against NaN is false, including <=]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [float("nan"), 1.0], (2,), "float32")[0],
                _pair(torch_module, c_module, [float("nan"), 1.0], (2,), "float32")[0],
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module, [float("nan"), 1.0], (2,), "float32")[1],
                _pair(torch_module, c_module, [float("nan"), 1.0], (2,), "float32")[1],
            ),
            note="nan <= nan is False even though the two sides are the same object; "
                 "1.0 <= 1.0 in the same call is True, so this is not a blanket False",
        )
    )
    cases.append(
        Case(
            name="le(int64, causal mask idiom) [arange(S)[None] <= arange(S)[:,None]]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [0, 1, 2, 3], (1, 1, 1, 4), "int64")[0],
                _pair(torch_module, c_module, [0, 1, 2, 3], (1, 1, 4, 1), "int64")[0],
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module, [0, 1, 2, 3], (1, 1, 1, 4), "int64")[1],
                _pair(torch_module, c_module, [0, 1, 2, 3], (1, 1, 4, 1), "int64")[1],
            ),
            note="the lower-triangular mask falcon/gptj/bloom/mpt all build, at the exact "
                 "(1,1,S,1) vs (1,1,1,S) broadcast they use",
        )
    )
    cases.append(
        Case(
            name="le(bool, [T,F] <= [F,T]) [False <= True, so bool ordering is False < True]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [1, 0], (2,), "bool")[0],
                _pair(torch_module, c_module, [0, 1], (2,), "bool")[0],
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module, [1, 0], (2,), "bool")[1],
                _pair(torch_module, c_module, [0, 1], (2,), "bool")[1],
            ),
            note="bool compares as 0/1, giving [False, True]",
        )
    )
    return cases


# --- the five ops docs/TAIL.md needed to open falcon/bloom/gpt_bigcode ------
#
# All five already had a kernel in rust/torch_c/src/aten.rs and showed up in
# `_aten_implemented()` before this file had a builder for any of them --
# `compare.py` was failing every one with `<no case builder registered>`.
# Each builder below re-measures the claims the kernel's own doc comment
# makes against torch 2.13.0 rather than trusting them, per this module's
# own rule (see the note above `_pair`): a doc comment is not a golden case.
#
# Two real, measured discrepancies came out of that re-measurement and are
# encoded as `expect="c_error"`/`"torch_error"` cases below rather than
# "fixed" (rust/torch_c/src/aten.rs is out of scope for this task):
#
#   * `aten.mul.Scalar(bool_tensor, scalar)` -- upstream computes this
#     *arithmetically* (`True`/`False` read as `1`/`0`, promoted exactly like
#     any other integral tensor: int scalar keeps int64, float scalar gives
#     float32). The shim's `arith_tag` refuses every bool tensor unconditionally,
#     which is right for the `.Tensor` overloads (upstream's bool `*` there
#     really is a logical and) but wrong for `.Scalar` -- `c_error`.
#   * `aten.add_.Tensor(bool_tensor, bool_tensor)` -- upstream computes this
#     too, as a logical or (`True`/`False` and `True`/`True` -> `[True, True]`,
#     measured). The shim's own doc comment above `add_inplace` claims this
#     "matches add.Tensor's own refusal", but that refusal is internal to the
#     shim, not upstream's -- upstream never refuses bool here. `c_error`.
#
# See docs/TAIL.md for the full measurement transcript and the exact
# `_aten_implemented()` counts this closed.


def _baddbmm_case(
    torch_module, c_module, torch_call, dtype_name,
    self_flat, self_shape, b1_flat, b1_shape, b2_flat, b2_shape,
    kwargs=None, expect="match", note="",
) -> Case:
    kwargs = kwargs or {}
    op = "aten.baddbmm.default"
    s_t, s_c = pair_from_flat(torch_module, c_module, self_flat, self_shape, dtype_name)
    a_t, a_c = pair_from_flat(torch_module, c_module, b1_flat, b1_shape, dtype_name)
    b_t, b_c = pair_from_flat(torch_module, c_module, b2_flat, b2_shape, dtype_name)
    return Case(
        name=f"baddbmm(dtype={dtype_name}, self={self_shape}, {b1_shape}x{b2_shape}, {kwargs}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(s_t, a_t, b_t, **kwargs),
        run_c=lambda: c_module._aten_dispatch(op, s_c, a_c, b_c, **kwargs),
        expect=expect,
        note=note,
    )


# (2,2,3) @ (2,3,2) -> (2,2,2), batch of 2 -- big enough that a kernel
# dropping the batch dimension (bmm_cases' own concern) would show up here
# too. Product, worked by hand per batch:
#   batch0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,0]] = [[4,2],[10,5]]... but
# with the actual b2 pattern used below the exact numbers are re-derived
# from beta=0/alpha=1 cases rather than transcribed here.
_BADDBMM_B1 = (list(range(1, 13)), (2, 2, 3))
_BADDBMM_B2 = ([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], (2, 3, 2))
_BADDBMM_SELF = ([0.0] * 8, (2, 2, 2))


def baddbmm_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.baddbmm.default"
    cases: list[Case] = []

    scenarios = [
        (None, "plain: self + batch1 @ batch2"),
        (dict(beta=2, alpha=3), "integer beta/alpha"),
        (dict(beta=0.5, alpha=0.25), "fractional beta/alpha"),
        # The two quick returns, addmm's rule reused batched (kernel doc).
        (dict(beta=0), "beta=0 -- self dropped"),
        (dict(alpha=0), "alpha=0 -- product dropped"),
        (dict(beta=0, alpha=0), "both zero -- a shaped tensor of zeros, not an error"),
        (dict(beta=True), "beta as a bool, which torch reads as 1"),
        (dict(beta=False), "beta as a bool, which torch reads as 0"),
    ]
    for dtype_name in _MM_MATCH_DTYPES:
        for kwargs, note in scenarios:
            cases.append(
                _baddbmm_case(
                    torch_module, c_module, torch_call, dtype_name,
                    *_BADDBMM_SELF, *_BADDBMM_B1, *_BADDBMM_B2, kwargs=kwargs, note=note,
                )
            )
        # `self` broadcasting into the (batch, n, p) target -- measured
        # against real torch: 1-D, 0-d, 2-D (no batch dim), and a
        # singleton-batch 3-D self all broadcast the same way addmm's bias
        # does, generalised to a 3-D target (kernel doc, re-checked above).
        for self_flat, self_shape, note in [
            ([1.0, 2.0], (2,), "1-D self -- broadcasts across the last dim (p=2)"),
            ([7.0], (), "0-d self"),
            ([1.0, 1.0, 1.0, 1.0], (2, 2), "2-D self, no batch dim -- broadcasts across the batch"),
            ([1.0, 1.0, 1.0, 1.0], (1, 2, 2), "singleton-batch 3-D self"),
        ]:
            cases.append(
                _baddbmm_case(
                    torch_module, c_module, torch_call, dtype_name,
                    self_flat, self_shape, *_BADDBMM_B1, *_BADDBMM_B2, note=note,
                )
            )

    # The beta=0 quick return, proven rather than asserted, batched --
    # measured: beta=0 with a NaN self gives the clean product (0*nan would
    # be nan if the multiply weren't skipped).
    nan_self = ([float("nan")] * 8, (2, 2, 2))
    cases.append(
        _baddbmm_case(
            torch_module, c_module, torch_call, "float32",
            *nan_self, *_BADDBMM_B1, *_BADDBMM_B2, kwargs=dict(beta=0),
            note="beta=0 with a NaN self -- 0*nan would be nan, torch gives the clean product",
        )
    )

    # alpha=0 is NOT the mirror-image quick return -- addmm's alpha=0 really
    # does skip the multiply entirely on real torch (addmm_cases above proves
    # it with an inf mat1), but baddbmm's does not: `baddbmm(self=zeros,
    # inf_batch1, batch2, alpha=0)` gives `[[nan, nan], [0, 0]], [[0, 0],
    # [0, 0]]]` on real torch -- NaN leaks through the multiply, which is
    # only *scaled* away afterward, not skipped. This used to be a live
    # regression trap (`expect="diverge"`, docs/TAIL.md §2.1) because the
    # kernel's `alpha_zero` branch skipped the matmul unconditionally
    # (copying addmm_scale's rule) and answered a clean `self` instead.
    # Fixed: the kernel now always runs the multiply and only skips the
    # *scale* on alpha==1 (inside `addmm_scale`), so this is a plain `match`
    # again -- promoted per compare.py's own instruction the moment the
    # divergence closed (re-measured: both sides now give the same NaN
    # pattern above).
    inf_b1 = ([float("inf")] + list(range(2, 13)), (2, 2, 3))
    cases.append(
        _baddbmm_case(
            torch_module, c_module, torch_call, "float32",
            *_BADDBMM_SELF, *inf_b1, *_BADDBMM_B2, kwargs=dict(alpha=0),
            note="alpha=0 with an inf in batch1 -- 0*inf is nan, and unlike addmm's bias "
                 "quick return, baddbmm's multiply is NOT skipped on real torch, so the NaN "
                 "leaks through on both sides now (was a KNOWN KERNEL BUG, fixed; see "
                 "docs/TAIL.md §2.1 and docs/KERNELS.md)",
        )
    )

    # int64 alpha truncation, re-measured rather than assumed from the
    # kernel's doc comment: upstream computes alpha=1.9 and alpha=1 as bit-
    # for-bit identical on an int64 triple (the Scalar truncates toward zero
    # before it multiplies), but the shim never gets that far for int64 --
    # candle has no integral matmul kernel at all (the inherited gap
    # _MM_C_ERROR_DTYPES already records for mm/addmm/bmm), so *any* nonzero
    # alpha on int64 raises "candle: unsupported dtype I64 for op matmul"
    # regardless of what alpha's fractional part is. The truncation claim
    # itself is therefore unverifiable as a match case today; this pins the
    # gap instead of the (currently unreachable) behaviour it claimed.
    int_b1 = (list(range(1, 13)), (2, 2, 3))
    int_b2 = ([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1], (2, 3, 2))
    int_self = ([0] * 8, (2, 2, 2))
    for alpha, note in [
        (1.9, "alpha=1.9 on int64 -- can't verify truncation, candle has no int64 matmul"),
        (1, "alpha=1 on int64 -- same inherited gap"),
    ]:
        cases.append(
            _baddbmm_case(
                torch_module, c_module, torch_call, "int64",
                *int_self, *int_b1, *int_b2,
                kwargs=dict(alpha=alpha), expect="c_error", note=note,
            )
        )

    # The inherited candle gap: no matmul kernel for the integral dtypes or
    # bfloat16 -- same split mm/addmm/bmm already carry.
    for dtype_name in _MM_C_ERROR_DTYPES:
        cases.append(
            _baddbmm_case(
                torch_module, c_module, torch_call, dtype_name,
                [0, 0, 0, 0], (1, 2, 2), [1, 2, 3, 4], (1, 2, 2), [1, 0, 0, 1], (1, 2, 2),
                expect="c_error",
                note=f"candle's matmul has no kernel for {dtype_name}; torch's CPU baddbmm does. "
                     "Same gap aten.mm.default/aten.addmm.default already carry.",
            )
        )
        cases.append(
            _baddbmm_case(
                torch_module, c_module, torch_call, dtype_name,
                [0, 0, 0, 0], (1, 2, 2), [1, 2, 3, 4], (1, 2, 2), [1, 0, 0, 1], (1, 2, 2),
                kwargs=dict(alpha=0),
                expect="c_error",
                note=f"{dtype_name} with alpha=0 -- used to dodge the gap above (the kernel's old "
                     "alpha_zero quick return skipped the matmul unconditionally, matching torch by "
                     "accident); now that the multiply always runs (docs/TAIL.md §2.1 fix), this hits "
                     "the exact same missing-integral-matmul gap as alpha!=0 above",
            )
        )

    # Refusals. Both sides only need to *refuse*, not agree on wording --
    # measured that upstream's actual check order for a malformed batch1/
    # batch2 rank differs from the shim's ("batch1 must be a 3D tensor" up
    # front), but both raise on every one of these, which is all
    # expect="both_error" requires (see the module docstring above).
    self_t, self_c = pair_from_flat(torch_module, c_module, *_BADDBMM_SELF, "float32")
    f32_b1_t, f32_b1_c = pair_from_flat(torch_module, c_module, *_BADDBMM_B1, "float32")
    b2_full_t, b2_full_c = pair_from_flat(torch_module, c_module, *_BADDBMM_B2, "float32")

    b1_2d_t, b1_2d_c = pair_from_flat(torch_module, c_module, list(range(1, 7)), (2, 3), "float32")
    cases.append(
        Case(
            name="baddbmm(batch1 not 3D rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(self_t, b1_2d_t, b2_full_t),
            run_c=lambda: c_module._aten_dispatch(op, self_c, b1_2d_c, b2_full_c),
            expect="both_error",
            note="batch1 is 2D; baddbmm must not silently fall back to mm's rank",
        )
    )
    b2_2d_t, b2_2d_c = pair_from_flat(torch_module, c_module, [1.0] * 6, (3, 2), "float32")
    cases.append(
        Case(
            name="baddbmm(batch2 not 3D rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(self_t, f32_b1_t, b2_2d_t),
            run_c=lambda: c_module._aten_dispatch(op, self_c, f32_b1_c, b2_2d_c),
            expect="both_error",
            note="batch2 is 2D",
        )
    )
    f64_b2_t, f64_b2_c = pair_from_flat(torch_module, c_module, *_BADDBMM_B2, "float64")
    cases.append(
        Case(
            name="baddbmm(batch1 float32 x batch2 float64 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(self_t, f32_b1_t, f64_b2_t),
            run_c=lambda: c_module._aten_dispatch(op, self_c, f32_b1_c, f64_b2_c),
            expect="both_error",
            note="torch: 'expected scalar type Float but found Double'",
        )
    )
    f64_self_t, f64_self_c = pair_from_flat(torch_module, c_module, *_BADDBMM_SELF, "float64")
    cases.append(
        Case(
            name="baddbmm(self float64 x batch2 float32 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(f64_self_t, f32_b1_t, b2_full_t),
            run_c=lambda: c_module._aten_dispatch(op, f64_self_c, f32_b1_c, b2_full_c),
            expect="both_error",
            note="torch: 'Input dtypes must be the same, got: input double, batch1: float, batch2: float'",
        )
    )
    b2_batch1_t, b2_batch1_c = pair_from_flat(
        torch_module, c_module, _BADDBMM_B2[0][:6], (1, 3, 2), "float32"
    )
    cases.append(
        Case(
            name="baddbmm(batch count mismatch rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(self_t, f32_b1_t, b2_batch1_t),
            run_c=lambda: c_module._aten_dispatch(op, self_c, f32_b1_c, b2_batch1_c),
            expect="both_error",
            note="batch1 has batch=2, batch2 has batch=1 -- baddbmm does not broadcast the batch dim",
        )
    )
    bad_inner_t, bad_inner_c = pair_from_flat(torch_module, c_module, [1.0] * 16, (2, 4, 2), "float32")
    cases.append(
        Case(
            name="baddbmm(inner dim mismatch rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(self_t, f32_b1_t, bad_inner_t),
            run_c=lambda: c_module._aten_dispatch(op, self_c, f32_b1_c, bad_inner_c),
            expect="both_error",
            note="batch1 is (2,2,3), batch2 is (2,4,2) -- 3 != 4",
        )
    )
    bad_self_t, bad_self_c = pair_from_flat(torch_module, c_module, [1.0] * 12, (3, 2, 2), "float32")
    cases.append(
        Case(
            name="baddbmm(self not expandable to target rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(bad_self_t, f32_b1_t, b2_full_t),
            run_c=lambda: c_module._aten_dispatch(op, bad_self_c, f32_b1_c, b2_full_c),
            expect="both_error",
            note="self is (3,2,2), target is (2,2,2) -- non-singleton mismatch at dim 0",
        )
    )
    too_many_t, too_many_c = pair_from_flat(torch_module, c_module, [1.0] * 8, (1, 2, 2, 2), "float32")
    cases.append(
        Case(
            name="baddbmm(self has more dims than target rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(too_many_t, f32_b1_t, b2_full_t),
            run_c=lambda: c_module._aten_dispatch(op, too_many_c, f32_b1_c, b2_full_c),
            expect="both_error",
            note="self is 4-D, target is 3-D",
        )
    )
    cases.append(
        Case(
            name="baddbmm(dtype=bool rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [1, 0, 1, 0], (1, 2, 2), "bool")[0],
                _pair(torch_module, c_module, [1, 0, 1, 0], (1, 2, 2), "bool")[0],
                _pair(torch_module, c_module, [1, 0, 1, 0], (1, 2, 2), "bool")[0],
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module, [1, 0, 1, 0], (1, 2, 2), "bool")[1],
                _pair(torch_module, c_module, [1, 0, 1, 0], (1, 2, 2), "bool")[1],
                _pair(torch_module, c_module, [1, 0, 1, 0], (1, 2, 2), "bool")[1],
            ),
            expect="both_error",
            note='torch: "baddbmm" not implemented for \'Bool\' -- measured to match the shim\'s '
                 "own wording exactly, unlike the rank refusals above",
        )
    )
    cases.append(
        Case(
            name="baddbmm(dtype=uint32 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [0, 0, 0, 0], (1, 2, 2), "uint32")[0],
                _pair(torch_module, c_module, [1, 2, 3, 4], (1, 2, 2), "uint32")[0],
                _pair(torch_module, c_module, [1, 0, 0, 1], (1, 2, 2), "uint32")[0],
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                _pair(torch_module, c_module, [0, 0, 0, 0], (1, 2, 2), "uint32")[1],
                _pair(torch_module, c_module, [1, 2, 3, 4], (1, 2, 2), "uint32")[1],
                _pair(torch_module, c_module, [1, 0, 0, 1], (1, 2, 2), "uint32")[1],
            ),
            expect="both_error",
            note='torch: "baddbmm" not implemented for \'UInt32\' -- unlike UInt16/UInt64, '
                 "uint32 is at least constructible on the c side, so this exercises baddbmm's "
                 "own refusal rather than a _tensor_from_flat storage gap",
        )
    )

    # Model-scale, batched with the bias -- attention's QK^T scale-and-add,
    # the shape docs/TAIL.md measured bloom reaching for.
    for dtype_name, note in [
        ("float32", "batched depth 512 with a bias -- bloom's scaled QK^T"),
        ("float16", "the same, in the dtype a device would actually run"),
    ]:
        cases.append(
            _big_gemm_case(torch_module, c_module, torch_call, "aten.baddbmm.default",
                           dtype_name, 8, 512, 8, with_bias=True, batch=2, note=note)
        )
    return cases


# --- aten.split_with_sizes.default -------------------------------------
# `split.Tensor` with the chunk sizes spelled out individually -- the
# spelling `gpt_bigcode`'s `c_attn(x).split((embed_dim, kv_dim, kv_dim),
# dim=2)` reaches for. Reuses `_chunk_list_check` (`split_cases`' own
# comparator) since this also answers with a list of tensors.
#
# Unlike `split.Tensor`, sizes must sum *exactly* to the dimension's extent
# -- no "last chunk is short" leniency -- and a size of 0 is fine even when
# the dimension is not itself empty. Both measured against torch 2.13.0
# (see the kernel's own doc comment, re-checked in the refusal cases below).


def _split_with_sizes_case(
    torch_module, c_module, torch_call, dtype_name, flat, shape, sizes, dim=0,
    expect="match", note="",
) -> Case:
    op = "aten.split_with_sizes.default"
    a_t, a_c = pair_from_flat(torch_module, c_module, flat, shape, dtype_name)
    return Case(
        name=f"split_with_sizes(dtype={dtype_name}, shape={shape}, sizes={sizes}, dim={dim}) [{note}]",
        op=op,
        run_torch=lambda: torch_call(a_t, sizes, dim),
        run_c=lambda: c_module._aten_dispatch(op, a_c, sizes, dim),
        expect=expect,
        value_check=_chunk_list_check if expect == "match" else None,
        note=note + " -- returns a list of tensors, see _chunk_list_check",
    )


def split_with_sizes_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.split_with_sizes.default"
    cases: list[Case] = []
    ten = list(range(10))
    for dtype_name in _SPLIT_DTYPES:
        for sizes, note in [
            ([3, 3, 4], "uneven three-way, no leftover"),
            ([10], "one chunk, the whole tensor"),
            ([1] * 10, "ten chunks of one"),
            ([0, 10], "a leading zero-length chunk, dimension not empty"),
            ([10, 0], "a trailing zero-length chunk"),
            ([4, 0, 6], "a zero-length chunk in the middle"),
        ]:
            cases.append(_split_with_sizes_case(torch_module, c_module, torch_call, dtype_name, ten, (10,), sizes, note=note))

    # The GPT-2/gpt_bigcode shape this op exists for: an uneven three-way
    # unpack of a fused QKV projection along the last dim -- query gets the
    # full embedding width, key/value share a narrower one.
    qkv = list(range(24))
    cases.append(
        _split_with_sizes_case(
            torch_module, c_module, torch_call, "float32", qkv, (2, 2, 6), [3, 1, 2], dim=2,
            note="gpt_bigcode's c_attn(x).split((3,1,2), dim=2) -- q wider than k/v",
        )
    )
    cases.append(
        _split_with_sizes_case(
            torch_module, c_module, torch_call, "float32", qkv, (2, 2, 6), [3, 1, 2], dim=-1,
            note="same split, addressed with a negative dim",
        )
    )

    # Refusals. Measured against torch 2.13.0: exact wording is reproduced
    # by the kernel and pinned here (unlike baddbmm's rank checks above,
    # these match verbatim -- see the kernel's own doc comment).
    cases.append(
        Case(
            name="split_with_sizes(sizes sum too small, rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(_pair(torch_module, c_module, ten, (10,), "float32")[0], [3, 3, 3], 0),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, ten, (10,), "float32")[1], [3, 3, 3], 0
            ),
            expect="both_error",
            note="sizes sum to 9, dimension extent is 10 -- unlike split.Tensor there is no "
                 "leniency here, the caller already spelled out every length",
        )
    )
    cases.append(
        Case(
            name="split_with_sizes(sizes sum too large, rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(_pair(torch_module, c_module, ten, (10,), "float32")[0], [5, 10], 0),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, ten, (10,), "float32")[1], [5, 10], 0
            ),
            expect="both_error",
            note="sizes sum to 15, dimension extent is 10",
        )
    )
    cases.append(
        Case(
            name="split_with_sizes(negative size entry, rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(_pair(torch_module, c_module, ten, (10,), "float32")[0], [5, -5, 10], 0),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, ten, (10,), "float32")[1], [5, -5, 10], 0
            ),
            expect="both_error",
            note="a negative entry is refused even though the sum happens to come out to 10",
        )
    )
    cases.append(
        Case(
            name="split_with_sizes(0-d tensor, rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(_pair(torch_module, c_module, [5.0], (), "float32")[0], [], 0),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [5.0], (), "float32")[1], [], 0
            ),
            expect="both_error",
            note="torch: 'split expects at least a 1-dimensional tensor' -- the same wording "
                 "split.Tensor gives; upstream does not distinguish the two overloads here",
        )
    )
    return cases


# --- aten._safe_softmax.default -----------------------------------------
# torch's own decomposition (`torch/_decomp/decompositions.py::safe_softmax`):
# `out = softmax(self, dim); masked = all(self == -inf, dim); where(masked, 0, out)`.
# The one place this disagrees with `_softmax.default` (`softmax_cases`
# above) is a row that is *entirely* -inf: plain softmax gives NaN there
# (0/0 from `-inf - (-inf)`), `_safe_softmax` gives a clean 0 -- exactly the
# shape of a fully-masked attention row. Both re-measured against torch
# 2.13.0 below rather than trusted from the kernel's own doc comment.


def safe_softmax_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten._safe_softmax.default"
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
                    name=f"_safe_softmax(dtype={dtype_name}, shape={shape}, dim={dim}) [{note}]",
                    op=op,
                    run_torch=lambda flat=flat, shape=shape, dim=dim, dtype_name=dtype_name: torch_call(
                        _pair(torch_module, c_module, flat, shape, dtype_name)[0], dim
                    ),
                    run_c=lambda flat=flat, shape=shape, dim=dim, dtype_name=dtype_name: c_module._aten_dispatch(
                        op, _pair(torch_module, c_module, flat, shape, dtype_name)[1], dim
                    ),
                    note=note,
                )
            )

    # The divergence from `_softmax.default` this op exists for: a fully
    # -inf row gives a clean 0 (not NaN), while a partially-masked row still
    # matches plain softmax (the max-subtraction already makes that one
    # safe). Both measured against real torch.
    edge = [
        ([1.0, float("-inf"), 2.0], (3,), "one masked position -- same as plain softmax here"),
        ([float("-inf"), float("-inf")], (2,), "a fully masked row -- 0, not NaN (the divergence)"),
        ([float("-inf"), float("-inf"), float("-inf")], (1, 3), "fully masked, batched over dim 0"),
        ([1.0, 2.0, float("-inf"), float("-inf")], (2, 2), "one masked row, one live row, same call"),
        ([1000.0, 1001.0, 999.0], (3,), "large logits -- max subtraction still applies"),
    ]
    for flat, shape, note in edge:
        cases.append(
            Case(
                name=f"_safe_softmax(float32, {shape}) [{note}]",
                op=op,
                run_torch=lambda flat=flat, shape=shape: torch_call(
                    _pair(torch_module, c_module, flat, shape, "float32")[0], -1
                ),
                run_c=lambda flat=flat, shape=shape: c_module._aten_dispatch(
                    op, _pair(torch_module, c_module, flat, shape, "float32")[1], -1
                ),
                note=note,
            )
        )

    # `dtype` casts *before* the integral refusal runs -- measured:
    # _safe_softmax(int64_tensor, 0, torch.float32) succeeds, unlike plain
    # _softmax which has no dtype-cast path exercised here at all.
    cases.append(
        Case(
            name="_safe_softmax(int64 input, dtype=float32 -- casts before the integral refusal)",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [1, 2, 3], (3,), "int64")[0], 0,
                dt.torch_dtype(torch_module, "float32"),
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [1, 2, 3], (3,), "int64")[1], 0,
                dt.c_dtype(c_module, "float32"),
            ),
            note="dtype casts self before the floating-point check runs, so this succeeds "
                 "even though a bare int64 input (below) is refused",
        )
    )
    cases.append(
        Case(
            name="_safe_softmax(int64 rejected on both sides)",
            op=op,
            run_torch=lambda: torch_call(_pair(torch_module, c_module, [1, 2, 3], (3,), "int64")[0], -1),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [1, 2, 3], (3,), "int64")[1], -1
            ),
            expect="both_error",
            note='torch: NotImplementedError, "softmax_lastdim_kernel_impl" not implemented for \'Long\'',
        )
    )
    return cases


# --- aten.add_.Tensor -----------------------------------------------------
# The in-place sibling of `add.Tensor`, needed for falcon's residual
# connections (`hidden_states += attn_output`, which traces to this
# overload rather than a rebind). Fresh operands per case, same as
# fill__cases/copy__cases above (the module note on in-place ops) --
# sharing an operand across cases would carry one case's mutation into the
# next.
#
# Two real gaps came out of re-measuring the kernel's own doc comment
# against actual torch 2.13.0 (see the block comment above `_baddbmm_case`
# for the other one):
#
#   * `int32.add_(float32_tensor)` -- upstream refuses ("result type Float
#     can't be cast to the desired output type Int"); the shim casts
#     `other` into the receiver's dtype and computes. `torch_error`.
#   * `bool.add_(bool)` -- upstream computes a logical or (`[True,False]
#     .add_([True,True])` -> `[True,True]`, measured); the shim refuses
#     unconditionally. `c_error`, not the `both_error` the kernel's doc
#     comment implies (it matches `add.Tensor`'s *own* refusal, not
#     upstream's actual behaviour).


def add__tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.add_.Tensor"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "float16", "bfloat16", "int64", "int32", "uint8"]:
        dst_t, dst_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), dtype_name)
        src_t, src_c = pair_from_flat(torch_module, c_module, [10, 20, 30, 40], (2, 2), dtype_name)
        cases.append(
            Case(
                name=f"add_(dtype={dtype_name}, same shape)",
                op=op,
                run_torch=lambda dst_t=dst_t, src_t=src_t: torch_call(dst_t, src_t),
                run_c=lambda dst_c=dst_c, src_c=src_c: c_module._aten_dispatch(op, dst_c, src_c),
                note="in-place: compares the mutated dst operand add_ returns",
            )
        )

    for dtype_name, alpha, note in [
        ("float32", 2.0, "alpha scales other before it's added"),
        ("float32", -1.0, "negative alpha -- effectively an in-place subtract"),
        ("int32", 3, "integer alpha"),
    ]:
        dst_t, dst_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), dtype_name)
        src_t, src_c = pair_from_flat(torch_module, c_module, [10, 20, 30, 40], (2, 2), dtype_name)
        cases.append(
            Case(
                name=f"add_(dtype={dtype_name}, alpha={alpha}) [{note}]",
                op=op,
                run_torch=lambda dst_t=dst_t, src_t=src_t, alpha=alpha: torch_call(dst_t, src_t, alpha=alpha),
                run_c=lambda dst_c=dst_c, src_c=src_c, alpha=alpha: c_module._aten_dispatch(op, dst_c, src_c, alpha=alpha),
                note=note,
            )
        )

    dst2_t, dst2_c = pair_from_flat(torch_module, c_module, [0, 0, 0, 0], (2, 2), "float32")
    src2_t, src2_c = pair_from_flat(torch_module, c_module, [9, 8], (1, 2), "float32")
    cases.append(
        Case(
            name="add_(dtype=float32, broadcast src)",
            op=op,
            run_torch=lambda: torch_call(dst2_t, src2_t),
            run_c=lambda: c_module._aten_dispatch(op, dst2_c, src2_c),
            note="src (1,2) broadcasts to fill dst (2,2) in place",
        )
    )

    # The two measured gaps -- see the module note above.
    int32_dst_t, int32_dst_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), "int32")
    float_src_t, float_src_c = pair_from_flat(torch_module, c_module, [1.5, 2.5, 3.5, 4.5], (2, 2), "float32")
    cases.append(
        Case(
            name="add_(dtype=int32, other=float32 -- c computes, torch refuses)",
            op=op,
            run_torch=lambda: torch_call(int32_dst_t, float_src_t),
            run_c=lambda: c_module._aten_dispatch(op, int32_dst_c, float_src_c),
            expect="torch_error",
            note="torch: 'result type Float can't be cast to the desired output type Int'; "
                 "the shim casts other into the receiver's dtype instead of refusing the unsafe cast",
        )
    )
    bool_dst_t, bool_dst_c = pair_from_flat(torch_module, c_module, [1, 0], (2,), "bool")
    bool_src_t, bool_src_c = pair_from_flat(torch_module, c_module, [1, 1], (2,), "bool")
    cases.append(
        Case(
            name="add_(dtype=bool -- torch computes a logical or, c refuses)",
            op=op,
            run_torch=lambda: torch_call(bool_dst_t, bool_src_t),
            run_c=lambda: c_module._aten_dispatch(op, bool_dst_c, bool_src_c),
            expect="c_error",
            note="torch: [True,False].add_([True,True]) -> [True,True] (logical or, measured); "
                 "the shim's blanket bool refusal in arith_tag over-refuses here",
        )
    )
    return cases


# --- aten.mul.Scalar -------------------------------------------------------
# `wrapped_scalar_tensor`'s counterpart to `mul.Tensor` -- reached only when
# the *parser* keeps the RHS as a `Scalar` rather than promoting it to a 0-d
# tensor first (see `arith_scalar`'s own doc comment). The promotion rule is
# `arith_tag`'s, the same one `rsub_scalar_cases` already pins for
# `rsub.Scalar`: an integral tensor stays integral under an int scalar and
# promotes to the default float under a float one.


def mul_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.mul.Scalar"
    cases: list[Case] = []
    for dtype_name in _MUL_DIV_FLOAT_DTYPES:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1.0, -2.0, 0.0, 3.5], (2, 2), dtype_name)
        for scalar, note in [
            (2.0, "plain scalar multiply"),
            (0.0, "multiply by zero"),
            (-1.5, "negative scalar"),
        ]:
            cases.append(
                Case(
                    name=f"mul.Scalar(dtype={dtype_name}, other={scalar}) [{note}]",
                    op=op,
                    run_torch=lambda a_t=a_t, scalar=scalar: torch_call(a_t, scalar),
                    run_c=lambda a_c=a_c, scalar=scalar: c_module._aten_dispatch(op, a_c, scalar),
                    note=note,
                )
            )

    # The "wrapped number" dtype rule: an integral tensor stays integral
    # under an int scalar and promotes to the default float under a float
    # one -- re-measured here rather than assumed from rsub.Scalar's case.
    for dtype_name, flat, scalar, note in [
        ("int64", [1, 2, 3, 4], 5, "int tensor, int scalar -> int64"),
        ("int64", [1, 2, 3, 4], 5.0, "int tensor, FLOAT scalar -> float32"),
        ("int32", [1, 2, 3, 4], -3, "negative int scalar"),
        ("int16", [10, 20, 30, 40], 2, "int16 * int scalar"),
        ("uint8", [80, 90, 100, 110], 3, "uint8 wraps mod 256: 90*3=270 -> 14, 100*3=300 -> 44"),
    ]:
        a_t, a_c = pair_from_flat(torch_module, c_module, flat, (2, 2), dtype_name)
        cases.append(
            Case(
                name=f"mul.Scalar(dtype={dtype_name}, other={scalar!r}) [{note}]",
                op=op,
                run_torch=lambda a_t=a_t, scalar=scalar: torch_call(a_t, scalar),
                run_c=lambda a_c=a_c, scalar=scalar: c_module._aten_dispatch(op, a_c, scalar),
                note=note,
            )
        )

    # The measured gap: upstream computes `bool * scalar` arithmetically
    # (True/False read as 1/0, promoted exactly like any other integral
    # tensor -- int scalar keeps int64, float scalar gives float32). The
    # shim's `arith_tag` refuses every bool tensor unconditionally, which is
    # right for `mul.Tensor` (upstream's bool `*` there really is a logical
    # and) but wrong here. `c_error`, re-measured rather than assumed.
    bool_t, bool_c = pair_from_flat(torch_module, c_module, [1, 0, 1], (3,), "bool")
    cases.append(
        Case(
            name="mul.Scalar(dtype=bool, other=3 -- torch computes arithmetically, c refuses)",
            op=op,
            run_torch=lambda: torch_call(bool_t, 3),
            run_c=lambda: c_module._aten_dispatch(op, bool_c, 3),
            expect="c_error",
            note="torch: bool*3 -> tensor([3,0,3], dtype=int64) (arithmetic, not logical); "
                 "the shim's blanket bool refusal in arith_tag over-refuses the .Scalar overload",
        )
    )
    bool_float_t, bool_float_c = pair_from_flat(torch_module, c_module, [1, 0, 1], (3,), "bool")
    cases.append(
        Case(
            name="mul.Scalar(dtype=bool, other=2.5 -- float scalar promotes to float32)",
            op=op,
            run_torch=lambda: torch_call(bool_float_t, 2.5),
            run_c=lambda: c_module._aten_dispatch(op, bool_float_c, 2.5),
            expect="c_error",
            note="torch: bool*2.5 -> tensor([2.5,0.,2.5], dtype=float32); same gap, float scalar",
        )
    )
    return cases


# --- mamba / mixtral -- the last two of the 20 measured architectures ------
# (docs/OPS4.md) with anything unimplemented. Every rule pinned below was
# re-measured against torch 2.13.0 with a real `TorchDispatchMode` over
# `transformers` 5.15.1 rather than copied from a kernel's doc comment --
# docs/OPS4.md's own note is that doc comments have been wrong about
# upstream three times before.


def exp_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.exp.default"
    cases: list[Case] = []
    scenarios = [
        ([0.0, 1.0, -1.0, 2.0, -2.0], (5,), "assorted"),
        ([0.0], (), "0-d"),
        ([float("nan"), float("inf"), float("-inf")], (3,), "NaN/+-inf -- nan, inf, 0.0"),
        # Overflows to +inf and underflows to 0.0 in every floating dtype
        # tested here -- not the mamba-relevant range (mamba's A_log stays
        # small), but exercising the boundary rather than assuming a naive
        # `exp` implementation gets it right.
        ([-1000.0, 1000.0], (2,), "far outside range -- underflow to 0.0, overflow to inf"),
    ]
    for dtype_name in _TANH_DTYPES:
        for flat, shape, note in scenarios:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))
    # The promotion rule, which is `cos`/`sin`/`tanh`'s and not `silu`'s: an
    # integral input gives float32 rather than raising. `mamba`'s `A =
    # -exp(A_log)` always calls with `A_log` already `float32`, but the rule
    # is re-measured here rather than assumed from the other unary ops.
    for dtype_name in _TANH_PROMOTING_DTYPES:
        cases.append(
            _unary_case(
                torch_module, c_module, op, torch_call, dtype_name, [0, 1, 2, 3], (2, 2),
                "integral input promotes to the default float, same rule as tanh/cos/sin",
            )
        )
    return cases


def softplus_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.softplus.default"
    cases: list[Case] = []
    scenarios = [
        ([-5.0, -1.0, 0.0, 1.0, 5.0], (5,), "assorted, well inside the default threshold"),
        ([-1000.0, 1000.0], (2,), "far outside the default threshold -- 0.0 and exactly x"),
        ([0.0], (), "0-d"),
    ]
    for dtype_name in ["float64", "float32", "float16", "bfloat16"]:
        for flat, shape, note in scenarios:
            cases.append(_unary_case(torch_module, c_module, op, torch_call, dtype_name, flat, shape, note))

    # beta/threshold overrides -- mamba always calls with the defaults, but
    # the schema accepts both and the kernel's numerically-stable formula
    # has to answer them correctly too, not just the default pair.
    for beta, threshold, note in [
        (2.0, 20.0, "beta scales the input before the formula"),
        (1.0, 5.0, "a lower threshold moves the linear cutoff earlier"),
    ]:
        cases.append(
            _unary_case(
                torch_module, c_module, op, torch_call, "float32",
                [-5.0, -1.0, 0.0, 1.0, 5.0, 10.0], (6,), note,
                kwargs={"beta": beta, "threshold": threshold},
            )
        )

    # Refused, not promoted -- unlike exp/tanh, measured `softplus_cpu`
    # raises `NotImplementedError` naming the dtype rather than widening an
    # integral input to the default float.
    int_t, int_c = pair_from_flat(torch_module, c_module, [1, 2, 3], (3,), "int64")
    cases.append(
        Case(
            name="softplus(dtype=int64) [refused, NOT promoted -- unlike exp/tanh]",
            op=op,
            run_torch=lambda: torch_call(int_t),
            run_c=lambda: c_module._aten_dispatch(op, int_c),
            expect="both_error",
            note="torch: NotImplementedError('\"softplus_cpu\" not implemented for \\'Long\\'')",
        )
    )
    return cases


def convolution_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.convolution.default"
    cases: list[Case] = []

    def make(dtype_name, in_flat, in_shape, w_flat, w_shape, bias_flat, stride, padding, dilation, groups, note):
        x_t, x_c = pair_from_flat(torch_module, c_module, in_flat, in_shape, dtype_name)
        w_t, w_c = pair_from_flat(torch_module, c_module, w_flat, w_shape, dtype_name)
        if bias_flat is None:
            b_t, b_c = None, None
        else:
            b_t, b_c = pair_from_flat(torch_module, c_module, bias_flat, (w_shape[0],), dtype_name)
        return Case(
            name=f"convolution(dtype={dtype_name}, in={in_shape}, w={w_shape}, groups={groups}) [{note}]",
            op=op,
            run_torch=lambda: torch_call(
                x_t, w_t, b_t, list(stride), list(padding), list(dilation), False, [0], groups
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, x_c, w_c, b_c, list(stride), list(padding), list(dilation), False, [0], groups
            ),
            note=note,
        )

    for dtype_name in ["float64", "float32"]:
        # mamba's exact shape: depthwise causal 1-D conv -- groups ==
        # in_channels == out_channels, padding == kernel_size - 1 (both
        # sides; the model slices `[..., :seq_len]` afterwards to keep it
        # causal, which is the caller's job, not this kernel's).
        cases.append(
            make(
                dtype_name,
                [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 0.5, 2.0, -2.0, 1.0, 0.0, 1.0, -1.0, 2.0, 3.0],
                (1, 3, 5),
                [1.0, -1.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, -1.0, 1.0, 0.0, 2.0],
                (3, 1, 4),
                [0.1, -0.2, 0.3],
                (1,), (3,), (1,), 3,
                "depthwise causal 1-D conv, mamba's exact shape (padding=kernel-1, groups=channels)",
            )
        )
        # groups=1 (ordinary, non-depthwise conv), no bias -- broader
        # coverage of `conv1d`'s groups argument than mamba alone exercises.
        cases.append(
            make(
                dtype_name,
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                (1, 2, 3),
                [1.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
                (2, 2, 3),
                None,
                (1,), (1,), (1,), 1,
                "groups=1, no bias",
            )
        )

    x_t, x_c = pair_from_flat(torch_module, c_module, list(range(1, 16)), (1, 3, 5), "float32")
    w_t, w_c = pair_from_flat(torch_module, c_module, list(range(1, 13)), (3, 1, 4), "float32")

    # transposed=True: torch computes it (the weight-shape convention it
    # uses is compatible here too, measured), the shim refuses by name --
    # not measured/needed by mamba or mixtral, so a documented gap rather
    # than a guess.
    cases.append(
        Case(
            name="convolution(transposed=True) [c_error -- torch computes, shim refuses]",
            op=op,
            run_torch=lambda: torch_call(x_t, w_t, None, [1], [3], [1], True, [0], 3),
            run_c=lambda: c_module._aten_dispatch(op, x_c, w_c, None, [1], [3], [1], True, [0], 3),
            expect="c_error",
            note="transposed convolution is not implemented in torch._C shim",
        )
    )
    # A non-zero output_padding: also computes on real torch (measured),
    # also refused here for the same reason.
    cases.append(
        Case(
            name="convolution(output_padding=[1]) [c_error -- torch computes, shim refuses]",
            op=op,
            run_torch=lambda: torch_call(x_t, w_t, None, [1], [3], [1], False, [1], 3),
            run_c=lambda: c_module._aten_dispatch(op, x_c, w_c, None, [1], [3], [1], False, [1], 3),
            expect="c_error",
            note="a non-zero output_padding is not implemented in torch._C shim",
        )
    )
    # A 2-D input: both sides refuse (measured on real torch: "Expected
    # 3-dimensional input for 3-dimensional weight").
    x2d_t, x2d_c = pair_from_flat(torch_module, c_module, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), "float32")
    cases.append(
        Case(
            name="convolution(2-D input) [both_error -- neither side does 2-D here]",
            op=op,
            run_torch=lambda: torch_call(x2d_t, w_t, None, [1], [3], [1], False, [0], 3),
            run_c=lambda: c_module._aten_dispatch(op, x2d_c, w_c, None, [1], [3], [1], False, [0], 3),
            expect="both_error",
            note="torch: 'Expected 3-dimensional input for 3-dimensional weight [3, 1, 4], "
                 "but got 2-dimensional input of size [2, 3] instead'",
        )
    )
    return cases


def zeros_like_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.zeros_like.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "int64", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        cases.append(
            Case(
                name=f"zeros_like(dtype={dtype_name}, shape=(2,3)) [dtype/shape inherited from self]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                note="mamba seeds the SSM's running state this way",
            )
        )
    for dtype_name in dt.DEFAULT_DTYPES:
        t_dt = dt.torch_dtype(torch_module, dtype_name)
        c_dt = dt.c_dtype(c_module, dtype_name)
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4], (2, 2), "float32")
        cases.append(
            Case(
                name=f"zeros_like(self_dtype=float32, dtype_override={dtype_name})",
                op=op,
                run_torch=lambda a_t=a_t, t_dt=t_dt: torch_call(a_t, dtype=t_dt),
                run_c=lambda a_c=a_c, c_dt=c_dt: c_module._aten_dispatch(op, a_c, dtype=c_dt),
                note="explicit dtype override beats the self tensor's dtype",
            )
        )
    return cases


def empty_like_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.empty_like.default"
    cases: list[Case] = []
    for dtype_name in ["float64", "float32", "int64", "uint8"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 2, 3, 4, 5, 6], (2, 3), dtype_name)
        cases.append(
            Case(
                name=f"empty_like(dtype={dtype_name}, shape=(2,3))",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c),
                value_check=_dtype_shape_only_check,
                note="uninitialized memory -- only dtype/shape are meaningful (mixtral's routing "
                     "immediately overwrites every element via index_put_ before reading it)",
            )
        )
    return cases


def ge_scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.ge.Scalar"
    cases: list[Case] = []
    for dtype_name in _CMP_DTYPES:
        cases.append(
            _binary_scalar_case(
                torch_module, c_module, op, torch_call, dtype_name,
                [1, 2, 3, 4], (2, 2), 3,
                "x >= 3 -- mixtral's sentinel_mask = expert_ids_g >= num_experts",
            )
        )
    cases.append(
        Case(
            name="ge(float32, nan >= 1.0) [every comparison against NaN is false]",
            op=op,
            run_torch=lambda: torch_call(
                _pair(torch_module, c_module, [float("nan"), 1.0], (2,), "float32")[0], 1.0
            ),
            run_c=lambda: c_module._aten_dispatch(
                op, _pair(torch_module, c_module, [float("nan"), 1.0], (2,), "float32")[1], 1.0
            ),
            note="NaN is not >= anything, including itself",
        )
    )
    return cases


def floor_divide_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.floor_divide.default"
    cases: list[Case] = []

    # Tensor,Tensor -- mixed sign, re-measured against real torch: floors
    # toward -inf like Python's `//`, not toward zero like C's truncation.
    for dtype_name in ["int64", "int32"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [-7, -6, -1, 0, 1, 6, 7], (7,), dtype_name)
        b_t, b_c = pair_from_flat(torch_module, c_module, [2, 2, 2, 2, 2, 2, 2], (7,), dtype_name)
        cases.append(
            Case(
                name=f"floor_divide(dtype={dtype_name}, tensor/tensor, mixed-sign self / positive divisor)",
                op=op,
                run_torch=lambda a_t=a_t, b_t=b_t: torch_call(a_t, b_t),
                run_c=lambda a_c=a_c, b_c=b_c: c_module._aten_dispatch(op, a_c, b_c),
                note="[-7,-6,-1,0,1,6,7] // 2 == [-4,-3,-1,0,0,3,3], measured -- floors toward "
                     "-inf, does not truncate toward zero",
            )
        )
        c_t, c_c = pair_from_flat(torch_module, c_module, [-7, -6, -1, 0, 1, 6, 7], (7,), dtype_name)
        d_t, d_c = pair_from_flat(torch_module, c_module, [-2, -2, -2, -2, -2, -2, -2], (7,), dtype_name)
        cases.append(
            Case(
                name=f"floor_divide(dtype={dtype_name}, tensor/tensor, mixed-sign self / negative divisor)",
                op=op,
                run_torch=lambda c_t=c_t, d_t=d_t: torch_call(c_t, d_t),
                run_c=lambda c_c=c_c, d_c=d_c: c_module._aten_dispatch(op, c_c, d_c),
                note="[-7,-6,-1,0,1,6,7] // -2 == [3,3,0,0,-1,-3,-4], measured",
            )
        )

    # Tensor,Scalar -- mixtral's exact call shape (`perm // num_top_k`, a
    # bare Python int reaching the (Tensor, Tensor) overload's `other` slot
    # -- see the kernel's own doc comment).
    e_t, e_c = pair_from_flat(torch_module, c_module, [-7, -6, -1, 0, 1, 6, 7], (7,), "int64")
    cases.append(
        Case(
            name="floor_divide(dtype=int64, tensor // python-int scalar)",
            op=op,
            run_torch=lambda: torch_call(e_t, 2),
            run_c=lambda: c_module._aten_dispatch(op, e_c, 2),
            note="mixtral's exact call shape: perm // num_top_k",
        )
    )

    # Floating dtype: division by zero is not an error (IEEE inf/-inf/nan).
    f_t, f_c = pair_from_flat(torch_module, c_module, [1.0, -1.0, 0.0], (3,), "float32")
    g_t, g_c = pair_from_flat(torch_module, c_module, [0.0, 0.0, 0.0], (3,), "float32")
    cases.append(
        Case(
            name="floor_divide(dtype=float32, division by zero -- inf/-inf/nan, not an error)",
            op=op,
            run_torch=lambda: torch_call(f_t, g_t),
            run_c=lambda: c_module._aten_dispatch(op, f_c, g_c),
            note="measured: [1.,-1.,0.] // [0.,0.,0.] == [inf,-inf,nan]",
        )
    )

    # Integral dtype: division by zero raises, matching upstream's exact
    # message (measured: RuntimeError('ZeroDivisionError')).
    h_t, h_c = pair_from_flat(torch_module, c_module, [1, 0, -1], (3,), "int64")
    i_t, i_c = pair_from_flat(torch_module, c_module, [2, 0, -2], (3,), "int64")
    cases.append(
        Case(
            name="floor_divide(dtype=int64, division by zero) [raises]",
            op=op,
            run_torch=lambda: torch_call(h_t, i_t),
            run_c=lambda: c_module._aten_dispatch(op, h_c, i_c),
            expect="both_error",
            note="torch: RuntimeError('ZeroDivisionError'), same wording reproduced here",
        )
    )
    return cases


def histc_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.histc.default"
    cases: list[Case] = []

    # mixtral's exact call shape: float32 input, bins=num_experts, min=0,
    # max=num_experts-1.
    a_t, a_c = pair_from_flat(torch_module, c_module, [0.0, 1.0, 2.0, 3.0, 3.0, -1.0, 4.0], (7,), "float32")
    cases.append(
        Case(
            name="histc(dtype=float32, bins=4, min=0, max=3) [out-of-range values dropped]",
            op=op,
            run_torch=lambda: torch_call(a_t, 4, 0, 3),
            run_c=lambda: c_module._aten_dispatch(op, a_c, 4, 0, 3),
            note="-1.0 and 4.0 fall outside [0,3] and are not counted, measured",
        )
    )

    for dtype_name in ["float64", "float32", "float16"]:
        b_t, b_c = pair_from_flat(torch_module, c_module, [1.0, 2.0, 3.0], (3,), dtype_name)
        cases.append(
            Case(
                name=f"histc(dtype={dtype_name}, bins=4, min=0, max=3)",
                op=op,
                run_torch=lambda b_t=b_t: torch_call(b_t, 4, 0, 3),
                run_c=lambda b_c=b_c: c_module._aten_dispatch(op, b_c, 4, 0, 3),
                note="output dtype follows the input's, not a fixed default",
            )
        )

    # min == max: falls back to the data's own [min, max] -- re-measured on
    # real torch with a *non-zero* equal bound too, not only min=max=0.
    c_t, c_c = pair_from_flat(torch_module, c_module, [1.0, 2.0, 3.0], (3,), "float32")
    cases.append(
        Case(
            name="histc(min==max==5, non-zero) [falls back to the data's own min/max]",
            op=op,
            run_torch=lambda: torch_call(c_t, 4, 5, 5),
            run_c=lambda: c_module._aten_dispatch(op, c_c, 4, 5, 5),
            note="measured: ignores the literal value 5 entirely, uses the data's range [1,3]",
        )
    )

    # Degenerate: the data itself is constant, so even the auto-detected
    # range collapses -- falls back a second time to [value-1, value+1].
    d_t, d_c = pair_from_flat(torch_module, c_module, [2.0, 2.0, 2.0], (3,), "float32")
    cases.append(
        Case(
            name="histc(constant data, min=max=0) [degenerate range -> [value-1, value+1]]",
            op=op,
            run_torch=lambda: torch_call(d_t, 4, 0, 0),
            run_c=lambda: c_module._aten_dispatch(op, d_c, 4, 0, 0),
            note="measured: [2,2,2] bins=4 -> [0,0,3,0], i.e. range [1,3], not [2,2]",
        )
    )

    # Refusals, both with upstream's exact wording (measured).
    e_t, e_c = pair_from_flat(torch_module, c_module, [1.0, 2.0], (2,), "float32")
    cases.append(
        Case(
            name="histc(bins=0) [refused]",
            op=op,
            run_torch=lambda: torch_call(e_t, 0, 0, 3),
            run_c=lambda: c_module._aten_dispatch(op, e_c, 0, 0, 3),
            expect="both_error",
            note="torch: 'bins must be > 0, but got 0 for dimension 0'",
        )
    )
    f_t, f_c = pair_from_flat(torch_module, c_module, [1.0, 2.0], (2,), "float32")
    cases.append(
        Case(
            name="histc(min > max, explicit) [refused]",
            op=op,
            run_torch=lambda: torch_call(f_t, 4, 3, 1),
            run_c=lambda: c_module._aten_dispatch(op, f_c, 4, 3, 1),
            expect="both_error",
            note="torch: 'torch.histc: max must be larger than min'",
        )
    )

    # Refused dtype: no CPU kernel for integral input, matching upstream's
    # exact wording.
    g_t, g_c = pair_from_flat(torch_module, c_module, [1, 2, 3], (3,), "int64")
    cases.append(
        Case(
            name="histc(dtype=int64) [refused -- histc has no CPU kernel for integral dtypes]",
            op=op,
            run_torch=lambda: torch_call(g_t, 4, 0, 3),
            run_c=lambda: c_module._aten_dispatch(op, g_c, 4, 0, 3),
            expect="both_error",
            note="torch: NotImplementedError('\"histogram_cpu\" not implemented for \\'Long\\'')",
        )
    )
    return cases


def clamp__default_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.clamp_.default"
    cases: list[Case] = []

    # mixtral's exact call shape: max only, min absent (None).
    for dtype_name in ["int64", "int32", "float32", "float64"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1, 5, 10, -3], (4,), dtype_name)
        cases.append(
            Case(
                name=f"clamp_(dtype={dtype_name}, min=None, max=3) [mixtral's exact call shape]",
                op=op,
                run_torch=lambda a_t=a_t: torch_call(a_t, None, 3),
                run_c=lambda a_c=a_c: c_module._aten_dispatch(op, a_c, None, 3),
                note="in-place: compares the mutated receiver clamp_ returns",
            )
        )

    for dtype_name in ["int64", "float32"]:
        b_t, b_c = pair_from_flat(torch_module, c_module, [1, 5, 10, -3], (4,), dtype_name)
        cases.append(
            Case(
                name=f"clamp_(dtype={dtype_name}, min=2, max=8)",
                op=op,
                run_torch=lambda b_t=b_t: torch_call(b_t, 2, 8),
                run_c=lambda b_c=b_c: c_module._aten_dispatch(op, b_c, 2, 8),
                note="both bounds present",
            )
        )

    # min > max: NOT refused -- collapses to a constant (measured formula
    # min(max(x,min_val),max_val), applied unconditionally regardless of
    # whether min_val <= max_val).
    c_t, c_c = pair_from_flat(torch_module, c_module, [1, 5, 10, -3], (4,), "int64")
    cases.append(
        Case(
            name="clamp_(min=8, max=2) [min > max collapses to a constant, NOT refused]",
            op=op,
            run_torch=lambda: torch_call(c_t, 8, 2),
            run_c=lambda: c_module._aten_dispatch(op, c_c, 8, 2),
            note="measured: [1,5,10,-3].clamp_(min=8,max=2) == [2,2,2,2]",
        )
    )

    # NaN propagates through both the floor and the ceiling.
    d_t, d_c = pair_from_flat(torch_module, c_module, [float("nan"), 1.0, -1.0], (3,), "float32")
    cases.append(
        Case(
            name="clamp_(dtype=float32, [nan,1.,-1.], min=0, max=2) [NaN propagates]",
            op=op,
            run_torch=lambda: torch_call(d_t, 0.0, 2.0),
            run_c=lambda: c_module._aten_dispatch(op, d_c, 0.0, 2.0),
            note="measured: [nan,1.,-1.].clamp_(0,2) == [nan,1.,0.]",
        )
    )

    # Both bounds absent: refused -- NOT a no-op. Measured on real torch
    # (naively guessable as "nothing to do, return self unchanged", and
    # wrong): `tensor.clamp_(None, None)` raises "At least one of 'min' or
    # 'max' must not be None".
    e_t, e_c = pair_from_flat(torch_module, c_module, [1.0, 2.0, 3.0], (3,), "float32")
    cases.append(
        Case(
            name="clamp_(min=None, max=None) [refused, NOT a no-op]",
            op=op,
            run_torch=lambda: torch_call(e_t, None, None),
            run_c=lambda: c_module._aten_dispatch(op, e_c, None, None),
            expect="both_error",
            note="torch: \"torch.clamp: At least one of 'min' or 'max' must not be None\"",
        )
    )

    # Refused: a float bound against an integral receiver, regardless of
    # whether the bound's value is exactly representable.
    f_t, f_c = pair_from_flat(torch_module, c_module, [1, 2, 3], (3,), "int32")
    cases.append(
        Case(
            name="clamp_(dtype=int32, max=2.0 [a FLOAT]) [refused, even though 2.0 is exact]",
            op=op,
            run_torch=lambda: torch_call(f_t, None, 2.0),
            run_c=lambda: c_module._aten_dispatch(op, f_c, None, 2.0),
            expect="both_error",
            note="torch: \"result type Float can't be cast to the desired output type Int\"",
        )
    )
    return cases


def div__tensor_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.div_.Tensor"
    cases: list[Case] = []

    for dtype_name in ["float64", "float32", "float16", "bfloat16"]:
        a_t, a_c = pair_from_flat(torch_module, c_module, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), dtype_name)
        b_t, b_c = pair_from_flat(torch_module, c_module, [2.0, 4.0, 5.0], (3, 1), dtype_name)
        cases.append(
            Case(
                name=f"div_(dtype={dtype_name}, other (3,1) broadcasts into receiver (3,2))",
                op=op,
                run_torch=lambda a_t=a_t, b_t=b_t: torch_call(a_t, b_t),
                run_c=lambda a_c=a_c, b_c=b_c: c_module._aten_dispatch(op, a_c, b_c),
                note="mixtral's exact call shape: "
                     "top_k_weights.div_(top_k_weights.sum(-1, keepdim=True))",
            )
        )

    c_t, c_c = pair_from_flat(torch_module, c_module, [1.0, -1.0, 0.0, 5.0], (2, 2), "float32")
    d_t, d_c = pair_from_flat(torch_module, c_module, [0.0, 0.0, 0.0, 2.0], (2, 2), "float32")
    cases.append(
        Case(
            name="div_(dtype=float32, division by zero -- inf/-inf/nan, not an error)",
            op=op,
            run_torch=lambda: torch_call(c_t, d_t),
            run_c=lambda: c_module._aten_dispatch(op, c_c, d_c),
            note="in-place true division, IEEE 0-division rules, same as div.Tensor",
        )
    )

    # Refused: an integral receiver cannot hold the true-division result.
    e_t, e_c = pair_from_flat(torch_module, c_module, [1, 2, 3], (3,), "int64")
    f_t, f_c = pair_from_flat(torch_module, c_module, [2, 2, 2], (3,), "int64")
    cases.append(
        Case(
            name="div_(dtype=int64) [refused -- true division can't write back into int64]",
            op=op,
            run_torch=lambda: torch_call(e_t, f_t),
            run_c=lambda: c_module._aten_dispatch(op, e_c, f_c),
            expect="both_error",
            note="torch: \"result type Float can't be cast to the desired output type Long\"",
        )
    )

    # Refused: other's shape does not broadcast INTO the receiver's own
    # shape -- in-place can only shrink what's applied, never grow self.
    g_t, g_c = pair_from_flat(torch_module, c_module, [1.0, 2.0, 3.0], (3, 1), "float32")
    h_t, h_c = pair_from_flat(torch_module, c_module, [1.0, 2.0], (1, 2), "float32")
    cases.append(
        Case(
            name="div_(receiver (3,1), other (1,2)) [refused -- other would grow the receiver]",
            op=op,
            run_torch=lambda: torch_call(g_t, h_t),
            run_c=lambda: c_module._aten_dispatch(op, g_c, h_c),
            expect="both_error",
            note="torch: \"output with shape [3, 1] doesn't match the broadcast shape [3, 2]\"",
        )
    )
    return cases


def masked_fill__scalar_cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.masked_fill_.Scalar"
    cases: list[Case] = []
    # Same bool-mask construction workaround `masked_fill_cases` documents
    # above -- `_C`'s `_tensor_from_flat` refuses to build a bool tensor
    # directly, so the mask is built from an int 0/1 flat list with an
    # explicit `dtype=c_module.bool`.
    a_flat, a_shape = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)
    mask_flat = [True, False, True, False, True, False]
    for dtype_name in ["float64", "float32", "float16", "bfloat16"]:
        big = _FLOAT_ADD_MAGNITUDE[dtype_name]
        for value, note in [
            (0.0, "zero fill"),
            (-big, "large negative -- sentinel masking, mixtral's exact use"),
        ]:
            cases.append(
                Case(
                    name=f"masked_fill_(dtype={dtype_name}, value={value}) [in-place]",
                    op=op,
                    run_torch=lambda dtype_name=dtype_name, value=value: torch_call(
                        torch_module.tensor(a_flat, dtype=dt.torch_dtype(torch_module, dtype_name)).reshape(
                            list(a_shape)
                        ),
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

    # Broadcasting mask -- mixtral's exact shape: `sentinel_mask` is
    # `(N,1)`, the receiver is `(N,hidden_dim)`.
    b_flat = [float(v) for v in range(1, 9)]
    b_shape = (4, 2)
    bmask_flat = [True, False, True, False]
    cases.append(
        Case(
            name="masked_fill_(dtype=float32, mask (4,1) broadcasts into receiver (4,2))",
            op=op,
            run_torch=lambda: torch_call(
                torch_module.tensor(b_flat).reshape(list(b_shape)),
                torch_module.tensor(bmask_flat).reshape([4, 1]),
                -1.0,
            ),
            run_c=lambda: c_module._aten_dispatch(
                op,
                c_module._tensor_from_flat(b_flat, list(b_shape), dtype=c_module.float32),
                c_module._tensor_from_flat([int(v) for v in bmask_flat], [4, 1], dtype=c_module.bool),
                -1.0,
            ),
            note="mixtral's exact call shape: "
                 "selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)",
        )
    )
    return cases


def index_put__cases(torch_module, c_module, torch_call) -> list[Case]:
    op = "aten.index_put_.default"
    cases: list[Case] = []

    # mixtral's exact call shape: inv_perm[perm] = torch.arange(perm.size(0))
    # -- a single int64 index tensor, no accumulate, 1-D self/index/values.
    for dtype_name in ["int64", "int32", "float32"]:
        zero_flat = [0.0] * 5 if dtype_name == "float32" else [0] * 5
        values_flat = [10.0, 20.0, 30.0, 40.0, 50.0] if dtype_name == "float32" else [10, 20, 30, 40, 50]
        self_t, self_c = pair_from_flat(torch_module, c_module, zero_flat, (5,), dtype_name)
        idx_t, idx_c = pair_from_flat(torch_module, c_module, [4, 3, 2, 1, 0], (5,), "int64")
        values_t, values_c = pair_from_flat(torch_module, c_module, values_flat, (5,), dtype_name)
        cases.append(
            Case(
                name=f"index_put_(dtype={dtype_name}, self[index]=values, no accumulate)",
                op=op,
                run_torch=lambda self_t=self_t, idx_t=idx_t, values_t=values_t: torch_call(
                    self_t, [idx_t], values_t, False
                ),
                run_c=lambda self_c=self_c, idx_c=idx_c, values_c=values_c: c_module._aten_dispatch(
                    op, self_c, [idx_c], values_c, False
                ),
                note="mixtral's exact call shape: inv_perm[perm] = torch.arange(perm.size(0))",
            )
        )

    # Repeated index -- last write wins (measured), the same rule
    # `scatter.src`'s doc comment already documents and this reuses.
    self2_t, self2_c = pair_from_flat(torch_module, c_module, [0, 0, 0], (3,), "int64")
    idx2_t, idx2_c = pair_from_flat(torch_module, c_module, [0, 0, 0], (3,), "int64")
    values2_t, values2_c = pair_from_flat(torch_module, c_module, [1, 2, 3], (3,), "int64")
    cases.append(
        Case(
            name="index_put_(repeated index) [last write wins]",
            op=op,
            run_torch=lambda: torch_call(self2_t, [idx2_t], values2_t, False),
            run_c=lambda: c_module._aten_dispatch(op, self2_c, [idx2_c], values2_c, False),
            note="measured: index [0,0,0] with values [1,2,3] leaves self[0] == 3",
        )
    )

    # Refused: accumulate=True is not measured/implemented.
    self3_t, self3_c = pair_from_flat(torch_module, c_module, [0, 0, 0], (3,), "int64")
    idx3_t, idx3_c = pair_from_flat(torch_module, c_module, [0, 1, 2], (3,), "int64")
    values3_t, values3_c = pair_from_flat(torch_module, c_module, [1, 2, 3], (3,), "int64")
    cases.append(
        Case(
            name="index_put_(accumulate=True) [c_error -- torch computes, shim refuses]",
            op=op,
            run_torch=lambda: torch_call(self3_t, [idx3_t], values3_t, True),
            run_c=lambda: c_module._aten_dispatch(op, self3_c, [idx3_c], values3_c, True),
            expect="c_error",
            note="torch computes accumulate=True; the shim refuses it by name (not measured/needed)",
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
    # The four docs/GPT2.md measured a 2-layer GPT-2 stopping on, after the
    # Llama-shaped work had already cleared everything else.
    "aten.addmm.default": addmm_cases,
    "aten.native_layer_norm.default": native_layer_norm_cases,
    "aten.split.Tensor": split_cases,
    "aten.tanh.default": tanh_cases,
    # What widening past the Llama/GPT-2 family asks for (docs/ARCH.md).
    "aten.gelu.default": gelu_cases,
    "aten.gather.default": gather_cases,
    "aten.zero_.default": zero__cases,
    # The four docs/ARCH.md measured falcon, gptj, bloom and mpt all missing --
    # the same four, in all four models -- plus the next two by architecture
    # count (docs/OPS4.md).
    "aten.le.Tensor": le_tensor_cases,
    "aten.scalar_tensor.default": scalar_tensor_cases,
    "aten.where.self": where_self_cases,
    "aten.permute.default": permute_cases,
    "aten.stack.default": stack_cases,
    "aten.relu.default": relu_cases,
    # The five ops docs/TAIL.md needed to open falcon/bloom/gpt_bigcode.
    "aten.baddbmm.default": baddbmm_cases,
    "aten.split_with_sizes.default": split_with_sizes_cases,
    "aten._safe_softmax.default": safe_softmax_cases,
    "aten.add_.Tensor": add__tensor_cases,
    "aten.mul.Scalar": mul_scalar_cases,
    # docs/KERNELS.md: the in-place sibling `F.relu(..., inplace=True)` traces
    # to, landed as its own kernel (was a measured gap, docs/SPELLINGS.md §6.6).
    "aten.relu_.default": relu__cases,
    # mamba and mixtral, the last two of the 20 measured architectures
    # (docs/OPS4.md) with anything unimplemented.
    "aten.exp.default": exp_cases,
    "aten.softplus.default": softplus_cases,
    "aten.convolution.default": convolution_cases,
    "aten.zeros_like.default": zeros_like_cases,
    "aten.empty_like.default": empty_like_cases,
    "aten.ge.Scalar": ge_scalar_cases,
    "aten.floor_divide.default": floor_divide_cases,
    "aten.histc.default": histc_cases,
    "aten.clamp_.default": clamp__default_cases,
    "aten.div_.Tensor": div__tensor_cases,
    "aten.masked_fill_.Scalar": masked_fill__scalar_cases,
    "aten.index_put_.default": index_put__cases,
}
