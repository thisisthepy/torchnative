#!/usr/bin/env python3
"""Golden comparison harness: does `torch._C` compute the same answer as
upstream torch?

DESIGN.md §5 names the risk this exists to catch: rebuilding dtype
promotion, broadcasting, and view/stride aliasing on top of candle can
silently diverge from torch's answers, and linking + importing proves
nothing about whether the numbers agree. This harness asks `_C` what it has
implemented (`_C._aten_implemented()`) -- never a hardcoded list, so
coverage grows automatically as rust/torch_c grows -- and for every op it
finds, runs a battery of dtype/shape/boundary-value cases against both
`_C._aten_dispatch(...)` and the matching `torch.ops.aten.*` overload,
comparing dtype, shape, and value.

Usage
-----
Needs a Python environment with real upstream torch installed (this repo's
scratch venv has it: /Volumes/macMini/caches/spike-venv, torch 2.13.0), and
a built host artefact for `_C` (default host build location per
docs/TORCH_C.md §7: /Volumes/macMini/caches/cargo-target/release/lib_C.dylib,
or built fresh via `cd rust/torch_c && ./pytests/run.sh` first).

    /Volumes/macMini/caches/spike-venv/bin/python tools/golden/compare.py
    /Volumes/macMini/caches/spike-venv/bin/python tools/golden/compare.py \
        --artefact /path/to/lib_C.dylib -v

Exit code is 0 iff every case matched its expectation; non-zero (1)
otherwise. Never grep stdout for a success marker -- this project's own
traceback text has, twice, printed source lines that accidentally matched a
success grep. Read the exit code.

Self-test: `--inject-fault {value,shape,dtype}` deliberately corrupts one
real, already-computed `_C` result before comparison, to prove the
comparator actually rejects a wrong answer rather than rubber-stamping
everything. See the final report for what this caught.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dtypes as dt
from cases import CASE_BUILDERS, Case
from loader import ShimLoadError, load_shim, resolve_torch_overload


@dataclass
class Outcome:
    case: Case
    passed: bool
    detail: str


def _flatten(x) -> list:
    if isinstance(x, list):
        out: list = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]


def _values_close(torch_nested, c_nested, atol: float, rtol: float) -> tuple[bool, str]:
    ta, ca = _flatten(torch_nested), _flatten(c_nested)
    if len(ta) != len(ca):
        return False, f"flattened length differs: torch has {len(ta)}, c has {len(ca)}"
    for i, (x, y) in enumerate(zip(ta, ca)):
        if isinstance(x, bool) or isinstance(y, bool):
            if x != y:
                return False, f"index {i}: torch={x!r} c={y!r}"
            continue
        if isinstance(x, int) and isinstance(y, int):
            if x != y:
                return False, f"index {i}: torch={x!r} c={y!r} (integer, exact match required)"
            continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            if x != y:
                return False, f"index {i}: torch={x!r} c={y!r}"
            continue
        if math.isnan(xf) or math.isnan(yf):
            if not (math.isnan(xf) and math.isnan(yf)):
                return False, f"index {i}: torch={x!r} c={y!r} (NaN mismatch)"
            continue
        if math.isinf(xf) or math.isinf(yf):
            if xf != yf:
                return False, f"index {i}: torch={x!r} c={y!r} (inf mismatch)"
            continue
        if not math.isclose(xf, yf, rel_tol=rtol, abs_tol=atol):
            return (
                False,
                f"index {i}: torch={x!r} c={y!r} "
                f"(|diff|={abs(xf - yf):.6g} exceeds atol={atol} rtol={rtol})",
            )
    return True, ""


def _summarize(result) -> str:
    try:
        return f"{result.tolist()!r} dtype={dt.dtype_name(result.dtype)} shape={tuple(result.shape)}"
    except Exception:
        return repr(result)


def _run_one(case: Case, inject_fault: str | None) -> Outcome:
    t_res = t_exc = c_res = c_exc = None
    try:
        t_res = case.run_torch()
    except Exception as e:  # noqa: BLE001 - deliberately broad, this is a probe
        t_exc = e
    try:
        c_res = case.run_c()
    except Exception as e:  # noqa: BLE001
        c_exc = e

    fault_tag = ""
    if inject_fault and t_res is not None and c_res is not None:
        c_res, fault_tag = _corrupt(c_res, inject_fault)

    t_ok, c_ok = t_exc is None, c_exc is None

    if case.expect == "both_error":
        if t_ok or c_ok:
            return Outcome(
                case,
                False,
                f"expected both sides to refuse; torch_ok={t_ok} c_ok={c_ok} "
                f"(torch={_summarize(t_res) if t_ok else t_exc!r}, "
                f"c={_summarize(c_res) if c_ok else c_exc!r})",
            )
        return Outcome(case, True, f"both refused as expected (torch={t_exc!r}, c={c_exc!r})")

    if case.expect == "c_error":
        if not (t_ok and not c_ok):
            extra = " -- gap appears CLOSED: both sides now succeed, promote this case to expect=match and diff real values" if (t_ok and c_ok) else ""
            return Outcome(
                case,
                False,
                f"expected torch to succeed and c to refuse (known gap: {case.note}); "
                f"got torch_ok={t_ok} c_ok={c_ok}{extra}",
            )
        return Outcome(case, True, f"known gap still present: c refused ({c_exc!r})")

    if case.expect == "torch_error":
        if not (c_ok and not t_ok):
            extra = " -- gap appears CLOSED: both sides now succeed" if (t_ok and c_ok) else ""
            return Outcome(
                case,
                False,
                f"expected c to succeed and torch to refuse (known gap: {case.note}); "
                f"got torch_ok={t_ok} c_ok={c_ok}{extra}",
            )
        return Outcome(case, True, f"known gap still present: torch refused ({t_exc!r})")

    # expect == "match"
    if t_ok != c_ok:
        refusing_side = "torch" if not t_ok else "c"
        computing_side = "c" if not t_ok else "torch"
        exc = t_exc if not t_ok else c_exc
        val = c_res if not t_ok else t_res
        return Outcome(
            case,
            False,
            f"SILENT DIVERGENCE: {refusing_side} raised {exc!r} but {computing_side} "
            f"computed a value: {_summarize(val)}",
        )
    if not t_ok and not c_ok:
        return Outcome(
            case, True, f"both refused (unregistered as expected, but consistent): torch={t_exc!r} c={c_exc!r}"
        )

    prefix = f"[{fault_tag}] " if fault_tag else ""

    t_dtype = dt.dtype_name(t_res.dtype)
    c_dtype = dt.dtype_name(c_res.dtype)
    if t_dtype != c_dtype:
        return Outcome(
            case,
            False,
            f"{prefix}dtype mismatch: torch={t_dtype} c={c_dtype} "
            f"(torch value={t_res.tolist()!r}, c value={c_res.tolist()!r})",
        )

    t_shape = tuple(int(x) for x in t_res.shape)
    c_shape = tuple(int(x) for x in c_res.shape)
    if t_shape != c_shape:
        return Outcome(case, False, f"{prefix}shape mismatch: torch={t_shape} c={c_shape}")

    tol = dt.tolerance_for(t_dtype)
    ok, detail = _values_close(t_res.tolist(), c_res.tolist(), tol.atol, tol.rtol)
    if not ok:
        return Outcome(
            case,
            False,
            f"{prefix}value mismatch ({detail}); torch={t_res.tolist()!r} c={c_res.tolist()!r} dtype={t_dtype}",
        )
    suffix = f" [{fault_tag} did not get caught -- COMPARATOR BUG]" if fault_tag else ""
    return Outcome(case, True, f"dtype={t_dtype} shape={t_shape}{suffix}")


def _corrupt(result, mode: str) -> tuple[Any, str]:
    """Only used by --inject-fault, to prove the comparator rejects a wrong
    answer. Mutates a copy of an already-correct `_C` result."""
    if mode == "value":
        flat = _flatten(result.tolist())
        if not flat:
            return result, ""
        return _FakeResult(_bump_first(result), result.dtype, result.shape), "INJECTED value fault"
    if mode == "shape":
        return _FakeResult(result.tolist(), result.dtype, tuple(result.shape) + (1,)), "INJECTED shape fault"
    if mode == "dtype":
        return _FakeResult(result.tolist(), _FakeDtype("torch.int16" if "int" not in str(result.dtype) else "torch.float32"), result.shape), "INJECTED dtype fault"
    return result, ""


class _FakeDtype:
    def __init__(self, s):
        self._s = s

    def __str__(self):
        return self._s


class _FakeResult:
    def __init__(self, values, dtype, shape):
        self._values = values
        self.dtype = dtype
        self.shape = shape

    def tolist(self):
        return self._values


def _bump_first(result):
    values = result.tolist()

    def bump(x):
        if isinstance(x, list):
            return [bump(v) for v in x]
        if isinstance(x, bool):
            return not x
        if isinstance(x, int):
            return x + 1000
        return float(x) + 1000.0

    flat_done = [False]

    def bump_once(x):
        if isinstance(x, list):
            return [bump_once(v) for v in x]
        if flat_done[0]:
            return x
        flat_done[0] = True
        return bump(x)

    return bump_once(values)


def run(artefact: str | None, verbose: bool, inject_fault: str | None) -> int:
    try:
        c_module = load_shim(artefact)
    except ShimLoadError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    import torch  # imported lazily so --help works even without torch installed

    implemented = list(c_module._aten_implemented())
    print(f"target={c_module._shim_target()} implemented={implemented}")
    if dt.EXCLUDED_DTYPES:
        for name, reason in dt.EXCLUDED_DTYPES.items():
            print(f"NOTE: dtype {name!r} excluded from all cases -- {reason}")

    all_outcomes: list[Outcome] = []
    missing_builders: list[str] = []
    already_injected = inject_fault is None

    for op_name in implemented:
        builder = CASE_BUILDERS.get(op_name)
        if builder is None:
            missing_builders.append(op_name)
            continue
        try:
            torch_call = resolve_torch_overload(torch, op_name)
        except ShimLoadError as e:
            all_outcomes.append(
                Outcome(
                    Case(name=f"<resolve {op_name}>", op=op_name, run_torch=lambda: None, run_c=lambda: None),
                    False,
                    f"could not resolve torch overload for {op_name!r}: {e}",
                )
            )
            continue

        cases = builder(torch, c_module, torch_call)
        for case in cases:
            fault_for_this_case = None
            if not already_injected and case.expect == "match":
                fault_for_this_case = inject_fault
                already_injected = True
            outcome = _run_one(case, fault_for_this_case)
            all_outcomes.append(outcome)
            if verbose or not outcome.passed:
                status = "PASS" if outcome.passed else "FAIL"
                print(f"{status} {case.op} :: {case.name} -- {outcome.detail}")

    for op_name in missing_builders:
        print(
            f"FAIL {op_name} :: <no case builder registered> -- "
            f"_aten_implemented() advertises this op but tools/golden/cases.py "
            f"has no entry in CASE_BUILDERS for it. Add one; do not silently skip."
        )

    total = len(all_outcomes) + len(missing_builders)
    failed = sum(1 for o in all_outcomes if not o.passed) + len(missing_builders)
    passed = total - failed

    print()
    print(f"SUMMARY: {passed}/{total} cases passed, {failed} failed, ops covered={len(implemented)}")
    if inject_fault and already_injected and inject_fault is not None:
        pass  # informational; actual detection is visible in the per-case FAIL line above

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artefact", default=None, help="path to the built lib_C.dylib/.so (default: TORCH_C_ARTEFACT env var, then docs/TORCH_C.md §7 default host path)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every case, not just failures")
    parser.add_argument(
        "--inject-fault",
        choices=["value", "shape", "dtype"],
        default=None,
        help="self-test: deliberately corrupt the first eligible result before comparison, to prove the comparator catches it. Should always exit non-zero when set.",
    )
    args = parser.parse_args()
    return run(args.artefact, args.verbose, args.inject_fault)


if __name__ == "__main__":
    raise SystemExit(main())
