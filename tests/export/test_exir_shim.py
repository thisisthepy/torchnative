"""`executorch.exir` on the shim, compared with upstream stage by stage (issue #306).

ExecuTorch step P3 (owner direction relayed by the leader, 2026-11-13/14: ExecuTorch
runs on the device at `.to(device.npu)`, JIT, programs in memory only). Before a
runtime can execute anything, exir itself has to run on torchnative's torch: capture,
the edge dialect, the passes, the serializer, the XNNPACK partitioner.

`tests/_support/exir_probe.py` runs one lowering of Linear(8, 4) + ReLU and reports
each stage. This suite runs it twice, in subprocesses: with the vendored tree on
`PYTHONPATH` (the shim) and without it (upstream torch, the oracle), and compares the
reports.

**Grade (AGENTS section 16): reaches, plus structural agreement of the program.** The
operator list and the delegate ids of the shim's program equal upstream's. Whether
the program computes upstream's numbers is step P5 (the runtime); nothing here runs
it.

**What makes these tests able to fail.** Before #227 the shim recorded Python
`torch.library` kernels and dropped them, and `import executorch.exir` stopped at
`exir/dialects/backend/_ops.py`'s kernel assertion (measured 2026-11-13). Restoring
that drop turns `test_exir_imports_on_the_shim` red. The comparison tests fail if the
shim's capture decomposes differently from upstream's (a different operator list) or
if the XNNPACK partitioner delegates nothing under the shim (an empty delegate list).
`test_the_two_reports_come_from_the_shim_and_from_upstream` catches the trap of
AGENTS section 15.2: with the vendored tree missing, both subprocesses would import
upstream and every comparison would pass against itself.

**What this cannot find.** Anything about a real model (this is one Linear), about
symbolic shapes in exir, about Android, and about the `flatc` subprocess the XNNPACK
serializer spawns (it works on a CI runner; an Android app cannot exec it, step P4).
The byte-for-byte equality of the two programs is printed, not asserted: nothing in
the specification says the bytes must be equal, only that the program is the same
graph.

executorch is a test dependency of the gate only (leader decision 2026-11-14), never
a dependency of the torchnative wheel. It lives in its own interpreter, named by
`TORCHNATIVE_EXIR_PYTHON`; without that every test here skips by name.
"""

import json
import os
import pathlib
import subprocess
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
_PROBE = str(_SUPPORT_DIR / "exir_probe.py")
_MARK = "EXIR_PROBE_JSON "

import _skip  # noqa: E402

_reports = {}


def _python():
    """The interpreter that has executorch, or None.

    `TORCHNATIVE_EXIR_PYTHON` only, the way `test_qnn.py` takes
    `TORCHNATIVE_QNN_PYTHON`. executorch is deliberately not installed in the gate
    interpreter: `tests/devices/qnn/test_qnn.py` asserts what a host WITHOUT
    executorch answers, and those tests would change meaning. On CI the variable
    points at a venv layered over the gate interpreter's packages (test.yml), so
    both sides of this suite run the gate's own torch oracle and vendored tree.
    """
    path = os.environ.get("TORCHNATIVE_EXIR_PYTHON")
    if not path or not os.path.isfile(path):
        return None
    return path


def _unavailable():
    """Why the suite cannot run here, or None."""
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        return "no vendored torch tree"
    if _python() is None:
        return (
            "TORCHNATIVE_EXIR_PYTHON unset or not a file: no interpreter with "
            "executorch (the gate builds one with executorch==1.4.1; issue #306)"
        )
    return None


def _run(side):
    """The probe's report on one side ('shim' or 'upstream'), run once per suite."""
    if side in _reports:
        return _reports[side]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_USE_RTLD_GLOBAL", None)
    if side == "shim":
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [_python(), _PROBE],
        env=env,
        cwd=str(_SUPPORT_DIR),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    # The last report: the probe prints one before its report-only strict
    # export stage and one after, so a crash there keeps the first.
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(_MARK)), None)
    if line is None:
        raise AssertionError(
            f"the {side} probe exited {proc.returncode} without a report; stderr tail:\n"
            + "\n".join(proc.stderr.splitlines()[-25:])
        )
    report = json.loads(line[len(_MARK):])
    _reports[side] = report
    return report


def _chain(record, indent="  "):
    """A stopped stage's error, where it was raised, and its cause chain, root
    last (`exir_probe._error`): exir's pass manager wraps the real failure in
    an `Exception` that only names the pass, and Dynamo wraps its own."""
    text = f"{record['error']}\n{indent}at " + f"\n{indent}at ".join(record.get("where", []))
    text += _recursion(record, indent)
    for c in record.get("causes", []):
        text += (f"\n{indent}caused by {c['error']}\n{indent}  at "
                 + f"\n{indent}  at ".join(c.get("where", [])))
        text += _recursion(c, indent + "  ")
    return text


def _recursion(record, indent):
    """A `RecursionError`'s deduplicated traceback (`exir_probe._repeat`): the
    head once, then the repeating window once with its length and count."""
    rec = record.get("recursion")
    if not rec:
        return ""
    text = (f"\n{indent}recursion of {rec['frames']} frames; the first "
            f"{len(rec['head'])}:\n{indent}  at " + f"\n{indent}  at ".join(rec["head"]))
    if rec.get("cycle") is None:
        return text + f"\n{indent}no repeating window of up to 64 frames found at the end"
    if rec["skipped"]:
        text += f"\n{indent}... {rec['skipped']} frames ..."
    text += (f"\n{indent}then a window of {rec['length']} frames repeated "
             f"{rec['count']} times:\n{indent}  at " + f"\n{indent}  at ".join(rec["cycle"]))
    return text


def _stage(side, name):
    """A completed stage's record; a stage that stopped fails the test with its name."""
    record = _run(side)["stages"][name]
    if "error" in record:
        strict = _run(side)["stages"].get("strict_export_interpolate")
        # The report-only strict export (the XNNPACK stage's first pass makes
        # the same capture) is printed with its own chain, so the wall behind
        # the one this stage names is readable in the same failure: after
        # `as_strided_` (issues #306, #380) the next one shows here first.
        if strict is None or name != "xnnpack":
            tail = ""
        elif "error" in strict:
            tail = ("\n  (strict torch.export of F.interpolate on this side stopped: "
                    + _chain(strict, "    ") + ")")
        else:
            tail = f"\n  (strict torch.export of F.interpolate on this side completed: {strict})"
        raise AssertionError(
            f"exir stage '{name}' stopped on {side} torch {_run(side)['torch']}: "
            + _chain(record) + tail
        )
    return record


def test_the_probe_names_the_repeating_window_of_a_recursion():
    """`exir_probe._repeat` on synthetic frames: the shortest window that
    repeats at least three times at the end, the head before it, the count.
    No executorch needed; runs in every gate. Before the change the probe had
    no such function and a RecursionError's record held only the message and
    the last six frames, one fragment of the cycle."""
    import exir_probe

    rep = exir_probe._repeat(["a", "b", "c"] + ["x", "y"] * 50)
    assert rep == {"head": ["a", "b", "c"], "skipped": 0, "cycle": ["x", "y"],
                   "length": 2, "count": 50}, rep
    assert exir_probe._repeat(["a", "b", "c", "d"]) is None
    # a phase-shifted tail: the window starts where the repetition starts
    rep = exir_probe._repeat(list("abcdefghijklmnopqrstuvwxyz") + ["p", "q", "r"] * 7)
    assert rep["cycle"] == ["p", "q", "r"] and rep["count"] == 7, rep
    assert rep["head"] == list("abcdefghijklmnopqrst") and rep["skipped"] == 6, rep
    # a window that repeats only twice is not a cycle
    assert exir_probe._repeat(["a"] + ["x", "y"] * 2) is None


def test_the_probe_records_a_real_recursion_errors_cycle():
    """`exir_probe._error` on a RecursionError raised by two mutually recursive
    functions: the record names the two frames as the window, and the chain
    text the suite prints shows them once with the count."""
    import exir_probe

    def ping(n):
        return pong(n)

    def pong(n):
        return ping(n)

    try:
        ping(0)
    except RecursionError as exc:
        record = exir_probe._error(exc)
    rec = record["recursion"]
    assert rec["length"] == 2 and rec["count"] >= 100, rec
    assert sorted(f.split()[-1] for f in rec["cycle"]) == ["ping", "pong"], rec["cycle"]
    assert rec["frames"] >= rec["count"] * 2, rec
    # the repetition starts right under this function: at most two head frames
    assert len(rec["head"]) <= 2 and rec["skipped"] == 0, rec
    text = _chain(record)
    assert f"window of 2 frames repeated {rec['count']} times" in text, text


def test_the_two_reports_come_from_the_shim_and_from_upstream():
    why = _unavailable()
    if why:
        _skip.skip(why)
        return
    shim, upstream = _run("shim"), _run("upstream")
    assert shim["shim"] is True, f"the shim side imported {shim['torch_file']}"
    assert upstream["shim"] is False, f"the upstream side imported {upstream['torch_file']}"
    print(f"ok   exir_shim: shim torch {shim['torch']} and upstream torch {upstream['torch']}, one probe each")


def test_exir_imports_on_the_shim():
    why = _unavailable()
    if why:
        _skip.skip(why)
        return
    _stage("shim", "import_exir")
    _stage("shim", "import_xnnpack")
    print("ok   exir_shim: executorch.exir and the XNNPACK partitioner import on the shim")


def test_to_edge_on_the_shim_gives_upstreams_edge_operators():
    why = _unavailable()
    if why:
        _skip.skip(why)
        return
    want = _stage("upstream", "edge")["targets"]
    got = _stage("shim", "edge")["targets"]
    assert want, "upstream's edge graph has no operator calls; the probe module is wrong"
    assert got == want, f"edge operators differ:\n  shim     {got}\n  upstream {want}"
    print(f"ok   exir_shim: to_edge gives upstream's {len(want)} edge operator calls: {want}")


def test_the_edge_dtype_table_for_linear_is_upstreams():
    """Wall W8 (CI run 43793269977): the XNNPACK stage keeps `aten.linear`, and
    the edge verifier reported float32 against float32 as a dtype mismatch.
    Its table holds an argument only when `str(arg.type)` is `Tensor`,
    `Optional[Tensor]` or `List[Tensor]`; the shim printed `Tensor?`, so
    `bias` was missing from the table and every call failed validation. The
    record is the verifier's own objects, compared with upstream's."""
    why = _unavailable()
    if why:
        _skip.skip(why)
        return
    want = _stage("upstream", "edge_dtypes")
    got = _stage("shim", "edge_dtypes")
    assert want["float32_everywhere_valid"] is True, f"upstream's table refuses float32: {want}"
    assert got == want, f"the edge dtype table for linear differs:\n  shim     {got}\n  upstream {want}"
    print(f"ok   exir_shim: the edge dtype table for linear holds {got['table_names']}, as upstream")


def test_clone_node_memory_format_gives_upstreams_dim_order():
    """Issue #306 (CI run 44423485167): `ConvertToSDPAPass` stopped in
    `executorch/exir/dim_order_utils.py::get_dim_order` with `Failed to
    generate dim_order for a given memory format: torch.contiguous_format` on
    `aten.clone.default(memory_format=torch.contiguous_format)`. Compares
    `kwargs['memory_format']` (`is`, `==`, `in [torch.contiguous_format]`) on
    the exported `clone` node and after `MemoryFormatOpsPass`'s
    `copy.deepcopy(kwargs)`, plus `get_dim_order`'s answer, against
    upstream's."""
    why = _unavailable()
    if why:
        _skip.skip(why)
        return
    want = _stage("upstream", "clone_memory_format")
    got = _stage("shim", "clone_memory_format")
    assert want["deepcopy_is"] is True and want["dim_order"] == [0, 1], want
    assert got == want, f"clone memory_format differs:\n  shim     {got}\n  upstream {want}"
    print(f"ok   exir_shim: clone kwargs['memory_format'] agrees with upstream before and after deepcopy: {got}")


def test_the_portable_program_has_upstreams_operators_and_no_delegate():
    why = _unavailable()
    if why:
        _skip.skip(why)
        return
    want = _stage("upstream", "portable")
    got = _stage("shim", "portable")
    assert want["operators"], "upstream's portable program lists no operators"
    assert got["operators"] == want["operators"], (got["operators"], want["operators"])
    assert got["delegates"] == want["delegates"] == [], (got["delegates"], want["delegates"])
    assert got["method"] == want["method"], (got["method"], want["method"])
    same = "equal" if got["sha256"] == want["sha256"] else "different"
    print(
        f"ok   exir_shim: portable program, operators {got['operators']}, "
        f"{got['bytes']} bytes on the shim and {want['bytes']} upstream, bytes {same}"
    )


def test_xnnpack_lowering_on_the_shim_delegates_what_upstream_delegates():
    why = _unavailable()
    if why:
        _skip.skip(why)
        return
    want = _stage("upstream", "xnnpack")
    got = _stage("shim", "xnnpack")
    assert want["delegates"] == ["XnnpackBackend"], f"upstream delegated {want['delegates']}"
    assert got["delegates"] == want["delegates"], (got["delegates"], want["delegates"])
    assert got["operators"] == want["operators"], (got["operators"], want["operators"])
    same = "equal" if got["sha256"] == want["sha256"] else "different"
    print(
        f"ok   exir_shim: XNNPACK program, delegates {got['delegates']}, operators left "
        f"{got['operators']}, {got['bytes']} bytes on the shim and {want['bytes']} upstream, bytes {same}"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        _skip._state["current"] = name
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            if name in _skip._state["skipped"]:
                print(f"SKIP test_exir_shim: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
