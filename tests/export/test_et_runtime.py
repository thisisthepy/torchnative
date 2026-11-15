"""The ExecuTorch runtime bound into `torch._C`, against upstream (issue #315, step R1).

ExecuTorch step P5, round R1 (leader decision 2026-11-14; SPEC S5.9): the
ATen-free ExecuTorch v1.4.1 runtime, the prim ops and the whole portable kernel
library, compiled from the vendored tree by `torchnative/rust/executorch_rt` and
linked into `torch._C` behind the cargo feature `executorch` on macOS arm64.

The program is produced by upstream: `tests/_support/et_program_probe.py` lowers
Linear(8, 4) + ReLU with `to_edge(...).to_executorch()` and no partitioner in the
interpreter named by `TORCHNATIVE_EXIR_PYTHON` (executorch 1.4.1 over upstream
torch), and sends the bytes back over the pipe (base64 in its JSON line), never
through a file. The same probe reports upstream's outputs for the same bytes.

**Grade (AGENTS.md section 16): agrees.** Outputs are compared element-wise
with upstream's own runtime (`portable_lib`) on the same program when that
loads in the exir interpreter, otherwise with upstream eager torch (the test
prints which). The tolerance is derived, not chosen: the p90 of upstream's own
float32-vs-float64 error on these inputs, floored at 8 ulp of each value. Exact
equality is printed, not asserted: both runtimes compile the same portable
kernel source, but with different compilers and flags, and nothing promises
the same bits.

**What makes each test able to fail.**

* `test_the_runtime_is_in_torch_c_where_the_gate_builds_it`: on macOS arm64 a
  build without the feature (`TORCHNATIVE_SHIM_FEATURES=none`, or an install
  script that stops passing `--features executorch`) has no runtime, and this
  is a FAIL there, never a skip, so the gate cannot go green without it. On
  other platforms the feature cannot exist (torch_c/Cargo.toml's target table)
  and the test skips by name with `_et_runtime_info()`'s reason.
* `test_every_generated_kernel_registration_survived_the_link`: the kernels
  register from static initializers that nothing references. Drop
  `+whole-archive` from executorch_rt/build.rs and the linker discards those
  objects: the registry is empty and this test lists all 237 missing names. The
  expected names are read from the vendored registration sources themselves,
  not from the binding.
* `test_garbage_buffers_raise_invalid_program_and_do_not_abort`: runs in a
  subprocess, so an abort (an `ET_CHECK` reached by a malformed buffer) is a
  FAIL naming the signal, not the death of this suite. Remove the shim's size
  pre-check and the short buffers raise `EtInvalidArgument` instead; drop
  `InternalConsistency` and garbage may reach the flatbuffer accessors.
* `test_truncated_programs_raise_invalid_program_and_do_not_abort`: the same,
  for prefixes of a real program (piped to the subprocess on stdin).
* `test_the_portable_program_agrees_with_upstream`: wrong kernels, a stale
  output (the second input checks the arena is rewritten), a wrong dtype
  mapping or a copy of the wrong bytes all move values past the tolerance.
* `test_operator_calls_name_the_kernels_that_ran`: the counts come from the
  profiling scope the generated kernel wrapper opens when the kernel is
  entered (`native_call_<op>`), so they are runtime evidence; they must equal
  the program's KernelCall instructions times the executions. Build without
  `ET_EVENT_TRACER_ENABLED` and they are empty.
* `test_wrong_inputs_raise_named_errors`: a wrong shape, dtype, count or type
  must raise the named class, and the method must still execute afterwards.
* `test_a_method_keeps_its_program_alive`: drop the Python program handle and
  the method must still run (the `Arc` in executorch_rt); a use-after-free
  here is a crash or garbage values.

**What this suite cannot find.** Anything about a delegate (R1 has none; R2 adds
XNNPACK), about Android (R3), about programs exir produces on the shim (the
program here comes from upstream torch), about dynamic shapes or non-contiguous
dim orders (refused by name, not exercised), about ET_CHECK aborts inside a
kernel given valid-looking but semantically wrong inputs, and about content
corruption that the flatbuffer verifier accepts. One small program exercises
three or four of the 237 kernels; the registration test checks the rest exist,
not that they compute.

executorch is a test dependency of the gate only (leader decision 2026-11-14),
never of the torchnative wheel. Without `TORCHNATIVE_EXIR_PYTHON` the tests that
need a program skip by name; the binding's own tests still run.
"""

import base64
import gc
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

_ET_VENDOR = _REPO_ROOT / "torchnative" / "rust" / "vendor" / "executorch"
_PROBE = str(_SUPPORT_DIR / "et_program_probe.py")
_MARK = "ET_PROGRAM_JSON "
_CHILD_MARK = "ET_CHILD_JSON "

import _skip  # noqa: E402

_cache = {}


def _on_macos_arm64():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _torch():
    """The shim's torch, or None when the vendored tree is not built."""
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        return None
    import torch

    assert hasattr(torch._C, "_aten_implemented"), f"upstream torch, not the shim: {torch.__file__}"
    return torch


def _runtime():
    """(torch, None) when the binding is in this build, else (None, why).

    On macOS arm64 the gate builds the runtime, so its absence raises (a FAIL)
    instead of returning a reason to skip with.
    """
    torch = _torch()
    if torch is None:
        return None, "no vendored torch tree"
    info_fn = getattr(torch._C, "_et_runtime_info", None)
    if info_fn is None:
        reason = "torch._C has no _et_runtime_info at all (a build from before issue #315)"
    else:
        info = info_fn()
        if info.get("available"):
            return torch, None
        reason = info.get("reason", repr(info))
    if _on_macos_arm64():
        raise AssertionError(
            "the ExecuTorch runtime is missing from torch._C on macOS arm64, where "
            f"install_shim.sh and run.sh build it with --features executorch: {reason}"
        )
    return None, f"ExecuTorch runtime not in this build: {reason}"


def _exir_python():
    path = os.environ.get("TORCHNATIVE_EXIR_PYTHON")
    if not path or not os.path.isfile(path):
        return None
    return path


def _report():
    """The probe's report, produced once per suite in the exir interpreter."""
    if "report" in _cache:
        return _cache["report"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [_exir_python(), _PROBE],
        env=env,
        cwd=str(_SUPPORT_DIR),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(_MARK)), None)
    if line is None:
        raise AssertionError(
            f"the program probe exited {proc.returncode} without a report; stderr tail:\n"
            + "\n".join(proc.stderr.splitlines()[-25:])
        )
    report = json.loads(line[len(_MARK):])
    if "error" in report:
        raise AssertionError(f"the program probe stopped: {report['error']}\n" + "\n".join(report.get("where", [])))
    assert report["shim"] is False, f"the probe imported the shim, not upstream: {report['torch_file']}"
    report["program_bytes"] = base64.b64decode(report["program"])
    _cache["report"] = report
    return report


def _program_or_skip():
    """(torch, report) for the tests that need a program; (None, None) after a skip."""
    torch, why = _runtime()
    if torch is None:
        _skip.skip(why)
        return None, None
    if _exir_python() is None:
        _skip.skip(
            "TORCHNATIVE_EXIR_PYTHON unset or not a file: no interpreter with executorch "
            "to produce the program and the oracle (the gate builds one with "
            "executorch==1.4.1; issue #306)"
        )
        return None, None
    return torch, _report()


def _floats(hexes):
    return [float.fromhex(h) for h in hexes]


def _rows(values, shape):
    assert len(shape) == 2, shape
    return [values[r * shape[1]:(r + 1) * shape[1]] for r in range(shape[0])]


def _input(torch, spec):
    return torch.tensor(_rows(_floats(spec["values"]), spec["shape"]), dtype=torch.float32)


def _flatten(nested):
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(_flatten(item))
        else:
            out.append(float(item))
    return out


def _ulp32(v):
    v = abs(v)
    if v == 0.0:
        return 2.0 ** -149
    _, e = math.frexp(v)
    return max(2.0 ** (e - 24), 2.0 ** -149)


def _tolerances(report):
    """Per element: max(p90 of upstream's f32-vs-f64 error, 8 ulp of the oracle value)."""
    errs = []
    for e32, e64 in zip(report["eager32"], report["eager64"]):
        errs.extend(abs(a - b) for a, b in zip(_floats(e32), _floats(e64)))
    errs.sort()
    p90 = errs[max(0, math.ceil(0.9 * len(errs)) - 1)] if errs else 0.0
    return p90, [[max(p90, 8 * _ulp32(v)) for v in _floats(o)] for o in report["oracle"]]


def _child(code, stdin_bytes=None):
    """Run `code` in a fresh interpreter on the shim; return (returncode, report, stderr)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_VENDOR_DIR, str(_SUPPORT_DIR)])
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        input=stdin_bytes,
        capture_output=True,
        timeout=600,
    )
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    line = next((ln for ln in stdout.splitlines() if ln.startswith(_CHILD_MARK)), None)
    report = json.loads(line[len(_CHILD_MARK):]) if line else None
    return proc.returncode, report, stderr


_LOAD_EACH = r"""
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented"), torch.__file__
cases = CASES
out = {}
for label, buf in cases:
    try:
        torch._C._et_load(buf)
        out[label] = {"raised": None}
    except Exception as exc:
        out[label] = {
            "raised": type(exc).__name__,
            "mro": [c.__name__ for c in type(exc).__mro__],
            "message": str(exc)[:600],
        }
print("ET_CHILD_JSON " + json.dumps(out))
"""


def _assert_invalid_program(returncode, report, stderr, what):
    if returncode < 0:
        raise AssertionError(
            f"{what}: the interpreter was killed by signal {-returncode} (an ExecuTorch "
            f"abort, not an exception); stderr tail:\n" + "\n".join(stderr.splitlines()[-15:])
        )
    assert returncode == 0 and report is not None, (
        f"{what}: child exited {returncode} without a report; stderr tail:\n"
        + "\n".join(stderr.splitlines()[-15:])
    )
    for label, r in sorted(report.items()):
        assert r["raised"] == "EtInvalidProgram", f"{what} {label}: raised {r['raised']}: {r.get('message')}"
        assert "EtError" in r["mro"] and "RuntimeError" in r["mro"], (label, r["mro"])
        assert "InvalidProgram" in r["message"], (label, r["message"])


# ---------------------------------------------------------------- the binding


def test_the_runtime_is_in_torch_c_where_the_gate_builds_it():
    torch, why = _runtime()
    if torch is None:
        _skip.skip(why)
        return
    info = torch._C._et_runtime_info()
    assert info["version"] == "1.4.1", info
    assert len(info["commit"]) == 40, info
    assert info["pal"] == "posix", info
    assert info["event_tracer"] is True, f"built without ET_EVENT_TRACER_ENABLED: {info}"
    assert info["program_verification"] is True, f"built without program verification: {info}"
    assert info["backends"] == [], f"R1 compiles no delegate, yet backends are registered: {info}"
    assert info["kernels"] > 0, f"the kernel registry is empty: {info}"
    for name in ("EtError", "EtInvalidProgram", "EtInvalidArgument", "EtOperatorMissing", "EtNotSupported"):
        cls = getattr(torch._C, name)
        assert issubclass(cls, torch._C.EtError) and issubclass(cls, RuntimeError), name
    print(
        f"ok   et_runtime: ExecuTorch {info['version']} ({info['commit'][:8]}) in torch._C on "
        f"{info['target']}: {info['kernels']} kernels, backends {info['backends']}"
    )


def test_every_generated_kernel_registration_survived_the_link():
    torch, why = _runtime()
    if torch is None:
        _skip.skip(why)
        return
    sources = {
        "portable": _ET_VENDOR / "generated" / "portable_ops_lib" / "RegisterCodegenUnboxedKernelsEverything.cpp",
        "prim_ops": _ET_VENDOR / "kernels" / "prim_ops" / "register_prim_ops.cpp",
    }
    expected = {}
    for group, path in sources.items():
        names = re.findall(r'Kernel\(\s*"([^"]+)"', path.read_text(encoding="utf-8"))
        assert names, f"no Kernel(\"...\") entries found in {path}; the pattern no longer reads it"
        expected[group] = names
    registered = torch._C._et_registered_kernels()
    have = set(registered)
    for group, names in expected.items():
        missing = sorted(set(names) - have)
        assert not missing, (
            f"{len(missing)} of {len(names)} {group} kernels are not registered at run time "
            f"(registry has {len(registered)}); their static initializers did not survive the "
            f"link (executorch_rt/build.rs, +whole-archive). First missing: {missing[:8]}"
        )
    total = sum(len(n) for n in expected.values())
    assert len(registered) == total, (
        f"registry has {len(registered)} kernels, the sources register {total}: "
        f"extra {sorted(have - set(sum(expected.values(), [])))[:8]}"
    )
    assert torch._C._et_runtime_info()["kernels"] == total
    print(
        f"ok   et_runtime: all {len(expected['portable'])} portable and {len(expected['prim_ops'])} "
        f"prim-op kernels are registered at run time"
    )


def test_garbage_buffers_raise_invalid_program_and_do_not_abort():
    torch, why = _runtime()
    if torch is None:
        _skip.skip(why)
        return
    cases = [
        ("empty", b""),
        ("ten_bytes", b"\x00" * 10),
        ("garbage_4k", bytes((i * 37 + 11) % 256 for i in range(4096))),
    ]
    code = _LOAD_EACH.replace("CASES", repr(cases))
    returncode, report, stderr = _child(code)
    _assert_invalid_program(returncode, report, stderr, "garbage buffer")
    print(f"ok   et_runtime: {len(cases)} garbage buffers raise EtInvalidProgram in a child that exits 0")


# ------------------------------------------------------- with an upstream program


def test_truncated_programs_raise_invalid_program_and_do_not_abort():
    torch, report = _program_or_skip()
    if torch is None:
        return
    n = report["bytes"]
    cuts = sorted({64, n // 4, n // 2, n - 1})
    code = (
        "import sys\n"
        "data = sys.stdin.buffer.read()\n"
        f"CUTS = {cuts!r}\n"
        + _LOAD_EACH.replace("CASES", "[(f'prefix_{c}', data[:c]) for c in CUTS]")
    )
    returncode, child, stderr = _child(code, stdin_bytes=report["program_bytes"])
    _assert_invalid_program(returncode, child, stderr, "truncated program")
    assert len(child) == len(cuts), child
    print(f"ok   et_runtime: prefixes {cuts} of the {n}-byte program raise EtInvalidProgram")


def test_the_portable_program_agrees_with_upstream():
    torch, report = _program_or_skip()
    if torch is None:
        return
    info = torch._C._et_runtime_info()
    assert report["executorch"].startswith(info["version"]), (report["executorch"], info["version"])
    assert report["delegates"] == [], f"the probe's program is delegated: {report['delegates']}"
    program = torch._C._et_load(report["program_bytes"])
    assert program.method_names() == [report["method"]], program.method_names()
    meta = program.method_meta(report["method"])
    assert [i["tag"] for i in meta["inputs"]] == ["Tensor"], meta
    assert meta["inputs"][0]["dtype"] == "float32" and meta["inputs"][0]["sizes"] == [2, 8], meta
    assert meta["backends"] == [], meta
    method = program.load_method(report["method"])
    p90, tolerances = _tolerances(report)
    worst = 0.0
    bit_equal = 0
    total = 0
    for k, spec in enumerate(report["inputs"]):
        outs = method.execute([_input(torch, spec)])
        assert len(outs) == 1, outs
        out = outs[0]
        assert isinstance(out, torch.Tensor) and out.dtype == torch.float32, (type(out), out.dtype)
        assert list(out.shape) == [2, 4], out.shape
        got = _flatten(out.tolist())
        want = _floats(report["oracle"][k])
        assert len(got) == len(want), (len(got), len(want))
        for i, (g, w, tol) in enumerate(zip(got, want, tolerances[k])):
            err = abs(g - w)
            assert err <= tol, (
                f"input {k} element {i}: torchnative {g!r} vs {report['oracle_kind']} {w!r}, "
                f"|diff| {err:.3e} > derived tolerance {tol:.3e} (p90 {p90:.3e})"
            )
            worst = max(worst, err)
            bit_equal += g == w
            total += 1
    note = f"; portable_lib did not load: {report['oracle_error']}" if "oracle_error" in report else ""
    print(
        f"ok   et_runtime: {total} outputs over 2 inputs agree with {report['oracle_kind']} "
        f"(worst |diff| {worst:.3e}, p90 tolerance {p90:.3e}, {bit_equal}/{total} bit-equal){note}"
    )


def test_operator_calls_name_the_kernels_that_ran():
    torch, report = _program_or_skip()
    if torch is None:
        return
    program = torch._C._et_load(report["program_bytes"])
    method = program.load_method(report["method"])
    method.reset_counters()
    x = _input(torch, report["inputs"][0])
    runs = 2
    for _ in range(runs):
        method.execute([x])
    counters = method.counters()
    want = {}
    for op in report["kernel_calls"]:
        key = op.split("::")[-1]
        want[key] = want.get(key, 0) + runs
    got = dict(counters["operator_calls"])
    assert got == want, f"operator_calls {got} != KernelCall instructions x {runs} {want}"
    assert "addmm.out" in got and "relu.out" in got, f"Linear + ReLU without addmm/relu kernels: {got}"
    assert counters["scopes"].get("OPERATOR_CALL") == runs * len(report["kernel_calls"]), counters["scopes"]
    assert counters["executions"] == runs, counters
    assert counters["delegate_ops"] == {}, counters
    registered = set(torch._C._et_registered_kernels())
    assert set(report["kernel_calls"]) <= registered, sorted(set(report["kernel_calls"]) - registered)
    method.reset_counters()
    assert method.counters()["operator_calls"] == {} and method.counters()["executions"] == 0
    print(f"ok   et_runtime: operator_calls after {runs} runs = {got} (program: {report['kernel_calls']})")


def test_wrong_inputs_raise_named_errors():
    torch, report = _program_or_skip()
    if torch is None:
        return
    program = torch._C._et_load(report["program_bytes"])
    method = program.load_method(report["method"])
    x = _input(torch, report["inputs"][0])
    cases = [
        ("shape (3, 8)", [torch.zeros(3, 8, dtype=torch.float32)], torch._C.EtInvalidArgument, "size"),
        ("rank 1", [torch.zeros(16, dtype=torch.float32)], torch._C.EtInvalidArgument, "rank"),
        ("float64", [x.double()], torch._C.EtInvalidArgument, "expects"),
        ("no inputs", [], torch._C.EtInvalidArgument, "takes 1 inputs"),
        ("two inputs", [x, x], torch._C.EtInvalidArgument, "takes 1 inputs"),
        ("not a tensor", [1.5], torch._C.EtInvalidArgument, "not a tensor"),
    ]
    for label, inputs, cls, needle in cases:
        try:
            method.execute(inputs)
        except cls as exc:
            assert needle in str(exc), f"{label}: {type(exc).__name__} without {needle!r}: {exc}"
        else:
            raise AssertionError(f"{label}: execute accepted it")
    good = _flatten(method.execute([x])[0].tolist())
    assert good == _flatten(method.execute([x])[0].tolist())
    print(f"ok   et_runtime: {len(cases)} wrong inputs raise EtInvalidArgument by name; the method still runs")


def test_a_method_keeps_its_program_alive():
    torch, report = _program_or_skip()
    if torch is None:
        return
    program = torch._C._et_load(report["program_bytes"])
    method = program.load_method(report["method"])
    del program
    gc.collect()
    x = _input(torch, report["inputs"][0])
    got = _flatten(method.execute([x])[0].tolist())
    _, tolerances = _tolerances(report)
    want = _floats(report["oracle"][0])
    assert all(abs(g - w) <= t for g, w, t in zip(got, want, tolerances[0])), (got, want)
    print("ok   et_runtime: a method runs after its EtProgram handle is gone")


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
                print(f"SKIP test_et_runtime: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
