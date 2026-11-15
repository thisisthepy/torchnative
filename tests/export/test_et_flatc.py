"""exir's flatc, linked into `torch._C` instead of exec'd (issue #357, step R4, SPEC S5.15).

Leader decision 2026-11-14 item 4: "flatbuffers C++ parser linked into the
extension, replacing exir's _run_flatc in process; _get_flatc_path refuses by
name"; leader decision 2026-11-16: R4 now. Owner direction 2026-11-08:
ExecuTorch compiles on the device (in-memory programs, never a .pte file).

Pieces under test: `torch._C._et_flatc_json_to_binary` (the vendored
flatbuffers v24.3.25 parser compiled into `executorch_rt`, shim
`tn_et_flatc.cpp`) and `torchnative.export.et_flatc.install_flatc_hook`, which
replaces `executorch.exir._serialize._flatbuffer._run_flatc` with an in-process
call of it and makes `_get_flatc_path` refuse by name.

The measured mode (executorch 1.4.1, Linear(8, 4) + ReLU through
`to_edge_transform_and_lower(..., XnnpackPartitioner())` and `to_executorch()`):
one flatc call, `flatc --binary -o D D/schema.fbs D/schema.json`, from the
XNNPACK serializer. The binding covers that mode and refuses every other.

How the two interpreters meet. exir needs upstream torch, so it runs in the
interpreter named by `TORCHNATIVE_EXIR_PYTHON` (`tests/_support/et_flatc_probe.py`),
where the shim's `torch._C` cannot be imported. The binding therefore runs in
this process, on the schema and JSON the exir side captured, and its bytes go
back to the exir side as a table the hook answers from. The hook itself
(argument checks, the files it reads, the file it writes, the refusal of
`_get_flatc_path`) runs inside real exir.

**What makes each test able to fail.**

* `test_the_binding_is_in_torch_c_where_the_gate_builds_it`: on macOS arm64 a
  build without the `executorch` feature, or one whose build.rs stopped
  compiling the parser, has no `_et_flatc_json_to_binary`: a FAIL there, never
  a skip. Elsewhere it skips by name with `_et_runtime_info()`'s reason.
* `test_exir_makes_exactly_the_measured_flatc_call`: if exir (another version,
  another partitioner path) makes a flatc call of another shape, or none, this
  names it; the hook would refuse it.
* `test_every_flatc_call_of_the_xnnpack_lowering_is_reproduced_byte_for_byte`:
  the binding's bytes against the file the real flatc 24.3.25 wrote, for every
  call. Different parser options (`force_defaults`, a different builder
  alignment), a different flatbuffers version or a builder that was not reset
  between schema and JSON all change bytes.
* `test_with_the_hook_and_no_flatc_the_lowering_completes_and_equals_flatcs_program`:
  flatc is unreachable (PATH and FLATC_EXECUTABLE point nowhere, and the exir
  side refuses to start any program named flatc). The lowering must complete,
  with as many conversions through the hook as flatc calls captured, no flatc
  start attempted, `_get_flatc_path` raising `FlatcNotLinked`, and a program
  byte-identical to the one built with flatc. A hook that wrote the output
  under another name, or wrote other bytes, changes the program or stops the
  serializer.
* `test_without_the_hook_and_no_flatc_the_lowering_fails`: the same lowering
  with the same unreachable flatc and no hook must fail and must have tried to
  start flatc, so the previous test's success is the hook's work and not a
  lowering that never needed flatc.
* `test_malformed_input_raises_the_named_error_and_does_not_abort`: in a child
  interpreter, truncated JSON, an unknown field, a NUL in the text, a JSON
  that is not an object, an empty JSON, a garbage schema and a schema with an
  `include` must each raise `torch._C.EtFlatcError` (an `EtError`, a
  `RuntimeError`) with the parser's message; an abort is a FAIL naming the
  signal. The include case runs with the
  included file present in the child's working directory and must still be
  refused: the binding reads no file. A valid conversion must still work
  after all of them.
* `test_the_measured_truncated_message_is_flatcs`: the truncated XNNPACK JSON
  flatc 24.3.25 was measured on must raise with flatc's own message
  ("bad.json:1: 28: error: expecting: { instead got: end of file"); a binding
  that swallowed the parser's message, or another parser, fails it.
* `test_the_hook_refuses_every_other_flatc_mode_by_name`: on a stand-in
  module, in this process: the decompile mode, a call whose `-o` directory is
  not its inputs' directory, and `_get_flatc_path` raise `FlatcNotLinked`,
  and nothing is written; the measured mode writes exactly `<json stem>.bin`
  with the binding's bytes and nothing else.

**What this suite cannot find.** The hook and the binding in one interpreter
with real exir: that needs exir on the shim (issue #306); here the exir side
answers from bytes the binding produced in another process. Any flatc mode
other than the measured one (Vulkan's decompile, QNN, flat_tensor, ETDump,
bundled programs, schemas with includes): refused by name, not reproduced. A
locale other than "C" for `LC_NUMERIC` (number parsing; see executorch_rt's
build.rs). Android: these tests run where the gate builds the runtime (macOS
arm64); the Android artefact is only compiled. Whether XNNPACK accepts the
bytes is test_et_runtime's question, and it is the same bytes.

executorch is a test dependency of the gate only (leader decision 2026-11-14).
Without `TORCHNATIVE_EXIR_PYTHON` the tests that need exir skip by name.
"""

import base64
import hashlib
import atexit
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import types

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

_PROBE = str(_SUPPORT_DIR / "et_flatc_probe.py")
_MARK = "ET_FLATC_JSON "
_CHILD_MARK = "ET_CHILD_JSON "
_MEASURED_TRUNCATED = "expecting: { instead got: end of file"

import _skip  # noqa: E402

_cache = {}


def _on_macos_arm64():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _scratch():
    """Temporary files live in the repository's git-ignored .scratch/ (AGENTS.md section 2)."""
    base = _REPO_ROOT / ".scratch"
    base.mkdir(exist_ok=True)
    work = tempfile.mkdtemp(prefix="test_et_flatc-", dir=str(base))
    atexit.register(shutil.rmtree, work, ignore_errors=True)
    return work


def _torch():
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        return None
    import torch

    assert hasattr(torch._C, "_aten_implemented"), f"upstream torch, not the shim: {torch.__file__}"
    return torch


def _binding():
    """(torch, None) when the binding is in this build, else (None, why).

    On macOS arm64 the gate builds it, so its absence raises (a FAIL).
    """
    torch = _torch()
    if torch is None:
        return None, "no vendored torch tree"
    if hasattr(torch._C, "_et_flatc_json_to_binary"):
        return torch, None
    info_fn = getattr(torch._C, "_et_runtime_info", None)
    info = info_fn() if info_fn is not None else {}
    reason = info.get("reason") or (
        "the runtime is linked but has no _et_flatc_json_to_binary (a build from before issue #357)"
        if info.get("available") else repr(info))
    if _on_macos_arm64():
        raise AssertionError(
            "torch._C._et_flatc_json_to_binary is missing on macOS arm64, where install_shim.sh and run.sh "
            f"build the runtime with --features executorch: {reason}"
        )
    return None, f"the linked flatc is not in this build: {reason}"


def _exir_python():
    path = os.environ.get("TORCHNATIVE_EXIR_PYTHON")
    if not path or not os.path.isfile(path):
        return None
    return path


def _probe(mode, stdin_text="", unreachable=False):
    """Run the exir probe in `mode`; its report. exir's temp dirs land in .scratch."""
    work = _scratch()
    try:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
        env["TMPDIR"] = work
        if unreachable:
            nowhere = os.path.join(work, "no-such-dir")
            env["PATH"] = nowhere
            env["FLATC_EXECUTABLE"] = os.path.join(nowhere, "flatc")
        proc = subprocess.run(
            [_exir_python(), _PROBE, mode],
            env=env,
            cwd=str(_SUPPORT_DIR),
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(_MARK)), None)
        if line is None:
            raise AssertionError(
                f"the flatc probe ({mode}) exited {proc.returncode} without a report; stderr tail:\n"
                + "\n".join(proc.stderr.splitlines()[-25:])
            )
        report = json.loads(line[len(_MARK):])
        if "error" in report:
            raise AssertionError(f"the flatc probe ({mode}) stopped: {report['error']}\n"
                                 + "\n".join(report.get("where", [])))
        assert report["shim"] is False, f"the probe imported the shim, not upstream: {report['torch_file']}"
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _needs_exir():
    """(torch, capture report) or (None, None) after a skip."""
    torch, why = _binding()
    if torch is None:
        _skip.skip(why)
        return None, None
    if _exir_python() is None:
        _skip.skip(
            "TORCHNATIVE_EXIR_PYTHON unset or not a file: no interpreter with executorch to run the "
            "XNNPACK lowering in (the gate builds one with executorch==1.4.1; issue #306)"
        )
        return None, None
    if "capture" not in _cache:
        _cache["capture"] = _probe("capture")
    return torch, _cache["capture"]


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _call_parts(call):
    """(schema name, schema text, json name, json text, flatc's output bytes, output name)."""
    args = call["args"]
    schema_name, json_name = args[3], args[4]
    out_name = os.path.splitext(json_name)[0] + ".bin"
    assert out_name in call["wrote"], f"flatc {args} wrote {sorted(call['wrote'])}, not {out_name}"
    return (schema_name, call["read"][schema_name], json_name, call["read"][json_name],
            base64.b64decode(call["wrote"][out_name]), out_name)


def _child(code, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_VENDOR_DIR, str(_SUPPORT_DIR)])
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, timeout=600)
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    line = next((ln for ln in stdout.splitlines() if ln.startswith(_CHILD_MARK)), None)
    return proc.returncode, (json.loads(line[len(_CHILD_MARK):]) if line else None), stderr


# A schema small enough to state here, for the tests that must run without exir.
_TINY_SCHEMA = 'namespace tn_test;\nfile_identifier "TNT0";\ntable T { a:int; b:[ubyte]; }\nroot_type T;\n'
_TINY_JSON = '{"a": 7, "b": [1, 2, 3]}'


# ---------------------------------------------------------------- the binding

def test_the_binding_is_in_torch_c_where_the_gate_builds_it():
    torch, why = _binding()
    if torch is None:
        _skip.skip(why)
        return
    out = torch._C._et_flatc_json_to_binary(_TINY_SCHEMA, _TINY_JSON)
    assert isinstance(out, bytes) and len(out) > 8, out
    assert out[4:8] == b"TNT0", f"the schema's file_identifier is not at offset 4: {out[:16]!r}"
    assert issubclass(torch._C.EtFlatcError, torch._C.EtError), torch._C.EtFlatcError.__mro__
    assert issubclass(torch._C.EtFlatcError, RuntimeError), torch._C.EtFlatcError.__mro__


def test_exir_makes_exactly_the_measured_flatc_call():
    torch, cap = _needs_exir()
    if torch is None:
        return
    calls = cap["calls"]
    assert calls, "the XNNPACK lowering made no flatc call: nothing here exercises the hook"
    for call in calls:
        args = call["args"]
        assert len(args) == 5 and args[:2] == ["--binary", "-o"] and args[2] == ".", (
            f"flatc {args}: not the measured `--binary -o D D/schema.fbs D/schema.json`")
        assert args[3].endswith(".fbs") and args[4].endswith(".json"), args
        assert sorted(call["read"]) == sorted([args[3], args[4]]), (args, sorted(call["read"]))
        assert "include " not in call["read"][args[3]].replace("// include", ""), (
            f"the schema of flatc {args} has an include, which the binding refuses")
        assert list(call["wrote"]) == [os.path.splitext(args[4])[0] + ".bin"], (args, list(call["wrote"]))
    print(f"ok   et_flatc: {len(calls)} flatc call(s), all `--binary -o D D/schema.fbs D/schema.json`")


def test_every_flatc_call_of_the_xnnpack_lowering_is_reproduced_byte_for_byte():
    torch, cap = _needs_exir()
    if torch is None:
        return
    for i, call in enumerate(cap["calls"]):
        schema_name, schema, json_name, data, flatc_bytes, _ = _call_parts(call)
        ours = torch._C._et_flatc_json_to_binary(schema, data, schema_name, json_name)
        assert ours == flatc_bytes, (
            f"call {i} (flatc {call['args']}): the linked parser gave {len(ours)} bytes "
            f"(sha256 {hashlib.sha256(ours).hexdigest()[:16]}), flatc 24.3.25 wrote {len(flatc_bytes)} "
            f"(sha256 {hashlib.sha256(flatc_bytes).hexdigest()[:16]}); first difference at byte "
            f"{next((k for k, (a, b) in enumerate(zip(ours, flatc_bytes)) if a != b), min(len(ours), len(flatc_bytes)))}"
        )
    print(f"ok   et_flatc: {len(cap['calls'])} call(s) byte-identical to flatc 24.3.25")


def test_with_the_hook_and_no_flatc_the_lowering_completes_and_equals_flatcs_program():
    torch, cap = _needs_exir()
    if torch is None:
        return
    table = {}
    for call in cap["calls"]:
        schema_name, schema, json_name, data, _, _ = _call_parts(call)
        out = torch._C._et_flatc_json_to_binary(schema, data, schema_name, json_name)
        table[f"{_sha(schema)}:{_sha(data)}"] = base64.b64encode(out).decode("ascii")
    hooked = _probe("hooked", json.dumps(table), unreachable=True)
    assert "lower_error" not in hooked, (
        f"with the hook installed the lowering failed: {hooked['lower_error']}\n" + "\n".join(hooked.get("where", [])))
    assert hooked["flatc_attempts"] == [], f"flatc was started with the hook installed: {hooked['flatc_attempts']}"
    assert len(hooked["conversions"]) == len(cap["calls"]), (
        f"the hook converted {len(hooked['conversions'])} time(s); flatc was called {len(cap['calls'])} time(s)")
    replaced = [tuple(x) for x in hooked["installed"]["replaced"]]
    assert ("executorch.exir._serialize._flatbuffer", "_run_flatc") in replaced, replaced
    gfp = hooked["get_flatc_path"]
    assert gfp["raised"] == "FlatcNotLinked" and "#357" in gfp["message"], gfp
    assert hooked["program"] == cap["program"], (
        "the program built through the hook differs from the one built with flatc "
        f"({len(base64.b64decode(hooked['program']))} vs {len(base64.b64decode(cap['program']))} bytes)")
    print(f"ok   et_flatc: lowering with no flatc completed through the hook ({len(hooked['conversions'])} "
          "conversion(s)); program byte-identical to flatc's")


def test_without_the_hook_and_no_flatc_the_lowering_fails():
    torch, cap = _needs_exir()
    if torch is None:
        return
    bare = _probe("unhooked", unreachable=True)
    assert bare["flatc_attempts"], "the lowering never tried to start flatc: the hook test proves nothing"
    assert "lower_error" in bare and "program" not in bare, (
        "the XNNPACK lowering completed with flatc unreachable and no hook")
    print(f"ok   et_flatc: without the hook the lowering fails ({bare['lower_error'][:120]})")


_MALFORMED = r"""
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented"), torch.__file__
f = torch._C._et_flatc_json_to_binary
SCHEMA = SCHEMA_TEXT
GOOD = GOOD_JSON
cases = {
    "truncated": (SCHEMA, '{"a": 7, "b": [1, 2'),
    "unknown_field": (SCHEMA, '{"a": 7, "zzz": 1}'),
    "nul": (SCHEMA, '{"a": 7}\x00{"a": 8}'),
    "not_an_object": (SCHEMA, '[1, 2, 3]'),
    "empty": (SCHEMA, ''),
    "garbage_schema": ('table { ;;; root_type', GOOD),
    "include": ('include "other.fbs";\n' + SCHEMA, GOOD),
}
out = {}
for label, (schema, data) in cases.items():
    try:
        f(schema, data, "schema.fbs", "schema.json")
        out[label] = {"raised": None}
    except Exception as exc:
        out[label] = {"raised": type(exc).__name__, "mro": [c.__name__ for c in type(exc).__mro__],
                      "message": str(exc)[:800]}
good = f(SCHEMA, GOOD, "schema.fbs", "schema.json")
out["after"] = {"bytes": len(good), "id": good[4:8].decode("latin-1")}
print("ET_CHILD_JSON " + json.dumps(out))
"""


def test_malformed_input_raises_the_named_error_and_does_not_abort():
    torch, why = _binding()
    if torch is None:
        _skip.skip(why)
        return
    work = _scratch()
    try:
        # Present on disk, so a binding that read includes from the working
        # directory would accept the schema.
        with open(os.path.join(work, "other.fbs"), "w") as fh:
            fh.write("namespace other;\ntable O { x:int; }\n")
        code = _MALFORMED.replace("SCHEMA_TEXT", repr(_TINY_SCHEMA)).replace("GOOD_JSON", repr(_TINY_JSON))
        rc, report, stderr = _child(code, cwd=work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if rc < 0:
        raise AssertionError(f"the interpreter was killed by signal {-rc} (an abort, not an exception); "
                             "stderr tail:\n" + "\n".join(stderr.splitlines()[-15:]))
    assert rc == 0 and report is not None, f"child exited {rc} without a report:\n" + "\n".join(stderr.splitlines()[-15:])
    after = report.pop("after")
    assert after["bytes"] > 8 and after["id"] == "TNT0", f"a valid conversion failed after the errors: {after}"
    for label, r in sorted(report.items()):
        assert r["raised"] == "EtFlatcError", f"{label}: raised {r['raised']}: {r.get('message')}"
        assert "EtError" in r["mro"] and "RuntimeError" in r["mro"], (label, r["mro"])
        assert "FlatcError" in r["message"] and "flatbuffers parser:" in r["message"], (label, r["message"])
    assert "end of file" in report["truncated"]["message"], report["truncated"]["message"]
    # flatc 24.3.25 on an empty JSON file: "schema.json:1: 0: error: input file
    # is empty" (measured 2026-11-16; bundle run 43698075642 showed the linked
    # parser giving the same words, where this used to expect another message).
    assert "schema.json:1: 0: error: input file is empty" in report["empty"]["message"], report["empty"]["message"]
    assert "unable to locate include file" in report["include"]["message"], report["include"]["message"]
    assert "other.fbs" in report["include"]["message"], report["include"]["message"]
    assert "appears to be binary" in report["nul"]["message"], report["nul"]["message"]
    assert "unknown field" in report["unknown_field"]["message"], report["unknown_field"]["message"]
    print(f"ok   et_flatc: {len(report)} malformed inputs raise EtFlatcError in a child that exits 0")


def test_the_measured_truncated_message_is_flatcs():
    """The truncation case flatc 24.3.25 was measured on (issue #357): the
    XNNPACK schema's root table, JSON cut after its opening bracket."""
    torch, cap = _needs_exir()
    if torch is None:
        return
    schema_name, schema, _, _, _, _ = _call_parts(cap["calls"][0])
    try:
        torch._C._et_flatc_json_to_binary(schema, '{"version": "0", "xnodes": [', schema_name, "bad.json")
    except torch._C.EtFlatcError as exc:
        assert "bad.json:1: 28: error: " + _MEASURED_TRUNCATED in str(exc), str(exc)
    else:
        raise AssertionError("a truncated JSON was accepted")


# --------------------------------------------------------------------- the hook

def _stand_in_module(name="tn_test_fake_flatbuffer"):
    """A module with exir's three functions, `_flatc_compile` looking
    `_run_flatc` up as a global at call time, as exir's does."""
    m = types.ModuleType(name)

    def _get_flatc_path():
        return "/no/such/flatc"

    def _run_flatc(args):
        raise AssertionError(f"the original _run_flatc ran: {args}")

    def _flatc_compile(output_dir, schema_path, json_path):
        m.__dict__["_run_flatc"](["--binary", "-o", output_dir, schema_path, json_path])

    m._get_flatc_path, m._run_flatc, m._flatc_compile = _get_flatc_path, _run_flatc, _flatc_compile
    return m


def test_the_hook_refuses_every_other_flatc_mode_by_name():
    torch, why = _binding()
    if torch is None:
        _skip.skip(why)
        return
    from torchnative.export import et_flatc

    fake = _stand_in_module()
    original_run = fake._run_flatc
    info = et_flatc.install_flatc_hook(fake)
    work = _scratch()
    try:
        assert fake._run_flatc is not original_run, "install_flatc_hook did not replace _run_flatc"
        assert (fake.__name__, "_run_flatc") in info["replaced"], info
        try:
            fake._get_flatc_path()
        except et_flatc.FlatcNotLinked as exc:
            assert "#357" in str(exc) and "_get_flatc_path" in str(exc), str(exc)
        else:
            raise AssertionError("_get_flatc_path did not refuse")

        d = os.path.join(work, "d")
        other = os.path.join(work, "elsewhere")
        os.mkdir(d)
        os.mkdir(other)
        schema_path, json_path = os.path.join(d, "schema.fbs"), os.path.join(d, "schema.json")
        with open(schema_path, "w") as fh:
            fh.write(_TINY_SCHEMA)
        with open(json_path, "w") as fh:
            fh.write(_TINY_JSON)
        bin_path = os.path.join(d, "schema.bin")
        refused = {
            "decompile": ["--json", "--defaults-json", "--strict-json", "-o", d, schema_path, "--", bin_path],
            "raw_binary_decompile": ["--raw-binary", "--json", "--defaults-json", "--strict-json", "-o", d,
                                     schema_path, "--", bin_path],
            "strict_json_compile": ["--binary", "--strict-json", "-o", d, schema_path, json_path],
            "two_inputs": ["--binary", "-o", d, schema_path, json_path, json_path],
            "out_elsewhere": ["--binary", "-o", other, schema_path, json_path],
        }
        for label, args in refused.items():
            before = (sorted(os.listdir(d)), sorted(os.listdir(other)))
            try:
                fake._run_flatc(args)
            except et_flatc.FlatcNotLinked as exc:
                assert "linked flatc" in str(exc), (label, str(exc))
            else:
                raise AssertionError(f"{label}: flatc {args} was not refused")
            assert (sorted(os.listdir(d)), sorted(os.listdir(other))) == before, f"{label} wrote a file"

        fake._flatc_compile(d, schema_path, json_path)
        assert sorted(os.listdir(d)) == ["schema.bin", "schema.fbs", "schema.json"], os.listdir(d)
        assert os.listdir(other) == [], os.listdir(other)
        with open(bin_path, "rb") as fh:
            written = fh.read()
        assert written == torch._C._et_flatc_json_to_binary(_TINY_SCHEMA, _TINY_JSON), "the hook wrote other bytes"
    finally:
        et_flatc.uninstall_flatc_hook()
        shutil.rmtree(work, ignore_errors=True)
    assert fake._run_flatc is original_run, "uninstall_flatc_hook did not restore _run_flatc"


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
                print(f"SKIP test_et_flatc: {name} -- {_skip._state['skipped'][name]}")
            else:
                print(f"ok   {name}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
