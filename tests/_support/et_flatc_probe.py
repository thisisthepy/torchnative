"""exir's flatc calls for the Linear+ReLU XNNPACK lowering, as JSON on stdout (issue #357).

Run as a script by `tests/export/test_et_flatc.py` in the interpreter named by
`TORCHNATIVE_EXIR_PYTHON` (executorch 1.4.1 over upstream torch), never with the
vendored tree on the path. A helper, not a suite: it asserts nothing about
torchnative. `TMPDIR` is set by the caller to a directory inside the
repository, so exir's own temporary directories land there.

    et_flatc_probe.py capture
        lower Linear(8, 4) + ReLU with `to_edge_transform_and_lower` and
        `XnnpackPartitioner`, then `to_executorch()`, with the real flatc, and
        record every `_run_flatc` call: its arguments (paths relative to the
        call's own directory), the schema and JSON texts it read and every
        file it wrote, base64. Also the program's bytes.

    et_flatc_probe.py hooked
        the same lowering with flatc unreachable (see below) and
        `torchnative/python/torchnative/export/et_flatc.py`'s
        `install_flatc_hook` installed. The shim's binding cannot be imported
        in this interpreter (its torch is upstream's), so the hook is given a
        `json_to_binary` that answers from a table on stdin: for each (schema,
        JSON) pair, keyed by their sha256, the bytes the shim's
        `torch._C._et_flatc_json_to_binary` produced in the test's own
        process. A pair not in the table is an error naming it. Reports the
        program, how many conversions the hook made, every attempt to start
        flatc, and what `_get_flatc_path()` now does.

    et_flatc_probe.py unhooked
        the same lowering with flatc unreachable and no hook: reports the
        error, and every attempt to start flatc.

"flatc unreachable": the caller sets `PATH` and `FLATC_EXECUTABLE` to paths
that do not exist, and this script replaces `subprocess.Popen` with a class
that refuses (FileNotFoundError, as for a missing executable) any command
whose program is named `flatc`, and records it. The second part is needed
because exir finds the flatc packaged in its wheel
(`executorch/data/bin/flatc`) through `importlib.resources`, not through
`PATH`.
"""

import base64
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import traceback

MARK = "ET_FLATC_JSON "
_HOOK = pathlib.Path(__file__).resolve().parents[2] / "torchnative" / "python" / "torchnative" / "export" / "et_flatc.py"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model():
    import torch

    lin = torch.nn.Linear(8, 4)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 - 0.5)
        lin.bias.copy_(torch.arange(4, dtype=torch.float32) / 4)
    model = torch.nn.Sequential(lin, torch.nn.ReLU()).eval()
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5
    return model, x


def _lower():
    import torch
    from executorch import exir
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    model, x = _model()
    ep = torch.export.export(model, (x,))
    lowered = exir.to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    buffer = lowered.to_executorch().buffer
    assert isinstance(buffer, bytes), type(buffer)
    return buffer


def _guard_flatc():
    """Refuse and record every attempt to start a program named flatc."""
    attempts = []
    real = subprocess.Popen

    class _NoFlatc(real):
        def __init__(self, args, *a, **k):
            argv = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
            program = os.path.basename(os.fsdecode(argv[0])) if argv else ""
            if program == "flatc" or program.startswith("flatc."):
                attempts.append([os.fsdecode(x) for x in argv])
                raise FileNotFoundError(2, "flatc is unreachable in this probe", os.fsdecode(argv[0]))
            super().__init__(args, *a, **k)

    subprocess.Popen = _NoFlatc
    return attempts


def _capture(report):
    import executorch.exir._serialize._flatbuffer as fb

    calls = []
    real = fb._run_flatc

    def logged(args):
        args = list(args)
        base = None
        for a in args:
            if os.path.isdir(a):
                base = os.path.realpath(a)
        before = set(os.listdir(base)) if base else set()
        entry = {"args": [os.path.relpath(os.path.realpath(a), base) if base and os.path.exists(a) else a
                          for a in args],
                 "read": {}, "wrote": {}}
        for a in args:
            if os.path.isfile(a):
                with open(a, "rb") as f:
                    entry["read"][os.path.basename(a)] = f.read().decode("utf-8")
        real(args)
        if base:
            for name in sorted(set(os.listdir(base)) - before):
                with open(os.path.join(base, name), "rb") as f:
                    entry["wrote"][name] = base64.b64encode(f.read()).decode("ascii")
        calls.append(entry)

    fb._run_flatc = logged
    buffer = _lower()
    report["calls"] = calls
    report["program"] = base64.b64encode(buffer).decode("ascii")
    report["flatc_path"] = fb._get_flatc_path()


def _load_hook():
    spec = importlib.util.spec_from_file_location("torchnative_et_flatc_under_test", _HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hooked(report, table):
    import executorch.exir._serialize._flatbuffer as fb

    attempts = _guard_flatc()
    hook = _load_hook()
    conversions = []

    def from_table(schema, data, schema_name, json_name):
        key = f"{_sha(schema)}:{_sha(data)}"
        conversions.append({"key": key, "schema_name": os.path.basename(schema_name),
                            "json_name": os.path.basename(json_name)})
        if key not in table:
            raise KeyError(f"no binding output for schema {_sha(schema)[:12]} / JSON {_sha(data)[:12]}: exir "
                           "produced inputs the capture run did not")
        return base64.b64decode(table[key])

    report["installed"] = hook.install_flatc_hook(fb, json_to_binary=from_table)
    report["installed"]["replaced"] = [list(x) for x in report["installed"]["replaced"]]
    try:
        buffer = _lower()
        report["program"] = base64.b64encode(buffer).decode("ascii")
    except Exception as exc:  # noqa: BLE001 - reported
        report["lower_error"] = f"{type(exc).__name__}: {exc}"[:1200]
        report["where"] = traceback.format_exc().splitlines()[-10:]
    report["conversions"] = conversions
    report["flatc_attempts"] = attempts
    try:
        fb._get_flatc_path()
        report["get_flatc_path"] = {"raised": None}
    except Exception as exc:  # noqa: BLE001 - reported
        report["get_flatc_path"] = {"raised": type(exc).__name__, "message": str(exc)[:600]}


def _unhooked(report):
    attempts = _guard_flatc()
    try:
        buffer = _lower()
        report["program"] = base64.b64encode(buffer).decode("ascii")
    except Exception as exc:  # noqa: BLE001 - reported
        report["lower_error"] = f"{type(exc).__name__}: {exc}"[:1200]
    report["flatc_attempts"] = attempts


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    report = {"mode": mode}
    try:
        import torch

        report["shim"] = hasattr(torch._C, "_aten_implemented")
        report["torch_file"] = torch.__file__
        if mode == "capture":
            _capture(report)
        elif mode == "hooked":
            _hooked(report, json.loads(sys.stdin.read() or "{}"))
        elif mode == "unhooked":
            _unhooked(report)
        else:
            report["error"] = f"unknown mode {mode!r}; expected capture, hooked or unhooked"
    except Exception as exc:  # noqa: BLE001 - the report names it
        report["error"] = f"{type(exc).__name__}: {exc}"[:1200]
        report["where"] = traceback.format_exc().splitlines()[-10:]
    print(MARK + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
