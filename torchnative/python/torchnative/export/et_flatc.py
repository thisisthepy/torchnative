"""exir's flatc, linked instead of exec'd (issue #357, step R4).

Leader decision 2026-11-14 item 4: "flatbuffers C++ parser linked into the
extension, replacing exir's _run_flatc in process; _get_flatc_path refuses by
name". Owner direction 2026-11-08: ExecuTorch compiles on the device, so the
lowering cannot depend on a flatc executable being present there.

`install_flatc_hook()` replaces `executorch.exir._serialize._flatbuffer._run_flatc`
with `_linked_run_flatc`, which does in process what flatc does for the one
invocation exir makes for an XNNPACK lowering, and replaces `_get_flatc_path`
with a refusal. Measured on executorch 1.4.1, Linear(8, 4) + ReLU through
`to_edge_transform_and_lower(..., XnnpackPartitioner())` then
`to_executorch()`: exactly one flatc call, from
`backends/xnnpack/serialization/xnnpack_graph_serialize.py`
`convert_to_flatbuffer` through `_flatbuffer._flatc_compile`:

    flatc --binary -o D D/schema.fbs D/schema.json

where `D` is a `tempfile.TemporaryDirectory()` the serializer created and
deletes, `schema.fbs` the XNNPACK schema (no `include`), `schema.json` the
graph, and flatc writes `D/schema.bin`, which the serializer reads back. The
program itself is packed by the `flatbuffers` Python builder and calls no
flatc; the portable lowering (no partitioner) calls none either.

Why replacing `_run_flatc` is enough: every caller imports `_flatc_compile` by
name, and `_flatc_compile` looks `_run_flatc` up as a module global at call
time. `install_flatc_hook` still scans `sys.modules` and replaces any other
module attribute that is bound to the original `_run_flatc` or
`_get_flatc_path` (none in executorch 1.4.1), and says which it replaced.

Files: `_linked_run_flatc` reads the two files flatc would read and writes the
one file flatc would write, `<out dir>/<json stem>.bin`, and only when the out
directory is the directory both inputs live in (the serializer's temporary
directory). Anything else, and every other flatc mode (decompile with `--json
--defaults-json --strict-json`, `--raw-binary`, includes, several inputs),
raises `FlatcNotLinked` naming the arguments. No program file is ever written:
the program stays in memory as `ExecutorchProgramManager.buffer`. exir's own
temporary files (the `.fbs`, `.json` and `.bin` in that directory) are exir's
business and are deleted by exir; with `logging.DEBUG` on its logger the
XNNPACK serializer also writes `./xnnpack_delegate_graph_<n>.json` itself.

The conversion is `torch._C._et_flatc_json_to_binary` (the shim built with the
`executorch` feature); a parse failure raises `torch._C.EtFlatcError` with the
parser's own message, the one flatc would print.
"""

import os
import sys

__all__ = ["FlatcNotLinked", "install_flatc_hook", "uninstall_flatc_hook", "MEASURED_MODE"]

#: The one flatc invocation shape exir makes for an XNNPACK lowering
#: (executorch 1.4.1, measured): `--binary -o OUT SCHEMA JSON`.
MEASURED_MODE = ("--binary", "-o", "<out dir>", "<schema>.fbs", "<data>.json")

_FLATBUFFER_MODULE = "executorch.exir._serialize._flatbuffer"

_state = {"installed": None}


class FlatcNotLinked(RuntimeError):
    """exir asked for something the linked flatc does not do, or for flatc itself."""


def _refuse_get_flatc_path():
    raise FlatcNotLinked(
        "executorch.exir._serialize._flatbuffer._get_flatc_path: torchnative runs no flatc executable "
        "(issue #357, leader decision 2026-11-14 item 4); exir's JSON-to-binary step goes through "
        "torch._C._et_flatc_json_to_binary, the linked flatbuffers v24.3.25 parser"
    )


def _binding():
    import torch

    fn = getattr(torch._C, "_et_flatc_json_to_binary", None)
    if fn is None:
        info = getattr(torch._C, "_et_runtime_info", None)
        why = info().get("reason", "") if info is not None else "torch._C has no _et_runtime_info"
        raise FlatcNotLinked(
            "torch._C._et_flatc_json_to_binary is not in this build, so the linked flatc hook cannot be "
            f"installed (issue #357): {why}"
        )
    return fn


def _read_text(path, what, args):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        raise FlatcNotLinked(f"linked flatc: cannot read the {what} {path!r} of flatc {list(args)}: {exc}") from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FlatcNotLinked(f"linked flatc: the {what} {path!r} is not UTF-8 ({exc}); flatc {list(args)}") from exc


def _make_linked_run_flatc(json_to_binary):
    def _linked_run_flatc(args):
        args = list(args)
        shape_ok = (
            len(args) == 5
            and all(isinstance(a, str) for a in args)
            and args[0] == "--binary"
            and args[1] == "-o"
            and args[3].endswith(".fbs")
            and not args[4].endswith(".fbs")
            and not args[4].startswith("-")
        )
        if not shape_ok:
            raise FlatcNotLinked(
                f"linked flatc: flatc {args} is not the one mode torchnative links "
                f"(flatc {' '.join(MEASURED_MODE)}, issue #357); refusing rather than running flatc"
            )
        out_dir, schema_path, json_path = args[2], args[3], args[4]
        here = os.path.realpath(out_dir)
        if not (os.path.isdir(here)
                and os.path.dirname(os.path.realpath(schema_path)) == here
                and os.path.dirname(os.path.realpath(json_path)) == here):
            raise FlatcNotLinked(
                f"linked flatc: flatc {args} writes outside the directory its inputs live in; the linked "
                "flatc writes only next to them (exir's own temporary directory)"
            )
        schema = _read_text(schema_path, "schema", args)
        data = _read_text(json_path, "JSON", args)
        binary = json_to_binary(schema, data, schema_path, json_path)
        # flatc's BinaryFileName: the JSON's name without directory and
        # extension, plus ".bin", in the -o directory.
        stem = os.path.splitext(os.path.basename(json_path))[0]
        with open(os.path.join(here, stem + ".bin"), "wb") as f:
            f.write(binary)

    _linked_run_flatc.__torchnative_linked_flatc__ = True
    return _linked_run_flatc


def install_flatc_hook(flatbuffer_module=None, json_to_binary=None):
    """Make exir's flatc calls run in process; returns where it replaced what.

    `flatbuffer_module` defaults to `executorch.exir._serialize._flatbuffer`;
    `json_to_binary(schema_text, json_text, schema_name, json_name) -> bytes`
    defaults to `torch._C._et_flatc_json_to_binary` and is a parameter only so
    a test can run the hook in an interpreter without the shim. Idempotent.
    """
    if flatbuffer_module is None:
        import importlib

        flatbuffer_module = importlib.import_module(_FLATBUFFER_MODULE)
    if json_to_binary is None:
        json_to_binary = _binding()
    current = getattr(flatbuffer_module, "_run_flatc", None)
    if getattr(current, "__torchnative_linked_flatc__", False):
        return {"replaced": list((_state["installed"] or {}).get("replaced", []))}
    original_run = flatbuffer_module._run_flatc
    original_path = flatbuffer_module._get_flatc_path
    linked = _make_linked_run_flatc(json_to_binary)
    replaced = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        for attr, original, new in (("_run_flatc", original_run, linked),
                                    ("_get_flatc_path", original_path, _refuse_get_flatc_path)):
            try:
                bound = module.__dict__.get(attr)
            except Exception:  # noqa: BLE001 - a lazy module that refuses introspection
                continue
            if bound is original:
                setattr(module, attr, new)
                replaced.append((name, attr))
    if (flatbuffer_module.__name__, "_run_flatc") not in replaced:
        flatbuffer_module._run_flatc = linked
        flatbuffer_module._get_flatc_path = _refuse_get_flatc_path
        replaced += [(flatbuffer_module.__name__, "_run_flatc"), (flatbuffer_module.__name__, "_get_flatc_path")]
    _state["installed"] = {
        "module": flatbuffer_module,
        "originals": {"_run_flatc": original_run, "_get_flatc_path": original_path},
        "replaced": sorted(set(replaced)),
    }
    return {"replaced": sorted(set(replaced))}


def uninstall_flatc_hook():
    """Put back what `install_flatc_hook` replaced."""
    installed = _state["installed"]
    if not installed:
        return
    for name, attr in installed["replaced"]:
        module = sys.modules.get(name)
        if module is not None:
            setattr(module, attr, installed["originals"][attr])
    module = installed["module"]
    for attr, original in installed["originals"].items():
        setattr(module, attr, original)
    _state["installed"] = None
