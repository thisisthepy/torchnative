"""exir's serializer step, in Rust (issue #423, step R5-S1).

Owner, 2026-11-21 (asked by the leader, option (c) of decision 4): "Rust로
재구현": the serializer step is re-implemented in Rust and only exir's passes
are vendored, so no executorch Python package goes into the wheel.

Two functions keep exir 1.4.1's signatures and return what exir's return, byte
for byte:

* `serialize_pte_binary(pte_file, *, extract_delegate_segments=False,
  segment_alignment=128, constant_tensor_alignment=None,
  delegate_alignment=None)`: exir's `exir/_serialize/_program.py` function of
  that name. The work is `torch._C._et_serialize_pte` (`et_pte.rs`): segments,
  the extended header, and the program packed by a port of the `flatbuffers`
  Python `Builder` in the generated `Pack` order of `program.fbs`.
* `serialize_xnnpack_binary(xnnpack_graph, constant_data_bytes)`: exir's
  `backends/xnnpack/serialization/xnnpack_graph_serialize.py` function. The work
  is `torch._C._et_serialize_xnnpack` (`et.rs`): exir's JSON text of the graph,
  reproduced in Rust (`et_json.rs`), converted by the linked flatbuffers parser
  (#357) with the vendored `schema.fbs`, framed with the `XH00` header. It
  exists only where the runtime is built (`torch._C._et_runtime_info()`).

`serialize_pte_binary` returns exir's `Cord` when exir is importable (exir's
callers take `bytes(cord)` or `cord.write_to_file`), else `bytes`.

`install_serializer_hook()` puts both into exir in place of its own, wherever a
module holds them by name (the measured holders in executorch 1.4.1:
`exir._serialize._program`, `exir._serialize._serialize`, the package
`exir._serialize` as `_serialize_pte_binary`, `exir.lowered_backend_module`,
`backends.xnnpack.serialization.xnnpack_graph_serialize` and
`backends.xnnpack.xnnpack_preprocess`). With the hook installed, a lowering
runs no flatc and no exir serializer code.
"""

import sys

__all__ = [
    "SerializerNotLinked",
    "serialize_pte_binary",
    "serialize_xnnpack_binary",
    "install_serializer_hook",
    "uninstall_serializer_hook",
    "HOLDERS",
]


class SerializerNotLinked(RuntimeError):
    """The Rust serializer is not in this `torch._C`."""


#: (module, attribute, which) of every name exir 1.4.1 binds a serializer to.
HOLDERS = (
    ("executorch.exir._serialize._program", "serialize_pte_binary", "pte"),
    ("executorch.exir._serialize._serialize", "serialize_pte_binary", "pte"),
    ("executorch.exir._serialize", "_serialize_pte_binary", "pte"),
    ("executorch.exir.lowered_backend_module", "_serialize_pte_binary", "pte"),
    ("executorch.backends.xnnpack.serialization.xnnpack_graph_serialize", "serialize_xnnpack_binary", "xnnpack"),
    ("executorch.backends.xnnpack.xnnpack_preprocess", "serialize_xnnpack_binary", "xnnpack"),
)

_state = {"installed": None}


def _c(name):
    import torch

    fn = getattr(torch._C, name, None)
    if fn is None:
        why = ""
        info = getattr(torch._C, "_et_runtime_info", None)
        if info is not None:
            why = ": " + str(info().get("reason", ""))
        raise SerializerNotLinked(f"torch._C.{name} is not in this build (issue #423){why}")
    return fn


def _cord(data):
    try:
        from executorch.exir._serialize._cord import Cord
    except ImportError:
        return data
    return Cord(data)


def serialize_pte_binary(
    pte_file,
    *,
    extract_delegate_segments=False,
    segment_alignment=128,
    constant_tensor_alignment=None,
    delegate_alignment=None,
):
    """exir's `serialize_pte_binary`, computed by `torch._C._et_serialize_pte`."""
    mutable = None
    if getattr(pte_file, "mutable_data", None) is not None:
        mutable = []
        for i, buf in enumerate(pte_file.mutable_data):
            if not isinstance(buf.storage, bytes):
                # Cord.append's refusal in exir, kept so both fail alike.
                raise TypeError(f"Can only append bytes or Cords, received {type(buf.storage)} (mutable_data[{i}])")
            mutable.append(buf.storage)
    buffers = entries = None
    named = getattr(pte_file, "named_data", None)
    if named is not None:
        buffers = list(named.buffers)
        entries = [(key, entry.buffer_index, entry.alignment) for key, entry in named.pte_data.items()]
    data = _c("_et_serialize_pte")(
        pte_file.program,
        mutable,
        buffers,
        entries,
        extract_delegate_segments=bool(extract_delegate_segments),
        segment_alignment=segment_alignment,
        constant_tensor_alignment=constant_tensor_alignment,
        delegate_alignment=delegate_alignment,
    )
    return _cord(data)


def serialize_xnnpack_binary(xnnpack_graph, constant_data_bytes):
    """exir's `serialize_xnnpack_binary`, computed by `torch._C._et_serialize_xnnpack`."""
    return _c("_et_serialize_xnnpack")(xnnpack_graph, bytes(constant_data_bytes))


def install_serializer_hook(replacements=None):
    """Bind the Rust serializers wherever exir holds its own; returns what was replaced.

    `replacements` maps "pte" / "xnnpack" to the function to install (defaults:
    this module's two); it is a parameter only so a test can install stand-ins.
    The XNNPACK one is installed only when `torch._C._et_serialize_xnnpack`
    exists or a replacement is given. Idempotent.
    """
    import importlib

    if _state["installed"] is not None:
        return {"replaced": list(_state["installed"]["replaced"])}
    new = {"pte": serialize_pte_binary, "xnnpack": serialize_xnnpack_binary}
    if replacements:
        new.update(replacements)
    else:
        import torch

        if getattr(torch._C, "_et_serialize_xnnpack", None) is None:
            new.pop("xnnpack")
    originals = {}
    for module, attr, which in HOLDERS:
        if which not in new:
            continue
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        fn = getattr(mod, attr, None)
        if fn is not None and which not in originals:
            originals[which] = fn
    replaced = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            items = list(mod.__dict__.items())
        except Exception:  # noqa: BLE001 - a lazy module that refuses introspection
            continue
        for attr, value in items:
            for which, original in originals.items():
                if value is original:
                    setattr(mod, attr, new[which])
                    replaced.append((name, attr, which))
    _state["installed"] = {"originals": originals, "replaced": sorted(replaced)}
    return {"replaced": sorted(replaced)}


def uninstall_serializer_hook():
    """Put back what `install_serializer_hook` replaced."""
    installed = _state["installed"]
    if not installed:
        return
    for name, attr, which in installed["replaced"]:
        mod = sys.modules.get(name)
        if mod is not None:
            setattr(mod, attr, installed["originals"][which])
    _state["installed"] = None
