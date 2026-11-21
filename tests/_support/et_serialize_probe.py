"""exir's serializers against torchnative's Rust ones, as JSON on stdout (issue #423, step R5-S1).

Run as a script by `tests/export/test_et_serialize.py` in the interpreter named by
`TORCHNATIVE_EXIR_PYTHON` (executorch 1.4.1) with the vendored tree on
`PYTHONPATH`, so `torch` is torchnative's shim (exir runs on it: issue #306) and
`torch._C` has the Rust serializers. A helper, not a suite: it compares and
reports, and the suite asserts. `TMPDIR` is set by the caller to a directory
inside the repository.

What it reports:

    lowered   for each lowering below, every call exir made to
              `serialize_pte_binary` and `serialize_xnnpack_binary` (recorded by
              wrapping exir's own functions), with exir's bytes and the Rust
              bytes for the same arguments, and for XNNPACK also exir's JSON
              text against `torch._C._et_dataclass_json`'s
    synthetic hand-built `Program`s that reach every `KernelTypes` and
              `InstructionArguments` arm and every serializer option
              (constants, mutable data, named data with alignments, inline and
              extracted delegates, shared and empty delegate blobs, stack
              traces, inf/nan/-0.0 doubles, non-ASCII strings), and hand-built
              `XNNGraph`s, each serialised by exir and by the Rust code
    errors    inputs exir refuses; the Rust code must refuse each too
    hooked    the XNNPACK lowering with `install_serializer_hook()` and flatc
              unreachable: the program's bytes, whether exir's serializer code
              ran, flatc start attempts, and the program run by
              `torch._C._et_load` against eager where the runtime exists

Comparisons are of bytes; a difference reports the first differing offset and
both lengths.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback

MARK = "ET_SERIALIZE_JSON "


def _err(exc):
    tail = traceback.extract_tb(exc.__traceback__)[-4:]
    return {"error": f"{type(exc).__name__}: {exc}"[:600],
            "where": [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}" for f in tail]}


def _cmp(want, got):
    want, got = bytes(want), bytes(got)
    out = {"equal": want == got, "upstream_bytes": len(want), "rust_bytes": len(got),
           "sha256": hashlib.sha256(want).hexdigest()[:16]}
    if want != got:
        n = min(len(want), len(got))
        out["first_difference"] = next((i for i in range(n) if want[i] != got[i]), n)
    return out


def _guard_flatc():
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
    return attempts, lambda: setattr(subprocess, "Popen", real)


# ------------------------------------------------------------------ models

def _linear():
    import torch

    lin = torch.nn.Linear(8, 4)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 - 0.5)
        lin.bias.copy_(torch.arange(4, dtype=torch.float32) / 4)
    return torch.nn.Sequential(lin, torch.nn.ReLU()).eval(), (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5,)


def _scalars():
    import torch

    class Scalars(torch.nn.Module):
        def forward(self, x):
            y = torch.clamp(x, min=-0.25, max=float("inf"))
            z = y.sum(dim=[1], keepdim=True)
            return torch.cat([y * 1.5, z.expand(-1, 8) - 2], dim=0)

    return Scalars().eval(), (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5,)


def _buffered():
    import torch

    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("acc", torch.arange(8, dtype=torch.float32))

        def forward(self, x):
            self.acc.add_(x.sum(dim=0))
            return x * self.acc

    return Counter().eval(), (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16,)


def _xnn_mix():
    import torch

    class Mix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(8, 8)
            self.b = torch.nn.Linear(8, 3)
            with torch.no_grad():
                self.a.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64 - 0.5)
                self.a.bias.zero_()
                self.b.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24 - 0.5)
                self.b.bias.copy_(torch.arange(3, dtype=torch.float32))

        def forward(self, x):
            h = torch.nn.functional.hardtanh(self.a(x), -0.5, 0.5)
            return self.b(h * x + x).sigmoid()

    return Mix().eval(), (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5,)


def _lower(kind, factory, config_kw=None):
    import torch
    from executorch import exir

    model, args = factory()
    ep = torch.export.export(model, args)
    config = exir.ExecutorchBackendConfig(**(config_kw or {}))
    if kind == "portable":
        manager = exir.to_edge(ep)
    else:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        manager = exir.to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    return manager.to_executorch(config=config).buffer, model, args


LOWERINGS = [
    ("portable_linear", "portable", _linear, None),
    ("portable_linear_stacktrace", "portable", _linear, {"emit_stacktrace": True}),
    ("portable_scalars", "portable", _scalars, None),
    ("portable_buffer", "portable", _buffered, None),
    ("xnnpack_linear", "xnnpack", _linear, None),
    ("xnnpack_linear_inline", "xnnpack", _linear, {"extract_delegate_segments": False}),
    ("xnnpack_mix", "xnnpack", _xnn_mix, None),
    ("xnnpack_mix_aligned", "xnnpack", _xnn_mix, {"segment_alignment": 4096, "constant_tensor_alignment": 64,
                                                   "delegate_alignment": 32}),
]


# --------------------------------------------------------------- synthetic

def _synthetic_program(with_delegates=True, empty_constant=False):
    from executorch.exir import schema as S
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from executorch.exir.scalar_type import ScalarType

    tensor = S.Tensor(
        scalar_type=ScalarType.FLOAT, storage_offset=0, sizes=[2, 3], dim_order=[0, 1], requires_grad=False,
        layout=0, data_buffer_idx=1, allocation_info=S.AllocationDetails(memory_id=1, memory_offset_low=64,
                                                                            memory_offset_high=1),
        shape_dynamism=S.TensorShapeDynamism.DYNAMIC_BOUND,
        extra_tensor_info=S.ExtraTensorInfo(mutable_data_segments_idx=0, fully_qualified_name="layer.wé",
                                            location=S.TensorDataLocation.SEGMENT,
                                            device_type=S.DeviceType.CPU, device_index=0),
    )
    plain = S.Tensor(scalar_type=ScalarType.LONG, storage_offset=0, sizes=[], dim_order=[], requires_grad=True,
                     layout=0, data_buffer_idx=0, allocation_info=None, shape_dynamism=S.TensorShapeDynamism.STATIC)
    values = [
        S.EValue(S.Null()), S.EValue(S.Int(-(2 ** 40))), S.EValue(S.Bool(True)), S.EValue(S.Double(float("inf"))),
        S.EValue(S.Double(float("-inf"))), S.EValue(S.Double(-0.0)), S.EValue(S.Double(1e-7)),
        S.EValue(S.Double(float("nan"))), S.EValue(tensor), S.EValue(plain), S.EValue(S.String("héllo \U0001F600")),
        S.EValue(S.IntList([1, -2, 3])), S.EValue(S.DoubleList([0.5, -1.25])), S.EValue(S.BoolList([True, False, True])),
        S.EValue(S.TensorList([8, 9])), S.EValue(S.OptionalTensorList([8, -1])), S.EValue(S.IntList([])),
    ]
    instructions = [
        S.Instruction(S.KernelCall(op_index=0, args=[8, 1, 9])),
        S.Instruction(S.DelegateCall(delegate_index=0, args=[8, 9])),
        S.Instruction(S.MoveCall(move_from=8, move_to=9)),
        S.Instruction(S.JumpFalseCall(cond_value_index=2, destination_instruction=0)),
        S.Instruction(S.FreeCall(value_index=9)),
    ]
    frames = [S.FrameList(items=[S.Frame(filename="m.py", lineno=3, name="forward", context="y = f(x)")]),
              S.FrameList(items=[])]
    chain = S.Chain(inputs=[8], outputs=[9], instructions=instructions, stacktrace=frames)
    delegates = []
    inline = []
    if with_delegates:
        delegates = [
            S.BackendDelegate(id="XnnpackBackend", processed=S.BackendDelegateDataReference(S.DataLocation.INLINE, 0),
                              compile_specs=[CompileSpec("k", b"\x01\x02"), CompileSpec("empty", b"")]),
            S.BackendDelegate(id="Same", processed=S.BackendDelegateDataReference(S.DataLocation.INLINE, 1),
                              compile_specs=[]),
            S.BackendDelegate(id="Empty", processed=S.BackendDelegateDataReference(S.DataLocation.INLINE, 2),
                              compile_specs=[]),
        ]
        inline = [S.BackendDelegateInlineData(b"blob" * 9), S.BackendDelegateInlineData(b"blob" * 9),
                  S.BackendDelegateInlineData(b"")]
    plan = S.ExecutionPlan(
        name="forward", container_meta_type=S.ContainerMetadata("iné", "out"), values=values, inputs=[8],
        outputs=[9], chains=[chain], operators=[S.Operator("aten::add", "out"), S.Operator("aten::mm", "")],
        delegates=delegates, non_const_buffer_sizes=[0, 4096, 2 ** 33],
        non_const_buffer_device=[S.NonConstBufferDevice(buffer_idx=1, device_type=S.DeviceType.CUDA, device_index=1)],
    )
    second = copy.deepcopy(plan)
    second.name = "other"
    second.delegates = []
    second.non_const_buffer_device = None
    second.chains[0].stacktrace = None
    constants = [S.Buffer(storage=b"")] + ([] if empty_constant else
                                          [S.Buffer(storage=bytes(range(24))), S.Buffer(storage=b"\x07" * 5)])
    return S.Program(version=0, execution_plan=[plan, second], constant_buffer=constants,
                     backend_delegate_data=inline, segments=[], constant_segment=S.SubsegmentOffsets(0, []),
                     mutable_data_segments=None, named_data=None)


def _named():
    from executorch.exir._serialize._named_data_store import NamedDataStore

    store = NamedDataStore()
    store.add_named_data("w0", b"\x01" * 40, alignment=64)
    store.add_named_data("w1", b"\x02" * 7)
    store.add_named_data("w0_again", b"\x01" * 40, alignment=8)
    return store.get_named_data_store_output()


def _synthetic_cases():
    from executorch.exir import schema as S
    from executorch.exir._serialize._program import PTEFile

    mutable = [S.Buffer(storage=b"\x05" * 12), S.Buffer(storage=b""), S.Buffer(storage=b"\x06" * 3)]
    cases = []
    for name, kw, file_kw in [
        ("default", {}, {}),
        ("extract", {"extract_delegate_segments": True}, {}),
        ("extract_named_mutable", {"extract_delegate_segments": True},
         {"mutable_data": mutable, "named_data": "named"}),
        ("aligned", {"extract_delegate_segments": True, "segment_alignment": 64, "constant_tensor_alignment": 32,
                     "delegate_alignment": 64}, {"named_data": "named"}),
        ("big_alignment", {"segment_alignment": 4096, "constant_tensor_alignment": 4}, {"mutable_data": mutable}),
        ("only_empty_constant", {}, {"empty_constant": True}),
        ("no_delegates_extract", {"extract_delegate_segments": True}, {"no_delegates": True}),
    ]:
        program = _synthetic_program(with_delegates=not file_kw.get("no_delegates"),
                                     empty_constant=file_kw.get("empty_constant", False))
        named = _named() if file_kw.get("named_data") == "named" else None
        cases.append((name, PTEFile(program=program, mutable_data=file_kw.get("mutable_data"), named_data=named), kw))
    return cases


def _synthetic_graphs():
    from executorch.backends.xnnpack.serialization import xnnpack_graph_schema as X

    def tv(i, dims, dt=X.XNNDatatype.xnn_datatype_fp32, buf=0, ext=0xFFFFFFFF, flags=0):
        return X.XValue(X.XNNTensorValue(datatype=dt, num_dims=len(dims), dims=dims, constant_buffer_idx=buf,
                                         external_id=ext, flags=flags, id_out=i))

    values = [tv(0, [2, 8], ext=0, flags=1), tv(1, [4, 8], buf=1), tv(2, [2, 4], ext=1, flags=2),
              X.XValue(X.XNNQuantizedTensorValue(tv(3, [4], X.XNNDatatype.xnn_datatype_qcint8).xvalue_union,
                                                 X.PerChannelQuant(scale=[0.5, 0.25, 1e-7, 3.0], channel_dim=0))),
              X.XValue(X.XNNQuantizedTensorValue(tv(4, [1], X.XNNDatatype.xnn_datatype_qint8).xvalue_union,
                                                 X.PerTensorQuant(scale=0.125, zero_point=-3)))]
    nodes = [
        X.XNode(X.XNNFullyConnected(input1_id=0, filter_id=1, bias_id=4, output_id=2, flags=0), debug_handle=7,
                output_min_max=X.OutputMinMax(output_min="-inf", output_max=6.0)),
        X.XNode(X.XNNClamp(input_id=2, output_id=2, flags=0), debug_handle=8,
                output_min_max=X.OutputMinMax(output_min=-0.5, output_max="inf")),
        X.XNode(X.XNNAdd(input1_id=2, input2_id=2, output_id=2, flags=0), debug_handle=9),
        X.XNode(X.XNNLeakyReLU(negative_slope=0.01, input_id=2, output_id=2, flags=0), debug_handle=10),
        X.XNode(X.XNNStaticReshape(num_dims=2, new_shape=[4, 2], input_id=2, output_id=2, flags=0), debug_handle=11),
    ]
    graph = X.XNNGraph(version="0", xnodes=nodes, xvalues=values, num_externs=2, input_ids=[0], output_ids=[2],
                       constant_data=[X.ConstantDataOffset(offset=0, size=0),
                                      X.ConstantDataOffset(offset=0, size=128, named_key="wéight")])
    empty = X.XNNGraph(version="0", xnodes=[], xvalues=[], num_externs=0, input_ids=[], output_ids=[],
                       constant_data=[])
    return [("mixed", graph, bytearray(b"\x09" * 33)), ("empty_graph", empty, bytearray())]


# -------------------------------------------------------------------- main

def main():
    import torch

    report = {"shim": hasattr(torch._C, "_aten_implemented"),
              "pte_binding": hasattr(torch._C, "_et_serialize_pte"),
              "xnnpack_binding": hasattr(torch._C, "_et_serialize_xnnpack"),
              "runtime": hasattr(torch._C, "_et_load")}
    if not (report["shim"] and report["pte_binding"]):
        print(MARK + json.dumps(report))
        return

    from executorch.backends.xnnpack.serialization import xnnpack_graph_serialize as XS
    from executorch.exir._serialize import _program as P
    from executorch.exir._serialize._dataclass import _DataclassEncoder
    from torchnative.export import et_serialize as ours

    exir_pte, exir_xnn = P.serialize_pte_binary, XS.serialize_xnnpack_binary
    calls = []

    def rec_pte(pte_file, **kw):
        out = exir_pte(pte_file, **kw)
        entry = {"kind": "pte", "kwargs": {k: v for k, v in kw.items()}}
        try:
            entry.update(_cmp(out, ours.serialize_pte_binary(pte_file, **kw)))
        except Exception as exc:  # noqa: BLE001
            entry.update(_err(exc))
        calls.append(entry)
        return out

    def rec_xnn(graph, constant):
        out = exir_xnn(graph, constant)
        entry = {"kind": "xnnpack"}
        try:
            entry["json_equal"] = (torch._C._et_dataclass_json(graph) == json.dumps(graph, cls=_DataclassEncoder))
            if report["xnnpack_binding"]:
                entry.update(_cmp(out, ours.serialize_xnnpack_binary(graph, constant)))
        except Exception as exc:  # noqa: BLE001
            entry.update(_err(exc))
        calls.append(entry)
        return out

    lowered = {}
    report["recorder_replaced"] = ours.install_serializer_hook({"pte": rec_pte, "xnnpack": rec_xnn})["replaced"]
    try:
        for name, kind, factory, cfg in LOWERINGS:
            calls.clear()
            try:
                buf, _, _ = _lower(kind, factory, cfg)
                lowered[name] = {"ok": True, "bytes": len(buf), "calls": list(calls)}
            except Exception as exc:  # noqa: BLE001
                lowered[name] = {"ok": False, "calls": list(calls), **_err(exc)}
    finally:
        ours.uninstall_serializer_hook()
    report["lowered"] = lowered

    synthetic = {}
    for name, pte_file, kw in _synthetic_cases():
        try:
            want = bytes(exir_pte(pte_file, **kw))
        except Exception as exc:  # noqa: BLE001
            synthetic[name] = {"upstream": _err(exc)}
            continue
        try:
            synthetic[name] = _cmp(want, ours.serialize_pte_binary(pte_file, **kw))
        except Exception as exc:  # noqa: BLE001
            synthetic[name] = _err(exc)
    for name, graph, constant in _synthetic_graphs():
        entry = {}
        try:
            entry["json_equal"] = torch._C._et_dataclass_json(graph) == json.dumps(graph, cls=_DataclassEncoder)
            if report["xnnpack_binding"]:
                entry.update(_cmp(exir_xnn(graph, constant), ours.serialize_xnnpack_binary(graph, constant)))
        except Exception as exc:  # noqa: BLE001
            entry.update(_err(exc))
        synthetic["xnnpack_" + name] = entry
    report["synthetic"] = synthetic

    report["errors"] = _errors(exir_pte, ours)
    report["hooked"] = _hooked(ours)
    print(MARK + json.dumps(report))


def _errors(exir_pte, ours):
    from executorch.exir import schema as S
    from executorch.exir._serialize._program import PTEFile

    out = {}

    def both(name, make_file, **kw):
        res = {}
        for side, fn in (("upstream", exir_pte), ("rust", ours.serialize_pte_binary)):
            try:
                fn(make_file(), **kw)
                res[side] = None
            except Exception as exc:  # noqa: BLE001
                res[side] = type(exc).__name__
        out[name] = res

    def segment_delegate():
        p = _synthetic_program()
        p.execution_plan[0].delegates[0].processed = S.BackendDelegateDataReference(S.DataLocation.SEGMENT, 0)
        return PTEFile(program=p)

    def bad_index():
        p = _synthetic_program()
        p.execution_plan[0].delegates[0].processed.index = 7
        return PTEFile(program=p)

    def unvisited():
        p = _synthetic_program()
        p.backend_delegate_data.append(S.BackendDelegateInlineData(b"x"))
        return PTEFile(program=p)

    def named_twice():
        p = _synthetic_program()
        p.named_data = [S.NamedData("k", 0)]
        return PTEFile(program=p, named_data=_named())

    both("delegate_not_inline", segment_delegate, extract_delegate_segments=True)
    both("delegate_index_out_of_range", bad_index, extract_delegate_segments=True)
    both("delegate_data_not_visited", unvisited, extract_delegate_segments=True)
    both("named_data_already_present", named_twice)
    both("bad_tensor_alignment", lambda: PTEFile(program=_synthetic_program()), constant_tensor_alignment=3)
    both("bad_delegate_alignment", lambda: PTEFile(program=_synthetic_program()), delegate_alignment=12)
    return out


def _hooked(ours):
    import torch

    out = {}
    want = None
    try:
        want_buf, _, _ = _lower("xnnpack", _linear)
        want = bytes(want_buf)
    except Exception as exc:  # noqa: BLE001
        return {"unhooked": _err(exc)}
    from executorch.backends.xnnpack.serialization import xnnpack_graph_serialize as XS
    from executorch.exir._serialize import _program as P

    exir_ran = []
    real_pte, real_xnn = P.serialize_pte_binary, XS.serialize_xnnpack_binary
    out["replaced"] = ours.install_serializer_hook()["replaced"]
    attempts, restore = _guard_flatc()
    # exir's own functions are no longer reachable through the holders; any
    # call that still lands in them is recorded through their code objects.
    probe = {"pte": real_pte.__code__, "xnnpack": real_xnn.__code__}

    def tracer(frame, event, arg):
        if event == "call":
            for k, code in probe.items():
                if frame.f_code is code:
                    exir_ran.append(k)
        return None

    sys.setprofile(tracer)
    try:
        got_buf, model, args = _lower("xnnpack", _linear)
        got = bytes(got_buf)
    except Exception as exc:  # noqa: BLE001
        sys.setprofile(None)
        restore()
        ours.uninstall_serializer_hook()
        out.update(_err(exc))
        return out
    sys.setprofile(None)
    restore()
    ours.uninstall_serializer_hook()
    out.update(_cmp(want, got))
    out["exir_serializer_ran"] = sorted(set(exir_ran))
    out["flatc_attempts"] = attempts
    if hasattr(torch._C, "_et_load"):
        try:
            prog = torch._C._et_load(got)
            method = prog.load_method("forward")
            res = method.execute(list(args))
            ref = model(*args)
            out["ran"] = {"max_abs_diff": float((res[0] - ref).abs().max()),
                          "shape": list(res[0].shape)}
        except Exception as exc:  # noqa: BLE001
            out["ran"] = _err(exc)
    return out


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(MARK + json.dumps({"fatal": _err(exc)}))
