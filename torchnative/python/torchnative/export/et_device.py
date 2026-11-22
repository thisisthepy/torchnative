"""`.to(device.npu)` through ExecuTorch, on the device, in memory (issue #423, step R5a).

Owner's direction (docs/devices/ETDEVICE.md section 2): the Android NPU back end
is ExecuTorch run on the device at `.to(device.npu)` time, JIT, with programs
held in memory only and never a `.pte` file (owner, 2026-11-08, relayed
2026-11-14; no offline or CI per-model pipeline, owner 2026-11-07). Delegate
order XNNPACK, then Vulkan, then Qualcomm HTP, then Samsung/MediaTek (owner,
2026-11-08). The sequence length is symbolic (owner, 2026-11-13), marked with a
range at `.to(device)` (owner, 2026-11-21, #231 Q4). Only exir's passes are
vendored and the serializer is the Rust one (owner, 2026-11-21, "Rust로 재구현").

The shape of this module is the agent proposal (i) to (iv) of ETDEVICE section 2
(relayed as standing by the leader, 2026-11-21):

* `_compile_model(model, max_seq_len=...)` keeps the same `nn.Module` and swaps
  its `forward` for an `ExecuTorchForward`. Nothing is lowered at `.to()`: the
  first call captures with `torch.export` on that call's arguments, lowers
  through the vendored exir with the XNNPACK partitioner, serialises in memory
  through `torch._C._et_serialize_pte` / `_et_serialize_xnnpack`, loads the
  bytes with `torch._C._et_load`, and runs the method. Later calls whose
  arguments fit a compiled program's guards run it; others compile again.
* XNNPACK runs on the **CPU**. The resolution says so ("CPU (ExecuTorch XNNPACK
  delegate)") and so does every report; nothing here says NPU until the HTP
  delegate is wired.
* Step R5a is the forward without a KV cache. `generate(..., use_cache=False)`
  runs every step as a full forward through the program. A KV cache (R5b) is
  refused by name rather than run with the cache dropped.

What is an agent proposal, not confirmed: which inputs get the symbolic length
(integer tensors' last dimension equal to the first integer 2-D input's, that is
token ids, masks and positions), the default `max_seq_len` (the model config's
`max_position_embeddings`, else 512), recompiling with a doubled range for a
longer input, and falling back to static shapes when export refuses the
symbolic one.

Runtime evidence, never inferred from a right answer (AGENTS section 16): each
program's method counters (`delegate_calls`, `operator_calls`) are read after
every call and kept on the report, so "ran in the XNNPACK delegate" is a count.
"""

import warnings

__all__ = [
    "BACKEND",
    "DELEGATE",
    "UNIT",
    "ExecuTorchForward",
    "ExecuTorchUnsupported",
    "runtime_refusal",
]

BACKEND = "executorch"
DELEGATE = "XnnpackBackend"
UNIT = "CPU (ExecuTorch XNNPACK delegate)"
VENDORED = "torchnative._vendor.executorch"
DEFAULT_MAX_SEQ_LEN = 512


class ExecuTorchUnsupported(NotImplementedError):
    """What the ExecuTorch path does not do, by name."""


def runtime_refusal():
    """Why this build cannot run `.to(device.npu)` through ExecuTorch, or None.

    Each piece is asked of the build itself: the runtime (`torch._C._et_load`),
    its XNNPACK backend registered (`_et_runtime_info()["xnnpack_available"]`),
    both Rust serializers, and the vendored exir importable."""
    import importlib.util

    import torch

    info = getattr(torch._C, "_et_runtime_info", None)
    if info is None:
        return "torch._C has no _et_runtime_info: this build has no ExecuTorch binding"
    got = dict(info())
    if not got.get("available"):
        return f"the ExecuTorch runtime is not built here: {got.get('reason', 'no reason given')}"
    if not hasattr(torch._C, "_et_load"):
        return "torch._C has no _et_load, so no program can be loaded"
    if not got.get("xnnpack_available"):
        return (f"the runtime has no {DELEGATE} registered "
                f"(registered: {got.get('backends')})")
    for name in ("_et_serialize_pte", "_et_serialize_xnnpack"):
        if not hasattr(torch._C, name):
            return f"torch._C has no {name}, so no program can be serialised in memory"
    try:
        spec = importlib.util.find_spec(VENDORED + ".exir")
    except ImportError as exc:
        spec, why = None, f"{type(exc).__name__}: {exc}"
    else:
        why = "not found"
    if spec is None:
        return f"the vendored exir ({VENDORED}) is not importable: {why}"
    return None


# ------------------------------------------------------------------ guards

def _flatten_call(args, kwargs):
    import torch
    from torch.utils import _pytree as pytree

    leaves, spec = pytree.tree_flatten((tuple(args), dict(kwargs)))
    tensor_pos = [i for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)]
    consts = [None if i in set(tensor_pos) else x for i, x in enumerate(leaves)]
    return leaves, spec, tensor_pos, consts


def _seq_dims(leaves, tensor_pos):
    """{leaf index: dim} for the inputs that take the symbolic sequence length.

    Agent proposal: integer tensors (token ids, attention masks, positions)
    whose last dimension equals the last dimension of the first integer 2-D
    input. Float inputs keep static shapes: a feature dimension is never a
    sequence length here."""
    ints = [i for i in tensor_pos if not leaves[i].is_floating_point() and not leaves[i].is_complex()]
    first = next((i for i in ints if leaves[i].dim() == 2), None)
    if first is None:
        return {}, None
    length = int(leaves[first].shape[-1])
    return {i: leaves[i].dim() - 1 for i in ints
            if leaves[i].dim() >= 1 and int(leaves[i].shape[-1]) == length}, length


def _same_consts(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        try:
            if type(x) is not type(y) or not bool(x == y):
                return False
        except Exception:  # noqa: BLE001 - an uncomparable constant never matches
            return False
    return True


class _Program:
    """One compiled program and the guards its inputs must meet."""

    def __init__(self, spec, tensor_pos, consts, metas, out_spec, out_tensor_pos,
                 out_consts, method, info):
        self.spec = spec
        self.tensor_pos = tensor_pos
        self.consts = consts
        self.metas = metas            # per tensor input: (dtype, [int | (lo, hi)])
        self.out_spec = out_spec
        self.out_tensor_pos = out_tensor_pos
        self.out_consts = out_consts
        self.method = method
        self.info = info

    def fits(self, spec, tensor_pos, consts, leaves):
        if spec != self.spec or tensor_pos != self.tensor_pos or not _same_consts(consts, self.consts):
            return False
        for i, (dtype, dims) in zip(self.tensor_pos, self.metas):
            t = leaves[i]
            if t.dtype != dtype or t.dim() != len(dims):
                return False
            for size, want in zip(t.shape, dims):
                if isinstance(want, tuple):
                    if not want[0] <= int(size) <= want[1]:
                        return False
                elif int(size) != want:
                    return False
        return True


# ------------------------------------------------------------------ lowering

def _bound_module(model, forward, spec, tensor_pos, consts, record):
    """A module whose forward takes only the call's tensors, in leaf order.

    The non-tensor arguments are this call's constants, baked into the
    program; a call with different constants does not fit its guards. Outputs
    are flattened; their tree and non-tensor leaves are recorded at trace time
    and put back after every run."""
    import torch
    from torch.utils import _pytree as pytree

    class _Bound(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, *tensors):
            leaves = list(consts)
            for i, t in zip(tensor_pos, tensors):
                leaves[i] = t
            a, kw = pytree.tree_unflatten(leaves, spec)
            out = forward(*a, **kw)
            flat, out_spec = pytree.tree_flatten(out)
            pos = [i for i, x in enumerate(flat) if isinstance(x, torch.Tensor)]
            record["out_spec"] = out_spec
            record["out_tensor_pos"] = pos
            record["out_consts"] = [None if i in set(pos) else x for i, x in enumerate(flat)]
            return tuple(flat[i] for i in pos)

    return _Bound()


def _graph_report(ep):
    """Which ops the partitioner left outside the delegate, from the edge graph."""
    delegated, left = 0, {}
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", str(node.target))
        if name == "executorch_call_delegate":
            delegated += 1
        elif name == "getitem":
            continue
        else:
            left[name] = left.get(name, 0) + 1
    return delegated, left


def _lower(model, forward, leaves, spec, tensor_pos, consts, max_seq_len):
    import torch

    from torchnative._vendor.executorch import exir
    from torchnative._vendor.executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    record = {}
    bound = _bound_module(model, forward, spec, tensor_pos, consts, record)
    sample = tuple(leaves[i] for i in tensor_pos)
    seq, length = _seq_dims(leaves, tensor_pos)
    shapes_mode = "static"
    ep = None
    export_refusal = None
    if seq and length is not None and length <= max_seq_len:
        dim = torch.export.Dim("seq", min=1, max=max_seq_len)
        dynamic = tuple({seq[i]: dim} if i in seq else None for i in tensor_pos)
        try:
            ep = torch.export.export(bound, sample, dynamic_shapes=dynamic)
            shapes_mode = "symbolic-seq"
        except Exception as exc:  # noqa: BLE001 - recorded, then static shapes
            export_refusal = f"{type(exc).__name__}: {str(exc)[:400]}"
            ep = None
    if ep is None:
        ep = torch.export.export(bound, sample)
    # Read before lowering, which runs its own passes over the program.
    metas = _guards(ep, leaves, tensor_pos)
    edge = exir.to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    delegated, left = _graph_report(edge.exported_program())
    if delegated == 0:
        raise ExecuTorchUnsupported(
            f"nn.Module.to(torchnative.device.npu): the {DELEGATE} partitioner took no "
            f"part of this graph (ops left: {sorted(left)}), so nothing would run in "
            f"the delegate. Refusing rather than running the whole program on "
            f"ExecuTorch's portable kernels under an npu label.")
    buf = bytes(edge.to_executorch().buffer)
    prog = torch._C._et_load(buf)
    method = prog.load_method("forward")
    meta = dict(prog.method_meta("forward"))
    info = {
        "shapes": shapes_mode,
        "max_seq_len": max_seq_len if shapes_mode == "symbolic-seq" else None,
        "export_refusal": export_refusal,
        "program_bytes": len(buf),
        "delegated_subgraphs": delegated,
        "left_outside_delegate": left,
        "method_backends": list(meta.get("backends") or []),
        "counters": None,
    }
    return _Program(spec, tensor_pos, consts, metas, record["out_spec"], record["out_tensor_pos"],
                    record["out_consts"], method, info)


def _guards(ep, leaves, tensor_pos):
    """Per tensor input, its dtype and each dimension as an int or the
    (lower, upper) range the exported program holds for it, read from
    `ep.range_constraints`, not from what was asked for."""
    placeholders = {n.name: n for n in ep.graph.nodes if n.op == "placeholder"}
    names = list(ep.graph_signature.user_inputs)
    metas = []
    for k, i in enumerate(tensor_pos):
        t = leaves[i]
        val = placeholders[names[k]].meta.get("val") if k < len(names) else None
        dims = []
        for d, size in enumerate(t.shape):
            got = val.shape[d] if val is not None else int(size)
            if isinstance(got, int):
                dims.append(int(got))
                continue
            rng = ep.range_constraints.get(got.node.expr)
            lo = int(rng.lower) if rng is not None else int(size)
            up = rng.upper if rng is not None else int(size)
            try:
                up = int(up)
            except (TypeError, ValueError, OverflowError):
                up = int(size)  # an unbounded range: keep only what was seen
            dims.append((lo, up))
        metas.append((t.dtype, dims))
    return metas


# ------------------------------------------------------------------ forward

class ExecuTorchForward:
    """The `forward` `.to(device.npu)` installs: compile on first fit, then run.

    Holds the class's own forward, bound to the module, so the module is
    captured through the code it was written with. `programs` is the list of
    compiled programs; `report` (also the module's `.torchnative_offload`) is
    updated after each compile and each call."""

    def __init__(self, model, report, max_seq_len):
        import functools
        import inspect

        self._model = model
        self._forward = type(model).forward.__get__(model)
        # transformers reads `inspect.signature(model.forward)` to decide what
        # generate passes (cache_position, logits_to_keep, ...): it must see
        # the module's own signature, not `*args, **kwargs`.
        functools.update_wrapper(self, self._forward)
        self.__signature__ = inspect.signature(self._forward)
        self._max_seq_len = int(max_seq_len)
        self.programs = []
        self.report = report

    def __call__(self, *args, **kwargs):
        import torch
        from torch.utils import _pytree as pytree

        self._refuse_cache(kwargs)
        if "use_cache" not in kwargs and "use_cache" in self.__signature__.parameters:
            # Unset, a transformers forward reads `config.use_cache` (True for
            # a causal LM) and builds a DynamicCache inside the traced forward
            # (run 44980924096: Qwen3's `past_key_values.update` was in the
            # graph). R5a has no cache, so it says so: the output's
            # `past_key_values` is None where upstream's is a DynamicCache,
            # until R5b (issue #423).
            kwargs = dict(kwargs, use_cache=False)
        leaves, spec, tensor_pos, consts = _flatten_call(args, kwargs)
        for i in tensor_pos:
            if leaves[i].requires_grad and torch.is_grad_enabled():
                raise ExecuTorchUnsupported(
                    "nn.Module.to(torchnative.device.npu): an ExecuTorch program has no "
                    "backward; input requires grad. Call under torch.no_grad().")
        program = next((p for p in self.programs if p.fits(spec, tensor_pos, consts, leaves)), None)
        if program is None:
            program = self._compile(leaves, spec, tensor_pos, consts)
        outs = program.method.execute([leaves[i].contiguous() for i in tensor_pos])
        program.info["counters"] = dict(program.method.counters())
        flat = list(program.out_consts)
        for i, t in zip(program.out_tensor_pos, outs):
            flat[i] = t
        return pytree.tree_unflatten(flat, program.out_spec)

    def _refuse_cache(self, kwargs):
        if kwargs.get("use_cache"):
            raise ExecuTorchUnsupported(
                "nn.Module.to(torchnative.device.npu): the ExecuTorch path runs full "
                "forwards without a KV cache in this build (issue #423, step R5a; the "
                "StaticCache as mutable buffers is step R5b). Call generate(..., "
                "use_cache=False). Refusing rather than dropping the cache.")
        past = kwargs.get("past_key_values")
        if past is not None:
            raise ExecuTorchUnsupported(
                f"nn.Module.to(torchnative.device.npu): past_key_values "
                f"({type(past).__name__}) is a KV cache, which the ExecuTorch path does "
                f"not take in this build (issue #423, step R5b).")

    def _compile(self, leaves, spec, tensor_pos, consts):
        max_seq_len = self._max_seq_len
        _, length = _seq_dims(leaves, tensor_pos)
        while length is not None and length > max_seq_len:
            max_seq_len *= 2
        program = _lower(self._model, self._forward, leaves, spec, tensor_pos, consts, max_seq_len)
        self.programs.append(program)
        self.report["programs"].append(program.info)
        self.report["compiles"] = len(self.programs)
        if program.info["left_outside_delegate"]:
            self.report["fully_delegated"] = False
            left = ", ".join(f"{k} x{v}" for k, v in sorted(program.info["left_outside_delegate"].items()))
            warnings.warn(
                f"nn.Module.to(torchnative.device.npu): a PARTIAL delegation. "
                f"{program.info['delegated_subgraphs']} subgraph(s) run in the {DELEGATE} "
                f"(on the CPU); ops left on ExecuTorch's portable kernels: {left}. The "
                f"report is on the model as `.torchnative_offload`.",
                UserWarning,
                stacklevel=4,
            )
        if program.info["export_refusal"]:
            warnings.warn(
                f"nn.Module.to(torchnative.device.npu): torch.export refused a symbolic "
                f"sequence length for this call ({program.info['export_refusal'][:200]}), "
                f"so this program has static shapes and another length compiles again.",
                UserWarning,
                stacklevel=4,
            )
        return program

    def counters(self):
        """Each program's method counters, read now."""
        return [dict(p.method.counters()) for p in self.programs]


def _compile_model(model, *, max_seq_len=None):
    """Swap `model.forward` for an `ExecuTorchForward`; return `(model, report)`.

    In place, as the openvino and coreml arms are: the same `nn.Module` comes
    back with its parameters, `state_dict()` and children."""
    refusal = runtime_refusal()
    if refusal:
        raise ExecuTorchUnsupported(f"nn.Module.to(torchnative.device.npu) through ExecuTorch: {refusal}")
    if isinstance(model.__dict__.get("forward"), ExecuTorchForward):
        raise ExecuTorchUnsupported("nn.Module.to(torchnative.device.npu): this module is already "
                                    "on the ExecuTorch path")
    if max_seq_len is None:
        cfg = getattr(model, "config", None)
        max_seq_len = getattr(cfg, "max_position_embeddings", None) or DEFAULT_MAX_SEQ_LEN
    if int(max_seq_len) < 1:
        raise ValueError(f"max_seq_len must be at least 1, got {max_seq_len!r}")
    report = {
        "backend": BACKEND,
        "delegate": DELEGATE,
        "unit": UNIT,
        "max_seq_len": int(max_seq_len),
        "compiles": 0,
        "fully_delegated": True,
        "programs": [],
    }
    model.eval()
    model.forward = ExecuTorchForward(model, report, max_seq_len)
    return model, report
