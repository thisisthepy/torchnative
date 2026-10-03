"""Intel NPU: a fused gated-MLP lowering and a dynamic row axis (GitHub issue #3).

`docs/devices/NPUFUSE.md` is the design record. Two capabilities, both behind
the one public entry point `model.to(torchnative.device.npu)` and develop's
private `intelnpu._compile_model` / `_NPULinear` path:

1. `down(silu(gate(x)) * up(x))` lowered as **one** OpenVINO graph
   (`intelnpu._NPUGatedMLP`, IR from `intelnpu.mlp_ir`), not three leaves plus
   host-side elementwise work.
2. The row axis (batch x sequence, flattened) compiled as `-1`, so one compiled
   model serves every prompt length and `generate()` stops recompiling.

**What grade each test here reaches, stated per test and summarised here
(AGENTS.md §16).** There is no Intel NPU on the Mac this was written on.

* **Pure / faked** tests always run in the gate. They are evidence about the
  IR text, the matcher, the dispatch through `to(device.npu)` and the compile
  counter -- with `intelnpu.OpenVINO` replaced by a stand-in that computes
  the graph it was handed from the weight blob it was handed. They are **not**
  evidence that anything ran on an NPU.
* **Real-runtime** tests need `TORCHNATIVE_OPENVINO_C` naming an
  `openvino_c` library, and skip **by name** without it. Against OpenVINO's
  **CPU** plugin they reach IR acceptance, a real compile counter, and
  element-wise agreement with upstream torch. The NPU plugin may still refuse
  the dynamic axis; the Windows procedure in NPUFUSE.md is what settles that.
  **The gate does not set `TORCHNATIVE_OPENVINO_C`, so in the gate these
  skip** -- they were run by hand for this round with the runtime in
  `.caches/bw-intelnpu-ov`, and NPUFUSE.md records that run.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENDOR_DIR = os.path.join(_ROOT, "python")
sys.path.insert(0, _VENDOR_DIR)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import torchnative.device as D  # noqa: E402
from torchnative.export import intelnpu  # noqa: E402
from torchnative.export.intelnpu import (  # noqa: E402
    MAX_DIM,
    IntelNPUUnavailable,
    IntelNPUUnsupported,
    linear_ir,
    mlp_ir,
)
import _skip  # noqa: E402

_OV_ENV = "TORCHNATIVE_OPENVINO_C"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _layers(xml):
    root = ET.fromstring(xml)
    layers = {int(l.get("id")): l for l in root.find("layers")}
    edges = [
        (int(e.get("from-layer")), int(e.get("to-layer")), int(e.get("to-port")))
        for e in root.find("edges")
    ]
    return root, layers, edges


def _types(xml):
    _, layers, _ = _layers(xml)
    counts = {}
    for layer in layers.values():
        counts[layer.get("type")] = counts.get(layer.get("type"), 0) + 1
    return counts


def _tiny_llama(seed=0, **overrides):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = dict(vocab_size=97, hidden_size=32, intermediate_size=88,
               num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, max_position_embeddings=64)
    cfg.update(overrides)
    return LlamaForCausalLM(LlamaConfig(**cfg)).eval()


def _llama_mlp(seed=0, **overrides):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP

    torch.manual_seed(seed)
    cfg = dict(hidden_size=16, intermediate_size=40, hidden_act="silu",
               num_attention_heads=4, num_key_value_heads=2)
    cfg.update(overrides)
    return LlamaMLP(LlamaConfig(**cfg)).eval()


class _SwappedRoles(nn.Module):
    """Structurally a gated MLP; functionally `down(silu(up(x)) * gate(x))`.

    The three children, their names, their shapes, the absence of a bias and
    the activation are all exactly what the matcher looks for. Only the
    forward differs, so a matcher that reads structure alone accepts this and
    lowers a different function.
    """

    def __init__(self, hidden=16, intermediate=40):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x))


class _FakeOpenVINO:
    """Stands in for `intelnpu.OpenVINO`, and for nothing above it.

    It **computes the graph it was handed**: it reads the IR with an XML
    parser, decides whether it is `torchnative_linear` or `torchnative_mlp`,
    reads the declared row dimension (`-1` or a number), slices the weight
    blob the module packed, and evaluates that function in torch. So a wrong
    weight order, a dropped activation, or a static IR fed a different row
    count produces a wrong number or an exception here, not a green test.

    `dynamic_shapes = True` is the capability the real class declares; a
    stand-in without it takes the static path (which is why the older fakes
    in `test_npuwire.py` / `test_ovpar.py` keep their old behaviour).

    `refuse_dynamic=True` makes it behave like a device that will not compile
    a `-1` axis -- the case NPUFUSE.md section 3 says the NPU plugin may be.
    """

    dynamic_shapes = True
    refuse_dynamic = False
    instances = []

    def __init__(self, path=None, cache_dir=None):
        self.compiled = []
        self.infers = 0
        _FakeOpenVINO.instances.append(self)

    def devices(self):
        return ("CPU", "NPU")

    def compile_ir(self, xml, device, weights):
        root, layers, _ = _layers(xml)
        param = next(l for l in layers.values() if l.get("type") == "Parameter")
        rows, width = (int(d) for d in param.find("data").get("shape").split(","))
        if rows == -1 and self.refuse_dynamic:
            raise IntelNPUUnavailable(
                "torchnative intelnpu: ov_core_compile_model(device=NPU) failed "
                "with ov_status_e GENERAL_ERROR (-1) -- fake: dynamic shapes "
                "are not supported by this device"
            )
        n = len(weights) // 2
        flat = intelnpu.f16_tensor(weights, (n,)).to(torch.float32)
        handle = {"kind": root.get("name"), "rows": rows, "width": width,
                  "device": device, "types": _types(xml), "infers": 0}
        if handle["kind"] == "torchnative_mlp":
            consts = [l for l in layers.values() if l.get("type") == "Const"]
            mats = []
            for c in sorted(consts, key=lambda l: int(l.find("data").get("offset"))):
                a, b = (int(s) for s in c.find("data").get("shape").split(","))
                off = int(c.find("data").get("offset")) // 2
                mats.append(flat[off:off + a * b].reshape(a, b))
            handle["gate"], handle["up"], handle["down"] = mats
            handle["out"] = width
        else:
            out_f = int(next(
                l for l in layers.values() if l.get("name") == "weight"
            ).find("data").get("shape").split(",")[0])
            handle["w"] = flat[: out_f * width].reshape(out_f, width)
            handle["b"] = flat[out_f * width:] if n > out_f * width else None
            handle["out"] = out_f
        self.compiled.append(handle)
        return handle

    def execution_devices(self, compiled):
        return (compiled["device"],)

    def infer(self, compiled, blob, shape=None):
        if compiled["rows"] == -1:
            assert shape is not None, (
                "a dynamic model was run without a concrete shape -- the "
                "real runtime has no buffer to copy into in that case"
            )
            rows = int(shape[0])
        else:
            rows = compiled["rows"]
            if shape is not None:
                assert int(shape[0]) == rows, (shape, rows)
        x = intelnpu.f16_tensor(blob, (rows, compiled["width"])).to(torch.float32)
        if compiled["kind"] == "torchnative_mlp":
            g = x @ compiled["gate"].t()
            u = x @ compiled["up"].t()
            y = (torch.nn.functional.silu(g) * u) @ compiled["down"].t()
        else:
            y = x @ compiled["w"].t()
            if compiled["b"] is not None:
                y = y + compiled["b"]
        compiled["infers"] += 1
        self.infers += 1
        return intelnpu.f16_bytes(y)


class _RefusingOpenVINO(_FakeOpenVINO):
    refuse_dynamic = True


class _on_a_faked_intel_npu_host:
    """`TORCHNATIVE_DEVICE_HOST=windows`, the probe answering yes, and a fake runtime.

    The resolver, `_module_to`, `_compile_model`, `_NPULinear`,
    `_NPUGatedMLP`, the IR emitters and the verdict are all the real ones.
    """

    def __init__(self, runtime=_FakeOpenVINO):
        self.runtime = runtime

    def __enter__(self):
        self._host = os.environ.get("TORCHNATIVE_DEVICE_HOST")
        os.environ["TORCHNATIVE_DEVICE_HOST"] = "windows"
        self._saved = (intelnpu.npu_available, intelnpu.available_devices,
                       intelnpu.OpenVINO)
        intelnpu.npu_available = lambda *a, **k: True
        intelnpu.available_devices = lambda *a, **k: ("CPU", "NPU")
        intelnpu.OpenVINO = self.runtime
        _FakeOpenVINO.instances = []
        return self

    def __exit__(self, *exc):
        (intelnpu.npu_available, intelnpu.available_devices,
         intelnpu.OpenVINO) = self._saved
        if self._host is None:
            os.environ.pop("TORCHNATIVE_DEVICE_HOST", None)
        else:
            os.environ["TORCHNATIVE_DEVICE_HOST"] = self._host


def _all_compiled():
    return [h for ov in _FakeOpenVINO.instances for h in ov.compiled]


def _lowered_modules(model):
    return [m for m in model.modules()
            if type(m).__name__ in ("_NPULinear", "_NPUGatedMLP")]


# --------------------------------------------------------------------------
# Pure: the IR documents
# --------------------------------------------------------------------------


def test_mlp_ir_is_one_graph_with_the_gate_activated_and_not_the_up():
    """PURE (builds): the fused IR is one graph, and the SiLU is on the gate.

    Counted, not inferred: exactly one Parameter, one Result, three Consts,
    three MatMuls, one Swish, one Multiply. Then the edges: Swish reads the
    `gate` MatMul, Multiply reads (Swish, `up`), and `down` reads Multiply.
    Swapping gate and up is the fault a structural count cannot see -- the
    counts are identical -- and on random weights it is a different function.
    """
    hidden, inter = 6, 10
    xml = mlp_ir(hidden, inter, None)
    counts = _types(xml)
    assert counts == {"Parameter": 1, "Const": 3, "MatMul": 3, "Swish": 1,
                      "Multiply": 1, "Result": 1}, counts
    _, layers, edges = _layers(xml)
    by_name = {l.get("name"): i for i, l in layers.items()}
    into = {}
    for src, dst, port in edges:
        into.setdefault(dst, {})[port] = layers[src].get("name")
    assert into[by_name["silu"]] == {0: "gate"}, into[by_name["silu"]]
    assert into[by_name["gated"]] == {0: "silu", 1: "up"}, into[by_name["gated"]]
    assert into[by_name["down"]][0] == "gated", into[by_name["down"]]
    proj = inter * hidden * 2
    offsets = {
        l.get("name"): (int(l.find("data").get("offset")), int(l.find("data").get("size")))
        for l in layers.values() if l.get("type") == "Const"
    }
    assert offsets == {"gate_weight": (0, proj), "up_weight": (proj, proj),
                       "down_weight": (2 * proj, proj)}, offsets
    print("ok   npufuse: mlp_ir is one graph (3 MatMul, 1 Swish, 1 Multiply) "
          "with the activation on the gate and the blob in gate/up/down order "
          "[pure]")


def test_the_dynamic_axis_is_minus_one_on_activations_and_never_on_weights():
    """PURE (builds): `batch=None` spells the row axis `-1`, and only there."""
    for xml in (linear_ir(6, 4, None, True), mlp_ir(6, 10, None)):
        _, layers, _ = _layers(xml)
        param = next(l for l in layers.values() if l.get("type") == "Parameter")
        assert param.find("data").get("shape").startswith("-1,"), param.find("data").attrib
        for layer in layers.values():
            dims_per_port = [
                [int(d.text) for d in port.findall("dim")]
                for port in layer.iter("port")
            ]
            if layer.get("type") == "Const":
                assert all(-1 not in dims for dims in dims_per_port), layer.attrib
                assert "-1" not in layer.find("data").get("shape"), layer.attrib
            else:
                activation_ports = [d for d in dims_per_port if d and d[0] == -1]
                assert activation_ports, (layer.get("name"), dims_per_port)
    # A static spelling is still available and still static.
    assert "<dim>-1</dim>" not in linear_ir(6, 4, 3, True)
    assert "<dim>-1</dim>" not in mlp_ir(6, 10, 3)
    for bad in (0, -1):
        for build in (lambda: linear_ir(6, 4, bad), lambda: mlp_ir(6, 10, bad)):
            try:
                build()
            except IntelNPUUnsupported as exc:
                assert "row" in str(exc), exc
            else:
                raise AssertionError(f"batch={bad} was accepted")
    print("ok   npufuse: batch=None puts -1 on every activation port and on no "
          "weight; a non-positive static batch refuses by name [pure]")


def test_mlp_ir_refuses_an_oversized_dimension_by_name():
    """PURE: MAX_DIM is enforced for the fused IR exactly as for linear_ir."""
    for args in ((MAX_DIM + 1, 8), (8, MAX_DIM + 1)):
        try:
            mlp_ir(*args)
        except IntelNPUUnsupported as exc:
            assert "MAX_DIM" in str(exc), exc
        else:
            raise AssertionError(f"mlp_ir{args} was accepted")
    print("ok   npufuse: mlp_ir refuses a dimension above MAX_DIM by name [pure]")


# --------------------------------------------------------------------------
# Pure: the matcher
# --------------------------------------------------------------------------


def test_the_matcher_accepts_transformers_own_llama_mlp():
    """PURE (reaches): the real `LlamaMLP` class is accepted, by behaviour."""
    mlp = _llama_mlp()
    reason = intelnpu._gated_mlp_refusal(mlp)
    assert reason is None, reason
    lowered = intelnpu._NPUGatedMLP.from_torch(mlp, "CPU")
    assert type(lowered).__name__ == "_NPUGatedMLP", type(lowered).__name__
    assert isinstance(lowered, nn.Module)
    assert (lowered.hidden, lowered.intermediate) == (16, 40)
    print("ok   npufuse: transformers' LlamaMLP is accepted as a gated MLP [pure]")


def test_each_unsupported_pattern_refuses_by_name_and_says_what_is_missing():
    """PURE: every refusal names the thing that is not there.

    Each case is also handed to `_NPUGatedMLP` directly, which must raise
    `IntelNPUUnsupported` carrying the same reason -- the matcher and the
    constructor cannot disagree, because the constructor calls the matcher.
    """
    big = nn.Module()
    big.gate_proj = nn.Linear(1, MAX_DIM + 1, bias=False)
    big.up_proj = nn.Linear(1, MAX_DIM + 1, bias=False)
    big.down_proj = nn.Linear(MAX_DIM + 1, 1, bias=False)

    not_linear = _llama_mlp()
    not_linear.gate_proj = nn.Identity()

    mismatched = _llama_mlp()
    mismatched.up_proj = nn.Linear(16, 24, bias=False)

    cases = [
        ("a GELU activation", _llama_mlp(hidden_act="gelu_pytorch_tanh"),
         ("does not compute", "silu")),
        ("gate and up swapped in forward", _SwappedRoles(),
         ("does not compute", "silu")),
        ("a biased projection", _llama_mlp(mlp_bias=True), ("bias",)),
        ("gate_proj that is not a Linear", not_linear, ("gate_proj", "torch.nn.Linear")),
        ("mismatched shapes", mismatched, ("shape", "up_proj")),
        ("an oversized dimension", big, ("MAX_DIM",)),
    ]
    for label, mod, needles in cases:
        reason = intelnpu._gated_mlp_refusal(mod)
        assert reason is not None, f"{label}: the matcher accepted it"
        for needle in needles:
            assert needle in reason, f"{label}: {needle!r} not in {reason!r}"
        try:
            intelnpu._NPUGatedMLP(mod, "CPU")
        except IntelNPUUnsupported as exc:
            assert reason in str(exc), (label, str(exc))
            assert str(exc).startswith("torchnative intelnpu:"), exc
        else:
            raise AssertionError(f"{label}: _NPUGatedMLP accepted it")
    # Not a candidate at all: no refusal is invented for a module that is not
    # shaped like a gated MLP.
    assert intelnpu._gated_mlp_candidate(nn.Sequential(nn.Linear(2, 2))) is False
    print(f"ok   npufuse: {len(cases)} unsupported gated-MLP patterns each "
          f"refuse by name, saying what is missing [pure]")


# --------------------------------------------------------------------------
# Faked host: the dispatch through to(device.npu)
# --------------------------------------------------------------------------


def test_to_device_npu_lowers_each_gated_mlp_as_one_compiled_model():
    """FAKED RUNTIME (dispatch): one compiled model per MLP, counted.

    Counted three ways that do not depend on each other: the compiled models
    the fake was handed whose IR is `torchnative_mlp` (one per MLP, each with
    three MatMuls), the report's `fused` list, and the modules in the tree.
    The three projections must no longer exist as separate leaves.
    """
    model = _tiny_llama()
    mlp_paths = [n for n, m in model.named_modules() if type(m).__name__ == "LlamaMLP"]
    assert len(mlp_paths) == 2, mlp_paths
    with _on_a_faked_intel_npu_host():
        result = model.to(D.npu)
        compiled = _all_compiled()
    assert result is model
    report = model.torchnative_offload
    assert report["fused"] == mlp_paths, report["fused"]
    assert report["unfused"] == [], report["unfused"]
    mlp_graphs = [h for h in compiled if h["kind"] == "torchnative_mlp"]
    assert len(mlp_graphs) == len(mlp_paths), [h["kind"] for h in compiled]
    for h in mlp_graphs:
        assert h["types"]["MatMul"] == 3 and h["types"]["Swish"] == 1, h["types"]
    for path in mlp_paths:
        mod = model.get_submodule(path)
        assert type(mod).__name__ == "_NPUGatedMLP", (path, type(mod).__name__)
        names = {n for n, _ in mod.named_children()}
        assert not names & {"gate_proj", "up_proj", "down_proj"}, names
    leaf_paths = [p for p in report["swapped"] if p not in mlp_paths]
    assert not any(p.startswith(tuple(f"{m}." for m in mlp_paths)) for p in leaf_paths), leaf_paths
    # One compile per lowered module, no more: the whole tree, counted.
    assert len(compiled) == len(report["swapped"]), (len(compiled), len(report["swapped"]))
    print(f"ok   npufuse: to(device.npu) lowered {len(mlp_paths)} gated MLPs as "
          f"{len(mlp_graphs)} compiled models, {len(compiled)} compiles for "
          f"{len(report['swapped'])} lowered modules (dispatch; the NPU is faked)")


def test_the_lowered_mlp_hands_the_runtime_the_right_function():
    """FAKED RUNTIME (dispatch): the blob and IR describe THIS module's function.

    The fake evaluates whatever graph and weight blob it was handed, so this
    catches a lowering that packs the weights in the wrong order or loses one
    -- which the counting tests above cannot see. The bound is f16's 8-ulp
    relative floor (the IR is f16); the control is the answer with gate and
    up exchanged, which must sit far outside it, or the check could not fail.
    Not agreement with upstream: that is the real-runtime test below.
    """
    import copy

    model = _tiny_llama()
    original = copy.deepcopy(model.model.layers[0].mlp)
    torch.manual_seed(7)
    x = torch.randn(5, 32)
    with torch.no_grad():
        want = original(x)
        g, u = original.gate_proj(x), original.up_proj(x)
        swapped = original.down_proj(torch.nn.functional.silu(u) * g)
    with _on_a_faked_intel_npu_host():
        model.to(D.npu)
        with torch.no_grad():
            got = model.model.layers[0].mlp(x)
    scale = float(want.abs().max())
    bound = 8 * 2.0 ** -10
    rel = float((got - want).abs().max()) / scale
    control = float((swapped - want).abs().max()) / scale
    assert control > 10 * bound, (control, bound)
    assert rel <= bound, (rel, bound)
    print(f"ok   npufuse: the fused module's blob and IR compute this MLP to "
          f"{rel:.2e} (bound {bound:.2e}; gate/up swapped would be {control:.2e}) "
          f"(dispatch; the NPU is faked)")


def test_generate_over_several_prompt_lengths_compiles_nothing_after_to():
    """FAKED RUNTIME (dispatch): the compile counter does not move in generate().

    Three prompt lengths, each with several decode steps, so the flattened row
    count takes at least four values (3, 5, 9 and 1). With a static IR that
    is at least four compiles per module. The counter is
    `intelnpu._compile_counters()["lowered"]`, and it is checked to be
    non-zero after `to()` so a counter that never increments cannot pass.
    The fake's own count of compiled models is checked alongside it, and the
    fused MLPs are checked to have been *run*, so a generate that bypassed
    them could not pass either.
    """
    model = _tiny_llama()
    with _on_a_faked_intel_npu_host():
        model.to(D.npu)
        after_to = intelnpu._compile_counters()["lowered"]
        fake_after_to = len(_all_compiled())
        modules = _lowered_modules(model)
        assert after_to >= len(modules) > 0, (after_to, len(modules))
        lengths = (3, 5, 9)
        for length in lengths:
            ids = torch.arange(1, length + 1).reshape(1, length)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
            assert tuple(out.shape) == (1, length + 3), out.shape
        after_generate = intelnpu._compile_counters()["lowered"]
        fake_after_generate = len(_all_compiled())
        mlp_runs = sum(h["infers"] for h in _all_compiled() if h["kind"] == "torchnative_mlp")
    assert after_generate == after_to, (after_to, after_generate)
    assert fake_after_generate == fake_after_to, (fake_after_to, fake_after_generate)
    assert all(m.compiles == 1 for m in modules), [m.compiles for m in modules]
    assert all(m.shape_mode == "dynamic" for m in modules), {m.shape_mode for m in modules}
    assert mlp_runs >= 2 * len(lengths) * 3, mlp_runs
    print(f"ok   npufuse: generate() over prompt lengths {lengths} compiled 0 "
          f"new models after to() ({after_to} at to(), each module exactly "
          f"once; fused MLPs ran {mlp_runs} times) (dispatch; the NPU is faked)")


def test_a_device_that_refuses_the_dynamic_axis_is_named_and_recompiles_visibly():
    """FAKED RUNTIME (dispatch): the fallback is a warning and a report, not silence.

    This is also the positive control for the test above: with the dynamic
    axis refused, the same generate() over the same lengths **does** move the
    compile counter. A counter that could not move would make "compiles
    nothing" vacuous.
    """
    model = _tiny_llama()
    with _on_a_faked_intel_npu_host(_RefusingOpenVINO):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.to(D.npu)
        report = model.torchnative_offload
        before = intelnpu._compile_counters()["lowered"]
        for length in (3, 5, 9):
            ids = torch.arange(1, length + 1).reshape(1, length)
            with torch.no_grad():
                model.generate(ids, max_new_tokens=2, do_sample=False, pad_token_id=0)
        after = intelnpu._compile_counters()["lowered"]
    modules = _lowered_modules(model)
    assert report["shape_modes"] == {"static-per-length": len(modules)}, report["shape_modes"]
    assert len(report["shape_fallbacks"]) == len(modules), report["shape_fallbacks"]
    path, why = report["shape_fallbacks"][0]
    assert "dynamic" in why and "GENERAL_ERROR" in why, why
    texts = [str(w.message) for w in caught]
    assert any("recompile" in t and "dynamic" in t for t in texts), texts
    # Each fused MLP sees rows 3, 5 and 9 (prefill) beyond the eager 1, so at
    # least four compiles each. (`lm_head` is not counted on: generate() asks
    # it for the last position only, so its row count stays 1.)
    mlps = [m for m in modules if type(m).__name__ == "_NPUGatedMLP"]
    assert len(mlps) == 2 and all(m.compiles >= 4 for m in mlps), [m.compiles for m in mlps]
    assert after - before >= 3 * len(mlps), (before, after)
    print(f"ok   npufuse: a refused dynamic axis is named in the report and a "
          f"warning, and generate() then recompiled {after - before} times -- "
          f"the counter can move (dispatch; the NPU is faked)")


def test_an_unfusable_mlp_is_named_and_its_linears_are_still_lowered():
    """FAKED RUNTIME (dispatch): refusing the fusion does not drop the leaves.

    A GELU-activated gated MLP is not this function, so it is not fused --
    and the report says why, by path. Its three Linears still go through the
    leaf path, and the activation is named as left on the CPU.
    """
    model = _tiny_llama(hidden_act="gelu_pytorch_tanh")
    with _on_a_faked_intel_npu_host():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.to(D.npu)
    report = model.torchnative_offload
    assert report["fused"] == [], report["fused"]
    assert [p for p, _ in report["unfused"]] == ["model.layers.0.mlp", "model.layers.1.mlp"], report["unfused"]
    assert all("does not compute" in why for _, why in report["unfused"]), report["unfused"]
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        assert f"model.layers.0.mlp.{leaf}" in report["swapped"], report["swapped"]
    assert report["fully_offloaded"] is False, report
    texts = " ".join(str(w.message) for w in caught)
    assert "not fused" in texts and "model.layers.0.mlp" in texts, texts
    print("ok   npufuse: an unfusable gated MLP is named with its reason and its "
          "Linears still lowered as leaves (dispatch; the NPU is faked)")


def test_the_plan_and_the_lowering_agree_on_what_is_fused():
    """PURE + FAKED: `plan_lowering` answers the same question without OpenVINO."""
    plan = intelnpu.plan_lowering(_tiny_llama())
    model = _tiny_llama()
    with _on_a_faked_intel_npu_host():
        model.to(D.npu)
    report = model.torchnative_offload
    assert plan["fused"] == report["fused"], (plan["fused"], report["fused"])
    assert plan["eligible"] == report["swapped"], (plan["eligible"], report["swapped"])
    assert plan["parameters_moved"] == report["parameters_moved"], (plan, report)
    print(f"ok   npufuse: plan_lowering and the lowering agree: {len(plan['fused'])} "
          f"fused, {len(plan['eligible'])} lowered modules")


# --------------------------------------------------------------------------
# Real OpenVINO runtime (CPU plugin). Skips by name without it.
# --------------------------------------------------------------------------

_HIDDEN, _INTER = 64, 172
_ROWS = (1, 3, 7, 12, 5)

_UPSTREAM = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), "upstream probe imported the shim"
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
torch.manual_seed(1234)
cfg = LlamaConfig(hidden_size={hidden}, intermediate_size={inter}, hidden_act="silu")
mlp = LlamaMLP(cfg).eval()
out = {{"torch": torch.__version__, "eps16": torch.finfo(torch.float16).eps,
        "state": {{k: v.flatten().tolist() for k, v in mlp.state_dict().items()}},
        "rows": []}}
mlp16 = LlamaMLP(cfg).eval(); mlp16.load_state_dict(mlp.state_dict()); mlp16 = mlp16.half()
mlp64 = LlamaMLP(cfg).eval(); mlp64.load_state_dict(mlp.state_dict()); mlp64 = mlp64.double()
for k, n in enumerate({rows}):
    torch.manual_seed(100 + k)
    x = torch.randn(n, {hidden})
    with torch.no_grad():
        up32 = mlp(x); up16 = mlp16(x.half()).double(); up64 = mlp64(x.double())
    out["rows"].append({{"x": x.flatten().tolist(), "n": n,
                         "up32": up32.flatten().tolist(),
                         "up16": up16.flatten().tolist(),
                         "up64": up64.flatten().tolist()}})
print(json.dumps(out))
"""

_SHIM = r"""
import json, os, sys
sys.path.insert(0, {vendor!r})
import torch
assert hasattr(torch._C, "_aten_implemented"), "shim probe imported upstream torch"
from torchnative.export import intelnpu as I
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
path = os.environ["TORCHNATIVE_OPENVINO_C"]
data = json.loads(sys.stdin.read())
out = {{"cache_dir": I.openvino_cache_dir()}}

cfg = LlamaConfig(hidden_size={hidden}, intermediate_size={inter}, hidden_act="silu")
mlp = LlamaMLP(cfg).eval()
shapes = {{k: tuple(v.shape) for k, v in mlp.state_dict().items()}}
mlp.load_state_dict({{k: torch.tensor(v).reshape(shapes[k]) for k, v in data["state"].items()}})
holder = torch.nn.Sequential(mlp)
c0 = I._compile_counters()
holder, report = I._compile_model(holder, device="CPU", library=path)
c1 = I._compile_counters()
out["report"] = {{k: report[k] for k in ("fused", "unfused", "swapped", "shape_modes",
                                         "shape_fallbacks", "execution_devices")}}
out["at_lowering"] = {{k: c1[k] - c0[k] for k in c1}}
lowered = holder[0]
out["type"] = type(lowered).__name__
out["got"] = []
for row in data["rows"]:
    x = torch.tensor(row["x"]).reshape(row["n"], {hidden})
    with torch.no_grad():
        out["got"].append(lowered(x).flatten().tolist())
c2 = I._compile_counters()
out["during_rows"] = {{k: c2[k] - c1[k] for k in c2}}
out["module_compiles"] = lowered.compiles
out["shape_mode"] = lowered.shape_mode

# The same graph at an explicit f32 EXECUTION precision. The IR (weights and
# activations) is f16 either way; this only asks the CPU plugin not to
# accumulate in f16, which its ARM default does. The compile cache is off for
# this lowering: with it on, OpenVINO 2026.3.1 handed back the f16 blob for
# the f32-hint compile (NPUFUSE.md section 4).
os.environ["TORCHNATIVE_OPENVINO_CACHE_DIR"] = "0"
_orig = I.OpenVINO.compile_ir
def _f32(self, xml, device="NPU", weights=None, properties=None):
    return _orig(self, xml, device, weights, {{"INFERENCE_PRECISION_HINT": "f32"}})
I.OpenVINO.compile_ir = _f32
mlp32 = LlamaMLP(cfg).eval()
mlp32.load_state_dict({{k: torch.tensor(v).reshape(shapes[k]) for k, v in data["state"].items()}})
held32, _ = I._compile_model(torch.nn.Sequential(mlp32), device="CPU", library=path)
out["got_f32_exec"] = []
for row in data["rows"]:
    x = torch.tensor(row["x"]).reshape(row["n"], {hidden})
    with torch.no_grad():
        out["got_f32_exec"].append(held32[0](x).flatten().tolist())
I.OpenVINO.compile_ir = _orig
os.environ.pop("TORCHNATIVE_OPENVINO_CACHE_DIR")

# generate() over several prompt lengths on a whole (tiny) causal LM.
torch.manual_seed(0)
lm = LlamaForCausalLM(LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=88,
                                  num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, max_position_embeddings=64)).eval()
reference = {{}}
for n in (3, 5, 9):
    ids = torch.arange(1, n + 1).reshape(1, n)
    with torch.no_grad():
        reference[n] = lm.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0)[0].tolist()
lm, lm_report = I._compile_model(lm, device="CPU", library=path)
g0 = I._compile_counters()
tokens = {{}}
for n in (3, 5, 9):
    ids = torch.arange(1, n + 1).reshape(1, n)
    with torch.no_grad():
        tokens[n] = lm.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0)[0].tolist()
g1 = I._compile_counters()
mods = [m for m in lm.modules() if type(m).__name__ in ("_NPULinear", "_NPUGatedMLP")]
out["lm"] = {{"fused": lm_report["fused"], "modules": len(mods),
             "compiles_per_module": sorted({{m.compiles for m in mods}}),
             "shape_modes": lm_report["shape_modes"],
             "during_generate": {{k: g1[k] - g0[k] for k in g1}},
             "tokens_equal": tokens == reference,
             "tokens": {{str(k): v for k, v in tokens.items()}},
             "reference": {{str(k): v for k, v in reference.items()}}}}
print(json.dumps(out))
"""

_cache = []


def _real_run():
    if _cache:
        return _cache[0]
    path = os.environ.get(_OV_ENV)
    if not path:
        _cache.append((None, f"{_OV_ENV} is not set, so there is no OpenVINO C runtime to load"))
        return _cache[0]
    if not os.path.isfile(path):
        _cache.append((None, f"{_OV_ENV}={path!r} does not name a file"))
        return _cache[0]
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")):
        _cache.append((None, "the vendored shim is not installed (run scripts/vendor/install_shim.sh)"))
        return _cache[0]
    fmt = dict(hidden=_HIDDEN, inter=_INTER, rows=_ROWS, vendor=_VENDOR_DIR)
    up_env = dict(os.environ)
    up_env.pop("PYTHONPATH", None)
    up_env.pop("TORCH_USE_RTLD_GLOBAL", None)
    up = subprocess.run([sys.executable, "-c", _UPSTREAM.format(**fmt)],
                        capture_output=True, text=True, env=up_env, cwd=_ROOT, timeout=900)
    if up.returncode != 0:
        raise RuntimeError(f"upstream probe exited {up.returncode}\n{up.stderr[-3000:]}")
    upstream = json.loads(up.stdout.strip().splitlines()[-1])
    shim_env = dict(os.environ)
    shim_env["PYTHONPATH"] = _VENDOR_DIR
    shim_env["TORCH_USE_RTLD_GLOBAL"] = "1"
    # The compile cache is left ON, in a throwaway directory: on, because
    # passing CACHE_DIR is the variadic call that segfaulted on Apple arm64
    # until `ov_core_compile_model` got its argtypes (NPUFUSE.md section 6),
    # so this run is that defect's regression check; throwaway, because the
    # default is the user's home cache and a test must not write there.
    with tempfile.TemporaryDirectory(prefix="npufuse-ovcache-") as cache:
        shim_env["TORCHNATIVE_CACHE_DIR"] = cache
        shim_env.pop("TORCHNATIVE_OPENVINO_CACHE_DIR", None)
        shim = subprocess.run([sys.executable, "-c", _SHIM.format(**fmt)],
                              input=json.dumps({"state": upstream["state"],
                                                "rows": [{"x": r["x"], "n": r["n"]}
                                                         for r in upstream["rows"]]}),
                              capture_output=True, text=True, env=shim_env, cwd=_ROOT,
                              timeout=900)
    if shim.returncode != 0:
        raise RuntimeError(f"shim probe exited {shim.returncode}\n{shim.stderr[-3000:]}")
    _cache.append(((upstream, json.loads(shim.stdout.strip().splitlines()[-1])), None))
    return _cache[0]


def _real_or_skip(what):
    payload, reason = _real_run()
    if payload is None:
        _skip.skip(f"   (skipped {what}: {reason})")
    return payload


def test_a_gated_mlp_is_one_compiled_model_on_a_real_openvino():
    """REAL RUNTIME, CPU plugin (reaches): one `ov_core_compile_model` for the MLP.

    `ov_compile` counts successful calls into `ov_core_compile_model` inside
    `OpenVINO.compile_ir`, a level below the module's own counter, so both the
    module and the runtime agree there was one compile. Five different row
    counts are then run through it and neither counter moves.
    """
    payload = _real_or_skip("one compiled model on a real OpenVINO")
    if payload is None:
        return
    _, shim = payload
    assert shim["type"] == "_NPUGatedMLP", shim["type"]
    assert shim["report"]["fused"] == ["0"], shim["report"]
    assert shim["report"]["swapped"] == ["0"], shim["report"]
    assert shim["report"]["execution_devices"] == ["CPU"], shim["report"]
    assert shim["at_lowering"] == {"lowered": 1, "ov_compile": 1}, shim["at_lowering"]
    assert shim["during_rows"] == {"lowered": 0, "ov_compile": 0}, shim["during_rows"]
    assert shim["shape_mode"] == "dynamic", shim["shape_mode"]
    # The compile went through the CACHE_DIR property (the variadic call).
    assert shim["cache_dir"] and "npufuse-ovcache-" in shim["cache_dir"], shim["cache_dir"]
    print(f"ok   npufuse: a LlamaMLP compiled to ONE OpenVINO model and served "
          f"row counts {_ROWS} without recompiling [real runtime, CPU plugin]")


def _agreement_rows(upstream, got_rows):
    rows = []
    for row, got in zip(upstream["rows"], got_rows):
        up32, up16, up64 = row["up32"], row["up16"], row["up64"]
        scale = max(abs(v) for v in up32)
        rows.append({
            "rel": max(abs(a - b) for a, b in zip(got, up32)) / scale,
            "rel64": max(abs(a - b) for a, b in zip(got, up64)) / scale,
            "oracle": max(abs(a - b) for a, b in zip(up16, up64)) / scale,
        })
    return rows


def test_the_fused_mlp_agrees_with_upstream_at_a_derived_tolerance():
    """REAL RUNTIME, CPU plugin (agrees, at f32 execution): element-wise vs upstream.

    Upstream torch and the shim are separate subprocesses; upstream builds the
    weights and the inputs and computes the reference. The IR is f16, so the
    oracle is upstream's **own f16** answer against its float64 answer --
    the precision the IR declares, measured on upstream, not chosen here
    (AGENTS.md §16; docs/numerics/AGREE.md §2's rule, at the IR's precision).
    Tolerance = p90 of that oracle, floored at 8 ulp of f16; and the 4x-ratio
    rule against the float64 truth.

    **Two execution precisions, and only one is claimed as agreement.**
    The lowered graph is run twice: with `INFERENCE_PRECISION_HINT=f32`
    (f16 weights and activations, f32 accumulation -- what upstream's own f16
    path does) and at the ARM CPU plugin's default, which executes in f16.
    The first must pass both rules, and is the claim: the fused graph computes
    upstream's function. The second must pass the derived tolerance, and its
    ratio is **reported, not asserted**: measured at 7.1x upstream's own f16
    error (NPUFUSE.md section 4), which fails the 4x rule. That is the
    plugin's accumulation precision, not the lowering -- the f32 run of the
    same graph is at 0.92x -- and it is exactly the number the Windows run
    must measure again on the NPU, whose execution precision this Mac cannot
    observe.
    """
    payload = _real_or_skip("fused-MLP agreement")
    if payload is None:
        return
    upstream, shim = payload
    exact = _agreement_rows(upstream, shim["got_f32_exec"])
    default = _agreement_rows(upstream, shim["got"])
    oracles = sorted(r["oracle"] for r in exact)
    assert min(oracles) > 0.0, oracles
    index = min(len(oracles) - 1, int(round(0.9 * (len(oracles) - 1))))
    tolerance = max(oracles[index], 8 * upstream["eps16"])
    worst = max(r["rel"] for r in exact)
    assert worst <= tolerance, (
        f"f32-execution: worst relative error {worst:.3e} exceeds the derived "
        f"tolerance {tolerance:.3e} (p90 of upstream's own f16-vs-f64 error "
        f"{oracles[index]:.3e}, floor 8 ulp f16 {8 * upstream['eps16']:.3e}); "
        f"rows={exact}"
    )
    worst_ratio = max(r["rel64"] / r["oracle"] for r in exact)
    assert worst_ratio <= 4.0, (worst_ratio, exact)
    worst_default = max(r["rel"] for r in default)
    assert worst_default <= tolerance, (worst_default, tolerance, default)
    default_ratio = max(r["rel64"] / r["oracle"] for r in default)
    print(f"ok   npufuse: the fused MLP agrees with upstream torch {upstream['torch']} "
          f"at f32 execution to {worst:.3e} (derived tolerance {tolerance:.3e}, "
          f"worst ratio to upstream's own f16 error {worst_ratio:.2f}); at the "
          f"plugin's default f16 execution {worst_default:.3e}, ratio "
          f"{default_ratio:.2f} -- reported, not claimed [real runtime, CPU plugin]")


def test_generate_over_several_prompt_lengths_compiles_once_on_a_real_openvino():
    """REAL RUNTIME, CPU plugin (reaches): zero compiles inside generate().

    A whole tiny `LlamaForCausalLM`, lowered by the same `_compile_model`
    `to(device.npu)` calls, then `generate()` for prompt lengths 3, 5 and 9.
    Every lowered module compiled exactly once, at lowering; both counters
    are flat across all three generations.

    Tokens are compared with the unlowered model and **reported, not
    asserted**: f16 weights can legitimately flip an argmax on a random
    model, and the agreement claim is the test above, not this one.
    """
    payload = _real_or_skip("generate() compile count")
    if payload is None:
        return
    _, shim = payload
    lm = shim["lm"]
    assert lm["fused"] == ["model.layers.0.mlp", "model.layers.1.mlp"], lm["fused"]
    assert lm["compiles_per_module"] == [1], lm["compiles_per_module"]
    assert lm["shape_modes"] == {"dynamic": lm["modules"]}, lm["shape_modes"]
    assert lm["during_generate"] == {"lowered": 0, "ov_compile": 0}, lm["during_generate"]
    print(f"ok   npufuse: generate() over prompt lengths 3/5/9 compiled 0 models "
          f"({lm['modules']} modules, each compiled once at lowering; tokens equal "
          f"to the unlowered model: {lm['tokens_equal']}) [real runtime, CPU plugin]")


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
                print(f"SKIP test_npufuse: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
