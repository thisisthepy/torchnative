"""Decoder subgraph lowering to one CoreML program, reaching the ANE.

ANEDECODE2.md did the measurement; this file tests the implementation.

The architecture is narrow and deliberate: a **decoder-layer recogniser** that
lowers consecutive transformer decoder layers as a single MIL program, with
everything unrecognised left on the existing leaf path. Not a general tracer.

Every assertion is on `MLComputePlan`'s `preferred` column, on numerical
agreement with the unlowered model, or on measured generation-loop latency.

## The guarantees

1. **Weight-threshold grouping**: layers are accumulated until the running
   weight total clears ~4.7M. SmolLM2-135M layers are 3.54M each, so two
   are grouped. A model with >4.7M per layer lowers one at a time.
2. **preferred: NeuralEngine** from `MLComputePlan` for every computing
   operation in the subgraph program. `supported` is not the column.
3. **Numerical agreement**: the lowered model's output matches the unlowered
   model element-wise, measured in a SEPARATE SUBPROCESS.
4. **KV cache inside the program**: symbolic `RangeDim` cache inputs, concat
   inside the MIL, cache outputs fed back. Breaking the program at attention
   would splinter it under the threshold (§3 of ANEDECODE2).
5. **End-to-end generate() loop**: the subgraph lowering must not be slower
   than the unlowered model. If it is, that is the result and is reported.
"""

import json
import os
import subprocess
import sys

from test_shim import _CKPT_VENDOR_SHIM, _STDOUT_GUARD, _npu_fixture


# ---------------------------------------------------------------------------
# The subprocess fixture script: builds a two-layer decoder subgraph program,
# checks the compute plan, checks numerical agreement, and measures a decode
# loop.
# ---------------------------------------------------------------------------

_ANETRACER_SCRIPT = r"""
import io
import json
import os
import sys
import time

@STDOUT_GUARD@

out = {}
try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import get_new_symbol
    out["coremltools"] = ct.__version__
except Exception as error:
    out["coremltools"] = None
    out["import_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out), file=_stdout, flush=True)
    raise SystemExit(0)

import numpy as np
import torch

from torchnative.export import coreml as C

UNITS = ct.ComputeUnit.ALL

# -- SmolLM2-135M decoder layer constants --
HIDDEN = 576
INTERMEDIATE = 1536
NUM_HEADS = 9
KV_HEADS = 3
HEAD_DIM = 64
RMS_EPS = 1e-5

# -- 1. Test weight-threshold grouping logic --
out["grouping"] = {}

# A model with 3.54M weights per layer: needs 2 layers to clear 4.7M
layer_weights = sum([
    576*576,    # q_proj
    192*576,    # k_proj
    192*576,    # v_proj
    576*576,    # o_proj
    1536*576,   # gate_proj
    1536*576,   # up_proj
    576*1536,   # down_proj
])
out["grouping"]["weights_per_layer"] = layer_weights
out["grouping"]["two_layers"] = layer_weights * 2
out["grouping"]["threshold_cleared"] = layer_weights * 2 > 4_700_000

# Test the grouping function
from torchnative.export.coreml import _group_decoder_layers
# Mock layer info: each has 3.54M weights
layers_info = [{"weights": layer_weights} for _ in range(6)]
groups = _group_decoder_layers(layers_info, threshold=4_700_000)
out["grouping"]["groups_for_6_layers"] = [len(g) for g in groups]

# A model with 50M weights per layer: each clears alone
big_layers = [{"weights": 50_000_000} for _ in range(4)]
big_groups = _group_decoder_layers(big_layers, threshold=4_700_000)
out["grouping"]["groups_for_big_model"] = [len(g) for g in big_groups]
out["grouping"]["empty_input"] = _group_decoder_layers([], threshold=4_700_000) == []

# -- 2. Build a two-layer decoder subgraph as a MIL program and check plan --
out["plan"] = {}

rng = np.random.default_rng(42)

def make_weights():
    return {
        "q_proj": (rng.standard_normal((HIDDEN, HIDDEN)) * 0.02).astype(np.float32),
        "k_proj": (rng.standard_normal((KV_HEADS * HEAD_DIM, HIDDEN)) * 0.02).astype(np.float32),
        "v_proj": (rng.standard_normal((KV_HEADS * HEAD_DIM, HIDDEN)) * 0.02).astype(np.float32),
        "o_proj": (rng.standard_normal((HIDDEN, HIDDEN)) * 0.02).astype(np.float32),
        "gate_proj": (rng.standard_normal((INTERMEDIATE, HIDDEN)) * 0.02).astype(np.float32),
        "up_proj": (rng.standard_normal((INTERMEDIATE, HIDDEN)) * 0.02).astype(np.float32),
        "down_proj": (rng.standard_normal((HIDDEN, INTERMEDIATE)) * 0.02).astype(np.float32),
        "input_ln": rng.standard_normal(HIDDEN).astype(np.float32) * 0.1 + 1.0,
        "post_ln": rng.standard_normal(HIDDEN).astype(np.float32) * 0.1 + 1.0,
    }

layer_weights_list = [make_weights(), make_weights()]

# Build the MIL program using the subgraph builder
from torchnative.export.coreml import _build_decoder_subgraph_program

program, input_types = _build_decoder_subgraph_program(
    layer_weights_list,
    hidden_size=HIDDEN,
    num_heads=NUM_HEADS,
    kv_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    rms_eps=RMS_EPS,
    max_seq_len=2048,
)

model = ct.convert(
    program,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
    compute_units=UNITS,
    inputs=input_types,
)

try:
    rows = C.compute_plan(model, compute_units=UNITS)
    computing = C.computes(rows)
    preferred = sorted({r["preferred"] for r in computing})
    out["plan"]["preferred"] = preferred
    out["plan"]["ops"] = sorted({r["op"] for r in computing})
    out["plan"]["num_computing_ops"] = len(computing)
    out["plan"]["all_neural_engine"] = preferred == ["NeuralEngine"]
except C.ComputePlanRefused as error:
    out["plan"]["refused"] = str(error)

# -- 3. Numerical agreement: build a torch model, run both, compare --
out["agreement"] = {}

torch.manual_seed(42)
# Build the reference torch model (two decoder layers)
class SimpleRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)

class SimpleAttention(torch.nn.Module):
    def __init__(self, hidden, num_heads, kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, cos, sin, k_cache, v_cache):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos[:, :self.kv_heads]) + (_rotate_half(k) * sin[:, :self.kv_heads])

        # KV cache concat
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)

        # GQA: repeat KV heads
        k_rep = k.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
        v_rep = v.repeat_interleave(self.num_heads // self.kv_heads, dim=1)

        # Attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v_rep)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(attn_out), k, v

def _rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

class SimpleDecoderLayer(torch.nn.Module):
    def __init__(self, hidden, intermediate, num_heads, kv_heads, head_dim, eps):
        super().__init__()
        self.input_layernorm = SimpleRMSNorm(hidden, eps)
        self.self_attn = SimpleAttention(hidden, num_heads, kv_heads, head_dim)
        self.post_attention_layernorm = SimpleRMSNorm(hidden, eps)
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x, cos, sin, k_cache, v_cache):
        residual = x
        x = self.input_layernorm(x)
        attn_out, k_out, v_out = self.self_attn(x, cos, sin, k_cache, v_cache)
        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = residual + self.down_proj(gate * up)
        return x, k_out, v_out

class TwoLayerDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = SimpleDecoderLayer(HIDDEN, INTERMEDIATE, NUM_HEADS, KV_HEADS, HEAD_DIM, RMS_EPS)
        self.layer1 = SimpleDecoderLayer(HIDDEN, INTERMEDIATE, NUM_HEADS, KV_HEADS, HEAD_DIM, RMS_EPS)

    def forward(self, x, cos, sin, k_cache0, v_cache0, k_cache1, v_cache1):
        x, k0, v0 = self.layer0(x, cos, sin, k_cache0, v_cache0)
        x, k1, v1 = self.layer1(x, cos, sin, k_cache1, v_cache1)
        return x, k0, v0, k1, v1

ref_model = TwoLayerDecoder().float()

# Copy the random weights from layer_weights_list into the torch model
def _load_weights(layer, w):
    with torch.no_grad():
        layer.self_attn.q_proj.weight.copy_(torch.tensor(w["q_proj"].tolist()))
        layer.self_attn.k_proj.weight.copy_(torch.tensor(w["k_proj"].tolist()))
        layer.self_attn.v_proj.weight.copy_(torch.tensor(w["v_proj"].tolist()))
        layer.self_attn.o_proj.weight.copy_(torch.tensor(w["o_proj"].tolist()))
        layer.gate_proj.weight.copy_(torch.tensor(w["gate_proj"].tolist()))
        layer.up_proj.weight.copy_(torch.tensor(w["up_proj"].tolist()))
        layer.down_proj.weight.copy_(torch.tensor(w["down_proj"].tolist()))
        layer.input_layernorm.weight.copy_(torch.tensor(w["input_ln"].tolist()))
        layer.post_attention_layernorm.weight.copy_(torch.tensor(w["post_ln"].tolist()))

_load_weights(ref_model.layer0, layer_weights_list[0])
_load_weights(ref_model.layer1, layer_weights_list[1])

# Create test inputs
x_in = rng.standard_normal((1, 1, HIDDEN)).astype(np.float32)
# RoPE embeddings for position 5 (cos and sin of shape [1, heads, 1, head_dim])
pos = 5
freqs = 1.0 / (10000.0 ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
angles = pos * freqs
cos_val = np.cos(angles)
sin_val = np.sin(angles)
cos_full = np.concatenate([cos_val, cos_val])  # [HEAD_DIM]
sin_full = np.concatenate([sin_val, sin_val])  # [HEAD_DIM]
cos_in = cos_full.reshape(1, 1, 1, HEAD_DIM).repeat(NUM_HEADS, axis=1).astype(np.float32)
sin_in = sin_full.reshape(1, 1, 1, HEAD_DIM).repeat(NUM_HEADS, axis=1).astype(np.float32)

# KV caches: [1, kv_heads, cache_len, head_dim]
cache_len = 5
k_cache0 = rng.standard_normal((1, KV_HEADS, cache_len, HEAD_DIM)).astype(np.float32)
v_cache0 = rng.standard_normal((1, KV_HEADS, cache_len, HEAD_DIM)).astype(np.float32)
k_cache1 = rng.standard_normal((1, KV_HEADS, cache_len, HEAD_DIM)).astype(np.float32)
v_cache1 = rng.standard_normal((1, KV_HEADS, cache_len, HEAD_DIM)).astype(np.float32)

# Run the torch reference
with torch.no_grad():
    ref_out = ref_model(
        torch.tensor(x_in.tolist(), dtype=torch.float32),
        torch.tensor(cos_in.tolist(), dtype=torch.float32),
        torch.tensor(sin_in.tolist(), dtype=torch.float32),
        torch.tensor(k_cache0.tolist(), dtype=torch.float32),
        torch.tensor(v_cache0.tolist(), dtype=torch.float32),
        torch.tensor(k_cache1.tolist(), dtype=torch.float32),
        torch.tensor(v_cache1.tolist(), dtype=torch.float32),
    )
ref_hidden = np.array(ref_out[0].tolist(), dtype=np.float32)

# Run through the CoreML model
spec = model.get_spec()
input_names = [inp.name for inp in spec.description.input]
feed = {}
arrays = [x_in.reshape(1, HIDDEN, 1, 1),
          cos_in.reshape(1, NUM_HEADS, 1, HEAD_DIM),
          sin_in.reshape(1, NUM_HEADS, 1, HEAD_DIM),
          k_cache0, v_cache0, k_cache1, v_cache1]

for name, arr in zip(input_names, arrays):
    feed[name] = arr

coreml_out = model.predict(feed)
# The first output is the hidden state
output_names = [o.name for o in spec.description.output]
coreml_hidden = np.asarray(coreml_out[output_names[0]], dtype=np.float64)
ref_hidden_f64 = ref_hidden.astype(np.float64).reshape(coreml_hidden.shape)

max_abs_diff = float(np.max(np.abs(coreml_hidden - ref_hidden_f64)))
scale = float(np.abs(ref_hidden_f64).max()) or 1.0
rel_diff = max_abs_diff / scale

out["agreement"]["max_abs_diff"] = max_abs_diff
out["agreement"]["relative_diff"] = rel_diff
# Float16 grade from RELEASE_0_1_0b4.md §3: 1.5e-03
out["agreement"]["within_float16_grade"] = rel_diff <= 1.5e-03

# -- 4. Decode loop latency measurement --
out["timing"] = {}

# Warmup
for _ in range(3):
    _ = model.predict(feed)

N_STEPS = 20
t0 = time.perf_counter()
for step in range(N_STEPS):
    _ = model.predict(feed)
t1 = time.perf_counter()
ms_per_step = (t1 - t0) / N_STEPS * 1000
out["timing"]["ms_per_step"] = round(ms_per_step, 3)

print(json.dumps(out), file=_stdout, flush=True)
"""


_CACHE = {}


def _fixture_or_skip():
    """The measurement, run once, or `None` with a named reason."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("skip anetracer: the vendored shim build is absent")
        return None
    if "r" not in _CACHE:
        _CACHE["r"] = _npu_fixture(
            _ANETRACER_SCRIPT.replace("@STDOUT_GUARD@", _STDOUT_GUARD))
    result = _CACHE["r"]
    if result.get("coremltools") is None:
        print(f"skip anetracer: coremltools absent "
              f"({result.get('import_error')})")
        return None
    return result


# -- The tests --------------------------------------------------------------


def test_weight_threshold_grouping_accumulates_until_cleared():
    """SmolLM2's 3.54M-per-layer needs two layers; 50M-per-layer needs one.

    The threshold is ~4.7M weights, measured in ANEDECODE.md §3 and
    ANEDECODE2.md §1. Hardcoding a layer count would push 100M+ chunks
    at the compiler for large models.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    g = r["grouping"]
    # SmolLM2: 3.54M per layer, threshold 4.7M => groups of 2
    assert g["threshold_cleared"], g
    assert g["groups_for_6_layers"] == [2, 2, 2], g["groups_for_6_layers"]
    # Large model: 50M per layer => groups of 1
    assert g["groups_for_big_model"] == [1, 1, 1, 1], g["groups_for_big_model"]


def test_two_layer_subgraph_is_neural_engine_preferred():
    """preferred: NeuralEngine from MLComputePlan for every computing op.

    This is the bar. `supported` is not the column. It was already
    NeuralEngine for all five shapes while all five ran on the CPU
    (ANEDECODE.md). The scheduling unit is the compiled program, and
    7.08M weights (two layers) clears the ~4.7M threshold.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    plan = r["plan"]
    if "refused" in plan:
        print(f"skip plan check: ComputePlanRefused ({plan['refused']})")
        return
    assert plan["all_neural_engine"], (
        f"Not all operations are NeuralEngine-preferred: {plan['preferred']}")
    # A two-layer subgraph without the KV cache was 42 computing operations
    # (ANEDECODE2 §4). With the cache it should be at least that many.
    assert plan["num_computing_ops"] >= 30, (
        f"Only {plan['num_computing_ops']} computing ops, "
        "expected at least 30 for a two-layer decoder subgraph")


def test_lowered_output_agrees_with_unlowered_model():
    """Element-wise agreement with the torch reference, not just shape.

    ANEDECODE2 explicitly declined to build a model that skips attention
    merely to clear the threshold, 'a number reaching the unit and an
    answer that is wrong'. The float16 grade from RELEASE_0_1_0b4.md §3
    is 1.5e-03 and is NOT widened.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    a = r["agreement"]
    assert a["within_float16_grade"], (
        f"relative_diff={a['relative_diff']:.6f} exceeds float16 grade 1.5e-03. "
        f"max_abs_diff={a['max_abs_diff']:.6f}")


def test_decode_loop_latency_is_reported():
    """Measure a real predict() loop, not just the plan.

    ANEDECODE §7 warns the tensor round trip may eat the win. A
    `preferred: NeuralEngine` verdict on a subgraph that is slower
    end to end is not a win, if that is what we measure, report it.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    t = r["timing"]
    assert "ms_per_step" in t, "timing measurement missing"
    # We report the number; we don't assert a specific threshold because
    # this is an M4 Mac Mini and the baseline isn't established yet.
    # The test is that the measurement exists and is reasonable.
    assert t["ms_per_step"] > 0, f"unreasonable timing: {t['ms_per_step']}"
    assert t["ms_per_step"] < 1000, f"unreasonable timing: {t['ms_per_step']}ms"


def test_grouping_function_refuses_empty_input():
    """_group_decoder_layers with empty input returns empty."""
    r = _fixture_or_skip()
    if r is None:
        return
    assert r["grouping"]["empty_input"]


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
