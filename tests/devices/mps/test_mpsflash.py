"""`F.scaled_dot_product_attention` on `mps` runs one fused Metal kernel (S7.3, issue #179).

Before this, attention on `mps` was the unfused candle formulation of
`aten._scaled_dot_product_flash_attention_for_cpu`: a matmul, a transposed
copy, a scale-and-mask pass, a row max, `exp`, a row sum, a divide and a second
matmul, each its own compute encoder. `torch_c/src/sdpa_metal.rs` replaces that
with candle-metal-kernels' `steel_attention` (prefill) and `sdpa_vector`
(decode) wherever they compute the same attention (`sdpa_metal::eligible`).

What this file asserts, each case in fresh subprocesses (the shim's with
`hasattr(torch._C, "_aten_implemented")`, upstream's without):

* *agrees*: for head dims 64 and 128 (SmolLM2, Qwen3), causal prefill at an
  aligned and an unaligned length, GQA and MHA, a non-causal decode step
  (`L == 1`) against 37 keys, in float32, float16 and bfloat16, the mps result
  is within the derived bound of upstream's float64 answer: 4x upstream's own
  distance from float64 in the same dtype (docs/numerics/AGREE.md), the rule
  tests/devices/mps/test_paged_attention.py uses.
* *placement*: the call opens **exactly one** compute encoder and downloads
  nothing. The unfused path opens at least seven for the same call, so this
  is what separates the fused kernel from the old one; agreement alone cannot.
* *the fallbacks stay*: an explicit mask, and a causal call with `L != S`
  (where the kernel's bottom-right causal mask is not upstream's top-left
  one), still agree and open more than one encoder.
* *logsumexp refuses*: the second result of the aten op is a meta tensor on
  the fused path, so reading it raises instead of returning numbers nobody
  computed.

What this cannot find: speed. That is the bench's claim (tests/bench/).
"""

import json
import os
import random
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

_UPSTREAM = r"""
import json, sys
import torch
import torch.nn.functional as F
assert not hasattr(torch._C, "_aten_implemented"), "expected upstream torch in the oracle"
req = json.loads(sys.stdin.read())
def run(dt):
    q = torch.tensor(req["q"], dtype=torch.float64).to(dt)
    k = torch.tensor(req["k"], dtype=torch.float64).to(dt)
    v = torch.tensor(req["v"], dtype=torch.float64).to(dt)
    m = None
    if req.get("mask") is not None:
        m = torch.tensor(req["mask"], dtype=torch.float64).to(dt)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=req["causal"],
                                          enable_gqa=True).to(torch.float64).reshape(-1).tolist()
print(json.dumps({"dt": run(getattr(torch, req["dtype"])), "f64": run(torch.float64)}))
"""

_SHIM = r"""
import json, sys
import torch
import torch.nn.functional as F
assert hasattr(torch._C, "_aten_implemented"), "the child imported upstream, not the shim"
req = json.loads(sys.stdin.read())
out = {"no_mps": None}
try:
    torch.ones(1, device="mps")
except Exception as e:
    out["no_mps"] = "no mps device -- %s" % str(e).splitlines()[0]
    print(json.dumps(out)); raise SystemExit(0)
dt = getattr(torch, req["dtype"])
def t(x):
    return torch.tensor(x, dtype=torch.float32).to(dt).to("mps")
q, k, v = t(req["q"]), t(req["k"]), t(req["v"])
m = t(req["mask"]) if req.get("mask") is not None else None
c0 = torch._C._metal_counters()
o = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=req["causal"], enable_gqa=True)
c1 = torch._C._metal_counters()
out["encoders"] = c1["compute_encoders"] - c0["compute_encoders"]
out["downloads"] = c1["host_downloads"] - c0["host_downloads"]
out["device"] = str(o.device)
out["dtype"] = str(o.dtype)
out["values"] = o.to(torch.float32).cpu().reshape(-1).tolist()
if req.get("lse"):
    pair = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
        q, k.repeat_interleave(q.shape[1] // k.shape[1], 1),
        v.repeat_interleave(q.shape[1] // v.shape[1], 1), 0.0, req["causal"])
    lse = pair[1]
    out["lse_shape"] = list(lse.shape)
    try:
        lse.tolist()
        out["lse_read"] = None
    except Exception as e:
        out["lse_read"] = type(e).__name__
print(json.dumps(out))
"""


def _env(shim):
    env = dict(os.environ)
    env.pop("BW_SDPA_REFERENCE", None)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
    return env


def _child(script, req, shim):
    proc = subprocess.run([sys.executable, "-c", script], input=json.dumps(req),
                          capture_output=True, text=True, timeout=600, env=_env(shim))
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert proc.returncode == 0 and lines, (
        "%s child exited %d:\n%s" % ("shim" if shim else "upstream", proc.returncode,
                                     proc.stderr[-3000:]))
    return json.loads(lines[-1])


def _rand(rng, shape):
    if len(shape) == 1:
        return [rng.uniform(-1.0, 1.0) for _ in range(shape[0])]
    return [_rand(rng, shape[1:]) for _ in range(shape[0])]


def _case(seed, b, h, hkv, l, s, d, dtype, causal, mask=False, lse=False):
    rng = random.Random(seed)
    req = {"q": _rand(rng, [b, h, l, d]), "k": _rand(rng, [b, hkv, s, d]),
           "v": _rand(rng, [b, hkv, s, d]), "dtype": dtype, "causal": causal, "lse": lse}
    if mask:
        req["mask"] = _rand(rng, [b, 1, l, s])
    return req


def _check(req, label):
    if not os.path.isfile(_VENDOR_SHIM):
        raise _skip.Skip("vendored tree not built: no %s" % _VENDOR_SHIM)
    got = _child(_SHIM, req, shim=True)
    if got["no_mps"]:
        raise _skip.Skip(got["no_mps"])
    up = _child(_UPSTREAM, req, shim=False)
    assert got["device"] == "mps:0" and got["dtype"] == "torch." + req["dtype"], (label, got["device"], got["dtype"])
    vals, up_dt, up_f64 = got["values"], up["dt"], up["f64"]
    assert len(vals) == len(up_f64), (label, len(vals), len(up_f64))
    worst_up = max(abs(a - b) for a, b in zip(up_dt, up_f64))
    worst = max(abs(a - b) for a, b in zip(vals, up_f64))
    floor = max(max(abs(x) for x in up_f64), 1.0) * 2.0 ** -21
    bound = max(4.0 * worst_up, floor)
    assert worst <= bound, (
        f"{label}: max |mps - upstream f64| = {worst:.3e} > bound {bound:.3e} "
        f"(upstream {req['dtype']} is {worst_up:.3e} from f64)")
    assert got["downloads"] == 0, (label, got)
    return got, worst, bound


# (label, b, h, hkv, l, s, d, causal)
_FUSED = [
    ("smollm2-like causal prefill L=S=37", 1, 9, 3, 37, 37, 64, True),
    ("smollm2-like causal prefill L=S=64", 2, 9, 3, 64, 64, 64, True),
    ("qwen3-like causal prefill L=S=37", 1, 16, 8, 37, 37, 128, True),
    ("mha non-causal prefill L=5 S=40", 1, 4, 4, 5, 40, 64, False),
    ("qwen3-like decode L=1 S=37", 1, 16, 8, 1, 37, 128, False),
    ("smollm2-like decode L=1 S=37", 2, 9, 3, 1, 37, 64, False),
]


def _check_fused(dtype):
    lines = []
    for i, (label, b, h, hkv, l, s, d, causal) in enumerate(_FUSED):
        got, worst, bound = _check(_case(178 + i, b, h, hkv, l, s, d, dtype, causal), f"{dtype} {label}")
        assert got["encoders"] == 1, (
            f"{dtype} {label}: {got['encoders']} compute encoders; the fused kernel opens exactly one "
            "(the unfused path opens at least seven)")
        lines.append(f"{label}: {worst:.2e} <= {bound:.2e}")
    print(f"    mpsflash: {dtype}: {len(_FUSED)} shapes fused (1 encoder each) and agree: " + "; ".join(lines))


def test_fused_sdpa_on_mps_agrees_in_float32_with_one_encoder():
    _check_fused("float32")


def test_fused_sdpa_on_mps_agrees_in_float16_with_one_encoder():
    _check_fused("float16")


def test_fused_sdpa_on_mps_agrees_in_bfloat16_with_one_encoder():
    _check_fused("bfloat16")


def test_masked_and_offset_causal_calls_keep_the_unfused_path_and_agree():
    for label, req in (
        ("explicit mask", _case(1781, 1, 4, 2, 6, 6, 64, "float32", False, mask=True)),
        ("causal L=3 S=9 (top-left != bottom-right)", _case(1782, 1, 4, 2, 3, 9, 64, "float32", True)),
        ("head dim 8 (not instantiated)", _case(1783, 1, 4, 2, 6, 6, 8, "float32", True)),
    ):
        got, worst, bound = _check(req, label)
        assert got["encoders"] > 1, (label, got["encoders"])
    print("    mpsflash: masked, offset-causal and head-dim-8 calls stay unfused and agree")


def test_the_fused_path_returns_a_logsumexp_that_refuses_to_be_read():
    req = _case(1784, 1, 4, 4, 8, 8, 64, "float32", True, lse=True)
    if not os.path.isfile(_VENDOR_SHIM):
        raise _skip.Skip("vendored tree not built: no %s" % _VENDOR_SHIM)
    got = _child(_SHIM, req, shim=True)
    if got["no_mps"]:
        raise _skip.Skip(got["no_mps"])
    assert got["lse_shape"] == [1, 4, 8], got["lse_shape"]
    assert got["lse_read"] is not None, "the fused path's logsumexp read back as numbers"
    print(f"    mpsflash: the fused logsumexp has shape [1, 4, 8] and reading it raises {got['lse_read']}")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsflash")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
