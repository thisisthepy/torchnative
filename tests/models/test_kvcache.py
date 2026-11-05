"""`PreallocatedCache` gives `generate` the same answer as `DynamicCache`, without `torch.cat` (issue #180).

transformers' `DynamicLayer.update` concatenates the whole cache with each
step's keys and values, so an n-token decode moves O(n^2) bytes.
`torchnative.kernels.kv_cache.PreallocatedLayer` writes into a buffer in place
and returns a narrowed view of it.

What this file asserts, each case in a fresh subprocess that imports the
vendored tree and asserts `hasattr(torch._C, "_aten_implemented")`, on a tiny
random-weight Qwen3 (two layers, GQA, no download):

* *agrees*: greedy `generate` of 24 tokens with `PreallocatedCache(chunk=4)`
  (so the buffer grows four times) yields the same token ids as with
  `DynamicCache`, and every step's logits are **bit-identical**, on `cpu` and
  on `mps`. The attention inputs hold the same values in a different layout,
  so a difference here is an indexing defect, not rounding.
* *no cat*: one decode step issues exactly `2 * num_hidden_layers` fewer
  `aten.cat` calls than with `DynamicCache` (the cache's own, one each for keys
  and values per layer); the model's other cats (rotary `rotate_half`) remain.
* *in place on mps*: a decode SDPA over the narrowed view opens exactly one
  compute encoder (the fused kernel), so the view is not copied first.
* *refusal*: a config with a `sliding_attention` layer raises
  `UnsupportedCacheLayout` naming it.

What this cannot find: speed, and models whose attention reads the cache
through something other than `keys`/`values` (none of the bench's do).
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

_CHILD = r"""
import json, struct, sys
req = json.loads(sys.argv[1])
import torch
assert hasattr(torch._C, "_aten_implemented"), "the child imported upstream, not the shim"
from transformers import Qwen3Config, Qwen3ForCausalLM, DynamicCache
from torchnative.kernels.kv_cache import PreallocatedCache, UnsupportedCacheLayout
out = {"no_mps": None}
dev = req["device"]
if dev == "mps":
    try:
        torch.ones(1, device="mps")
    except Exception as e:
        out["no_mps"] = "no mps device -- %s" % str(e).splitlines()[0]
        print(json.dumps(out)); raise SystemExit(0)

cfg = Qwen3Config(vocab_size=96, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                  max_position_embeddings=256, tie_word_embeddings=True)
cfg._attn_implementation = "sdpa"
torch.manual_seed(179)
model = Qwen3ForCausalLM(cfg).eval().to(dev)
ids = torch.tensor([[3, 17, 5, 42, 8, 11, 60]], device=dev)

if req["case"] == "agree":
    def run(cache):
        with torch.no_grad():
            g = model.generate(ids, attention_mask=torch.ones_like(ids), past_key_values=cache,
                               max_new_tokens=24, min_new_tokens=24, do_sample=False,
                               output_logits=True, return_dict_in_generate=True,
                               pad_token_id=0)
        toks = g.sequences[0].tolist()
        logits = [struct.pack("<%df" % l.numel(), *l.float().cpu().reshape(-1).tolist()).hex()
                  for l in g.logits]
        return toks, logits
    t_dyn, l_dyn = run(DynamicCache(config=cfg))
    t_pre, l_pre = run(PreallocatedCache(config=cfg, chunk=4))
    out["tokens_equal"] = t_dyn == t_pre
    out["n_tokens"] = len(t_pre) - ids.shape[1]
    out["logits_equal_steps"] = sum(a == b for a, b in zip(l_dyn, l_pre))
    out["steps"] = len(l_dyn)
elif req["case"] == "cats":
    from torch.utils._python_dispatch import TorchDispatchMode
    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.cat = 0
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) == "aten.cat.default":
                self.cat += 1
            return func(*args, **(kwargs or {}))
    res = {}
    for name, cache in (("dynamic", DynamicCache(config=cfg)),
                        ("prealloc", PreallocatedCache(config=cfg, chunk=4))):
        with torch.no_grad():
            o = model(ids, past_key_values=cache, use_cache=True)
            nxt = o.logits[:, -1:].argmax(-1)
            pos = torch.tensor([[ids.shape[1]]], device=dev)
            with Count() as c:
                model(nxt, past_key_values=cache, use_cache=True, position_ids=pos,
                      cache_position=pos[0])
        res[name] = c.cat
    out["cats"] = res
    out["layers"] = cfg.num_hidden_layers
elif req["case"] == "view":
    import torch.nn.functional as F
    cache = PreallocatedCache(config=cfg, chunk=64)
    k0 = torch.randn(1, 2, 37, 64, device=dev)
    k, v = cache.layers[0].update(k0, k0.clone()) if cache.layers else cache.update(k0, k0.clone(), 0)
    q = torch.randn(1, 4, 1, 64, device=dev)
    c0 = torch._C._metal_counters()["compute_encoders"]
    F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    out["encoders"] = torch._C._metal_counters()["compute_encoders"] - c0
    out["stride"] = list(k.stride())
elif req["case"] == "refuse":
    sw = Qwen3Config(num_hidden_layers=2, layer_types=["full_attention", "sliding_attention"],
                     sliding_window=16, use_sliding_window=True)
    try:
        PreallocatedCache(config=sw)
        out["raised"] = None
    except UnsupportedCacheLayout as e:
        out["raised"] = str(e)
print(json.dumps(out))
"""


def _run(case, device):
    if not os.path.isfile(_VENDOR_SHIM):
        raise _skip.Skip("vendored tree not built: no %s" % _VENDOR_SHIM)
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps({"case": case, "device": device})],
                          capture_output=True, text=True, timeout=900, env=env)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert proc.returncode == 0 and lines, (
        "the %s child (%s) exited %d:\n%s" % (device, case, proc.returncode, proc.stderr[-3000:]))
    out = json.loads(lines[-1])
    if out["no_mps"]:
        raise _skip.Skip(out["no_mps"])
    return out


def _check_agree(device):
    out = _run("agree", device)
    assert out["n_tokens"] == 24, out
    assert out["tokens_equal"], f"{device}: PreallocatedCache generated different tokens"
    assert out["logits_equal_steps"] == out["steps"] == 24, (
        f"{device}: logits bit-identical on {out['logits_equal_steps']} of {out['steps']} steps")
    print(f"    kvcache: {device}: 24 greedy tokens and all 24 steps' logits bit-identical to "
          "DynamicCache (chunk=4, four growths)")


def test_preallocated_cache_generates_bit_identically_on_cpu():
    _check_agree("cpu")


def test_preallocated_cache_generates_bit_identically_on_mps():
    _check_agree("mps")


def test_a_decode_step_issues_no_cat_for_the_cache():
    out = _run("cats", "cpu")
    cats, layers = out["cats"], out["layers"]
    assert cats["dynamic"] - cats["prealloc"] == 2 * layers, (cats, layers)
    print(f"    kvcache: one decode step: {cats['dynamic']} aten.cat with DynamicCache, "
          f"{cats['prealloc']} with PreallocatedCache (2 x {layers} layers fewer)")


def test_the_fused_mps_decode_reads_the_cache_view_without_a_copy():
    """The view's head stride is the buffer's (64 rows), not 37: one encoder means no copy kernel ran."""
    out = _run("view", "mps")
    assert out["stride"] == [8192, 4096, 64, 1], out["stride"]
    assert out["encoders"] == 1, out
    print("    kvcache: a decode SDPA over the narrowed cache view (stride %s) opens 1 compute encoder"
          % out["stride"])


def test_a_sliding_window_layer_refuses_by_name():
    out = _run("refuse", "cpu")
    assert out["raised"] and "sliding_attention" in out["raised"], out
    print("    kvcache: a sliding_attention layer refuses by name")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_kvcache")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
