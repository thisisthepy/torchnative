"""Issue #146: what Gemstone does on the continuous-batching path beyond greedy
identity (tests/models/test_cbidentity.py): switching the attention back and
forth, reading the mean-pooled last hidden state, and q8_0 weights under
`paged|sdpa`. float32, cpu, a tiny Qwen3 with weights set by a seeded formula,
the same on both sides. Each side runs in its own process.

1. **Switching.** One model, `set_attn_implementation` sdpa -> paged|sdpa ->
   sdpa -> paged|sdpa, a greedy run at each stage (plain `generate` under sdpa,
   continuous batching under paged|sdpa). Every stage's tokens must be
   upstream's, and the relations upstream shows between stages (the two sdpa
   stages equal, the two paged stages equal, sdpa equal to paged) must be the
   relations the shim shows.
2. **Hidden state.** `output_hidden_states=True`, the last hidden state
   mean-pooled over the tokens, for three prompts of different lengths. Grade
   *agrees* (AGENTS.md section 16): the bound is derived from upstream re-run in
   float64, the p90 of its float32 error relative to the pooled vector's scale,
   floored at 8 ulp, or 4x upstream's own worst distance from float64
   (docs/numerics/AGREE.md section 2, as tests/devices/npu/test_etnpu.py).
3. **q8_0 with continuous batching.** `torchnative.quant.quantize_(model,
   "q8_0")`, then `paged|sdpa`: the batched greedy tokens must equal each
   request run alone and the same model's plain greedy `generate`. Upstream has
   no q8_0, so this is the shim against itself; the quantised kernels
   themselves are held to their reference in tests/numerics/test_qgemm.py.

What this cannot find: real checkpoints (the tiny model has the architecture,
not the size); bf16 (batched differs from alone upstream, issue #15); sampled
runs (the two torch RNGs differ); and transformers 5.18 or Python 3.14, which
need their own environments (issue #146's remaining legs).
"""

import json
import os
import subprocess
import sys

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests", "_support"))
import _skip  # noqa: E402

_VENDOR_DIR = os.environ.get("TORCHNATIVE_VENDOR_DIR", os.path.join(_REPO_ROOT, "torchnative", "python"))
_NO_CB = "transformers continuous batching (ContinuousBatchingConfig) is unavailable"
_EPS32 = 1.1920929e-07
_ORACLE_FACTOR = 4.0


def _run_side(script: str, shim: bool, timeout: int = 600) -> dict:
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{'shim' if shim else 'upstream'} side failed "
                           f"(rc={proc.returncode}):\n{proc.stderr[-4000:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out.pop("is_shim") is shim, "the side imported the wrong torch"
    return out


_SCRIPT = r"""
import json, random, time
import torch
import transformers
from transformers import GenerationConfig
IS_SHIM = hasattr(torch._C, "_aten_implemented")
MODE = __MODE__
try:
    from transformers.generation.configuration_utils import ContinuousBatchingConfig
    import transformers.generation.continuous_batching.input_outputs as cbio
except Exception as e:
    print(json.dumps({"is_shim": IS_SHIM, "unavailable": repr(e)})); raise SystemExit(0)
if not IS_SHIM:
    cbio.get_available_devices = lambda: ["cpu"]   # as test_cbpath.py (issue #30)

def build(dtype, attn):
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=16, max_position_embeddings=1024, tie_word_embeddings=False,
                                   attn_implementation=attn)
    model = transformers.Qwen3ForCausalLM(cfg)
    rng = random.Random(1234)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            vals = [rng.uniform(-0.2, 0.2) for _ in range(p.numel())]
            if name.endswith("norm.weight"):
                vals = [1.0 + v for v in vals]
            p.copy_(torch.tensor(vals, dtype=torch.float64).reshape(p.shape).to(dtype))
    return model.eval()

prng = random.Random(1)
SPEC = [(5, 3), (23, 9), (41, 6)]          # (prompt length, max_new_tokens)
PROMPTS = [[prng.randrange(3, 256) for _ in range(n)] for n, _ in SPEC]
REQS = list(zip(PROMPTS, [m for _, m in SPEC]))
BLOCK, BLOCKS = 16, 24

def cb(model, reqs):
    gen = dict(max_new_tokens=max(m for _, m in reqs), eos_token_id=-1, pad_token_id=0, do_sample=False)
    cbc = ContinuousBatchingConfig(block_size=BLOCK, num_blocks=BLOCKS, max_batch_tokens=32,
                                   use_cuda_graph=False, allow_block_sharing=False)
    errors, toks = [], {}
    with model.continuous_batching_context_manager(
            generation_config=GenerationConfig(**gen), continuous_batching_config=cbc,
            block=True, timeout=5, warmup=False) as mgr:
        ids = [mgr.add_request(p, request_id=f"r{i}", max_new_tokens=m) for i, (p, m) in enumerate(reqs)]
        done, t0 = 0, time.time()
        while done < len(reqs) and time.time() - t0 < 120:
            r = mgr.get_result(timeout=1)
            if r is None:
                continue
            if r.error:
                errors.append(str(r.error)[:200])
            if r.is_finished():
                toks[r.request_id] = list(map(int, r.generated_tokens))
                done += 1
        if getattr(mgr, "fatal_error", None) is not None:
            errors.append(f"fatal: {type(mgr.fatal_error).__name__}")
    return [toks.get(i) for i in ids], errors

def plain(model, reqs):
    out = []
    for p, m in reqs:
        x = torch.tensor([p], dtype=torch.long)
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=m,
                           do_sample=False, eos_token_id=-1, pad_token_id=0)
        out.append(y[0, len(p):].tolist())
    return out

out = {"is_shim": IS_SHIM}
with torch.no_grad():
    if MODE == "switch":
        model = build(torch.float32, "sdpa")
        stages, errors = [], []
        for attn in ("sdpa", "paged|sdpa", "sdpa", "paged|sdpa"):
            model.set_attn_implementation(attn)
            if attn == "sdpa":
                stages.append(plain(model, REQS))
            else:
                t, e = cb(model, REQS)
                stages.append(t); errors += e
        out.update(stages=stages, errors=errors,
                   attn_after=model.config._attn_implementation)
    elif MODE == "hidden":
        pooled = {}
        for name, dtype in (("f32", torch.float32), ("f64", torch.float64)):
            if name == "f64" and IS_SHIM:
                continue
            model = build(dtype, "sdpa")
            rows = []
            for p in PROMPTS:
                x = torch.tensor([p], dtype=torch.long)
                h = model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True).hidden_states
                rows.append(h[-1].mean(dim=1)[0].tolist())
            pooled[name] = rows
            out["layers"] = len(h)
        out["pooled"] = pooled
    elif MODE == "q8_0":
        if not IS_SHIM:
            print(json.dumps({"is_shim": IS_SHIM, "skipped": "upstream has no q8_0"})); raise SystemExit(0)
        from torchnative.quant import quantize_, QuantizedLinear
        model = build(torch.float32, "sdpa")
        quantize_(model, "q8_0")
        out["quantized"] = sorted(n for n, m in model.named_modules() if isinstance(m, QuantizedLinear))
        out["plain"] = plain(model, REQS)
        model.set_attn_implementation("paged|sdpa")
        out["batched"], e1 = cb(model, REQS)
        alone, e2 = [], []
        for r in REQS:
            t, e = cb(model, [r]); alone += t; e2 += e
        out.update(alone=alone, errors=e1 + e2)
print(json.dumps(out))
"""


def _side(mode: str, shim: bool) -> dict:
    out = _run_side(_SCRIPT.replace("__MODE__", repr(mode)), shim=shim)
    if "unavailable" in out:
        raise _skip.Skip(f"{_NO_CB}: {out['unavailable']}")
    return out


def _relations(stages):
    """Which stages equal which: the facts compared across sides."""
    n = len(stages)
    return [[stages[i] == stages[j] for j in range(n)] for i in range(n)]


def test_switching_sdpa_and_paged_back_and_forth_keeps_upstreams_tokens():
    up = _side("switch", shim=False)
    sh = _side("switch", shim=True)
    for name, side in (("upstream", up), ("shim", sh)):
        assert not side["errors"], (name, side["errors"])
        assert all(t is not None and [len(x) for x in t] == [3, 9, 6] for t in side["stages"]), (name, side["stages"])
        assert side["attn_after"] == "paged|sdpa", (name, side["attn_after"])
    for k, (a, b) in enumerate(zip(sh["stages"], up["stages"])):
        assert a == b, f"stage {k}: shim {a} != upstream {b}"
    assert _relations(sh["stages"]) == _relations(up["stages"]), (_relations(sh["stages"]), _relations(up["stages"]))
    print(f"CBFAMILY switch: 4 stages equal upstream's; stage relations {_relations(up['stages'])}")


def _bound(up, up64):
    """docs/numerics/AGREE.md section 2, as tests/devices/npu/test_etnpu.py derives it."""
    scale = max(abs(v) for v in up64)
    assert scale > 0, "degenerate output: every value is zero"
    own = sorted(abs(a - b) / scale for a, b in zip(up, up64))
    p90 = own[int(0.9 * (len(own) - 1))]
    tol = max(p90, 8 * _EPS32)
    return max(tol, _ORACLE_FACTOR * own[-1]) * scale, tol, own[-1], scale


def test_mean_pooled_last_hidden_state_agrees_with_upstream():
    up = _side("hidden", shim=False)
    sh = _side("hidden", shim=True)
    assert sh["layers"] == up["layers"] == 3, (sh["layers"], up["layers"])
    worst = 0.0
    for k, (got, want, want64) in enumerate(zip(sh["pooled"]["f32"], up["pooled"]["f32"], up["pooled"]["f64"])):
        assert len(got) == len(want) == 64, (k, len(got), len(want))
        bound, tol, own, scale = _bound(want, want64)
        dist = max(abs(a - b) for a, b in zip(got, want))
        assert dist <= bound, (f"prompt {k}: shim is {dist:.3e} from upstream, bound {bound:.3e} "
                               f"(p90 {tol:.3e}, upstream's own worst {own:.3e}, scale {scale:.3e})")
        worst = max(worst, dist / bound)
    print(f"CBFAMILY hidden: 3 pooled vectors agree, worst {worst:.2f} of the bound")


def test_q8_0_under_paged_sdpa_batched_equals_alone_equals_plain_generate():
    sh = _side("q8_0", shim=True)
    assert len(sh["quantized"]) == 15, sh["quantized"]   # 7 per layer x 2 + lm_head
    assert not sh["errors"], sh["errors"]
    assert [len(t) for t in sh["batched"]] == [3, 9, 6], sh["batched"]
    assert sh["batched"] == sh["alone"], (sh["batched"], sh["alone"])
    assert sh["batched"] == sh["plain"], (sh["batched"], sh["plain"])
    print(f"CBFAMILY q8_0: {len(sh['quantized'])} QuantizedLinear, batched == alone == plain generate")


def _main():
    failures = _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        "models/test_cbfamily")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
