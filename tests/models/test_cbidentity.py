"""Issue #15 core criteria: transformers continuous batching (`paged|sdpa`) on cpu,
float32, tiny random-init Qwen3.

  1. greedy tokens of a batched run == each request run alone (and == upstream);
  2. requests with different prompt lengths and different max_new_tokens finish at
     different steps and each still equals its alone run;
  3. seeded sampling. Measured on upstream transformers 5.15.1: there is NO
     per-request RNG stream. `ContinuousBatchingConfig.seed` is applied once with
     `torch.manual_seed` (`continuous_batching/distributed.py:182`, called from
     `continuous_api.py:245`) and every step draws one `torch.multinomial` over all
     scheduled rows (`model_runner.py:208`), so a request's draws depend on which
     rows share its step. The criterion therefore cannot hold upstream, and
     torchnative must produce upstream's behaviour (AGENTS.md §11), not a stronger
     one. The test pins the facts that hold on both sides: the same seed and the
     same batch repeat exactly, and the shim shows the same alone-vs-batched
     relation upstream shows. Sampled values are not compared across sides (the two
     torch RNGs differ).
  4. one request that alone cannot fit the KV cache fails every in-flight request
     upstream; the shim must fail the same way (issue #15, Gemstone's note).

bf16 is excluded: batched != alone there even upstream (issue #15 comment).
All other requests keep `prompt + max_new_tokens <= KV cache` (24 blocks x 16).
Each side runs in its own process (as test_cbpath.py). AGENTS.md §22: written
before the shim build that runs it.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())), "tests", "_support"))
import _skip  # noqa: E402

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.environ.get("TORCHNATIVE_VENDOR_DIR", os.path.join(_REPO_ROOT, "torchnative", "python"))
_NO_CB = "transformers continuous batching (generate_batch / ContinuousBatchingConfig) is unavailable"


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
import json, random
import torch
import transformers
from transformers import GenerationConfig
IS_SHIM = hasattr(torch._C, "_aten_implemented")
MODE = __MODE__
try:
    from transformers.generation.configuration_utils import ContinuousBatchingConfig
    import transformers.generation.continuous_batching.input_outputs as cbio
    assert hasattr(transformers.Qwen3ForCausalLM, "generate_batch")
except Exception as e:
    print(json.dumps({"is_shim": IS_SHIM, "unavailable": repr(e)})); raise SystemExit(0)
if not IS_SHIM:
    cbio.get_available_devices = lambda: ["cpu"]   # as test_cbpath.py (issue #30)

common = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              max_position_embeddings=1024, tie_word_embeddings=False,
              attn_implementation="paged|sdpa")
model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**common))
rng = random.Random(1234)
with torch.no_grad():
    for name, p in sorted(model.named_parameters()):
        vals = [rng.uniform(-0.2, 0.2) for _ in range(p.numel())]
        if name.endswith("norm.weight"):
            vals = [1.0 + v for v in vals]
        p.copy_(torch.tensor(vals, dtype=torch.float32).reshape(p.shape))
model.eval()

prng = random.Random(1)
SPEC = [(5, 3), (23, 9), (41, 6)]          # (prompt length, max_new_tokens)
PROMPTS = [[prng.randrange(3, 256) for _ in range(n)] for n, _ in SPEC]
BLOCK, BLOCKS = 16, 24                      # KV cache = 384 tokens
assert all(n + m <= BLOCK * BLOCKS for n, m in SPEC)

def run(reqs, sample, seed=None):
    # reqs: list of (prompt, max_new_tokens). Returns (tokens by index, errors)
    gen = dict(max_new_tokens=max(m for _, m in reqs), eos_token_id=-1, pad_token_id=0)
    if sample:
        gen.update(do_sample=True, temperature=0.7, top_k=20, top_p=0.9)
    else:
        gen.update(do_sample=False)
    cbc = ContinuousBatchingConfig(block_size=BLOCK, num_blocks=BLOCKS, max_batch_tokens=32,
                                   use_cuda_graph=False, allow_block_sharing=False, seed=seed)
    errors = []
    toks = {}
    with model.continuous_batching_context_manager(
            generation_config=GenerationConfig(**gen), continuous_batching_config=cbc,
            block=True, timeout=5, warmup=False) as mgr:
        ids = [mgr.add_request(p, request_id=f"r{i}", max_new_tokens=m)
               for i, (p, m) in enumerate(reqs)]
        done = 0
        import time
        t0 = time.time()
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

out = {"is_shim": IS_SHIM}
reqs = list(zip(PROMPTS, [m for _, m in SPEC]))
if MODE == "greedy":
    with torch.no_grad():
        batched, e1 = run(reqs, False)
        alone = []
        e2 = []
        for r in reqs:
            t, e = run([r], False)
            alone += t; e2 += e
    out.update(batched=batched, alone=alone, errors=e1 + e2,
               lens=[len(t) for t in batched], want=[m for _, m in SPEC])
elif MODE == "sample":
    with torch.no_grad():
        a, ea = run(reqs, True, seed=7)
        b, eb = run(reqs, True, seed=7)
        solo, es = run([reqs[1]], True, seed=7)
    out.update(batched_a=a, batched_b=b, solo1=solo[0], errors=ea + eb + es,
               repeat_equal=(a == b), alone_equals_batched=(solo[0] == a[1]))
elif MODE == "oversize":
    # request 1 alone needs 380 + 40 > 384 cache tokens; requests 0 and 2 fit
    big = [prng.randrange(3, 256) for _ in range(380)]
    mix = [(PROMPTS[0], 3), (big, 40), (PROMPTS[2], 6)]
    with torch.no_grad():
        toks, errors = run(mix, False)
    out.update(tokens=toks, errors=errors)
print(json.dumps(out))
"""


def _side(mode: str, shim: bool) -> dict:
    out = _run_side(_SCRIPT.replace("__MODE__", repr(mode)), shim=shim)
    if "unavailable" in out:
        raise _skip.Skip(f"{_NO_CB}: {out['unavailable']}")
    return out


def test_greedy_batched_equals_alone_equals_upstream_with_staggered_finishes():
    up = _side("greedy", shim=False)
    sh = _side("greedy", shim=True)
    for name, side in (("upstream", up), ("shim", sh)):
        assert not side["errors"], (name, side["errors"])
        # different max_new_tokens: the requests finish at different steps
        assert side["lens"] == side["want"] == [3, 9, 6], (name, side["lens"])
        assert side["batched"] == side["alone"], (name, side["batched"], side["alone"])
    assert sh["batched"] == up["batched"], (sh["batched"], up["batched"])
    assert sh["alone"] == up["alone"], (sh["alone"], up["alone"])


def test_seeded_sampling_matches_upstream_semantics_no_per_request_stream():
    up = _side("sample", shim=False)
    sh = _side("sample", shim=True)
    for name, side in (("upstream", up), ("shim", sh)):
        assert not side["errors"], (name, side["errors"])
        assert [len(t) for t in side["batched_a"]] == [3, 9, 6], (name, side["batched_a"])
        # the same seed and the same batch repeat exactly
        assert side["repeat_equal"] is True, (name, side["batched_a"], side["batched_b"])
        assert all(0 <= t < 256 for r in side["batched_a"] for t in r), name
    assert up["alone_equals_batched"] is False, "upstream grew a per-request stream: revisit criterion 3"
    # the relation upstream shows (global RNG, not per request) is the relation we show
    assert sh["alone_equals_batched"] == up["alone_equals_batched"], (
        "shim", sh["alone_equals_batched"], "upstream", up["alone_equals_batched"])


def test_oversize_request_fails_every_in_flight_request_on_both_sides():
    up = _side("oversize", shim=False)
    sh = _side("oversize", shim=True)
    # Measured on upstream 5.15.1: the oversize request and the request still in
    # flight both end with None tokens and the error "No requests can be scheduled
    # and no requests can be offloaded."; the manager then records a fatal
    # RuntimeError. A request that had already finished (index 0) keeps its tokens.
    phrase = "No requests can be scheduled and no requests can be offloaded."
    for name, side in (("upstream", up), ("shim", sh)):
        assert any(phrase in e for e in side["errors"]), (name, side["errors"])
        assert "fatal: RuntimeError" in side["errors"], (name, side["errors"])
        assert side["tokens"][1] is None and side["tokens"][2] is None, (name, side["tokens"])
    assert sh["tokens"] == up["tokens"], (sh["tokens"], up["tokens"])


def _main():
    failures = _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        "models/test_cbidentity")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
