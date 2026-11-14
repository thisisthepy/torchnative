"""A Qwen3 decode step traced through `torchnative.compile`, compared with eager.

Issue #304 (regression test for #269), SPEC S4.5. The owner's target path is
from_pretrained, `.to(device)`, `generate` on Qwen3. #269 recorded that tracing a Qwen3
decode step with a `DynamicCache` stopped at `Tensor._base`; real view metadata landed
with #272 (#279) and the StaticCache default with #263. This suite holds the standing
check that the real transformers `Qwen3ForCausalLM` (a tiny, randomly initialised one,
no download) goes through an attached engine and agrees with eager.

1. `test_static_cache_default_matches_eager_and_hits_one_decode_graph`: after
   `attach_to_module`, `generate()` with no cache argument gives eager's greedy tokens
   for two prompt lengths, with `fail == 0`, a bounded `translate`, and decode steps
   served from the cache (`hit`).
2. `test_a_caller_chosen_dynamic_cache_is_captured_and_matches_eager`: #269's case.
3. `test_one_decode_step_called_directly_is_captured_and_equals_eager`: a prefill, then
   one decode step through the attached forward; logits equal eager's and the step was
   captured, not skipped.

**What catches a silent eager run.** A forward that ran eagerly touches none of the
engine's counters except `skip`, and never calls the backend. So `translate > 0` or
`hit > 0`, a non-empty `eng.entries` and `backend.graphs > 0` all fail on a silent eager
run, and `skip == 0` fails on a forward the engine declined. Token equality alone would
pass on an eager run (it is the same model), so no test here relies on it alone.
`fail == 0` guards the other direction: a capture failure raises `CompileRefused`
(it is never answered by a silent eager run, S1.4), and `fail` counts it.

**These tests were written before CI ran them.** The bounds below come from reading
`torchnative/compile/__init__.py` and from the sibling suite's measured Llama numbers
(`test_compile_static_cache.py`: 2 graphs, `hit == 7 - 1` for 8 new tokens); the first
CI run confirms or refutes them. A refutation is a finding to report, not a bound to
loosen until green.

What this cannot find: anything on a device, and whether the backend's graph is
profitable. The recording backend returns `gm.forward`, so the numbers are the shim's
own kernels run through Dynamo's graph.
"""

import os
import pathlib
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch

assert hasattr(torch._C, "_aten_implemented"), "this suite must run on the torchnative shim"

import torchnative.compile as _tc
from torchnative.compile import attach_to_module

_tc._ensure_shim_ready()

_NEW_TOKENS = 8


class _Counter:
    def __init__(self):
        self.graphs = 0

    def __call__(self, gm, ex):
        self.graphs += 1
        return gm.forward


def _pair():
    """A tiny random Qwen3 and an un-attached copy with the same weights.

    Two layers, GQA 4/2, head_dim 16, SDPA (the config of tests/models/test_qwen3.py).
    `eos_token_id` is None so a random model cannot stop early and every run makes
    exactly `_NEW_TOKENS` tokens, which the bounds below count on.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        rope_theta=1e6,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(304)
    model = Qwen3ForCausalLM(cfg).eval()
    ref = Qwen3ForCausalLM(cfg).eval()
    ref.load_state_dict(model.state_dict())
    return model, ref


def _generate(m, ids, **kw):
    return m.generate(ids, max_new_tokens=_NEW_TOKENS, min_new_tokens=_NEW_TOKENS, do_sample=False, pad_token_id=0, **kw)


def test_static_cache_default_matches_eager_and_hits_one_decode_graph():
    for n in (5, 11):
        model, ref = _pair()
        c = _Counter()
        attach_to_module(model, backend=c)
        assert model.torchnative_compile["kv_cache"] == "static"
        eng = model.torchnative_compile["engine"]
        torch.manual_seed(n)
        ids = torch.randint(3, 128, (1, n))
        with torch.no_grad():
            out = _generate(model, ids)
            want = _generate(ref, ids)
        assert out.shape[1] == n + _NEW_TOKENS, out.shape
        assert torch.equal(out, want), f"prompt {n}: compiled tokens {out.tolist()} != eager {want.tolist()}"
        st = eng.stats
        assert st["fail"] == 0, f"prompt {n}: a forward fell back to eager: {st}"
        assert st["skip"] == 0, f"prompt {n}: the engine declined a forward: {st}"
        # Catches a silent eager run: nothing was translated, no graph reached the backend.
        assert st["translate"] > 0 and c.graphs > 0 and eng.entries, (st, c.graphs)
        # One prefill plus 7 decode forwards. The sibling Llama suite measured 2 graphs
        # (prefill, decode); the bound of 3 leaves room for the symbolic recompile of
        # the prefill length and no more.
        assert st["translate"] <= 3, f"prompt {n}: too many compiles: {st}"
        # Decode steps are cache hits: of the 7 decode forwards at most `translate`
        # compiled, the rest hit.
        assert st["hit"] >= (_NEW_TOKENS - 1) - st["translate"], f"prompt {n}: {st}"
        assert st["hit"] > 0, st


def test_a_caller_chosen_dynamic_cache_is_captured_and_matches_eager():
    """#269's case: before #279 this stopped at `Tensor._base` and raised by name."""
    from transformers import DynamicCache

    for n in (5, 11):
        model, ref = _pair()
        c = _Counter()
        attach_to_module(model, backend=c)
        eng = model.torchnative_compile["engine"]
        torch.manual_seed(n)
        ids = torch.randint(3, 128, (1, n))
        with torch.no_grad():
            want = _generate(ref, ids)
            out = _generate(model, ids, past_key_values=DynamicCache(config=model.config))
        assert torch.equal(out, want), f"prompt {n}: compiled tokens {out.tolist()} != eager {want.tolist()}"
        st = eng.stats
        assert st["fail"] == 0, f"prompt {n}: a forward fell back to eager (the wall is named in the error): {st}"
        assert st["skip"] == 0, f"prompt {n}: the engine declined a forward: {st}"
        # Catches a silent eager run.
        assert st["translate"] > 0, st
        assert len(eng.entries) >= 1, "no compiled entry was stored"
        assert c.graphs > 0, "the backend never saw a graph"


def test_one_decode_step_called_directly_is_captured_and_equals_eager():
    """Prefill, then one decode step through the attached forward, against eager.

    Tolerance: none. Both sides run the same shim kernels in the same order (the
    recording backend returns `gm.forward`, which replays the traced ops one by one), so
    the arithmetic is identical and the logits must be equal bit for bit. A difference
    would be a finding about the capture, and AGENTS section 16 forbids widening a
    tolerance to hide it.
    """
    from transformers import DynamicCache

    model, ref = _pair()
    c = _Counter()
    attach_to_module(model, backend=c)
    eng = model.torchnative_compile["engine"]
    torch.manual_seed(7)
    ids = torch.randint(3, 128, (1, 6))
    nxt = torch.randint(3, 128, (1, 1))
    cache_c = DynamicCache(config=model.config)
    cache_e = DynamicCache(config=ref.config)
    with torch.no_grad():
        model(ids, past_key_values=cache_c, use_cache=True)
        ref(ids, past_key_values=cache_e, use_cache=True)
        before = dict(eng.stats)
        graphs_before = c.graphs
        got = model(nxt, past_key_values=cache_c, use_cache=True).logits
        want = ref(nxt, past_key_values=cache_e, use_cache=True).logits
    after = eng.stats
    assert got.shape == want.shape == (1, 1, 128), (got.shape, want.shape)
    assert torch.equal(got, want), f"decode logits differ, max abs {(got - want).abs().max().item()}"
    assert after["fail"] == 0, after
    assert after["skip"] == before["skip"] == 0, (before, after)
    # Exactly one root forward ran: it either compiled or hit; a silent eager run does neither.
    assert (after["translate"] - before["translate"]) + (after["hit"] - before["hit"]) == 1, (before, after)
    if after["translate"] > before["translate"]:
        assert c.graphs > graphs_before, "the decode step compiled but handed the backend no graph"
    assert c.graphs > 0 and eng.entries


def _main():
    tests = [
        test_static_cache_default_matches_eager_and_hits_one_decode_graph,
        test_a_caller_chosen_dynamic_cache_is_captured_and_matches_eager,
        test_one_decode_step_called_directly_is_captured_and_equals_eager,
    ]
    failed = 0
    for fn in tests:
        try:
            torch._dynamo.pgo.reset_code_state()
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:400]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
