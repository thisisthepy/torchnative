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
`hit > 0`, the root forward's code in `eng.entries` and `backend.graphs > 0` all fail on
a silent eager run. Token equality alone would pass on an eager run (it is the same
model), so no test here relies on it alone. `fail == 0` guards the other direction: a
capture failure raises `CompileRefused` (it is never answered by a silent eager run,
S1.4), and `fail` counts it.

**`skip` is not only the root forward.** `Engine.run` counts `skip` for every frame it
runs without code, and the resume function Dynamo installs after a graph break
(`_ensure_resume_patch` wraps it in `eng.wrap`) is such a frame. CI run 43610884249
measured it: after a DynamicCache prefill, `translate == 2`, `resume_wrapped == 1`,
`skip == 1`, `fail == 0` (the root forward compiled, broke once, and its resume function
got no code), and the decode step after it added `translate + 1` and no skip. So these
tests no longer assert `skip == 0`; `_SkipLog` attributes each counted skip to the frame
and the root call it happened in, and the tests assert that no skip is the root forward,
that every skip is a Dynamo resume function, and that none happens after the prefill:
a decode step that ran any frame without a graph still fails.

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


#: The names Dynamo gives a resume function (`resume_execution.py`,
#: `ContinueExecutionCache.generate`). Measured on the shim (develop build, cf56414 or
#: earlier) for `y = x * 2; graph_break(); return y`: `__name__` is
#: `__resume_at_70_3`, `__qualname__` is `torch_dynamo_resume_in_f_at_15`.
_RESUME_PREFIX = "torch_dynamo_resume_in_"
_RESUME_NAME_PREFIX = "__resume_at_"


class _SkipLog:
    """Which frame each counted `skip` belongs to, and in which root call.

    Wraps `eng.run` on the instance: the attached forward and every wrapped resume
    function call `eng.run(...)` through the instance, so each frame passes here. A
    frame's own skips are the counter's growth over its call minus what nested frames
    added. `root_calls` counts the outermost calls (the attached forward), so the first
    one is the prefill. This reads the engine's own counter, so it works with or
    without `Engine.skips` (#303).
    """

    def __init__(self, eng):
        self.eng = eng
        self.root_calls = 0
        self.skipped = []  # (root call index, frame __name__, frame __qualname__)
        self._stack = []
        inner = eng.run

        def run(fn, args, kwargs):
            if not self._stack:
                self.root_calls += 1
            start = eng.stats["skip"]
            self._stack.append(0)
            try:
                return inner(fn, args, kwargs)
            finally:
                nested = self._stack.pop()
                grown = eng.stats["skip"] - start
                for _ in range(grown - nested):
                    self.skipped.append(
                        (self.root_calls, getattr(fn, "__name__", "?"), getattr(fn, "__qualname__", "?"))
                    )
                if self._stack:
                    self._stack[-1] += grown

        eng.run = run

    def why(self):
        """The evidence a failing assertion prints: frames, the engine's records, graph breaks."""
        import torch._dynamo.utils as du

        records = [str(r) for r in getattr(self.eng, "skips", ())]
        breaks = dict(du.counters.get("graph_break", {}))
        return f"skipped frames {self.skipped}; engine skip records {records}; graph breaks {breaks}"

    def check(self, root_qualname, *, prefill_only):
        """No skip is the root forward, each is a resume function, and (if asked) only in call 1."""
        assert len(self.skipped) == self.eng.stats["skip"], (
            f"the log attributed {len(self.skipped)} skips, the engine counted "
            f"{self.eng.stats['skip']}: {self.why()}"
        )
        for call, name, qualname in self.skipped:
            assert qualname != root_qualname, f"the root forward ran without a graph: {self.why()}"
            assert qualname.startswith(_RESUME_PREFIX) or name.startswith(_RESUME_NAME_PREFIX), (
                f"a frame other than a resume function was skipped: {self.why()}"
            )
            if prefill_only:
                assert call == 1, f"a frame ran without a graph after the prefill (root call {call}): {self.why()}"


def _generate(m, ids, **kw):
    return m.generate(ids, max_new_tokens=_NEW_TOKENS, min_new_tokens=_NEW_TOKENS, do_sample=False, pad_token_id=0, **kw)


def test_static_cache_default_matches_eager_and_hits_one_decode_graph():
    for n in (5, 11):
        model, ref = _pair()
        c = _Counter()
        attach_to_module(model, backend=c)
        assert model.torchnative_compile["kv_cache"] == "static"
        eng = model.torchnative_compile["engine"]
        log = _SkipLog(eng)
        root = type(model).forward
        torch.manual_seed(n)
        ids = torch.randint(3, 128, (1, n))
        with torch.no_grad():
            out = _generate(model, ids)
            want = _generate(ref, ids)
        assert out.shape[1] == n + _NEW_TOKENS, out.shape
        assert torch.equal(out, want), f"prompt {n}: compiled tokens {out.tolist()} != eager {want.tolist()}"
        st = eng.stats
        assert st["fail"] == 0, f"prompt {n}: a forward fell back to eager: {st}"
        # Was `st["skip"] == 0`. CI run 43610884249 never reached this line (generate
        # refused at `bool * True` first, now implemented), but test 3 of the same run
        # measured `skip == 1` from a resume function after a prefill graph break, with
        # the root compiled. `skip` counts that frame by the engine's design, so the
        # property asserted is the one `skip == 0` stood for: the root forward was never
        # run without a graph, and no frame of the 7 decode steps was (root calls 2..8).
        log.check(root.__qualname__, prefill_only=True)
        # Catches a silent eager run: nothing was translated, no graph reached the backend,
        # and the root forward's own code has no compiled entry.
        assert st["translate"] > 0 and c.graphs > 0 and root.__code__ in eng.entries, (st, c.graphs, log.why())
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
        log = _SkipLog(eng)
        root = type(model).forward
        torch.manual_seed(n)
        ids = torch.randint(3, 128, (1, n))
        with torch.no_grad():
            want = _generate(ref, ids)
            out = _generate(model, ids, past_key_values=DynamicCache(config=model.config))
        assert torch.equal(out, want), f"prompt {n}: compiled tokens {out.tolist()} != eager {want.tolist()}"
        st = eng.stats
        assert st["fail"] == 0, f"prompt {n}: a forward fell back to eager (the wall is named in the error): {st}"
        # Was `st["skip"] == 0`. CI run 43610884249 stopped before this line (the eager
        # `ref.generate` refused at `bool * True`, now implemented); test 3 of the same
        # run measured, for this very cache, a DynamicCache prefill whose resume function
        # after a graph break got no code (`skip == 1`, root compiled, `fail == 0`).
        # See `_SkipLog`: the root forward never ran without a graph and no decode step
        # ran any frame without one.
        log.check(root.__qualname__, prefill_only=True)
        # Catches a silent eager run.
        assert st["translate"] > 0, st
        assert root.__code__ in eng.entries, f"the root forward has no compiled entry: {log.why()}"
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
    log = _SkipLog(eng)
    root = type(model).forward
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
        # The prefill's root forward was captured: an eager root leaves no entry.
        assert root.__code__ in eng.entries, f"the prefill root forward has no compiled entry: {log.why()}"
        root_entries_before = len(eng.entries[root.__code__])
        got = model(nxt, past_key_values=cache_c, use_cache=True).logits
        want = ref(nxt, past_key_values=cache_e, use_cache=True).logits
    after = eng.stats
    assert got.shape == want.shape == (1, 1, 128), (got.shape, want.shape)
    assert torch.equal(got, want), f"decode logits differ, max abs {(got - want).abs().max().item()}"
    assert after["fail"] == 0, after
    # Was `after["skip"] == before["skip"] == 0`. CI run 43610884249 observed
    # before = {translate 2, hit 0, skip 1, fail 0, resume_wrapped 1} and
    # after = {translate 3, hit 0, skip 1, fail 0, resume_wrapped 1}: the decode step
    # added no skip, and the one skip was the prefill's. `translate 2` with
    # `resume_wrapped 1` is the root forward plus the one resume function Dynamo
    # installed after a graph break, and the skip is that resume function: the engine
    # on that run (#303's) refuses a root that gets no code, which `fail == 0`
    # excludes, and the root's entry asserted above and `_SkipLog` check it directly
    # on either engine. `before["skip"] == 0` therefore asserted something about
    # the prefill's nested frames that the engine does not promise; what this test is
    # for, the decode step, is held exactly: no new skip, and the prefill's skips are
    # resume functions and not the root (`_SkipLog.check`).
    assert after["skip"] == before["skip"], f"the decode step ran a frame without a graph: {(before, after)} {log.why()}"
    log.check(root.__qualname__, prefill_only=True)
    # Exactly one root forward ran: it either compiled or hit; a silent eager run does
    # neither. CI run 43610884249 measured the compile (`translate` 2 -> 3, `hit` 0).
    assert (after["translate"] - before["translate"]) + (after["hit"] - before["hit"]) == 1, (before, after)
    if after["translate"] > before["translate"]:
        assert c.graphs > graphs_before, "the decode step compiled but handed the backend no graph"
        assert len(eng.entries[root.__code__]) == root_entries_before + 1, (
            "the decode step compiled, but not the root forward's code"
        )
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
