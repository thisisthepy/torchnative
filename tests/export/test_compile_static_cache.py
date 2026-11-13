"""StaticCache as the default KV cache on compiled modules.

Issue #263, INTENT §5, SPEC S4.5. Owner, 2026-11-13 ("그렇게 하자"): on compiled devices
the KV cache is transformers' `StaticCache` by default (preallocated, written at
`cache_position`), set up by torchnative at `.to(device)` so user code is unchanged;
prefill compiles with a symbolic length and decode stays static-shaped.

1. `test_generate_uses_a_static_cache_and_matches_eager_tokens`: after
   `attach_to_module` (what `.to(<compiled device>)` calls), `generate()` with no cache
   argument runs on a `StaticCache`, compiles a prefill graph and a decode graph, serves
   every decode step from the decode graph, and its greedy tokens equal eager
   `generate` with transformers' default cache.
2. `test_prefill_compiles_once_across_prompt_lengths`: three prompt lengths give at most
   two prefill compiles (static, then symbolic) and the third prompt compiles nothing.
3. `test_a_cache_the_caller_chose_is_respected`: an explicit `DynamicCache`,
   `cache_implementation=` and `use_cache=False` all pass through untouched.
4. `test_a_baseless_view_argument_is_traced_as_a_plain_tensor`: a sliced argument has
   its real `_base` (#272) and compiles, aliasing included; a view the shim cannot
   track (no `_base`) compiles as a plain tensor, and refuses by name
   (`ViewWithoutBase`) when it aliases another argument.
5. `test_upstream_meta_stands_in_only_for_listed_ops`: the in-place KV-cache write used
   upstream's Python meta registration (counted in `UPSTREAM_META_USED`), and an op not
   on the list still refuses by the shim's own name.
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
from torchnative.compile import Engine, ViewWithoutBase, attach_to_module

_tc._ensure_shim_ready()


class _Counter:
    def __init__(self):
        self.graphs = 0

    def __call__(self, gm, ex):
        self.graphs += 1
        return gm.forward


def _pair():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg).eval()
    ref = LlamaForCausalLM(cfg).eval()
    ref.load_state_dict(model.state_dict())
    return model, ref


def _cache_spy(model):
    seen = []

    def hook(mod, args, kwargs):
        pkv = kwargs.get("past_key_values")
        seen.append(type(pkv).__name__ if pkv is not None else None)

    model.register_forward_pre_hook(hook, with_kwargs=True)
    return seen


def test_generate_uses_a_static_cache_and_matches_eager_tokens():
    model, ref = _pair()
    c = _Counter()
    attach_to_module(model, backend=c)
    assert model.torchnative_compile["kv_cache"] == "static"
    eng = model.torchnative_compile["engine"]
    seen = _cache_spy(model)
    ids = torch.randint(0, 128, (1, 5))
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
        want = ref.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(out, want), f"compiled tokens {out.tolist()} != eager {want.tolist()}"
    assert set(seen) == {"StaticCache"}, f"forward saw caches {set(seen)}"
    assert eng.stats["fail"] == 0, f"a forward fell back to eager: {eng.stats}"
    assert c.graphs == 2, f"expected a prefill and a decode graph, got {c.graphs}"
    assert eng.stats["hit"] == 7 - 1, f"decode steps 2..8 must hit the decode graph: {eng.stats}"


def test_prefill_compiles_once_across_prompt_lengths():
    model, ref = _pair()
    c = _Counter()
    attach_to_module(model, backend=c)
    counts = []
    with torch.no_grad():
        for n in (5, 9, 17):
            ids = torch.randint(0, 128, (1, n))
            out = model.generate(ids, max_new_tokens=4, do_sample=False)
            want = ref.generate(ids, max_new_tokens=4, do_sample=False)
            assert torch.equal(out, want), f"prompt {n}: tokens differ"
            counts.append(c.graphs)
    assert counts[2] == counts[1], f"the third prompt length compiled again: graphs per call {counts}"
    assert counts[2] <= 4, counts


def test_a_cache_the_caller_chose_is_respected():
    from transformers import DynamicCache

    model, ref = _pair()
    attach_to_module(model, backend=_Counter())
    seen = _cache_spy(model)
    ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        want = ref.generate(ids, max_new_tokens=3, do_sample=False)
        a = model.generate(ids, max_new_tokens=3, do_sample=False, past_key_values=DynamicCache(config=model.config))
        assert set(seen) == {"DynamicCache"}, set(seen)
        seen.clear()
        b = model.generate(ids, max_new_tokens=3, do_sample=False, cache_implementation="dynamic")
        assert set(seen) == {"DynamicCache"}, set(seen)
        seen.clear()
        d = model.generate(ids, max_new_tokens=3, do_sample=False, use_cache=False)
        assert set(seen) == {None}, set(seen)
    for got in (a, b, d):
        assert torch.equal(got, want)


def test_a_baseless_view_argument_is_traced_as_a_plain_tensor():
    """Two kinds of view argument, since #272 records view bases.

    A view the shim tracks (`base[:, 4:]`) answers its real `_base`, so Dynamo
    models it, and its aliasing with another argument, as upstream does. A view
    the shim cannot track (`nn.Parameter` of a slice goes through
    `_make_subclass`, so `_base` still refuses) keeps #263's handling: traced as
    the plain tensor it is when it aliases nothing, refused by name
    (`ViewWithoutBase`) when another argument shares its storage.
    """
    w = torch.randn(4, 4)

    def f(x):
        return x @ w

    def g(a, b):
        return a.sum() + b.sum()

    base = torch.randn(3, 8)

    # A tracked view: the base is real, so there is nothing to work around.
    v = base[:, 4:]
    assert v._is_view() and v._base is base, (v._is_view(), v._base)
    c = _Counter()
    eng = Engine(c)
    assert (eng.run(f, (v,), {}) - f(v)).abs().max().item() == 0.0
    assert c.graphs == 1, c.graphs
    assert "_is_view" not in v.__dict__, "the translation-only override touched a tracked view"
    c = _Counter()
    eng = Engine(c)
    assert (eng.run(g, (base, base[:, 4:]), {}) - g(base, base[:, 4:])).abs().max().item() == 0.0
    assert c.graphs == 1, c.graphs

    # A view the shim cannot track: #263's handling.
    p = torch.nn.Parameter(base[:, 4:], requires_grad=False)
    assert p._is_view()
    try:
        p._base
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected an untracked view; the fixture no longer builds one")
    c = _Counter()
    eng = Engine(c)
    assert (eng.run(f, (p,), {}) - f(p)).abs().max().item() == 0.0
    assert c.graphs == 1, c.graphs
    assert "_is_view" not in p.__dict__, "the translation-only override leaked"
    try:
        Engine(_Counter()).run(g, (base, p), {})
    except ViewWithoutBase as e:
        assert "shares its storage" in str(e)
    else:
        raise AssertionError("an untracked view aliasing another argument must refuse by name")


def test_upstream_meta_stands_in_only_for_listed_ops():
    from torchnative import _dynamo_symbolic as sy

    model, _ = _pair()
    attach_to_module(model, backend=_Counter())
    before = dict(sy.UPSTREAM_META_USED)
    with torch.no_grad():
        model.generate(torch.randint(0, 128, (1, 4)), max_new_tokens=2, do_sample=False)
    used = {k: v - before.get(k, 0) for k, v in sy.UPSTREAM_META_USED.items()}
    assert used.get("aten.index_put_.default", 0) > 0, used
    assert set(k for k, v in used.items() if v) <= set(sy._UPSTREAM_META_OPS), used
    assert sy._upstream_meta(torch.ops.aten.index_select.default) is None


def _main():
    tests = [
        test_generate_uses_a_static_cache_and_matches_eager_tokens,
        test_prefill_compiles_once_across_prompt_lengths,
        test_a_cache_the_caller_chose_is_respected,
        test_a_baseless_view_argument_is_traced_as_a_plain_tensor,
        test_upstream_meta_stands_in_only_for_listed_ops,
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
