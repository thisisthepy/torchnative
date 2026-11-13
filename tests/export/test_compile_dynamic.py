"""Dynamic shapes (SymInt) through torchnative's hook-free compile engine.

Issue #262, INTENT §5, SPEC S1.9. Owner, 2026-11-13: "시퀀스 길이를 기호로 다루는거
좋은거 같아. 진행해. 동적 형태 구현해."

1. `test_llama_compiles_once_across_sequence_lengths_7_to_64`: a tiny real
   `transformers` Llama compiles twice (static first call, then one symbolic graph after
   automatic dynamic marks the length) and serves every length 7..64 from that graph;
   logits agree with eager.
2. `test_mark_dynamic_compiles_a_single_graph`: with `torch._dynamo.mark_dynamic` on the
   length, one graph serves every length, and the backend receives a `SymInt` input.
3. `test_a_shape_outside_the_guards_recompiles`: length 1 is outside the symbolic range
   (Dynamo specializes 0 and 1), so it recompiles and still agrees.
4. `test_recompile_limit_still_refuses_with_dynamic_shapes`: `recompile_limit = 1` stops
   the dynamic recompile; the call runs eagerly and agrees.
5. `test_op_without_rule_refuses_by_name_and_frame_goes_static`: an op with no symbolic
   rule refuses by name (`SymbolicShapeUnsupported`, naming the op); the engine records
   the refusal and compiles that frame with static shapes.
6. `test_a_wrong_rule_is_caught_by_the_hint_check`: a rule that returns a wrong size is
   caught against the kernel's hint result (nullification of the rule table).
7. `test_no_symbol_is_silently_specialized`: no `SymInt.__index__` during a symbolic
   trace (a guard-suppressed read would bake the traced length in).
8. `test_eager_tensors_pay_nothing`: only `FakeTensor` carries the symbolic methods.
9. (The type-dict guard on a mappingproxy is #270's, tested in
   `bindings/test_dynamo_guards_unit.py`.)
10. `test_a_clamped_slice_guards_its_bound`: a slice clamped at a constant records the
    comparison as a guard, so a length on the other side recompiles.
11. `test_a_symbolic_size_with_hint_one_is_not_taken_for_a_broadcast`: a symbolic
    expression whose hint is 1 is not treated as a broadcast 1 without a guard.
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
import torch.nn as nn

assert hasattr(torch._C, "_aten_implemented"), "this suite must run on the torchnative shim"

import torchnative.compile as _tc
from torchnative.compile import Engine

# Before `transformers` imports `torch._dynamo` (which binds the guard classes by name).
# A current shim installs them at `import torch` already; this is a no-op there.
_tc._ensure_shim_ready()


class _Counter:
    def __init__(self):
        self.graphs = 0
        self.example_inputs = []

    def __call__(self, gm, ex):
        self.graphs += 1
        self.example_inputs.append(list(ex))
        return gm.forward


def _tiny_llama():
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
    return LlamaForCausalLM(cfg).eval()


def _logits(eng, model, ids):
    return eng.run(type(model).forward, (model, ids), {"use_cache": False}).logits


def _eager(model, ids):
    return model(ids, use_cache=False).logits


def _maxdiff(a, b):
    return float((a - b).abs().max().item())


def test_llama_compiles_once_across_sequence_lengths_7_to_64():
    model = _tiny_llama()
    c = _Counter()
    eng = Engine(c)
    torch.manual_seed(1)
    with torch.no_grad():
        for n in range(7, 65):
            ids = torch.randint(0, 128, (1, n))
            out = _logits(eng, model, ids)
            ref = _eager(model, ids)
            d = _maxdiff(out, ref)
            assert d <= 1e-5, f"length {n}: logits differ from eager by {d}"
            if n == 8:
                assert c.graphs == 2, f"expected the dynamic recompile at length 8, got {c.graphs} graphs"
    assert c.graphs == 2, (
        f"lengths 7..64 compiled {c.graphs} graphs; expected 2 "
        f"(static at 7, one symbolic graph from 8), stats {eng.stats}"
    )
    assert eng.stats["hit"] == 64 - 8, f"expected 56 cache hits, stats {eng.stats}"
    assert eng.stats.get("dynamic_refused", 0) == 0, eng.dynamic_refusals


def test_mark_dynamic_compiles_a_single_graph():
    model = _tiny_llama()
    c = _Counter()
    eng = Engine(c)
    with torch.no_grad():
        for n in (7, 13, 31, 64):
            ids = torch.randint(0, 128, (1, n))
            torch._dynamo.mark_dynamic(ids, 1)
            d = _maxdiff(_logits(eng, model, ids), _eager(model, ids))
            assert d <= 1e-5, f"length {n}: {d}"
    assert c.graphs == 1, f"mark_dynamic: expected one graph, got {c.graphs}"
    syms = [e for e in c.example_inputs[0] if isinstance(e, torch.SymInt)]
    assert syms, f"the graph received no SymInt input: {[type(e).__name__ for e in c.example_inputs[0]]}"


def test_a_shape_outside_the_guards_recompiles():
    model = _tiny_llama()
    c = _Counter()
    eng = Engine(c)
    with torch.no_grad():
        for n in (7, 8, 20):
            _logits(eng, model, torch.randint(0, 128, (1, n)))
        assert c.graphs == 2
        ids = torch.randint(0, 128, (1, 1))
        d = _maxdiff(_logits(eng, model, ids), _eager(model, ids))
    assert c.graphs == 3, f"length 1 must miss the symbolic guards (s >= 2) and recompile, got {c.graphs}"
    assert d <= 1e-5, d


def test_recompile_limit_still_refuses_with_dynamic_shapes():
    # A function of its own: the eval_frame cache-entry count that `recompile_limit`
    # reads walks every engine's entries for a code object, so a shared model's code
    # would carry the other tests' entries.
    import torch._dynamo.config as dc

    w = torch.randn(8, 8)

    def f(x):
        return torch.softmax(x @ w, -1) * 2

    old = dc.recompile_limit
    dc.recompile_limit = 1
    try:
        c = _Counter()
        eng = Engine(c)
        eng.run(f, (torch.randn(3, 8),), {})
        x = torch.randn(5, 8)
        d = _maxdiff(eng.run(f, (x,), {}), f(x))
        assert c.graphs == 1, f"recompile_limit=1 must stop the dynamic recompile, got {c.graphs}"
        assert eng.stats["skip"] >= 1, eng.stats
        assert d == 0.0, d
    finally:
        dc.recompile_limit = old
    # Control: without the limit the same shapes compile the symbolic graph.
    c2 = _Counter()
    eng2 = Engine(c2)

    def g(x):
        return torch.softmax(x @ w, -1) * 2

    for n in (3, 5, 9):
        eng2.run(g, (torch.randn(n, 8),), {})
    assert c2.graphs == 2, f"control: expected static + symbolic graphs, got {c2.graphs}"


def test_op_without_rule_refuses_by_name_and_frame_goes_static():
    from torchnative import _dynamo_symbolic as sy

    assert not sy.has_rule("flip"), "pick an op with no symbolic rule for this test"
    c = _Counter()
    eng = Engine(c)

    def f(x):
        return torch.flip(x, [0]) * 2

    for n in (3, 5, 7):
        x = torch.randn(n, 4)
        d = _maxdiff(eng.run(f, (x,), {}), f(x))
        assert d == 0.0, d
    assert eng.stats["dynamic_refused"] == 1, eng.stats
    assert "aten.flip" in eng.dynamic_refusals[0], eng.dynamic_refusals
    # static at 3, refused symbolic at 5 then static at 5, static at 7
    assert c.graphs == 3, f"after the refusal the frame compiles per shape, got {c.graphs}"


def test_a_wrong_rule_is_caught_by_the_hint_check():
    from torchnative import _dynamo_symbolic as sy

    saved = sy._RULES["mm"]

    def wrong(func, args, kwargs, out):
        a, b = sy.sizes_of(args[0]), sy.sizes_of(args[1])
        return [(a[0] + 1, b[1])]

    sy._RULES["mm"] = wrong
    try:
        c = _Counter()
        eng = Engine(c)
        w = torch.randn(4, 6)

        def f(x):
            return torch.mm(x, w)

        for n in (3, 5):
            x = torch.randn(n, 4)
            assert _maxdiff(eng.run(f, (x,), {}), f(x)) == 0.0
        assert eng.stats["dynamic_refused"] == 1, eng.stats
        assert "kernel produced" in eng.dynamic_refusals[0], eng.dynamic_refusals
    finally:
        sy._RULES["mm"] = saved


def test_no_symbol_is_silently_specialized():
    """No `SymInt.__index__` during a symbolic trace: each would read the hint as a constant.

    Under a guard-suppressing context (the fake-tensor cache's output synthesis) such a
    read adds no guard either, so the graph would bake in the traced length silently.
    """
    model = _tiny_llama()
    eng = Engine(_Counter())
    calls = []
    orig = torch.SymInt.__index__

    def counting(self):
        calls.append(str(self))
        return orig(self)

    torch.SymInt.__index__ = counting
    try:
        with torch.no_grad():
            ids = torch.randint(0, 128, (1, 11))
            torch._dynamo.mark_dynamic(ids, 1)
            _logits(eng, model, ids)
    finally:
        torch.SymInt.__index__ = orig
    assert not calls, f"{len(calls)} symbols specialized through __index__: {calls[:5]}"


def test_a_clamped_slice_guards_its_bound():
    """`x[:8]` traced at length 20 has length 8 only while the length is >= 8.

    The rule must record that as a guard, so length 5 recompiles instead of reusing a
    graph that baked in 8 (the reshape below would then fail on 20 elements vs 32).
    """
    c = _Counter()
    eng = Engine(c)

    def f(x):
        y = x[:8]
        return y.reshape(y.shape[0] * 4) * 2

    for n in (10, 20, 5):
        x = torch.randn(n, 4)
        assert torch.equal(eng.run(f, (x,), {}), f(x)), f"length {n}"
    assert c.graphs == 3, f"length 5 is outside the slice guard and must recompile, got {c.graphs}"


def test_a_symbolic_size_with_hint_one_is_not_taken_for_a_broadcast():
    """`x[1:]` traced at length 2 has size `s - 1` with hint 1; broadcasting it must guard.

    Taking it for a broadcastable 1 without a guard would claim an output length of 1
    for every length.
    """
    c = _Counter()
    eng = Engine(c)

    def g(x):
        y = x[1:] + x[:1]
        return y.reshape(y.shape[0] * 4)

    for n in (3, 2, 10):
        x = torch.randn(n, 4)
        assert torch.equal(eng.run(g, (x,), {}), g(x)), f"length {n}"


def test_eager_tensors_pay_nothing():
    Engine(_Counter())  # installs the bridge
    from torch._subclasses.fake_tensor import FakeTensor

    for name in ("size", "stride", "numel", "storage_offset", "shape"):
        assert name not in torch.Tensor.__dict__, f"torch.Tensor.{name} was patched; eager would pay for it"
        assert name in FakeTensor.__dict__, f"FakeTensor.{name} is not symbolic-aware"
    t = torch.ones(2, 3)
    assert t.size() == (2, 3) and "_tn_sym" not in t.__dict__


def _main():
    tests = [
        test_llama_compiles_once_across_sequence_lengths_7_to_64,
        test_mark_dynamic_compiles_a_single_graph,
        test_a_shape_outside_the_guards_recompiles,
        test_recompile_limit_still_refuses_with_dynamic_shapes,
        test_op_without_rule_refuses_by_name_and_frame_goes_static,
        test_a_wrong_rule_is_caught_by_the_hint_check,
        test_no_symbol_is_silently_specialized,
        test_a_clamped_slice_guards_its_bound,
        test_a_symbolic_size_with_hint_one_is_not_taken_for_a_broadcast,
        test_eager_tensors_pay_nothing,
    ]
    failed = 0
    for fn in tests:
        try:
            torch._dynamo.pgo.reset_code_state()  # automatic-dynamic memory is per code, process-wide
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:400]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
