"""Hook-free Dynamo compilation engine without PEP 523 and model.to(device.npu).

Issue #211, SPEC S1.8, S4.2:
Verifies torchnative.compile and Dynamo translation without set_eval_frame:
1. `test_control_flow_graph_breaks_inlining_and_monitoring`:
   5-pattern model (flag, shape, data-dependent, inlined helper, sys.monitoring
   trampoline on nested noisy_helper) produces exact eager outputs, cache hits,
   wrapped resume functions, and zero non-None set_eval_frame calls.
2. `test_negative_control_always_true_guards_break_branch_equality`:
   Monkeypatching RootGuardManager.check to always True causes flag=False to reuse
   the flag=True compiled graph producing max_diff > 1.0, proving guards are load-bearing.
3. `test_recompile_limit_enforced_through_eval_frame_cache_entries`:
   Dynamo recompile_limit = 3 limits cached compiled graphs for code object to 3,
   exposed via eval_frame._get_cache_entries_for_region and _get_total_cache_entry_count.
4. `test_wider_guard_coverage_containers_globals_kwargs_aliasing`:
   ModuleList, dict/list containers, floats, kwargs, aliasing x is y, requires_grad,
   shape changes, global dict and module attribute mutations.
5. `test_mini_transformer_block_across_sequence_lengths`:
   2-layer mini Transformer LM in eval mode across sequence lengths T=7, T=7 again
   (cache hit), and T=9 (recompilation).
6. `test_backward_and_inplace_mutation_through_compiled_module`:
   Backward pass parameter gradients match eager, and in-place tensor mutations
   mutate inputs and return values identically with cache hits.
7. `test_model_to_device_npu_attaches_hook_free_compile`:
   model.to(torchnative.device.npu) on faked Intel NPU host attaches
   torchnative_compile and executes first forward via compile, second forward
   via guard cache hit, and third forward with new shape recompilation.
"""

import contextlib
import io
import os
import pathlib
import re
import sys
import types
import warnings

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
_VENDOR_SHIM = str(_REPO_ROOT / "torchnative" / "python" / "torch" / "_C.abi3.so")
_MAIN_VENDOR_DIR = "/Volumes/macMini/thisisthepy/torchnative/torchnative/python"
_MAIN_VENDOR_SHIM = "/Volumes/macMini/thisisthepy/torchnative/torchnative/python/torch/_C.abi3.so"


def _find_vendor_dir():
    if os.path.isfile(_VENDOR_SHIM):
        return _VENDOR_DIR
    if os.path.isfile(_MAIN_VENDOR_SHIM):
        return _MAIN_VENDOR_DIR
    return _VENDOR_DIR


_vdir = _find_vendor_dir()
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
if _vdir not in sys.path:
    sys.path.insert(1, _vdir)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchnative.compile import Engine, Monitor, compile, attach_to_module
import torchnative.device as D
from torchnative.export import intelnpu


# ---------------------------------------------------------------------------
# Test helpers and fakes
# ---------------------------------------------------------------------------


def _diff(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def pure_helper(x):
    return torch.tanh(x) * 3 + 1


def noisy_helper(x):
    y = torch.relu(x) * 2
    print("noisy_helper called")
    return y + 1


class M(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)

    def forward(self, x, flag):
        h = self.lin(x)
        if flag:
            h = h + 1
        else:
            h = h - 1
        if x.shape[0] > 4:
            h = h * 2
        h = pure_helper(h)
        if h.sum() > 0:
            h = h + 10
        else:
            h = h - 10
        h = noisy_helper(h)
        return h


class _FakeOpenVINO:
    instances = []

    def __init__(self, path=None):
        self.compiled = []
        _FakeOpenVINO.instances.append(self)

    def devices(self):
        return ("CPU", "NPU")

    def compile_ir(self, xml, device, weights):
        dims = [int(d) for d in re.findall(r"<dim>(\d+)</dim>", xml)]
        batch, in_f = dims[0], dims[1]
        out_f = dims[-1]
        flat = intelnpu.unpack_f16(weights)
        n = out_f * in_f
        w = flat[:n]
        bias = flat[n : n + out_f] if len(flat) >= n + out_f else None
        handle = {
            "device": device,
            "batch": batch,
            "in": in_f,
            "out": out_f,
            "w": w,
            "bias": bias,
        }
        self.compiled.append(handle)
        return handle

    def execution_devices(self, compiled):
        return (compiled["device"],)

    def infer(self, compiled, blob):
        x = intelnpu.unpack_f16(blob)
        in_f, out_f, batch = compiled["in"], compiled["out"], compiled["batch"]
        w, bias = compiled["w"], compiled["bias"]
        out = []
        for b in range(batch):
            row = x[b * in_f : (b + 1) * in_f]
            for o in range(out_f):
                acc = sum(row[i] * w[o * in_f + i] for i in range(in_f))
                if bias is not None:
                    acc += bias[o]
                out.append(acc)
        return intelnpu.pack_f16(out)


class _on_a_faked_intel_npu_host:
    def __enter__(self):
        self._host = os.environ.get("TORCHNATIVE_DEVICE_HOST")
        os.environ["TORCHNATIVE_DEVICE_HOST"] = "windows"
        self._saved = (
            intelnpu.npu_available,
            intelnpu.available_devices,
            intelnpu.OpenVINO,
        )
        intelnpu.npu_available = lambda *a, **k: True
        intelnpu.available_devices = lambda *a, **k: ("CPU", "NPU")
        intelnpu.OpenVINO = _FakeOpenVINO
        _FakeOpenVINO.instances = []
        return self

    def __exit__(self, *exc):
        (
            intelnpu.npu_available,
            intelnpu.available_devices,
            intelnpu.OpenVINO,
        ) = self._saved
        if self._host is None:
            os.environ.pop("TORCHNATIVE_DEVICE_HOST", None)
        else:
            os.environ["TORCHNATIVE_DEVICE_HOST"] = self._host


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.fc1 = nn.Linear(8, 6)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


def _fresh_block():
    torch.manual_seed(0)
    return _TinyBlock()


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_control_flow_graph_breaks_inlining_and_monitoring():
    """Verify 5-pattern control flow model across 10-call sequence with monitoring."""
    class Counter:
        def __init__(self):
            self.graphs = 0
            self.calls = 0

        def __call__(self, gm, ex):
            self.graphs += 1

            def f(*a):
                self.calls += 1
                return gm.forward(*a)

            return f

    import torch._C._dynamo.eval_frame as ef

    orig_set_eval_frame = ef.set_eval_frame
    non_none_calls = []

    def spy_set_eval_frame(cb):
        if cb is not None:
            non_none_calls.append(cb)
        return orig_set_eval_frame(cb)

    ef.set_eval_frame = spy_set_eval_frame

    m = M()
    torch.manual_seed(1)
    xp = torch.rand(6, 8) + 1
    xn = -torch.rand(6, 8) * 50 - 50
    xs = torch.rand(3, 8) + 1

    c = Counter()
    eng = Engine(c)
    f = eng.compile_method(m)
    mon = Monitor(eng, [os.path.abspath(__file__)])

    seq = [
        ("xp,T", xp, True),
        ("xp,T", xp, True),
        ("xp,F", xp, False),
        ("xn,T", xn, True),
        ("xn,F", xn, False),
        ("xs,T", xs, True),
        ("xs,T", xs, True),
        ("xp,T", xp, True),
        ("xn,T", xn, True),
        ("xp,F", xp, False),
    ]

    try:
        for name, x, flag in seq:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = f(x, flag)
            ref = type(m).forward(m, x, flag)
            diff = _diff(out, ref)
            assert diff <= 1e-4, f"{name}: diff={diff} exceeds tolerance 1e-4"

        assert eng.stats["hit"] > 0, f"Expected cache hits, got {eng.stats}"
        assert (
            eng.stats["resume_wrapped"] > 0
        ), f"Expected resume wrapped functions, got {eng.stats}"
        assert (
            len(non_none_calls) == 0
        ), f"set_eval_frame called with non-None callback: {non_none_calls}"
    finally:
        mon.stop()
        ef.set_eval_frame = orig_set_eval_frame


def test_negative_control_always_true_guards_break_branch_equality():
    """Negative control: monkeypatching RootGuardManager.check breaks branch equality."""
    import torch._C._dynamo.guards as dg

    orig_check = dg.RootGuardManager.check

    class Counter:
        def __init__(self):
            self.graphs = 0

        def __call__(self, gm, ex):
            self.graphs += 1
            return gm.forward

    try:
        m = M()
        torch.manual_seed(1)
        xp = torch.rand(6, 8) + 1

        c = Counter()
        eng = Engine(c)
        f = eng.compile_method(m)

        # 1. Warm call with flag=True: compiles specialized graph
        with contextlib.redirect_stdout(io.StringIO()):
            out_true = f(xp, True)
        ref_true = type(m).forward(m, xp, True)
        assert _diff(out_true, ref_true) <= 1e-4

        # 2. Monkeypatch check to always True
        dg.RootGuardManager.check = lambda self, v: True

        # 3. Call with flag=False: reuses flag=True compiled graph
        with contextlib.redirect_stdout(io.StringIO()):
            out_bad = f(xp, False)
        ref_false = type(m).forward(m, xp, False)
        diff = _diff(out_bad, ref_false)
        assert diff > 1.0, (
            f"Nullification failed: expected wrong output diff > 1.0, "
            f"got {diff} (ref_false={ref_false}, out_bad={out_bad})"
        )
    finally:
        dg.RootGuardManager.check = orig_check


def test_recompile_limit_enforced_through_eval_frame_cache_entries():
    """Verify recompile_limit stops compilation and falls back to eager via cache entries."""
    import torch._dynamo.config as dc
    import torch._C._dynamo.eval_frame as ef

    orig_limit = dc.recompile_limit
    orig_acc_limit = dc.accumulated_recompile_limit
    orig_specialize_float = dc.specialize_float

    try:
        dc.recompile_limit = 3
        dc.accumulated_recompile_limit = 64
        # Each float value of `k` must be its own graph for the limit to be reached;
        # the engine's default lifts floats into the graph instead (#287).
        dc.specialize_float = True

        class Counter:
            def __init__(self):
                self.graphs = 0

            def __call__(self, gm, ex):
                self.graphs += 1
                return gm.forward

        c = Counter()
        eng = Engine(c)

        def f(x, k):
            return x * k + 1.0

        x = torch.ones(4, dtype=torch.float32)
        res = []
        for k in [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 5.0]:
            g0 = c.graphs
            out = eng.run(f, (x, k), {})
            ref = f(x, k)
            assert _diff(out, ref) <= 1e-5
            res.append((k, c.graphs - g0, len(eng.entries.get(f.__code__, ()))))

        code = f.__code__
        assert (
            c.graphs == 3
        ), f"Expected exactly 3 compiled graphs under recompile_limit=3, got {c.graphs}"
        assert (
            ef._get_total_cache_entry_count(code) == 3
        ), f"Expected 3 cache entries, got {ef._get_total_cache_entry_count(code)}"
        assert (
            len(ef._get_cache_entries_for_region(code, 0)) == 3
        ), f"Expected region 0 cache entries == 3"
        assert eng.stats["hit"] >= 1, f"Expected cache hits, got {eng.stats}"
    finally:
        dc.recompile_limit = orig_limit
        dc.accumulated_recompile_limit = orig_acc_limit
        dc.specialize_float = orig_specialize_float


def test_wider_guard_coverage_containers_globals_kwargs_aliasing():
    """Verify wide guard coverage: containers, kwargs, aliasing, attrs, globals."""
    class Counter:
        def __init__(self):
            self.graphs = 0

        def __call__(self, gm, ex):
            self.graphs += 1
            return gm.forward

    c = Counter()
    eng = Engine(c)

    G = {"scale": 2.0}

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
            self.act = nn.ReLU()
            self.mode = "a"

        def forward(self, x, opts, items, scale, maybe=None, *, bias=0.0):
            for layer in self.layers:
                x = self.act(layer(x))
            x = x * opts["mul"] + items[0] + items[1][0]
            x = x * scale * G["scale"] + bias
            if maybe is not None:
                x = x + maybe
            if self.mode == "a":
                x = x + 1
            return x

    m = Net()
    x = torch.rand(2, 4)
    y = torch.rand(2, 4)

    def call(*a, **k):
        return eng.run(type(m).forward, (m,) + a, k)

    def eager(*a, **k):
        return type(m).forward(m, *a, **k)

    cases = [
        ("base", lambda: ((x, {"mul": 3}, [y, (y,)], 0.5), {})),
        ("same", lambda: ((x, {"mul": 3}, [y, (y,)], 0.5), {})),
        ("alias x is y", lambda: ((x, {"mul": 3}, [x, (x,)], 0.5), {})),
        ("float change", lambda: ((x, {"mul": 3}, [y, (y,)], 0.75), {})),
        ("dict int chg", lambda: ((x, {"mul": 4}, [y, (y,)], 0.5), {})),
        ("maybe tensor", lambda: ((x, {"mul": 3}, [y, (y,)], 0.5, y), {})),
        ("kw bias", lambda: ((x, {"mul": 3}, [y, (y,)], 0.5), {"bias": 1.0})),
        ("x req grad", lambda: ((x.clone().requires_grad_(), {"mul": 3}, [y, (y,)], 0.5), {})),
        ("shape 5", lambda: ((torch.rand(5, 4), {"mul": 3}, [torch.rand(5, 4), (torch.rand(5, 4),)], 0.5), {})),
        ("global chg", lambda: ((x, {"mul": 3}, [y, (y,)], 0.5), {})),
        ("attr chg", lambda: ((x, {"mul": 3}, [y, (y,)], 0.5), {})),
        ("base again", lambda: ((x, {"mul": 3}, [y, (y,)], 0.5), {})),
    ]

    for name, mk in cases:
        if name == "global chg":
            G["scale"] = 3.0
        if name == "attr chg":
            m.mode = "b"
        a, k = mk()
        g0 = c.graphs
        out = call(*a, **k)
        ref = eager(*a, **k)
        diff = _diff(out, ref)
        assert diff <= 1e-4, f"{name}: diff={diff}"
        if name == "same":
            assert c.graphs - g0 == 0, f"{name} triggered recompilation instead of cache hit"

    assert eng.stats["hit"] > 0, f"Expected cache hits, got {eng.stats}"


def test_mini_transformer_block_across_sequence_lengths():
    """Verify 2-layer mini Transformer LM compiles across sequence lengths T=7, 7, 9."""
    class Block(nn.Module):
        def __init__(self, d=16, h=2):
            super().__init__()
            self.h = h
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.o = nn.Linear(d, d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            b_sz, t_sz, d_sz = x.shape
            q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
            q, k, v = (
                t.view(b_sz, t_sz, self.h, d_sz // self.h).transpose(1, 2)
                for t in (q, k, v)
            )
            att = (q @ k.transpose(-2, -1)) / (d_sz // self.h) ** 0.5
            mask = torch.tril(torch.ones(t_sz, t_sz, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
            att = self.drop(F.softmax(att, dim=-1))
            y = (att @ v).transpose(1, 2).reshape(b_sz, t_sz, d_sz)
            x = x + self.o(y)
            x = x + self.mlp(self.ln2(x))
            return x

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = nn.Embedding(50, 16)
            self.blocks = nn.ModuleList([Block() for _ in range(2)])
            self.head = nn.Linear(16, 50)

        def forward(self, ids):
            x = self.emb(ids)
            for b in self.blocks:
                x = b(x)
            return self.head(x)[:, -1, :].argmax(-1)

    class C:
        def __init__(self):
            self.graphs = 0

        def __call__(self, gm, ex):
            self.graphs += 1
            return gm.forward

    c = C()
    eng = Engine(c)
    m = LM().eval()
    ids7 = torch.randint(0, 50, (2, 7))
    ids9 = torch.randint(0, 50, (2, 9))
    f = eng.compile_method(m)

    with torch.no_grad():
        # 1. T=7 -> first compile
        g0 = c.graphs
        out1 = f(ids7)
        ref1 = type(m).forward(m, ids7)
        assert (out1 == ref1).all().item(), "T=7 output mismatch"
        assert c.graphs - g0 == 1, "Expected 1 compile on T=7"

        # 2. T=7 again -> cache hit
        g1 = c.graphs
        out2 = f(ids7)
        assert (out2 == ref1).all().item(), "T=7 again output mismatch"
        assert c.graphs - g1 == 0, "Expected cache hit on T=7 again"

        # 3. T=9 -> recompile for new static shape
        g2 = c.graphs
        out3 = f(ids9)
        ref3 = type(m).forward(m, ids9)
        assert (out3 == ref3).all().item(), "T=9 output mismatch"
        assert c.graphs - g2 == 1, "Expected recompile on T=9"


def test_backward_and_inplace_mutation_through_compiled_module():
    """Verify backward gradient agreement and in-place tensor mutation tracking."""
    # 1. Backward pass
    m = M()
    torch.manual_seed(1)
    x = torch.rand(6, 8) + 1

    eng = Engine(lambda gm, ex: gm.forward)
    f = eng.compile_method(m)
    mon = Monitor(eng, [os.path.abspath(__file__)])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            f(x, True)  # warm
            out = f(x, True)
            out.sum().backward()
            g_c = [p.grad.clone() for p in m.parameters()]
            m.zero_grad()
            ref = type(m).forward(m, x, True)
            ref.sum().backward()
            g_e = [p.grad.clone() for p in m.parameters()]

        assert len(g_c) == len(g_e)
        for gc, ge in zip(g_c, g_e):
            assert _diff(gc, ge) <= 1e-4, f"Grad diff {_diff(gc, ge)}"
    finally:
        mon.stop()

    # 2. In-place mutations
    n_graphs = [0]

    def be(gm, ex):
        n_graphs[0] += 1
        return gm.forward

    eng2 = Engine(be)

    def g(x_val, buf):
        x_val.add_(1)
        buf.mul_(2)
        return x_val * 2 + buf

    a = torch.zeros(3)
    b = torch.ones(3)
    a_ref = a.clone()
    b_ref = b.clone()

    o1 = eng2.run(g, (a, b), {})
    r1 = g(a_ref, b_ref)
    assert _diff(o1, r1) == 0.0, "Output 1 mismatch"
    assert _diff(a, a_ref) == 0.0, "Mutated input a mismatch"
    assert _diff(b, b_ref) == 0.0, "Mutated input b mismatch"
    assert n_graphs[0] == 1, "Expected 1 compile on first call"

    # Second call -> cache hit
    o2 = eng2.run(g, (a, b), {})
    r2 = g(a_ref, b_ref)
    assert _diff(o2, r2) == 0.0, "Output 2 mismatch"
    assert _diff(a, a_ref) == 0.0, "Mutated input a mismatch 2"
    assert _diff(b, b_ref) == 0.0, "Mutated input b mismatch 2"
    assert n_graphs[0] == 1, "Expected cache hit on second call"


def test_model_to_device_npu_attaches_hook_free_compile():
    """Verify model.to(torchnative.device.npu) attaches torchnative_compile and recompiles per shape."""
    reference = _fresh_block()
    x1 = torch.randn(1, 8)
    x2 = torch.randn(3, 8)
    ref_x1 = reference(x1)
    ref_x2 = reference(x2)

    model = _fresh_block()
    with _on_a_faked_intel_npu_host():
        res = model.to(D.npu)
        assert res is model, "model.to(D.npu) must return model (res is model)"
        assert hasattr(model, "torchnative_offload"), "missing torchnative_offload"
        assert hasattr(model, "torchnative_compile"), "missing torchnative_compile"
        assert model.torchnative_compile.get("compiled") is True

        eng = model.torchnative_compile["engine"]
        t0 = eng.stats["translate"]

        # 1. First forward on shape (1, 8) -> translates and compiles
        out1 = model(x1)
        assert eng.stats["translate"] > t0, "Expected translation on first forward"
        diff1 = _diff(out1, ref_x1)
        assert diff1 <= 2e-2, f"out1 diff {diff1} exceeds tolerance 2e-2"

        # 2. Second forward on same shape (1, 8) -> guard cache hit
        t1 = eng.stats["translate"]
        h1 = eng.stats["hit"]
        out2 = model(x1)
        assert eng.stats["translate"] == t1, "Expected zero new translations on same shape"
        assert eng.stats["hit"] > h1, "Expected cache hit on same shape"
        assert float((out2 - out1).abs().max().item()) == 0.0, "Deterministic output mismatch"

        # 3. Third forward on new shape (3, 8) -> misses shape guard and recompiles
        t2 = eng.stats["translate"]
        out3 = model(x2)
        assert eng.stats["translate"] > t2, "Expected recompilation on new shape (3, 8)"
        diff3 = _diff(out3, ref_x2)
        assert diff3 <= 2e-2, f"out3 diff {diff3} exceeds tolerance 2e-2"


def _main():
    tests = [
        (
            "test_control_flow_graph_breaks_inlining_and_monitoring",
            test_control_flow_graph_breaks_inlining_and_monitoring,
        ),
        (
            "test_negative_control_always_true_guards_break_branch_equality",
            test_negative_control_always_true_guards_break_branch_equality,
        ),
        (
            "test_recompile_limit_enforced_through_eval_frame_cache_entries",
            test_recompile_limit_enforced_through_eval_frame_cache_entries,
        ),
        (
            "test_wider_guard_coverage_containers_globals_kwargs_aliasing",
            test_wider_guard_coverage_containers_globals_kwargs_aliasing,
        ),
        (
            "test_mini_transformer_block_across_sequence_lengths",
            test_mini_transformer_block_across_sequence_lengths,
        ),
        (
            "test_backward_and_inplace_mutation_through_compiled_module",
            test_backward_and_inplace_mutation_through_compiled_module,
        ),
        (
            "test_model_to_device_npu_attaches_hook_free_compile",
            test_model_to_device_npu_attaches_hook_free_compile,
        ),
    ]

    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
