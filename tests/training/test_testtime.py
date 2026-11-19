"""Test-time learning: the module layer and the rule layer (SPEC S6.3-S6.5, #237).

INTENT section 4: test-time learning is a time axis, split into two layers whose
names were set by the leader by convention under the owner's delegation
("컨벤션에 맞게 편성해", 2026-11-13):

* `torch.nn.modules.test_time.TestTimeModuleMixin`: `online()` / `offline()` /
  `online_parameters()` with test-time-adapters' semantics (`online` is
  `eval()`, freeze everything, unfreeze `online_parameters()`; `offline` is
  `train()` and the original `requires_grad` back).
* `torch.optim.test_time`: `TestTimeAdaptation` holds an optimizer the way
  `LRScheduler` does, and the generic rules `EntropyMinimization`,
  `PseudoLabel`, `NormStatistics`.

Both are reached here through the add-hooks in the vendored tree, so the hooks
are load-bearing: delete either and every test below goes red.

**The grade is agrees.** The oracle is upstream torch in the spike venv running
the same step written out by hand (forward, objective, `autograd.grad`, SGD),
not this layer's code, so the comparison is between the rule and what the rule
is supposed to compute. The model source (`_MODEL_SRC`) is executed on both
sides. The tolerance is derived, not chosen (AGENTS.md section 16): upstream
re-runs its own loop in float64, and the shim may differ from upstream's float32
by at most `agree_sweep.ORACLE_FACTOR` times upstream's own float32 error,
floored at 8 ulp.

What this file cannot find: whether the rules help on a real distribution
shift (that is test-time-adapters' evaluation, not a property of the layer),
and anything about a device other than the host.

Nullifications, each one made and watched go red, are listed in the PR for #237.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip  # noqa: E402
from agree_sweep import F32_EPS, ORACLE_FACTOR  # noqa: E402

_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

try:
    import torch as _upstream_torch
except Exception:  # pragma: no cover - reported as a skip below
    _upstream_torch = None


def _available():
    return _upstream_torch is not None and os.path.isfile(_VENDOR_SHIM)


# Executed verbatim on both sides. RNG-free inputs, for the reason
# tests/training/test_stage0.py gives: two builds of torch, two RNGs.
_MODEL_SRC = r'''
C, K, N = 4, 3, 6
LR = 0.5
THRESHOLD = 0.45

def _det(n, seed, lo=-1000, hi=1000):
    return [((seed * 1103515245 + i * 12345) % (hi - lo) + lo) / 4000.0 for i in range(n)]

class Net(nn.Module):
    """Linear -> LayerNorm -> Linear: what entropy and pseudo-labels move."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C, C)
        self.norm = nn.LayerNorm(C)
        self.head = nn.Linear(C, K)
    def forward(self, x):
        return self.head(self.norm(self.fc(x)))

class BNNet(nn.Module):
    """Linear -> BatchNorm1d -> Linear: what has running statistics."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C, C, bias=False)
        self.norm = nn.BatchNorm1d(C)
        self.head = nn.Linear(C, K, bias=False)
    def forward(self, x):
        return self.head(self.norm(self.fc(x)))

def build(cls, dtype=torch.float32):
    m = cls()
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(m.named_parameters())):
            vals = _det(p.numel(), 3 + 7 * i)
            if name.startswith("norm.weight"):
                vals = [1.0 + v for v in vals]
            p.copy_(torch.tensor(vals).reshape(p.shape) * (3.0 if name.startswith("fc") else 1.0))
    return m.to(dtype).eval()

def batches(dtype=torch.float32):
    return [(torch.tensor(_det(N * C, 101 + 7 * k)).reshape(N, C) * 2.0 + 0.5 * k).to(dtype)
            for k in range(3)]

def entropy(y):
    p = y.log_softmax(-1)
    return -(p.exp() * p).sum(-1).mean()

def flat(d):
    return {n: t.detach().reshape(-1).tolist() for n, t in d.items()}
'''


_SHIM_SCRIPT = (
    "import json, sys, contextlib\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "assert hasattr(torch._C, '_aten_implemented'), 'not the shim'\n"
    + _MODEL_SRC
    + r'''
out = {"errors": {}}

@contextlib.contextmanager
def section(name):
    try:
        yield
    except BaseException as e:  # recorded, so each test reports its own wall
        out["errors"][name] = "%s: %s" % (type(e).__name__, e)

xs = batches()

# --- the module layer ----------------------------------------------------
with section("mixin"):
    from torch.nn.modules.test_time import TestTimeModuleMixin
    out["mixin_module"] = TestTimeModuleMixin.__module__
    out["mixin_at_torch_nn_top"] = hasattr(torch.nn, "TestTimeModuleMixin")
    out["mixin_in_torch_nn_modules"] = hasattr(torch.nn.modules, "TestTimeModuleMixin")

    class Child(TestTimeModuleMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(C, C)
            self.norm = nn.LayerNorm(C)
            self.calls = []
        def online_parameters(self):
            return [self.norm.weight, self.norm.bias]
        def online(self, mode=True):
            self.calls.append(mode)
            return super().online(mode)

    class Plain(nn.Module):
        """Has a submodule *named* online, and no online() of its own."""
        def __init__(self):
            super().__init__()
            self.online = nn.Linear(C, C)

    class Parent(TestTimeModuleMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Child()
            self.b = Plain()
            self.head = nn.Linear(C, C)
            self.frozen = nn.Linear(C, C)
        def online_parameters(self):
            return self.head.parameters()

    p = Parent().train()
    p.frozen.weight.requires_grad_(False)
    out["rg_before"] = {n: q.requires_grad for n, q in p.named_parameters()}
    out["online_returns_self"] = p.online() is p
    out["online_training"] = {n or "<root>": m.training for n, m in p.named_modules()}
    out["online_rg"] = {n: q.requires_grad for n, q in p.named_parameters()}
    out["online_adapting"] = [p.adapting, p.a.adapting]
    p.online()  # a second online() must not re-snapshot the frozen state
    out["offline_returns_self"] = p.offline() is p
    out["offline_training"] = {n or "<root>": m.training for n, m in p.named_modules()}
    out["offline_rg"] = {n: q.requires_grad for n, q in p.named_parameters()}
    out["offline_adapting"] = [p.adapting, p.a.adapting]
    out["child_calls"] = list(p.a.calls)

    class D(TestTimeModuleMixin, nn.Linear):
        pass
    d = D(C, C)
    d.weight.requires_grad_(False)
    d.online()
    out["default_online_rg"] = [q.requires_grad for q in d.parameters()]

    class G(TestTimeModuleMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Linear(C, C)
            self.y = nn.Linear(C, C)
        def online_parameters(self):
            return [{"params": [self.x.weight], "lr": 0.1}]
    g = G().online()
    out["group_online_rg"] = {n: q.requires_grad for n, q in g.named_parameters()}

with section("adapt_removed"):
    try:
        import torchnative.adapt  # noqa: F401
    except ModuleNotFoundError as e:
        out["adapt_import"] = "ModuleNotFoundError"
    else:
        out["adapt_import"] = "IMPORTED"

# --- the rule layer: entropy ---------------------------------------------
with section("entropy"):
    from torch.optim.test_time import TestTimeAdaptation, EntropyMinimization
    m = build(Net)
    opt = torch.optim.SGD([m.norm.weight, m.norm.bias], lr=LR)
    tta = EntropyMinimization(m, opt)
    out["is_optimizer"] = isinstance(tta, torch.optim.Optimizer)
    out["holds_optimizer"] = tta.optimizer is opt
    base = {n: q.detach().clone() for n, q in m.named_parameters()}
    with torch.no_grad():
        out["em_entropy_before"] = float(entropy(m(xs[0])))
    out["em_preds"] = [tta.step(x).detach().reshape(-1).tolist() for x in xs]
    with torch.no_grad():
        out["em_entropy_after"] = float(entropy(m(xs[0])))
    out["em_params"] = flat(dict(m.named_parameters()))
    out["em_has_grad"] = {n: q.grad is not None for n, q in m.named_parameters()}
    out["em_moved"] = sorted(n for n, q in m.named_parameters() if not torch.equal(q, base[n]))
    out["em_delta_covers"] = sorted(tta.delta.covers)
    out["em_delta_norm"] = tta.delta.norm()

with section("entropy_wrong_sign"):
    class Anti(EntropyMinimization):
        def loss(self, outputs):
            return -super().loss(outputs)
    m = build(Net)
    tta = Anti(m, torch.optim.SGD([m.norm.weight, m.norm.bias], lr=LR))
    for x in xs:
        tta.step(x)
    with torch.no_grad():
        out["anti_entropy_after"] = float(entropy(m(xs[0])))

with section("foreign"):
    m, other = build(Net), build(Net)
    try:
        EntropyMinimization(m, torch.optim.SGD([m.norm.weight, other.norm.bias], lr=LR))
    except Exception as e:
        out["foreign"] = "%s: %s" % (type(e).__name__, e)
    else:
        out["foreign"] = "ACCEPTED"

# --- reset ---------------------------------------------------------------
with section("reset"):
    def fresh():
        m = build(BNNet).train()   # training mode: the buffers move too
        opt = torch.optim.SGD([m.norm.weight, m.norm.bias], lr=LR, momentum=0.9)
        return m, opt, EntropyMinimization(m, opt)
    m, opt, tta = fresh()
    state0 = {n: t.detach().clone() for n, t in m.state_dict().items()}
    for x in xs:
        tta.step(x)
    out["reset_moved_before_reset"] = sorted(
        n for n, t in m.state_dict().items() if not torch.equal(t, state0[n]))
    out["reset_optimizer_state_before"] = len(opt.state)
    tta.reset()
    out["reset_differs_after"] = sorted(
        n for n, t in m.state_dict().items() if not torch.equal(t, state0[n]))
    out["reset_optimizer_state_after"] = len(opt.state)
    out["reset_delta_norm"] = tta.delta.norm()
    # The same two steps after reset as on a fresh rule: momentum really went.
    tta.step(xs[0]); tta.step(xs[1])
    m2, opt2, tta2 = fresh()
    tta2.step(xs[0]); tta2.step(xs[1])
    out["reset_replays_like_fresh"] = all(
        torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))

# --- pseudo-labels -------------------------------------------------------
with section("pseudo"):
    from torch.optim.test_time import PseudoLabel
    m = build(Net)
    tta = PseudoLabel(m, torch.optim.SGD([m.norm.weight, m.norm.bias], lr=LR), threshold=THRESHOLD)
    out["pl_preds"] = [tta.step(x).detach().reshape(-1).tolist() for x in xs]
    out["pl_params"] = flat(dict(m.named_parameters()))
    m = build(Net)
    base = {n: q.detach().clone() for n, q in m.named_parameters()}
    tta = PseudoLabel(m, torch.optim.SGD([m.norm.weight, m.norm.bias], lr=LR), threshold=1.01)
    out["pl_none_preds"] = [tta.step(x).detach().reshape(-1).tolist() for x in xs]
    out["pl_none_moved"] = sorted(n for n, q in m.named_parameters() if not torch.equal(q, base[n]))
    out["pl_none_has_grad"] = [q.grad is not None for q in m.parameters()]
    # A skipped batch, then reset(), then an update: the skipped forward must
    # not be left where reset()'s writes would invalidate the next backward.
    tta.reset()
    tta.threshold = 0.0
    try:
        tta.step(xs[0])
    except Exception as e:
        out["pl_skip_reset_update"] = "%s: %s" % (type(e).__name__, e)
    else:
        out["pl_skip_reset_update"] = sorted(
            n for n, q in m.named_parameters() if not torch.equal(q, base[n]))

# --- normalisation statistics --------------------------------------------
with section("norm"):
    from torch.optim.test_time import NormStatistics
    m = build(BNNet)
    params0 = {n: q.detach().clone() for n, q in m.named_parameters()}
    buf0 = {n: b.detach().clone() for n, b in m.named_buffers()}
    tta = NormStatistics(m)
    out["ns_optimizer_is_none"] = tta.optimizer is None
    out["ns_preds"] = [tta.step(x).detach().reshape(-1).tolist() for x in xs]
    out["ns_mean"] = m.norm.running_mean.tolist()
    out["ns_var"] = m.norm.running_var.tolist()
    out["ns_count"] = int(m.norm.num_batches_tracked)
    out["ns_training_after"] = {n or "<root>": mm.training for n, mm in m.named_modules()}
    out["ns_params_moved"] = sorted(n for n, q in m.named_parameters() if not torch.equal(q, params0[n]))
    tta.reset()
    out["ns_reset_differs"] = sorted(n for n, b in m.named_buffers() if not torch.equal(b, buf0[n]))

json.dump(out, sys.stdout)
'''
)


# A configuration without a backward (SPEC S6.5).
_OFF_SCRIPT = (
    "import json, sys\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "assert hasattr(torch._C, '_aten_implemented'), 'not the shim'\n"
    + _MODEL_SRC
    + r'''
out = {}
import torch.optim.test_time as tt
for name in ("EntropyMinimization", "PseudoLabel"):
    try:
        exec("from torch.optim.test_time import " + name)
    except ImportError as e:
        out[name] = "ImportError: %s" % e
    except Exception as e:
        out[name] = "%s: %s" % (type(e).__name__, e)
    else:
        out[name] = "IMPORTED"
try:
    class Mine(tt.TestTimeAdaptation):
        requires_backward = True
        def loss(self, outputs):
            return outputs.sum()
except Exception as e:
    out["subclass"] = "%s: %s" % (type(e).__name__, e)
else:
    out["subclass"] = "ACCEPTED"
m = build(BNNet)
buf0 = {n: b.detach().clone() for n, b in m.named_buffers()}
ns = tt.NormStatistics(m)
for x in batches():
    ns.step(x)
out["ns_moved"] = sorted(n for n, b in m.named_buffers() if not torch.equal(b, buf0[n]))
ns.reset()
out["ns_reset_differs"] = sorted(n for n, b in m.named_buffers() if not torch.equal(b, buf0[n]))
json.dump(out, sys.stdout)
'''
)


_BAD_SCRIPT = r'''
import json, sys
try:
    import torch.optim.test_time  # noqa: F401
except Exception as e:
    out = "%s: %s" % (type(e).__name__, e)
else:
    out = "IMPORTED"
json.dump(out, sys.stdout)
'''


# Upstream only, and none of this layer's code: the steps written by hand.
_ORACLE_SCRIPT = (
    "import json, sys\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    + _MODEL_SRC
    + r'''
out = {}

def gradient_loop(objective, dtype):
    m = build(Net, dtype)
    moving = [m.norm.weight, m.norm.bias]
    opt = torch.optim.SGD(moving, lr=LR)
    preds, counts = [], []
    for x in batches(dtype):
        y = m(x)
        preds.append(y.detach().reshape(-1).tolist())
        loss, n = objective(y)
        counts.append(n)
        if loss is not None:
            grads = torch.autograd.grad(loss, moving)
            for q, g in zip(moving, grads):
                q.grad = g
            opt.step()
            opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        ent = float(entropy(m(batches(dtype)[0])))
    return preds, flat(dict(m.named_parameters())), ent, counts

def em(y):
    return entropy(y), y.shape[0]

def pl(y):
    conf, label = y.softmax(-1).max(-1)
    keep = conf >= THRESHOLD
    n = int(keep.sum())
    if n == 0:
        return None, 0
    return F.cross_entropy(y[keep], label[keep].detach()), n

for tag, obj in (("em", em), ("pl", pl)):
    for dt, name in ((torch.float32, "32"), (torch.float64, "64")):
        preds, params, ent, counts = gradient_loop(obj, dt)
        out[tag + "_preds_" + name] = preds
        out[tag + "_params_" + name] = params
        out[tag + "_entropy_" + name] = ent
        out[tag + "_counts"] = counts
with torch.no_grad():
    out["em_entropy_before"] = float(entropy(build(Net)(batches()[0])))

# Normalisation statistics: predict with the statistics going in, then a
# training-mode forward of the normalisation module takes the batch in.
for dt, name in ((torch.float32, "32"), (torch.float64, "64")):
    m = build(BNNet, dt)
    preds = []
    with torch.no_grad():
        for x in batches(dt):
            m.norm.eval()
            preds.append(m(x).reshape(-1).tolist())
            m.norm.train()
            m(x)
            m.norm.eval()
    out["ns_preds_" + name] = preds
    out["ns_mean_" + name] = m.norm.running_mean.tolist()
    out["ns_var_" + name] = m.norm.running_var.tolist()
    out["ns_count"] = int(m.norm.num_batches_tracked)
json.dump(out, sys.stdout)
'''
)


def _run(script, vendored, extra_env=None):
    env = dict(os.environ)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env.pop("TORCHNATIVE_BACKWARD", None)
    if vendored:
        env["PYTHONPATH"] = _VENDOR_DIR
    else:
        env.pop("PYTHONPATH", None)
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "testtime subprocess (vendored=%s, env=%s) exited %d\n--- stdout ---\n%s\n"
            "--- stderr ---\n%s" % (vendored, extra_env, proc.returncode, proc.stdout, proc.stderr)
        )
    return json.loads(proc.stdout)


_CACHE = {}


def _get(key, *a, **kw):
    if key not in _CACHE:
        _CACHE[key] = _run(*a, **kw)
    return _CACHE[key]


def _shim():
    return _get("shim", _SHIM_SCRIPT, vendored=True)


def _oracle():
    return _get("oracle", _ORACLE_SCRIPT, vendored=False)


def _section(r, name):
    err = r["errors"].get(name)
    assert err is None, "section %r raised: %s" % (name, err)


def _flatten(v):
    if isinstance(v, dict):
        return [x for k in sorted(v) for x in _flatten(v[k])]
    if isinstance(v, list):
        return [x for e in v for x in _flatten(e)]
    return [float(v)]


def _agrees(shim, o32, o64, what):
    """Derived tolerance: upstream's own float32 error against float64, times
    `ORACLE_FACTOR`, floored at 8 ulp of the scale (AGENTS.md section 16)."""
    s, a, b = _flatten(shim), _flatten(o32), _flatten(o64)
    assert len(s) == len(a) == len(b), (what, len(s), len(a), len(b))
    scale = max(abs(x) for x in b)
    assert scale > 1e-6, (what, "degenerate scale", scale)
    rel = max(abs(x - y) for x, y in zip(s, a)) / scale
    oracle_rel = max(abs(x - y) for x, y in zip(a, b)) / scale
    tol = ORACLE_FACTOR * max(oracle_rel, 8 * F32_EPS)
    assert rel <= tol, "%s: shim vs upstream rel %.3g > tol %.3g (upstream f32 vs f64 %.3g)" % (
        what, rel, tol, oracle_rel)
    return rel, tol


def _guard():
    if not _available():
        _skip.skip("no vendored shim or no upstream torch")
        return False
    return True


# -- S6.3: the module layer -------------------------------------------------


def test_the_mixin_is_reached_through_torch_nn_modules_test_time_only():
    """Like `LazyModuleMixin`: a submodule of `torch.nn.modules`, not top level."""
    if not _guard():
        return
    r = _shim()
    _section(r, "mixin")
    assert r["mixin_module"] == "torchnative.nn.modules.test_time", r["mixin_module"]
    assert r["mixin_at_torch_nn_top"] is False, r
    assert r["mixin_in_torch_nn_modules"] is False, r
    print("ok   testtime: TestTimeModuleMixin is torch.nn.modules.test_time's, not torch.nn's")


def test_online_freezes_everything_but_the_online_parameters_and_evaluates():
    if not _guard():
        return
    r = _shim()
    _section(r, "mixin")
    assert r["online_returns_self"] is True
    assert not any(r["online_training"].values()), r["online_training"]
    rg = r["online_rg"]
    want_true = {"head.weight", "head.bias", "a.norm.weight", "a.norm.bias"}
    assert {n for n, v in rg.items() if v} == want_true, rg
    assert r["online_adapting"] == [True, True], r["online_adapting"]
    # The default is every parameter, including one that was frozen before.
    assert r["default_online_rg"] == [True, True], r["default_online_rg"]
    # Optimizer-style groups are accepted, as test-time-adapters' online() does.
    assert r["group_online_rg"] == {
        "x.weight": True, "x.bias": False, "y.weight": False, "y.bias": False
    }, r["group_online_rg"]
    print("ok   testtime: online() is eval(), freeze all, unfreeze online_parameters()")


def test_offline_trains_and_restores_the_original_requires_grad():
    """Restored to what each parameter had when online was entered, through a
    second online() call: a re-snapshot there would make `frozen.weight`
    trainable after offline(), because online() had frozen it."""
    if not _guard():
        return
    r = _shim()
    _section(r, "mixin")
    assert r["offline_returns_self"] is True
    assert all(r["offline_training"].values()), r["offline_training"]
    assert r["offline_rg"] == r["rg_before"], (r["offline_rg"], r["rg_before"])
    assert r["rg_before"]["frozen.weight"] is False, r["rg_before"]
    assert r["offline_adapting"] == [False, False], r["offline_adapting"]
    print("ok   testtime: offline() is train() and the original requires_grad")


def test_online_propagates_only_to_children_that_define_it():
    """`Plain` has a submodule named `online`; calling it would be
    `Linear(True)`, which raises, so the whole section would go red."""
    if not _guard():
        return
    r = _shim()
    _section(r, "mixin")
    assert r["child_calls"] == [True, True, False], r["child_calls"]
    assert r["online_rg"]["b.online.weight"] is False, r["online_rg"]
    print("ok   testtime: online() reaches children that define it and no other")


def test_torchnative_adapt_is_removed():
    """Replaced by this layer (INTENT section 4). Not a capability: a guard
    against the old wrapper coming back beside the new one."""
    if not _guard():
        return
    r = _shim()
    _section(r, "adapt_removed")
    assert r["adapt_import"] == "ModuleNotFoundError", r["adapt_import"]
    print("ok   testtime: torchnative.adapt is gone")


# -- S6.4: the rule layer ---------------------------------------------------


def test_the_rule_is_not_an_optimizer():
    if not _guard():
        return
    r = _shim()
    _section(r, "entropy")
    assert r["is_optimizer"] is False and r["holds_optimizer"] is True, r
    print("ok   testtime: TestTimeAdaptation holds an optimizer and is not one")


def test_entropy_minimization_agrees_with_upstream_and_lowers_entropy():
    if not _guard():
        return
    s, o = _shim(), _oracle()
    _section(s, "entropy")
    rp, tp = _agrees(s["em_params"], o["em_params_32"], o["em_params_64"], "entropy params")
    rq, tq = _agrees(s["em_preds"], o["em_preds_32"], o["em_preds_64"], "entropy predictions")
    assert abs(s["em_entropy_before"] - o["em_entropy_before"]) <= 8 * F32_EPS * abs(o["em_entropy_before"]), (
        s["em_entropy_before"], o["em_entropy_before"])
    assert s["em_entropy_after"] < s["em_entropy_before"] - 1e-3, (
        s["em_entropy_before"], s["em_entropy_after"])
    print(f"ok   testtime: EntropyMinimization agrees (params {rp:.3g}<= {tp:.3g}, preds {rq:.3g}<= {tq:.3g}),"
          f" entropy {s['em_entropy_before']:.4f} -> {s['em_entropy_after']:.4f}")


def test_the_wrong_sign_does_not_lower_entropy():
    if not _guard():
        return
    s = _shim()
    _section(s, "entropy_wrong_sign")
    assert s["anti_entropy_after"] > s["em_entropy_before"], (s["anti_entropy_after"], s["em_entropy_before"])
    print("ok   testtime: maximising entropy raises it, so the descent is real")


def test_only_the_optimizer_parameters_move_and_get_gradients():
    if not _guard():
        return
    s = _shim()
    _section(s, "entropy")
    moving = ["norm.bias", "norm.weight"]
    assert s["em_moved"] == moving, s["em_moved"]
    assert {n for n, v in s["em_has_grad"].items() if v} <= set(moving), s["em_has_grad"]
    assert s["em_delta_covers"] == moving, s["em_delta_covers"]
    assert s["em_delta_norm"] > 0.0, s["em_delta_norm"]
    print("ok   testtime: only the optimizer's parameters move; no other gets a .grad")


def test_a_foreign_optimizer_parameter_is_refused_by_name():
    if not _guard():
        return
    s = _shim()
    _section(s, "foreign")
    assert s["foreign"].startswith("ValueError:"), s["foreign"]
    assert "not a parameter of" in s["foreign"], s["foreign"]
    print("ok   testtime: an optimizer over another model's parameter is refused")


def test_reset_restores_parameters_buffers_and_optimizer_state_bit_for_bit():
    if not _guard():
        return
    s = _shim()
    _section(s, "reset")
    moved = s["reset_moved_before_reset"]
    assert "norm.weight" in moved and "norm.running_mean" in moved, moved
    assert s["reset_optimizer_state_before"] > 0, s
    assert s["reset_differs_after"] == [], s["reset_differs_after"]
    assert s["reset_optimizer_state_after"] == 0, s
    assert s["reset_delta_norm"] == 0.0, s["reset_delta_norm"]
    assert s["reset_replays_like_fresh"] is True, s
    print("ok   testtime: reset() restores parameters, buffers and optimizer state")


def test_pseudo_label_agrees_with_upstream_and_skips_unconfident_batches():
    if not _guard():
        return
    s, o = _shim(), _oracle()
    _section(s, "pseudo")
    counts = o["pl_counts"]
    # The threshold has to select some samples and not all, or the mask is untested.
    assert any(0 < c < 6 for c in counts), counts
    rp, tp = _agrees(s["pl_params"], o["pl_params_32"], o["pl_params_64"], "pseudo-label params")
    rq, tq = _agrees(s["pl_preds"], o["pl_preds_32"], o["pl_preds_64"], "pseudo-label predictions")
    assert s["pl_none_moved"] == [], s["pl_none_moved"]
    assert not any(s["pl_none_has_grad"]), s["pl_none_has_grad"]
    assert len(s["pl_none_preds"]) == 3, s["pl_none_preds"]
    assert s["pl_skip_reset_update"] == ["norm.bias", "norm.weight"], s["pl_skip_reset_update"]
    print(f"ok   testtime: PseudoLabel agrees (params {rp:.3g}<= {tp:.3g}, kept per batch {counts}),"
          " and no confident sample means no update")


def test_norm_statistics_agree_with_upstream():
    if not _guard():
        return
    s, o = _shim(), _oracle()
    _section(s, "norm")
    assert s["ns_optimizer_is_none"] is True, s
    rm, tm = _agrees(s["ns_mean"], o["ns_mean_32"], o["ns_mean_64"], "running mean")
    rv, tv = _agrees(s["ns_var"], o["ns_var_32"], o["ns_var_64"], "running var")
    _agrees(s["ns_preds"], o["ns_preds_32"], o["ns_preds_64"], "normstats predictions")
    assert s["ns_count"] == o["ns_count"] == 3, (s["ns_count"], o["ns_count"])
    assert not any(s["ns_training_after"].values()), s["ns_training_after"]
    assert s["ns_params_moved"] == [], s["ns_params_moved"]
    assert s["ns_reset_differs"] == [], s["ns_reset_differs"]
    print(f"ok   testtime: NormStatistics agree (mean {rm:.3g}<= {tm:.3g}, var {rv:.3g}<= {tv:.3g}) and reset")


# -- S6.5: a configuration without a backward -------------------------------


def test_a_backward_free_configuration_refuses_the_gradient_rules_by_name():
    if not _guard():
        return
    r = _get("off", _OFF_SCRIPT, vendored=True, extra_env={"TORCHNATIVE_BACKWARD": "off"})
    for name in ("EntropyMinimization", "PseudoLabel"):
        assert r[name].startswith("ImportError:"), (name, r[name])
        assert "TORCHNATIVE_BACKWARD" in r[name] and name in r[name], (name, r[name])
    assert r["subclass"].startswith("TypeError:"), r["subclass"]
    assert "Mine" in r["subclass"] and "TORCHNATIVE_BACKWARD" in r["subclass"], r["subclass"]
    print("ok   testtime: TORCHNATIVE_BACKWARD=off refuses the gradient rules by name")


def test_norm_statistics_still_run_in_a_backward_free_configuration():
    if not _guard():
        return
    r = _get("off", _OFF_SCRIPT, vendored=True, extra_env={"TORCHNATIVE_BACKWARD": "off"})
    assert "norm.running_mean" in r["ns_moved"], r["ns_moved"]
    assert r["ns_reset_differs"] == [], r["ns_reset_differs"]
    print("ok   testtime: NormStatistics adapts and resets without a backward")


def test_an_unrecognised_backward_setting_is_refused_by_name():
    if not _guard():
        return
    r = _get("bad", _BAD_SCRIPT, vendored=True, extra_env={"TORCHNATIVE_BACKWARD": "maybe"})
    assert r.startswith("ValueError:"), r
    assert "TORCHNATIVE_BACKWARD" in r and "maybe" in r, r
    print("ok   testtime: TORCHNATIVE_BACKWARD=maybe is refused by name")


if __name__ == "__main__":
    failures = _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")],
        suite="test_testtime",
    )
    raise SystemExit(1 if failures else 0)
