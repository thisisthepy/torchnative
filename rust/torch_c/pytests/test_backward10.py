"""Tests for issue #10 and docs/training/BACKWARD10.md -- double backward,
`torch.autograd.Function` and hooks, against upstream.

SPEC S6.1 said all three "refuse by name". Measured before this round
(docs/training/BACKWARD10.md §1), **two of the three did not refuse at all**:

  * `torch.autograd.Function` ran the forward with grad mode on, so the tape
    differentiated the forward's own ops and **never called `backward`**. A
    gradient-reversal layer came back with the wrong sign, silently.
  * a `register_hook` on a leaf was stored and never read. A hook that scaled
    the gradient by ten left `.grad` unscaled, silently.

Only `create_graph=True`, the non-leaf hook and the module backward hook
refused, and the module hook refused with upstream's own "please open an
issue" rather than a name.

Every comparison below is element-wise against **upstream torch running the
identical program in a second interpreter** (docs/training/BACKWARD9.md §1),
and every tolerance is derived, not chosen (AGENTS.md §16): upstream runs the
program a third time in float64, and the bound is the p90 of upstream's own
float32-vs-float64 relative error on that quantity, floored at 8 float32 ulp.
There is no constant in this file to widen.

The inputs are RNG-free (`det`): the two interpreters do not share a
generator, so a seed would compare two different programs.
"""

import json
import os
import subprocess
import sys

from test_shim import _upstream_torch, _CKPT_VENDOR_DIR, _CKPT_VENDOR_SHIM
import _skip

_EPS32 = 1.1920929e-07


def _run(script, env_overrides):
    env = dict(os.environ)
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _shim(script, dtype="float32"):
    return _run(script, {"PYTHONPATH": _CKPT_VENDOR_DIR, "TORCH_USE_RTLD_GLOBAL": "1",
                         "TN_DTYPE": dtype})


def _upstream(script, dtype):
    return _run(script, {"PYTHONPATH": None, "TORCH_USE_RTLD_GLOBAL": None,
                         "TN_DTYPE": dtype})


def _available():
    return os.path.isfile(_CKPT_VENDOR_SHIM)


def _flat(value):
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_flat(item))
        return out
    return [float(value)]


def _derived_tolerance(up32, up64):
    """docs/numerics/AGREE.md §2's method: the p90 of upstream's own f32-vs-f64
    error, relative to the quantity's largest magnitude, floored at 8 ulp."""
    scale = max(max(abs(v) for v in up32), 1e-30)
    rel = sorted(abs(a - b) / scale for a, b in zip(up32, up64))
    return max(rel[int(0.9 * (len(rel) - 1))], 8 * _EPS32) * scale


def _agree(shim, up32, up64, fields):
    """Element-wise agreement for each field, and the worst ratio seen.

    Every field must be non-trivial on upstream's side: a quantity that is
    all zeros would agree with a shim that computed nothing.
    """
    report = {}
    for field in fields:
        ours, theirs, oracle = _flat(shim[field]), _flat(up32[field]), _flat(up64[field])
        assert len(ours) == len(theirs) == len(oracle), (field, len(ours), len(theirs))
        assert ours, f"{field}: compared an empty quantity"
        assert max(abs(v) for v in theirs) > 0.0, f"{field}: upstream's value is all zero"
        tol = _derived_tolerance(theirs, oracle)
        worst = max(abs(a - b) for a, b in zip(ours, theirs))
        assert worst <= tol, (
            f"{field}: shim disagrees with upstream by {worst:.3e} > derived "
            f"tolerance {tol:.3e} (n={len(ours)})")
        report[field] = (worst, tol)
    return report


def _three(script):
    shim = _shim(script)
    assert shim["who"] == "shim", shim["who"]
    up32 = _upstream(script, "float32")
    up64 = _upstream(script, "float64")
    assert up32["who"] == up64["who"] == "upstream"
    return shim, up32, up64


_PRELUDE = r"""
import json, os, sys, warnings
warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = getattr(torch, os.environ["TN_DTYPE"])
out = {"who": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def det(n, seed):
    return [((seed * 1103515245 + i * 12345) % 2000 - 1000) / 1000.0 for i in range(n)]


def make(shape, seed, scale=1.0):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([v * scale for v in det(n, seed)], dtype=DT).reshape(shape)


def init(model, seed):
    with torch.no_grad():
        for k, p in enumerate(model.parameters()):
            p.copy_(make(tuple(p.shape), seed + 7 * k, 0.5))


def flat(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]
"""


# --- 1. double backward ------------------------------------------------------

_GRADIENT_PENALTY = _PRELUDE + r"""
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8, dtype=DT)
        self.l2 = nn.Linear(8, 8, dtype=DT)
        self.l3 = nn.Linear(8, 1, dtype=DT)

    def forward(self, x):
        return self.l3(torch.tanh(self.l2(torch.tanh(self.l1(x)))))


D = Critic()
init(D, 3)
opt = torch.optim.SGD(D.parameters(), lr=0.05)
real = make((6, 4), 41)
fake = make((6, 4), 43)
alpha = make((6, 1), 47).abs()

losses = []
for step in range(3):
    opt.zero_grad()
    interp = (alpha * real + (1 - alpha) * fake).detach().requires_grad_(True)
    d_interp = D(interp)
    (g,) = torch.autograd.grad(
        d_interp, interp, grad_outputs=torch.ones_like(d_interp), create_graph=True)
    if step == 0:
        out["g_requires_grad"] = bool(g.requires_grad)
        out["g"] = flat(g)
    gp = ((g.pow(2).sum(dim=1) + 1e-12).sqrt() - 1.0).pow(2).mean()
    loss = D(fake).mean() - D(real).mean() + 10.0 * gp
    loss.backward()
    if step == 0:
        out["first_grads"] = [flat(p.grad) for p in D.parameters()]
        out["interp_grad"] = flat(interp.grad)
    opt.step()
    losses.append(float(loss))
out["losses"] = losses
out["params"] = [flat(p) for p in D.parameters()]

# A Hessian-vector product: the second derivative asked for by
# `torch.autograd.grad` rather than by `.backward()`.
w = make((5,), 51).requires_grad_(True)
xx = make((5,), 53)
f = (torch.sin(w * xx) * w).sum() + w.pow(3).sum()
(gw,) = torch.autograd.grad(f, w, create_graph=True)
v = make((5,), 57)
(hv,) = torch.autograd.grad((gw * v).sum(), w)
out["hvp_g"] = flat(gw)
out["hvp"] = flat(hv)
print(json.dumps(out))
"""


def test_a_wgan_gradient_penalty_trains_and_agrees_with_upstream():
    """**The bar for double backward.** A WGAN-GP critic -- three `Linear`s and
    two `tanh`s -- trained for three SGD steps, where each step takes
    `torch.autograd.grad(D(interp), interp, create_graph=True)`, builds the
    penalty `(||g|| - 1)^2` from it and backpropagates the whole loss through
    that gradient. Plus a Hessian-vector product through `sin` and `pow`.

    `tanh` is the activation on purpose: its rule reads the forward's
    *result* (`1 - y^2`). A create_graph backward that treated the forward's
    values as constants would drop every second-order term through `y` and
    still return a number; this compares that number with upstream's.
    """
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    if _upstream_torch is None:
        _skip.skip("no upstream torch in this interpreter to compare against")
        return
    shim, up32, up64 = _three(_GRADIENT_PENALTY)
    assert shim["g_requires_grad"] is True and up32["g_requires_grad"] is True
    report = _agree(shim, up32, up64, (
        "g", "first_grads", "interp_grad", "losses", "params", "hvp_g", "hvp"))
    print("BACKWARD10 gradient penalty:", {k: f"{w:.2e}/{t:.2e}" for k, (w, t) in report.items()})


def test_create_graph_through_a_rule_with_no_graph_refuses_by_name():
    """`aten.embedding.default`'s weight gradient is an `index_put_` into a
    zero table -- an in-place write the eager tape does not record. Under
    `create_graph=True` that gradient would come back with no graph while
    its input had one, so a second derivative through it would be silently
    zero. It refuses, naming the op, instead."""
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    script = _PRELUDE + r"""
emb = nn.Embedding(5, 3, dtype=DT)
init(emb, 5)
ids = torch.tensor([1, 3, 3])
scale = make((3, 3), 61).requires_grad_(True)
y = (emb(ids) * scale).pow(2).sum()
try:
    torch.autograd.grad(y, [emb.weight], create_graph=True)
    out["raised"] = None
except NotImplementedError as exc:
    out["raised"] = str(exc)
print(json.dumps(out))
"""
    shim = _shim(script)
    assert shim["raised"] is not None, "create_graph through embedding was answered"
    assert "aten.embedding.default" in shim["raised"], shim["raised"]
    assert "create_graph" in shim["raised"], shim["raised"]


# --- 2. torch.autograd.Function -----------------------------------------------

_FUNCTION_SCRIPT = _PRELUDE + r"""
from torch.autograd.function import once_differentiable


class LowRank(torch.autograd.Function):
    # A LoRA-style low-rank update, scale * (x A^T) B^T, with its backward
    # written out by hand -- the shape PEFT-style layers take.
    @staticmethod
    def forward(ctx, x, A, B, scale):
        ctx.save_for_backward(x, A, B)
        ctx.scale = scale
        return (x @ A.t()) @ B.t() * scale

    @staticmethod
    def backward(ctx, g):
        x, A, B = ctx.saved_tensors
        gs = g * ctx.scale
        h = x @ A.t()
        gh = gs @ B
        gx = gh @ A if ctx.needs_input_grad[0] else None
        out.setdefault("needs_input_grad", []).append(list(ctx.needs_input_grad))
        return gx, gh.t() @ x, gs.t() @ h, None


class GradReverse(torch.autograd.Function):
    # Domain-adversarial training's gradient reversal: identity forward,
    # negated and scaled backward. The forward's own ops differentiate to +g.
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lam, None


class QuantSTE(torch.autograd.Function):
    # A straight-through estimator in the separate forward/setup_context
    # shape, with a non-differentiable second output and @once_differentiable.
    @staticmethod
    def forward(x, step):
        return torch.round(x / step) * step, (x > 0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, step = inputs
        ctx.save_for_backward(x)
        ctx.mark_non_differentiable(output[1])

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gpos):
        (x,) = ctx.saved_tensors
        # What arrives for the output no gradient reached: upstream
        # materialises it rather than passing None.
        out.setdefault("gpos", []).append(
            None if gpos is None else [str(gpos.dtype), list(gpos.shape),
                                       bool((gpos == 0).all())])
        return gq * (x.abs() <= 1.0).to(gq.dtype), None


class LoRALinear(nn.Module):
    def __init__(self, i, o, r):
        super().__init__()
        self.base = nn.Linear(i, o, dtype=DT)
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(make((r, i), 71 + i, 0.3))
        self.B = nn.Parameter(make((o, r), 73 + o, 0.3))

    def forward(self, x):
        return self.base(x) + LowRank.apply(x, self.A, self.B, 0.5)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = LoRALinear(6, 5, 2)
        self.cls = nn.Linear(5, 3, dtype=DT)
        self.dom = nn.Linear(5, 2, dtype=DT)

    def forward(self, x):
        h = torch.tanh(self.feat(x))
        q, pos = QuantSTE.apply(h, 0.25)
        return self.cls(q), self.dom(GradReverse.apply(h, 0.3)), pos


net = Net()
with torch.no_grad():
    net.feat.base.weight.copy_(make((5, 6), 81, 0.5))
    net.feat.base.bias.copy_(make((5,), 83, 0.5))
    net.cls.weight.copy_(make((3, 5), 85, 0.5))
    net.cls.bias.copy_(make((3,), 87, 0.5))
    net.dom.weight.copy_(make((2, 5), 89, 0.5))
    net.dom.bias.copy_(make((2,), 91, 0.5))
opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.2)
x = make((7, 6), 93)
yc = torch.tensor([0, 1, 2, 1, 0, 2, 1])
yd = torch.tensor([0, 1, 1, 0, 1, 0, 0])
out["names"] = [n for n, p in net.named_parameters() if p.requires_grad]

losses = []
for step in range(3):
    opt.zero_grad()
    logits, dom, pos = net(x)
    if step == 0:
        out["pos_requires_grad"] = bool(pos.requires_grad)
        out["dom_grad_fn"] = type(dom.grad_fn).__name__
        h = torch.tanh(net.feat(x))
        q, _ = QuantSTE.apply(h, 0.25)
        out["q_grad_fn"] = type(q.grad_fn).__name__
        out["q_is_leaf"] = bool(q.is_leaf)
        del h, q
    loss = F.cross_entropy(logits, yc) + F.cross_entropy(dom, yd)
    loss.backward()
    if step == 0:
        out["first_grads"] = [flat(p.grad) for n, p in net.named_parameters() if p.requires_grad]
    opt.step()
    losses.append(float(loss))
out["losses"] = losses
out["params"] = [flat(p) for n, p in net.named_parameters() if p.requires_grad]
out["base_weight_grad_is_none"] = net.feat.base.weight.grad is None
print(json.dumps(out))
"""


def test_custom_autograd_functions_train_and_agree_with_upstream():
    """**The bar for `autograd.Function`.** One network carrying three custom
    Functions, each a shape real training code uses, trained three SGD steps:

      * `LowRank` -- a LoRA-style update with a hand-written `backward`, three
        tensors through `ctx.save_for_backward`, a non-tensor argument, and a
        `None` gradient where `ctx.needs_input_grad` says so;
      * `GradReverse` -- identity forward, `-lam * g` backward. **The forward's
        own ops differentiate to `+g`**, so this is the case that was silently
        wrong before (docs/training/BACKWARD10.md §1): a tape that walked the
        forward instead of calling `backward` gets the domain head's
        contribution to the feature extractor with the opposite sign;
      * `QuantSTE` -- a straight-through `round` in the separate
        `forward`/`setup_context` shape, `@once_differentiable`, and a second
        output marked non-differentiable.
    """
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    if _upstream_torch is None:
        _skip.skip("no upstream torch in this interpreter to compare against")
        return
    shim, up32, up64 = _three(_FUNCTION_SCRIPT)
    assert shim["names"] == up32["names"], (shim["names"], up32["names"])
    for key in ("pos_requires_grad", "dom_grad_fn", "q_grad_fn", "q_is_leaf",
                "base_weight_grad_is_none", "needs_input_grad", "gpos"):
        assert shim[key] == up32[key], (key, shim[key], up32[key])
    assert shim["q_grad_fn"] == "QuantSTEBackward", shim["q_grad_fn"]
    report = _agree(shim, up32, up64, ("first_grads", "losses", "params"))
    print("BACKWARD10 autograd.Function:", {k: f"{w:.2e}/{t:.2e}" for k, (w, t) in report.items()})


def test_a_custom_function_backward_is_what_runs_not_the_forward_ops():
    """The defect, isolated: the gradient is the user's `backward`, not the
    derivative of what `forward` computed. A gradient reversal whose forward is
    `x.clone()` -- `clone` has a tape rule, so before this round the tape
    differentiated it and returned `+g` with `backward` never called
    (measured: `[3.0, 4.0]` where upstream gives `[-3.0, -4.0]`, no error).
    Against `-0.3 * g`, which is what upstream computes."""
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    script = _PRELUDE + r"""
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        out["backward_calls"] = out.get("backward_calls", 0) + 1
        return g.neg() * ctx.lam, None


x = make((4,), 101).requires_grad_(True)
w = make((4,), 103)
(GradReverse.apply(x, 0.3) * w).sum().backward()
out["grad"] = flat(x.grad)
out["expected"] = flat(-0.3 * w)
print(json.dumps(out))
"""
    shim = _shim(script)
    assert shim.get("backward_calls") == 1, shim
    assert shim["grad"] == shim["expected"], (shim["grad"], shim["expected"])


def test_a_custom_backward_is_validated_the_way_upstream_validates_it():
    """Upstream's `validate_outputs`, case by case, against upstream: a
    gradient with extra leading and size-1-broadcast dimensions is summed down
    to the input's shape; a float64 gradient for a float32 input is cast; a
    wrong number of gradients is a `RuntimeError` with upstream's message;
    trailing `None`s beyond the argument count are accepted."""
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    if _upstream_torch is None:
        _skip.skip("no upstream torch in this interpreter to compare against")
        return
    script = _PRELUDE + r"""
class Broadcasting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x * w

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        # [2, 4, 3] for an x of [4, 1]: two leading dims to drop, one to sum.
        big = (g * w).unsqueeze(0).expand(2, 4, 3) * 0.5
        return big.reshape(2, 4, 3), (g * x).double(), None, None


x = make((4, 1), 141).requires_grad_(True)
w = make((4, 3), 143).requires_grad_(True)
(Broadcasting.apply(x, w) * make((4, 3), 147)).sum().backward()
out["gx"] = flat(x.grad)
out["gw"] = flat(w.grad)
out["gw_dtype"] = str(w.grad.dtype)


class Short(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def backward(ctx, g):
        return g


try:
    Short.apply(make((2,), 151).requires_grad_(True), make((2,), 153)).sum().backward()
    out["short"] = None
except RuntimeError as exc:
    out["short"] = str(exc)
print(json.dumps(out))
"""
    shim, up32, up64 = _three(script)
    assert shim["gw_dtype"] == up32["gw_dtype"], (shim["gw_dtype"], up32["gw_dtype"])
    assert shim["short"] is not None and shim["short"] == up32["short"], (
        shim["short"], up32["short"])
    _agree(shim, up32, up64, ("gx", "gw"))


def test_node_pre_and_post_hooks_on_a_custom_function_agree_with_upstream():
    """`grad_fn.register_prehook` and `grad_fn.register_hook` on a custom
    Function's node, each replacing what it is handed. Written after the
    nullification that skipped every pre-hook left the whole suite green
    (docs/training/BACKWARD10.md §6, N13): `nn.Module`'s `BackwardHook` only
    uses post-hooks, so no other test here reaches a pre-hook."""
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    if _upstream_torch is None:
        _skip.skip("no upstream torch in this interpreter to compare against")
        return
    script = _PRELUDE + r"""
class Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x * w

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        out.setdefault("seen_g", []).append(flat(g))
        return g * w, g * x


x = make((4,), 161).requires_grad_(True)
w = make((4,), 163).requires_grad_(True)
y = Scale.apply(x, w)
y.grad_fn.register_prehook(lambda grad_outputs: (grad_outputs[0] * 3.0 + 0.5,))
y.grad_fn.register_hook(lambda grad_inputs, grad_outputs: (grad_inputs[0] - 1.0, grad_inputs[1]))
(y * make((4,), 167)).sum().backward()
out["gx"] = flat(x.grad)
out["gw"] = flat(w.grad)
print(json.dumps(out))
"""
    shim, up32, up64 = _three(script)
    _agree(shim, up32, up64, ("seen_g", "gx", "gw"))


def test_a_custom_function_refuses_mark_dirty_by_name():
    """What is not built, refused where a caller meets it: an input written in
    place and marked dirty would have to become a non-leaf, which this shim
    never does to an existing tensor (`mark_from_op`'s last clause)."""
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    script = _PRELUDE + r"""
class Dirty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x.mul_(2.0)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, g):
        return g * 2.0


a = make((3,), 111).requires_grad_(True)
b = a * 1.0
try:
    Dirty.apply(b)
    out["dirty"] = None
except NotImplementedError as exc:
    out["dirty"] = str(exc)
print(json.dumps(out))
"""
    shim = _shim(script)
    assert shim["dirty"] is not None and "mark_dirty" in shim["dirty"], shim["dirty"]


# --- 3. hooks -----------------------------------------------------------------

_HOOK_SCRIPT = _PRELUDE + r"""
model = nn.Sequential(nn.Linear(5, 6, dtype=DT), nn.Tanh(), nn.Linear(6, 4, dtype=DT),
                      nn.Tanh(), nn.Linear(4, 3, dtype=DT))
init(model, 13)
rec = {"tanh_out_grad": [], "gin": [], "gout": [], "pre_gout": [], "calls": []}

# Gradient clipping by value as a leaf hook, the way clipping helpers and
# some adapters install it; and a leaf hook that rescales a bias gradient.
clip = model[0].weight.register_hook(lambda g: g.clamp(-0.05, 0.05))
model[4].bias.register_hook(lambda g: g * 3.0)


# A non-leaf hook installed from a forward hook, every forward: logs the
# activation gradient and halves it.
def on_tanh(mod, inputs, output):
    def hook(g):
        rec["tanh_out_grad"].append(flat(g))
        rec["calls"].append("tanh")
        return g * 0.5
    output.register_hook(hook)


model[1].register_forward_hook(on_tanh)


# A module full backward hook that reads grad_input/grad_output and replaces
# grad_input, and a full backward pre-hook that replaces grad_output.
def full_hook(mod, grad_input, grad_output):
    rec["gin"].append(flat(grad_input[0]))
    rec["gout"].append(flat(grad_output[0]))
    rec["calls"].append("full")
    return (grad_input[0] * 2.0,)


def pre_hook(mod, grad_output):
    rec["pre_gout"].append(flat(grad_output[0]))
    rec["calls"].append("pre")
    return (grad_output[0] + 0.01,)


model[2].register_full_backward_hook(full_hook)
model[4].register_full_backward_pre_hook(pre_hook)

opt = torch.optim.SGD(model.parameters(), lr=0.3)
x = make((6, 5), 17)
y = torch.tensor([0, 1, 2, 1, 0, 2])
losses = []
for step in range(3):
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    if step == 0:
        out["first_grads"] = [flat(p.grad) for p in model.parameters()]
    if step == 1:
        clip.remove()
    opt.step()
    losses.append(float(loss))
out["losses"] = losses
out["params"] = [flat(p) for p in model.parameters()]
for k in ("tanh_out_grad", "gin", "gout", "pre_gout"):
    out[k] = rec[k]
out["calls"] = rec["calls"]

# The leaf hook through torch.autograd.grad, which upstream also applies.
z = make((3,), 19).requires_grad_(True)
z.register_hook(lambda g: g * 10.0)
(gz,) = torch.autograd.grad((z * z).sum(), z)
out["grad_hooked"] = flat(gz)
print(json.dumps(out))
"""


def test_tensor_and_module_hooks_train_and_agree_with_upstream():
    """**The bar for hooks.** A five-layer MLP trained three SGD steps with:

      * a leaf `register_hook` that **clips** `model[0].weight`'s gradient by
        value -- removed through its handle after step two, so removal is
        compared too -- and one that triples a bias gradient;
      * a **non-leaf** `register_hook`, installed from a forward hook on every
        forward, that logs and halves an activation gradient;
      * a module **full backward hook** that reads `grad_input`/`grad_output`
        and replaces `grad_input`, and a full backward **pre**-hook that
        replaces `grad_output` -- `nn.Module`'s own `BackwardHook`, which
        installs itself through `BackwardHookFunction`, an `autograd.Function`.

    Compared: the trajectory, the first step's gradients, the final
    parameters, every tensor each hook was handed, and the order the hooks
    fired in. The leaf-hook half was silently ignored before this round:
    `.grad` came back unclipped and unscaled.
    """
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    if _upstream_torch is None:
        _skip.skip("no upstream torch in this interpreter to compare against")
        return
    shim, up32, up64 = _three(_HOOK_SCRIPT)
    assert shim["calls"] == up32["calls"], (shim["calls"], up32["calls"])
    for key in ("tanh_out_grad", "gin", "gout", "pre_gout"):
        assert len(shim[key]) == len(up32[key]) == 3, (key, len(shim[key]), len(up32[key]))
    report = _agree(shim, up32, up64, (
        "first_grads", "losses", "params", "tanh_out_grad", "gin", "gout",
        "pre_gout", "grad_hooked"))
    print("BACKWARD10 hooks:", {k: f"{w:.2e}/{t:.2e}" for k, (w, t) in report.items()})


def test_the_hooks_this_round_did_not_build_refuse_by_name():
    """`register_post_accumulate_grad_hook` died with a bare `AttributeError`
    and the legacy `Module.register_backward_hook` needs `register_hook` on an
    aten node's `grad_fn`, which is the hollow node. Both now name themselves."""
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    script = _PRELUDE + r"""
p = make((3,), 121).requires_grad_(True)
try:
    p.register_post_accumulate_grad_hook(lambda t: None)
    out["post"] = None
except NotImplementedError as exc:
    out["post"] = str(exc)
lin = nn.Linear(3, 2, dtype=DT)
lin.register_backward_hook(lambda m, gi, go: None)
try:
    lin(make((1, 3), 123)).sum().backward()
    out["legacy"] = None
except NotImplementedError as exc:
    out["legacy"] = str(exc)
print(json.dumps(out))
"""
    shim = _shim(script)
    assert shim["post"] is not None and "register_post_accumulate_grad_hook" in shim["post"], shim
    assert shim["legacy"] is not None and "register_hook" in shim["legacy"], shim


def test_a_seeded_backward_from_a_non_scalar_output_agrees_with_upstream():
    """`y.backward(torch.ones_like(y))` -- and `torch.autograd.grad(...,
    grad_outputs=torch.ones_like(d))`, which is how every gradient penalty is
    written -- for a `y` with more than one row. Before this round the tape
    extracted the seed tensor as a *sequence* and refused with "1 output(s)
    and 3 gradient(s)" (measured: `x * 2` over a `[3, 2]` leaf)."""
    if not _available():
        _skip.skip("vendored tree has no _C.abi3.so")
        return
    if _upstream_torch is None:
        _skip.skip("no upstream torch in this interpreter to compare against")
        return
    script = _PRELUDE + r"""
x = make((3, 2), 131).requires_grad_(True)
y = torch.tanh(x) * make((3, 2), 133)
y.backward(make((3, 2), 137))
out["grad"] = flat(x.grad)
print(json.dumps(out))
"""
    shim, up32, up64 = _three(script)
    _agree(shim, up32, up64, ("grad",))


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_backward10")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
