"""The eager tape's recordings live as long as their outputs (#251).

Upstream's graph hangs off the output tensors: a forward whose outputs are all
gone leaves nothing behind, so a later in-place write into the parameters it
read constrains nothing. This stack keeps one eager tape per thread, and until
#251 a recording outlived its outputs: a forward that was never differentiated,
then a write into a weight it read (an optimizer step, a weight reset), and the
**next backward refused, even on another model**. Leader decision (2026-11-13):
tie each recording to its outputs' lifetime, as upstream does.

What is held down here, each against upstream torch running the same program:

* an abandoned forward, a write, and a backward on another model agree with
  upstream (the #251 repro);
* the same on the *same* model, which is what an evaluation forward outside
  `no_grad` followed by a training step is;
* a live graph beside a dead one keeps working when only the dead one's
  weights are written;
* the dead recording is actually released (`_eager_tape_size()`), not just
  forgiven;
* **the refusal this tape exists for is kept**: a write into a storage a
  *live* graph reads still refuses by name, the BACKWARD5 section 1 view case
  included.

What this file cannot find: liveness that is visible to Python only through a
weak reference (nothing in this tree differentiates through one), and the
free-threaded build, where reference counts are read across threads.
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


# Executed verbatim on both sides. Every case returns gradients (as lists) or
# the refusal message; upstream must produce the same gradients, and the two
# refusal cases are refusals on upstream too (its version counter).
_PROGRAM_SRC = r'''
import torch.nn as nn

def _det(n, seed):
    return [((seed * 1103515245 + i * 12345) % 2000 - 1000) / 1000.0 for i in range(n)]

def build(seed):
    m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 3))
    with torch.no_grad():
        for i, p in enumerate(m.parameters()):
            p.copy_(torch.tensor(_det(p.numel(), seed + 5 * i)).reshape(p.shape))
    return m

X = torch.tensor(_det(24, 41)).reshape(6, 4)

def grads(model):
    loss = model(X).log_softmax(-1).sum()
    return [g.reshape(-1).tolist() for g in torch.autograd.grad(loss, list(model.parameters()))]

def caught(fn):
    try:
        fn()
    except Exception as e:
        return "%s: %s" % (type(e).__name__, str(e))
    return "ACCEPTED"

def program(shim):
    out = {}
    size = (lambda: torch._C._eager_tape_size()) if shim else (lambda: None)

    # 1. #251: abandoned forward, a write into its weights, backward elsewhere.
    m = build(1)
    float(m(X).sum())                      # recorded, never differentiated
    with torch.no_grad():
        m[1].weight.add_(0.5)
    out["other_model"] = caught(lambda: out.__setitem__("other_grads", grads(build(2))))
    out["size_after_backward"] = size()

    # 2. The same model: an evaluation forward outside no_grad, then a step.
    m = build(3)
    float(m(X).sum())
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(0.9)
    out["same_model"] = caught(lambda: out.__setitem__("same_grads", grads(m)))

    # 3. A live graph beside a dead one; only the dead one's weights move.
    live, dead = build(4), build(5)
    y = live(X).log_softmax(-1).sum()      # held
    float(dead(X).sum())                   # dropped
    before = size()
    with torch.no_grad():
        dead[0].weight.zero_()
    out["pruned"] = None if before is None else before - size()
    out["live_beside_dead"] = caught(lambda: out.__setitem__(
        "live_grads", [g.reshape(-1).tolist() for g in torch.autograd.grad(y, list(live.parameters()))]))

    # 4. A live graph whose weight is written: still refused.
    m = build(6)
    y = m(X).sum()
    with torch.no_grad():
        m[2].weight.add_(1.0)
    out["live_write"] = caught(lambda: torch.autograd.grad(y, list(m.parameters())))
    # A refused tape stays refused while its graph is held (#251 keeps that),
    # so each refusal case starts from a fresh tape: case 5 must refuse on its
    # own write, not on case 4's.
    del y
    if shim:
        torch._C._eager_reset()

    # 5. BACKWARD5 section 1: a write through the base of a held view.
    x = torch.tensor([0.06, 0.14, 0.22], requires_grad=True)
    a = x * 1
    v = a.view(3)
    a.mul_(10)
    out["view_write"] = caught(lambda: torch.autograd.grad((v * v).sum(), [x]))
    del a, v
    if shim:
        torch._C._eager_reset()
    # 6. A dead autograd.Function whose ctx saved an intermediate: the ctx is
    #    the tape's own reference, so one pruning pass is not enough.
    class Scale(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h):
            ctx.save_for_backward(h)
            return h * 2.0
        @staticmethod
        def backward(ctx, g):
            (h,) = ctx.saved_tensors
            return g * 2.0 + h * 0.0
    m = build(7)
    float(Scale.apply(m(X)).sum())
    with torch.no_grad():
        m[0].weight.add_(0.25)
    out["dead_function"] = caught(lambda: out.__setitem__("function_grads", grads(build(8))))

    # 7. A dead recording whose weight is rebound through `.data`, which no
    #    in-place dispatch sees: the backward must not consult it.
    m = build(9)
    float(m(X).sum())
    m[0].weight.data = torch.tensor(_det(16, 77)).reshape(4, 4)
    out["data_rebind"] = caught(lambda: out.__setitem__("rebind_grads", grads(build(10))))

    # 8. The node bound counts live nodes: a grad-mode loop that drops every
    #    output never reaches it (shim only; upstream has no bound).
    if shim:
        torch._C._eager_reset()
        default = torch._C._eager_max_nodes()
        torch._C._eager_set_max_nodes(64)
        try:
            m = build(11)
            for _ in range(40):
                float(m(X).sum())
            out["bound_reason"] = torch._C._eager_reason()
            out["bound_size"] = torch._C._eager_tape_size()
        finally:
            torch._C._eager_set_max_nodes(default)
    if shim:
        torch._C._eager_reset()
    return out
'''

_SHIM_SCRIPT = (
    "import json, sys\nimport torch\n"
    "assert hasattr(torch._C, '_aten_implemented'), 'not the shim'\n"
    + _PROGRAM_SRC
    + "json.dump(program(True), sys.stdout)\n"
)


_CACHE = {}


def _shim():
    if "shim" not in _CACHE:
        env = dict(os.environ)
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
        env["PYTHONPATH"] = _VENDOR_DIR
        proc = subprocess.run([sys.executable, "-c", _SHIM_SCRIPT], capture_output=True,
                              text=True, env=env, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError("eagerlife subprocess exited %d\n%s\n%s"
                               % (proc.returncode, proc.stdout, proc.stderr))
        _CACHE["shim"] = json.loads(proc.stdout)
    return _CACHE["shim"]


def _oracle(dtype="float32"):
    if dtype not in _CACHE:
        ns = {"torch": _upstream_torch}
        exec(_PROGRAM_SRC, ns)
        previous = _upstream_torch.get_default_dtype()
        _upstream_torch.set_default_dtype(getattr(_upstream_torch, dtype))
        try:
            _CACHE[dtype] = ns["program"](False)
        finally:
            _upstream_torch.set_default_dtype(previous)
    return _CACHE[dtype]


def _close(shim, key):
    """Derived tolerance (AGENTS.md section 16): upstream's own float32 error
    against float64, times `ORACLE_FACTOR`, floored at 8 ulp of the scale."""
    flat = lambda rows: [v for row in rows for v in row]
    s, a, b = flat(shim[key]), flat(_oracle()[key]), flat(_oracle("float64")[key])
    assert len(s) == len(a) == len(b), (key, len(s), len(a), len(b))
    scale = max(abs(v) for v in b)
    rel = max(abs(x - y) for x, y in zip(s, a)) / scale
    oracle_rel = max(abs(x - y) for x, y in zip(a, b)) / scale
    tol = ORACLE_FACTOR * max(oracle_rel, 8 * F32_EPS)
    assert rel <= tol, "%s: rel %.3g > tol %.3g" % (key, rel, tol)


def _guard():
    if not _available():
        _skip.skip("no vendored shim or no upstream torch")
        return False
    return True


def test_an_abandoned_forward_does_not_poison_a_backward_on_another_model():
    if not _guard():
        return
    s, o = _shim(), _oracle()
    assert o["other_model"] == "ACCEPTED", o["other_model"]
    assert s["other_model"] == "ACCEPTED", s["other_model"]
    _close(s, "other_grads")
    print("ok   eagerlife: an abandoned forward plus a write leaves another model's backward alone")


def test_an_abandoned_forward_does_not_poison_the_next_step_on_the_same_model():
    if not _guard():
        return
    s, o = _shim(), _oracle()
    assert s["same_model"] == "ACCEPTED" == o["same_model"], (s["same_model"], o["same_model"])
    _close(s, "same_grads")
    print("ok   eagerlife: an evaluation forward outside no_grad, a step, and the next backward agree")


def test_a_dead_recording_is_released_and_a_live_one_beside_it_still_answers():
    if not _guard():
        return
    s, o = _shim(), _oracle()
    assert s["live_beside_dead"] == "ACCEPTED" == o["live_beside_dead"], s["live_beside_dead"]
    _close(s, "live_grads")
    # Released, not forgiven: the dead model's forward is gone from the tape.
    assert isinstance(s["pruned"], int) and s["pruned"] > 0, s["pruned"]
    assert s["size_after_backward"] == 0, s["size_after_backward"]
    print(f"ok   eagerlife: a dead recording is released ({s['pruned']} nodes) and the live one answers")


def test_a_write_into_a_live_graph_is_still_refused_by_name():
    """The guard #251 must not weaken.

    A weight a held graph saved, written before its backward: upstream refuses
    it with its version counter, and so does this stack, by storage. The
    BACKWARD5 section 1 view case is the documented trade
    (docs/training/AUTOGRAD.md section 6): upstream differentiates it, and
    this stack refuses rather than answer `[0.6, 1.4, 2.2]` for upstream's
    `[60, 140, 220]`. Pruning must keep both refusals.
    """
    if not _guard():
        return
    s, o = _shim(), _oracle()
    assert o["live_write"].startswith("RuntimeError:"), o["live_write"]
    assert o["view_write"] == "ACCEPTED", o["view_write"]
    for case in ("live_write", "view_write"):
        assert "in-place operation wrote into a tensor the eager graph holds" in s[case], (case, s[case])
    print("ok   eagerlife: a write into a live graph, view case included, is still refused")


def test_a_dead_function_whose_ctx_saved_an_intermediate_is_released_too():
    if not _guard():
        return
    s, o = _shim(), _oracle()
    assert s["dead_function"] == "ACCEPTED" == o["dead_function"], (s["dead_function"], o["dead_function"])
    _close(s, "function_grads")
    print("ok   eagerlife: a dead autograd.Function's saved tensors do not keep its graph alive")


def test_a_backward_does_not_consult_a_dead_recording_rebound_through_data():
    if not _guard():
        return
    s, o = _shim(), _oracle()
    assert s["data_rebind"] == "ACCEPTED" == o["data_rebind"], (s["data_rebind"], o["data_rebind"])
    _close(s, "rebind_grads")
    print("ok   eagerlife: a `.data` rebind under a dead recording leaves the next backward alone")


def test_the_node_bound_counts_only_live_nodes():
    """40 forwards of a 3-layer model record well past 64 nodes; each one's
    outputs are dropped, so none of them is live when the bound is reached."""
    if not _guard():
        return
    s = _shim()
    assert s["bound_reason"] is None, s["bound_reason"]
    assert s["bound_size"] < 64, s["bound_size"]
    print(f"ok   eagerlife: a loop that drops its outputs stays under the bound ({s['bound_size']} nodes held)")


if __name__ == "__main__":
    failures = _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")],
        suite="test_eagerlife",
    )
    raise SystemExit(1 if failures else 0)
