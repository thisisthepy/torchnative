"""Issue #333: functionalization end to end, through `FunctionalTensorMode`.

Direction: leader decision 2026-11-15, agent proposal; not owner-confirmed.
`torch.export.export(Linear+ReLU).run_decompositions()` is what
`executorch.exir.to_edge` always runs (SPEC S5.7), and it re-traces through
AOT autograd: `FunctionalTensorMode` (an infra mode in the FUNCTIONAL slot)
receives `FunctionalTensor` arguments, unwraps them to the C wrappers
(`.elem`), and calls `func._op_dk(DispatchKey.Functionalize, ...)` inside a
`_ForceDispatchKeyGuard` that includes Functionalize. That call lands on the
Functionalize kernel this round wrote (`functional.rs::functionalize`), which
dispatches the inner op on the unwrapped values back through the door, where
the remaining modes see it: Proxy, then Fake.

`_PROBE` runs on both sides and records, with the three modes'
`__torch_dispatch__` patched to log: (1) `make_fx(dispatch_functionalize(fn),
tracing_mode="fake")` of add, view, a factory op and mul, the smallest program
with one of each kind, and (2) `run_decompositions()` of Linear+ReLU, its node
targets, graph signature, agreement of `ep2.module()` with eager, and the ops
`FunctionalTensorMode` received. On the shim it also records how many ops the
kernel functionalized during each (`_shim_functionalize_counters`, AGENTS.md
13.1/16: the node list alone cannot show that the kernel, not some other path,
produced it).

**Expected red in CI, and why.** Upstream's dispatch order, measured
(`UPSTREAM`): for each op `Functional` receives `FunctionalTensor` arguments
with Functionalize not included, then `Proxy` and `Fake` receive the unwrapped
fake tensors with Functionalize included (the inner op of the kernel); a
factory op reaches `Proxy` and `Fake` with no tensor arguments. On the shim,
`FunctionalTensor.__new__` (functional_tensor.py:198) calls
`torch.Tensor._make_wrapper_subclass`, which is a surface stub in this shim
(it is listed in `surface.json` and nothing implements it), so both programs
are expected to stop there, before the first `_op_dk`, with the wall named in
the failure message. That wall is outside step 2 (it is the binding every
wrapper subclass needs, DTensor included). Behind it the next walls seen by
reading, not by running: CompositeImplicitAutograd decomposition (`aten.t` is
CIA upstream and Proxy decomposes it to `permute`; the shim's
`_dispatch_has_kernel_for_dispatch_key(..., CompositeImplicitAutograd)` is
False for aten ops, so `OpOverload.decompose` returns NotImplemented and the
shim's graph would hold `aten.t`), and the shim's own export recording
`aten.t, aten.addmm` where upstream records `aten.linear`
(docs/design/FUNCTIONALIZATION.md section 6). The test names whichever wall
it meets; it is never marked passing on the shim without upstream's node list.

These tests were written before the implementation and have not run on a
built shim (leader rule 2026-11-15). The upstream half was run.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR = ROOT / "torchnative" / "python"

sys.path.insert(0, str(ROOT / "tests" / "_support"))
import _skip  # noqa: E402


def _run(script, shim, timeout=1800):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = str(VENDOR)
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", script, "shim" if shim else "upstream"],
                          capture_output=True, text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"probe ({'shim' if shim else 'upstream'}) exited {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _have_shim():
    return (VENDOR / "torch" / "__init__.py").is_file()


def _have_upstream():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "TORCH_USE_RTLD_GLOBAL")}
    proc = subprocess.run([sys.executable, "-c", "import torch, sys; "
                           "sys.exit(1 if hasattr(torch._C, '_aten_implemented') else 0)"],
                          capture_output=True, env=env, timeout=600)
    return proc.returncode == 0


_PROBE = r'''
import json, sys, traceback
import torch

IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == (sys.argv[1] == "shim"), ("shim" if IS_SHIM else "upstream", sys.argv[1])  # AGENTS.md 15.2

from torch._subclasses.functional_tensor import FunctionalTensorMode, FunctionalTensor
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

FK = torch._C.DispatchKey.Functionalize
LOG = []


def kind(a):
    if isinstance(a, FunctionalTensor):
        return "FT"
    if isinstance(a, torch.Tensor):
        return ("W" if torch._is_functional_tensor(a) else "") + ("fake" if isinstance(a, FakeTensor) else "plain")
    if isinstance(a, (list, tuple)):
        ks = [k for k in (kind(x) for x in a) if k]
        return ks or None
    return None


def patch(cls, label):
    orig = cls.__torch_dispatch__

    def logged(self, func, types, args=(), kwargs=None):
        kinds = [k for k in (kind(a) for a in list(args) + list((kwargs or {}).values())) if k]
        LOG.append([label, str(func), kinds, torch._C._dispatch_tls_is_dispatch_key_included(FK)])
        return orig(self, func, types, args, kwargs)

    cls.__torch_dispatch__ = logged


patch(FunctionalTensorMode, "Functional")
patch(FakeTensorMode, "Fake")
patch(ProxyTorchDispatchMode, "Proxy")


def wall(exc):
    """The first frame inside torch that raised, and what it said."""
    frames = traceback.extract_tb(exc.__traceback__)
    inside = [f for f in frames if "/torch/" in f.filename.replace("\\", "/")]
    where = inside[-1] if inside else (frames[-1] if frames else None)
    site = None if where is None else where.filename.replace("\\", "/").split("/torch/")[-1] + f":{where.lineno}"
    return {"type": type(exc).__name__, "message": str(exc)[:500], "site": site}


def counters():
    return torch._C._shim_functionalize_counters() if IS_SHIM else None


out = {}

# 1. dispatch_functionalize under make_fx(fake): Functional above Proxy above Fake.
from torch._subclasses.functional_tensor import dispatch_functionalize  # noqa: E402
from torch.fx.experimental.proxy_tensor import make_fx  # noqa: E402


def fn(a):
    b = torch.ops.aten.add.Tensor(a, 1)
    c = torch.ops.aten.view.default(b, [-1])
    d = torch.ops.aten.ones.default([6])
    return torch.ops.aten.mul.Tensor(c, d)


T1 = torch.randn(2, 3)
LOG.clear()
c0 = counters()
try:
    g = make_fx(dispatch_functionalize(fn), tracing_mode="fake")(T1)
    out["make_fx"] = {"nodes": [str(n.target) for n in g.graph.nodes if n.op == "call_function"]}
except Exception as exc:  # noqa: BLE001
    out["make_fx"] = {"wall": wall(exc)}
out["make_fx"]["log"] = [e for e in LOG if not e[1].startswith("prim.")]
if IS_SHIM:
    c1 = counters()
    out["make_fx"]["counters"] = {k: c1[k] - c0[k] for k in c1 if c1[k] != c0[k]}

# 2. torch.export(Linear+ReLU).run_decompositions(): AOT's two passes.
m = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU())
x = torch.randn(2, 8)
try:
    ep = torch.export.export(m, (x,))
    out["export_targets"] = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
except Exception as exc:  # noqa: BLE001
    out["export_wall"] = wall(exc)
    ep = None
if ep is not None:
    LOG.clear()
    c0 = counters()
    try:
        ep2 = ep.run_decompositions()
        rd = {"targets": [str(n.target) for n in ep2.graph.nodes if n.op == "call_function"],
              "signature_inputs": [str(s.kind) for s in ep2.graph_signature.input_specs],
              "signature_outputs": [str(s.kind) for s in ep2.graph_signature.output_specs]}
        with torch.no_grad():
            rd["agrees_with_eager"] = bool(torch.allclose(ep2.module()(x), m(x), rtol=1e-5, atol=1e-6))
    except Exception as exc:  # noqa: BLE001
        rd = {"wall": wall(exc)}
    rd["functional_ops"] = [e[1] for e in LOG if e[0] == "Functional" and not e[1].startswith("prim.")]
    if IS_SHIM:
        c1 = counters()
        rd["counters"] = {k: c1[k] - c0[k] for k in c1 if c1[k] != c0[k]}
    out["run_decompositions"] = rd
print(json.dumps({"is_shim": IS_SHIM, "out": out}))
'''

#: Upstream's answers, torch 2.13.0, measured with `_PROBE` on 2026-11-15.
UPSTREAM = {'make_fx': {'nodes': ['aten.add.Tensor', 'aten.view.default', 'aten.ones.default', 'aten.mul.Tensor'],
             'log': [['Functional', 'aten.add.Tensor', ['FT'], False],
                     ['Proxy', 'aten.add.Tensor', ['fake'], True],
                     ['Fake', 'aten.add.Tensor', ['fake'], True],
                     ['Functional', 'aten.view.default', ['FT'], False],
                     ['Proxy', 'aten.view.default', ['fake'], True],
                     ['Fake', 'aten.view.default', ['fake'], True],
                     ['Fake', 'aten.as_strided.default', ['fake'], True],
                     ['Functional', 'aten.ones.default', [], False],
                     ['Proxy', 'aten.ones.default', [], True],
                     ['Fake', 'aten.ones.default', [], True],
                     ['Functional', 'aten.mul.Tensor', ['FT', 'FT'], False],
                     ['Proxy', 'aten.mul.Tensor', ['fake', 'fake'], True],
                     ['Fake', 'aten.mul.Tensor', ['fake', 'fake'], True]]},
 'export_targets': ['aten.linear.default', 'aten.relu.default'],
 'run_decompositions': {'targets': ['aten.permute.default', 'aten.addmm.default', 'aten.relu.default'],
                        'signature_inputs': ['InputKind.PARAMETER',
                                             'InputKind.PARAMETER',
                                             'InputKind.USER_INPUT'],
                        'signature_outputs': ['OutputKind.USER_OUTPUT'],
                        'agrees_with_eager': True,
                        'functional_ops': ['aten.linear.default',
                                           'aten.t.default',
                                           'aten.addmm.default',
                                           'aten.relu.default',
                                           'aten.linear.default',
                                           'aten.t.default',
                                           'aten.addmm.default',
                                           'aten.relu.default']}}

_cache = {}


def _side(shim):
    key = "shim" if shim else "up"
    if key not in _cache:
        _cache[key] = _run(_PROBE, shim)["out"]
    return _cache[key]


def _named_wall(what, row):
    w = row["wall"]
    return (f"{what} on the shim stops at a wall: {w['type']} at torch/{w['site']}: "
            f"{w['message']}")


def test_upstream_is_the_recorded_table():
    """Re-derive `UPSTREAM` on every run (the order table and the node list)."""
    if not _have_upstream():
        _skip.skip("upstream torch not importable without the vendored tree")
        return
    got = _side(False)
    assert got == UPSTREAM, {k: (UPSTREAM.get(k), got.get(k)) for k in set(UPSTREAM) | set(got)
                             if UPSTREAM.get(k) != got.get(k)}
    assert UPSTREAM["run_decompositions"]["targets"] == [
        "aten.permute.default", "aten.addmm.default", "aten.relu.default"]


def test_functional_tensor_mode_dispatches_in_upstreams_order():
    """`make_fx(dispatch_functionalize(fn), tracing_mode="fake")`: the shim's
    node list and the log of which mode saw which op with which argument kinds
    equal upstream's, and the kernel functionalized every op (4: add, view,
    the factory op, mul). Expected red at `_make_wrapper_subclass` until that
    binding exists (see the module docstring)."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _side(True)["make_fx"]
    assert "wall" not in got, _named_wall("make_fx(dispatch_functionalize(fn))", got)
    up = UPSTREAM["make_fx"]
    assert got["nodes"] == up["nodes"], (up["nodes"], got["nodes"])
    assert got["log"] == up["log"], (up["log"], got["log"])
    assert got["counters"].get("functionalized") == 4, got["counters"]
    assert got["counters"].get("view") == 1 and got["counters"].get("factory") == 1, got["counters"]


def test_run_decompositions_of_linear_relu_matches_upstream():
    """`torch.export.export(Linear+ReLU).run_decompositions()` on the shim
    gives upstream's node targets (`permute, addmm, relu`), graph signature
    and eager agreement, and the kernel ran for it (counters moved: at least
    one view, the permute, and the pure ops). Expected red: the first wall is
    named in the failure message (see the module docstring for the walls
    expected in order)."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _side(True)
    assert "export_wall" not in got, _named_wall("torch.export.export(Linear+ReLU)",
                                                 {"wall": got.get("export_wall")})
    rd = got["run_decompositions"]
    assert "wall" not in rd, _named_wall("ep.run_decompositions()", rd)
    up = UPSTREAM["run_decompositions"]
    for key in ("targets", "signature_inputs", "signature_outputs", "agrees_with_eager"):
        assert rd[key] == up[key], (key, up[key], rd[key])
    assert rd["counters"].get("functionalized", 0) >= 3, rd["counters"]
    assert rd["counters"].get("view", 0) >= 1, rd["counters"]


if __name__ == "__main__":
    sys.exit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        suite="test_functionalize_export",
    ) else 0)
