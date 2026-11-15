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

**State before #372, from CI run 43739516306 (batch 13, Linux).** All three
tests here were `ok`: `run_decompositions()` already gave upstream's
`permute, addmm, relu`, but only because the shim's export never recorded
`linear` (its `torch._C._nn.linear` composed `t` + `addmm` before the door),
so nothing on the shim had to decompose `linear`. The graph matched; the path
did not (docs/design/FUNCTIONALIZATION.md, "Next walls" (2) and (3)).

**Issue #372** (leader decision 2026-11-16, agent proposal; not
owner-confirmed): CompositeImplicitAutograd decomposition. `_nn.linear` now
enters the door as `aten.linear.default`; the door runs its body
(`bootstrap.py::_cia_linear`) above Functionalize and the modes, except where
upstream does not (a pre-dispatch mode active, which is export's tracer;
`run_decompositions` preserving the op; below autograd); `_op_dk(CIA)` runs
the body. So the shim's export now records `aten.linear.default` as
upstream's does (`test_export_records_linear_as_upstream_does`),
`run_decompositions()` must decompose it in `FunctionalTensorMode` through
`OpOverload.decompose` as upstream's does, and the CIA counters
(`_shim_cia_counters()`) show which entry ran the body. `aten.t` is not
CompositeImplicitAutograd (upstream: CompositeExplicitAutograd); Proxy turns it
into `permute` through the decomposition table, on both sides.

The #372 tests were written before the implementation and have not run on a
built shim (leader rule: agents compile only). The upstream half was run.
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
# `OpOverload.decompose` is wrapped to log which op it was asked about and
# whether it answered (issue #372: `linear` is decomposed there, by its
# CompositeImplicitAutograd body, and `t`, `addmm`, `relu` are not), and the
# shim's CIA counters show the body ran (AGENTS.md 13.1/16).
DECOMPOSED = []
_orig_decompose = torch._ops.OpOverload.decompose


def _logged_decompose(self, *args, **kwargs):
    r = _orig_decompose(self, *args, **kwargs)
    DECOMPOSED.append([str(self), r is not NotImplemented])
    return r


torch._ops.OpOverload.decompose = _logged_decompose
METADATA = {str(f) for f in FunctionalTensor.metadata_fns}


def cia_counters():
    return torch._C._shim_cia_counters() if IS_SHIM else None


def cia_delta(before, after):
    delta = {}
    for op, row in after.items():
        old = before.get(op, {})
        moved = {k: v - old.get(k, 0) for k, v in row.items() if v != old.get(k, 0)}
        if moved:
            delta[op] = moved
    return delta


m = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU())
x = torch.randn(2, 8)
k0 = cia_counters()
try:
    ep = torch.export.export(m, (x,))
    out["export_targets"] = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
except Exception as exc:  # noqa: BLE001
    out["export_wall"] = wall(exc)
    ep = None
if IS_SHIM:
    out["export_cia"] = cia_delta(k0, cia_counters())
if ep is not None:
    LOG.clear()
    DECOMPOSED.clear()
    c0 = counters()
    k0 = cia_counters()
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
    rd["functional_ops_nometa"] = [op for op in rd["functional_ops"] if op not in METADATA]
    rd["decomposed"] = [e for e in DECOMPOSED if not e[0].startswith("prim.") and e[0] not in METADATA]
    if IS_SHIM:
        c1 = counters()
        rd["counters"] = {k: c1[k] - c0[k] for k in c1 if c1[k] != c0[k]}
        rd["cia"] = cia_delta(k0, cia_counters())
    out["run_decompositions"] = rd

# 3. make_fx of F.linear with no export: the CIA kernel runs above the Proxy
# mode (upstream: the graph is `t, addmm`, and `linear` never reaches it).
w3, b3 = torch.randn(4, 8), torch.randn(4)
k0 = cia_counters()
try:
    g3 = make_fx(lambda a: torch.nn.functional.linear(a, w3, b3))(x)
    out["make_fx_linear"] = {"nodes": [str(n.target) for n in g3.graph.nodes if n.op == "call_function"]}
except Exception as exc:  # noqa: BLE001
    out["make_fx_linear"] = {"wall": wall(exc)}
if IS_SHIM:
    out["make_fx_linear"]["cia"] = cia_delta(k0, cia_counters())

# 4. The same under `make_fx(..., pre_dispatch=True)`, the tracer
# `torch.export.export` uses: its ProxyTorchDispatchMode sits on the
# pre-dispatch stack, and upstream's `PreDispatch` key is above the autograd
# keys where the CIA kernel runs, so the mode records `linear` itself and the
# decomposition happens when it re-dispatches below itself (#372).
k0 = cia_counters()
try:
    g4 = make_fx(lambda a: torch.nn.functional.linear(a, w3, b3), pre_dispatch=True)(x)
    out["make_fx_linear_pre_dispatch"] = {
        "nodes": [str(n.target) for n in g4.graph.nodes if n.op == "call_function"]}
except Exception as exc:  # noqa: BLE001
    out["make_fx_linear_pre_dispatch"] = {"wall": wall(exc)}
if IS_SHIM:
    out["make_fx_linear_pre_dispatch"]["cia"] = cia_delta(k0, cia_counters())
torch._ops.OpOverload.decompose = _orig_decompose
print(json.dumps({"is_shim": IS_SHIM, "out": out}))
'''

#: Upstream's answers, torch 2.13.0, measured with `_PROBE` on 2026-11-15;
#: `make_fx_linear`, `decomposed` and `functional_ops_nometa` added and
#: measured on 2026-11-16 (#372); `make_fx_linear_pre_dispatch` added and
#: measured on 2026-11-16 after the first CI run of #373. `decomposed` keeps upstream's FakeTensorMode
#: internals (`aten.to.dtype`, decomposed inside `addmm`'s Python
#: decomposition); the tests compare only the ops `FunctionalTensorMode`
#: received.
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
 'make_fx_linear': {'nodes': ['aten.t.default', 'aten.addmm.default']},
 'make_fx_linear_pre_dispatch': {'nodes': ['aten.linear.default']},
 'run_decompositions': {'targets': ['aten.permute.default', 'aten.addmm.default', 'aten.relu.default'],
                        'signature_inputs': ['InputKind.PARAMETER',
                                             'InputKind.PARAMETER',
                                             'InputKind.USER_INPUT'],
                        'signature_outputs': ['OutputKind.USER_OUTPUT'],
                        'agrees_with_eager': True,
                        'decomposed': [['aten.t.default', False],
                                       ['aten.to.dtype', True],
                                       ['aten.to.dtype', True],
                                       ['aten.to.dtype', True],
                                       ['aten.to.dtype', True],
                                       ['aten.linear.default', True],
                                       ['aten.relu.default', False],
                                       ['aten.t.default', False],
                                       ['aten.permute.default', False],
                                       ['aten.permute.default', False],
                                       ['aten.linear.default', True]],
                        'functional_ops': ['aten.linear.default',
                                           'aten.t.default',
                                           'aten.addmm.default',
                                           'aten.relu.default',
                                           'aten.linear.default',
                                           'aten.t.default',
                                           'aten.addmm.default',
                                           'aten.relu.default'],
                        'functional_ops_nometa': ['aten.linear.default',
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


def test_export_records_linear_as_upstream_does():
    """Issue #372. `torch.export.export(Linear+ReLU)` on the shim records
    upstream's targets, `aten.linear.default, aten.relu.default`. Before #372
    the shim's `torch._C._nn.linear` composed `t` + `addmm` before the door, so
    no mode ever saw `linear` and the shim's graph held `t, addmm`.

    How upstream keeps `linear` (measured on 2.13.0, 2026-11-16, after the
    first CI run of #373 showed the shim recording `t, addmm, relu`): export
    traces with `make_fx(..., pre_dispatch=True)`, so its
    ProxyTorchDispatchMode is on the pre-dispatch stack, and the `PreDispatch`
    key sits above the autograd keys where the CIA kernel runs. That mode
    receives `aten.linear.default` and records it; when it re-dispatches, below
    itself, the CIA kernel decomposes and the FakeTensorMode under it receives
    `t, addmm`. None of export's CIA-preserving machinery
    (`_override_composite_implicit_decomp`, `_collect_all_valid_cia_ops`,
    `_get_decomp_for_cia`) is called during the trace; that is
    `run_decompositions`'. On the shim the door's CIA phase declines while a
    pre-dispatch mode is active, the mode records `linear`, and its
    re-dispatch runs the body at the door (the `door` counter; `op_dk` is not
    the path here, since nothing calls `decompose` during the trace)."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _side(True)
    assert "export_wall" not in got, _named_wall("torch.export.export(Linear+ReLU)",
                                                 {"wall": got.get("export_wall")})
    assert got["export_targets"] == UPSTREAM["export_targets"], (
        UPSTREAM["export_targets"], got["export_targets"])
    row = got["export_cia"].get("aten.linear.default", {})
    assert row.get("door", 0) >= 1, ("the re-dispatch of the recorded linear node did not "
                                      "run the CIA body at the door", got["export_cia"])


def test_make_fx_pre_dispatch_records_linear_and_decomposes_below_the_mode():
    """Issue #372, the mechanism under `test_export_records_linear_as_upstream_does`
    without the rest of export: `make_fx(F.linear, pre_dispatch=True)` records
    `aten.linear.default` upstream (plain `make_fx` records `t, addmm`), because
    a pre-dispatch mode sees the op above the autograd keys. On the shim the
    door's CIA phase declines while that mode is active, and the body runs
    exactly once, at the door, when the mode re-dispatches below itself.
    Removing the pre-dispatch check from `_shim_cia_dispatch` makes the graph
    `t, addmm` again (the first CI run of #373)."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _side(True)["make_fx_linear_pre_dispatch"]
    assert "wall" not in got, _named_wall("make_fx(F.linear, pre_dispatch=True)", got)
    assert got["nodes"] == UPSTREAM["make_fx_linear_pre_dispatch"]["nodes"], (
        UPSTREAM["make_fx_linear_pre_dispatch"]["nodes"], got["nodes"])
    assert got["cia"].get("aten.linear.default", {}) == {"door": 1}, got["cia"]


def test_make_fx_of_linear_decomposes_above_the_proxy():
    """Issue #372. Outside export, `aten::linear`'s CIA kernel runs above the
    modes: `make_fx(F.linear)` records `t, addmm` upstream, and on the shim the
    door's CIA phase ran the body once (`door` counter), which is what put
    those two nodes there."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _side(True)["make_fx_linear"]
    assert "wall" not in got, _named_wall("make_fx(F.linear)", got)
    assert got["nodes"] == UPSTREAM["make_fx_linear"]["nodes"], (
        UPSTREAM["make_fx_linear"]["nodes"], got["nodes"])
    assert got["cia"].get("aten.linear.default", {}).get("door") == 1, got["cia"]


def _decomposed_received(rd):
    """The ops `FunctionalTensorMode` received that `OpOverload.decompose`
    answered (decomposed), in order: the decompositions that shape the graph.
    Declined calls are left out because which mode asks is not the same on
    both sides: `autograd_would_have_decomposed` reads
    `_dispatch_has_kernel_for_dispatch_key(op, "CPU")`, True upstream and
    False on the shim for every aten op, so the shim's modes also ask about
    `relu` and get NotImplemented, which changes nothing in the graph."""
    received = set(rd["functional_ops_nometa"])
    return [e[0] for e in rd["decomposed"] if e[0] in received and e[1]]


def test_run_decompositions_of_linear_relu_matches_upstream():
    """`torch.export.export(Linear+ReLU).run_decompositions()` on the shim
    gives upstream's node targets (`permute, addmm, relu`), graph signature
    and eager agreement; `FunctionalTensorMode` receives upstream's ops in
    upstream's order (metadata queries left out on both sides) and
    `OpOverload.decompose` answers for them as upstream's does (`linear`
    decomposed, `t` and `relu` not); the Functionalize kernel ran for it
    (counters moved: at least one view, the permute, and the pure ops); and
    `linear`'s CIA body ran through `_op_dk` exactly as many times as upstream
    decomposed `linear` (#372). The first wall is named in the failure
    message."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _side(True)
    assert "export_wall" not in got, _named_wall("torch.export.export(Linear+ReLU)",
                                                 {"wall": got.get("export_wall")})
    rd = got["run_decompositions"]
    assert "wall" not in rd, _named_wall("ep.run_decompositions()", rd)
    up = UPSTREAM["run_decompositions"]
    for key in ("targets", "signature_inputs", "signature_outputs", "agrees_with_eager",
                "functional_ops_nometa"):
        assert rd[key] == up[key], (key, up[key], rd[key])
    assert _decomposed_received(rd) == _decomposed_received(up), (
        _decomposed_received(up), _decomposed_received(rd))
    assert rd["counters"].get("functionalized", 0) >= 3, rd["counters"]
    assert rd["counters"].get("view", 0) >= 1, rd["counters"]
    want = _decomposed_received(up).count("aten.linear.default")
    assert rd["cia"].get("aten.linear.default", {}).get("op_dk") == want, (want, rd["cia"])


if __name__ == "__main__":
    sys.exit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        suite="test_functionalize_export",
    ) else 0)
