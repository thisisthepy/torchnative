"""Issue #333, functionalization step 2 of 4: the Functionalize kernel for
non-mutating operators and view operators.

Direction: leader decision 2026-11-15, agent proposal; not owner-confirmed.
Stacked on step 1 (#329, `test_functional_tensor.py`): a functional tensor is
`Repr::Functional`; step 2 is the one generic, schema-driven kernel at the door
(`functional.rs::functionalize`) that unwraps functional arguments (inside
lists too), runs the operator on the values back through `_aten_dispatch`, and
wraps the outputs, sharing the input's storage for a view.

One probe, `_PROBE`, runs in a subprocess on both sides: on the shim
(`PYTHONPATH` at the vendored tree) and on upstream torch 2.13.0
(`sys.executable` without it), and asserts AGENTS.md section 15.2's shim check
for the side it was asked to run on. Each row calls one operator on
`torch._to_functional_tensor` wrappers under a logging `TorchDispatchMode`
(`Log`) and records the output's functional-ness, metadata, `_is_view`,
`_base`, `_is_functional_tensor_base`, `_functionalize_is_multi_output_view`,
the unwrapped value and whether it is an autograd view of an unwrapped input,
the inputs' state before and after, and what `Log` saw. `Log` is a user mode,
and `Functionalize` sits above `Python`, so `Log` sees only what the kernel
dispatches below itself: this is how the dispatch order is measured, op by op.
`UPSTREAM_OPS` is upstream's answer, measured with this probe on 2026-11-15
and re-derived on every run (`test_the_recorded_table_is_upstreams_answer`).

Measured on upstream and recorded because it decides the design:

* the inner op of a view is the view itself with `reapply_views` True (what
  `FunctionalTensorMode` sets) and its `*_copy` variant with it False (the TLS
  default); the wrapper's layout is the inner value's either way;
* through the door (autograd above Functionalize) a view's output is an
  autograd view of its input wrapper (`_base` is the input) and its unwrapped
  value is not a view; through `_op_dk(Functionalize, ...)`, which skips
  autograd, the reverse;
* a view output shares its input's storage (`StorageWeakRef` equal), is not a
  functional base, and `split`/`unbind`/`chunk` outputs are multi-output views;
  `detach` is a functional view but not an autograd one; `_unsafe_view` has no
  alias annotation and is still a functional view; `reshape` of a
  non-contiguous tensor is not a view of its input (a fresh storage);
* a factory op's output is functional when the kernel runs (under `_op_dk` at
  the key, or with Functionalize in the TLS include set); with only plain
  tensor arguments it is not; `lift_fresh` of a plain tensor is functional;
* `_functionalize_unsafe_set(dst, src)` swaps only the storage: value, layout
  and base-ness stay; a functional `dst` takes a plain `src`'s storage and a
  plain `dst` a functional one's, upstream;
* a mutating op on a wrapper runs its functional variant (step 3); the shim
  refuses it by name here.

Which rows the shim must refuse is **derived, not listed**
(`_expected_refusal`): from the schema (an `(a!)` argument is a mutation,
functionalization step 3), from the shim's own kernel table (the inner op the
kernel must dispatch, `<view>_copy` with `reapply_views` False, has no kernel
here), and for `_functionalize_unsafe_set`, from whether both sides are
functional tensors (a plain side swaps a plain TensorImpl's storage, which the
shim cannot represent). Every other row must compute and agree with upstream.

Two fields are compared with a stated exception, both derived from upstream's
own rows rather than chosen: (1) where upstream's inner ops are not the op the
row called (or its `_copy`), upstream decomposed it above Functionalize
(`CompositeImplicitAutograd`) and the shim, which does not decompose aten CIA
ops, dispatches the op itself; the inner ops of those rows are reported, not
compared; (2) the `excluded` flag a mode reads is not compared, because
upstream's Python key shows every mode Functionalize excluded (measured: the
`ones([2])` row involves no functional tensor at all and reads True) and the
shim's mode path does not emulate that; `prim.*` entries are C++ reads of a
FakeTensor's device that the shim answers in Python without dispatching.

What these tests cannot find: anything about `FunctionalTensorMode` (that path
is `tests/export/test_functionalize_export.py`, and on the shim it stops
before the kernel at `_make_wrapper_subclass`); mutation (step 3) and replay
(step 4); a kernel that ran but recorded the wrong view step (the steps are
read back by `test_view_steps_are_recorded_for_replay`, but nothing replays
them until step 4); symbolic shapes (refused by name, unmeasured here).

These tests were written before the implementation and have not run on a
built shim: the round that wrote them could not build (leader rule
2026-11-15). The upstream half was run.
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


def _run(script, shim, timeout=1200):
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
import gc, json, sys
import torch
from torch.utils._python_dispatch import TorchDispatchMode

IS_SHIM = hasattr(torch._C, "_aten_implemented")
WANT_SHIM = sys.argv[1] == "shim"
# AGENTS.md 15.2: a probe meant for the shim must not silently import upstream.
assert IS_SHIM == WANT_SHIM, ("shim" if IS_SHIM else "upstream", "probe asked for", sys.argv[1])

FK = torch._C.DispatchKey.Functionalize
aten = torch.ops.aten
isf = torch._is_functional_tensor
unwrap = torch._from_functional_tensor
IMPLEMENTED = set(torch._C._aten_implemented()) if IS_SHIM else None


def mk(t=None):
    if t is None:
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return torch._to_functional_tensor(t)


def kind(a):
    if isinstance(a, torch.Tensor):
        k = "W" if isf(a) else "T"
        if type(a).__name__ == "FakeTensor":
            k += "fake"
        return k
    if isinstance(a, (list, tuple)):
        ks = [kind(x) for x in a]
        if any(k is not None for k in ks):
            return ks
    return None


class Log(TorchDispatchMode):
    """A user mode. Functionalize sits above Python, so this sees only what the
    Functionalize kernel dispatches below itself, with the TLS it set."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        ks = [kind(a) for a in list(args) + list(kwargs.values())]
        self.seen.append([str(func), [k for k in ks if k is not None],
                          torch._C._dispatch_tls_is_dispatch_key_included(FK),
                          torch._C._dispatch_tls_is_dispatch_key_excluded(FK),
                          torch._C._functionalization_reapply_views_tls()])
        return func(*args, **kwargs)


def meta(o):
    return [list(o.shape), list(o.stride()), o.storage_offset(), str(o.dtype), str(o.device)]


def describe(o, inputs):
    """Everything observable about one output, relative to the input wrappers."""
    if isinstance(o, (list, tuple)):
        return {"seq": type(o).__name__, "items": [describe(x, inputs) for x in o]}
    if not isinstance(o, torch.Tensor):
        return {"value": o if isinstance(o, (bool, int, float, type(None))) else repr(o)}
    d = {"functional": isf(o), "meta": meta(o)}
    d["is_view"] = o._is_view()
    b = o._base
    d["base"] = None if b is None else next((i for i, x in enumerate(inputs) if b is x), "other")
    if isf(o):
        v = unwrap(o)
        d["is_base"] = torch._is_functional_tensor_base(o)
        d["multi_output_view"] = torch._functionalize_is_multi_output_view(o)
        d["data_mutation"] = torch._functionalize_has_data_mutation(o)
        d["value_type"] = type(v).__name__
        if not v.is_meta and type(v).__name__ != "FakeTensor":
            d["value"] = v.tolist()
        d["value_meta"] = meta(v)
        # Whether the unwrapped output is an autograd view of an unwrapped
        # input (a view op re-applied to the value, below ADInplaceOrView).
        vb = v._base
        d["value_base"] = None if vb is None else next(
            (i for i, x in enumerate(inputs)
             if isinstance(x, torch.Tensor) and isf(x) and (vb is unwrap(x) or vb is unwrap(x)._base)), "other")
    return d


def schema_facts(outer):
    """What the schema says about the op a row calls, read on this side."""
    if outer is None:
        return None
    ns, name, overload = outer.split(".")
    func = getattr(getattr(getattr(torch.ops, ns), name), overload)
    s = func._schema
    mutable = any(a.alias_info is not None and a.alias_info.is_write for a in s.arguments)
    view = (not mutable) and any(r.alias_info is not None for r in s.returns)
    copy = f"{ns}.{name}_copy.{overload}"
    packet = getattr(getattr(torch.ops, ns), name + "_copy", None)
    has_copy = packet is not None and overload in packet.overloads()
    facts = {"mutable": mutable, "view": view, "copy_op": copy if has_copy else None}
    if IS_SHIM:
        facts["implemented"] = outer in IMPLEMENTED
        facts["copy_implemented"] = has_copy and copy in IMPLEMENTED
    return facts


out = {}


def case(name, build, outer, reapply=None):
    """`build()` returns (inputs, thunk); the thunk runs under `Log`."""
    old = None
    if reapply is not None:
        old = torch._functionalize_enable_reapply_views(reapply)
    try:
        inputs, thunk = build()
        before = [torch._functionalize_has_data_mutation(x) if isinstance(x, torch.Tensor) and isf(x) else None
                  for x in inputs]
        mode = Log()
        try:
            with mode:
                o = thunk()
        except Exception as exc:  # noqa: BLE001 -- the exception type is the answer
            out[name] = {"raises": type(exc).__name__, "message": str(exc)[:600], "inner": mode.seen,
                         "outer": outer, "schema": schema_facts(outer)}
            return
        rec = describe(o, inputs)
        rec["inner"] = mode.seen
        rec["outer"] = outer
        rec["schema"] = schema_facts(outer)
        rec["inputs_still_functional"] = [isf(x) if isinstance(x, torch.Tensor) else None for x in inputs]
        rec["inputs_data_mutation_before_after"] = [
            [b, torch._functionalize_has_data_mutation(x) if isinstance(x, torch.Tensor) and isf(x) else None]
            for b, x in zip(before, inputs)]
        out[name] = rec
    finally:
        if old is not None:
            torch._functionalize_enable_reapply_views(old)


def one(fn, *makers):
    def build():
        ins = [m() for m in makers]
        return ins, (lambda: fn(*ins))
    return build


W = mk
P = lambda: torch.ones(2, 3)  # noqa: E731
W0 = lambda: mk(torch.tensor(3.5))  # noqa: E731
W1 = lambda: mk(torch.arange(4, dtype=torch.float32))  # noqa: E731
WB = lambda: mk(torch.tensor([[True, False, True], [False, True, False]]))  # noqa: E731
W33 = lambda: mk(torch.arange(9, dtype=torch.float32).reshape(3, 3))  # noqa: E731
W34 = lambda: mk(torch.arange(12, dtype=torch.float32).reshape(3, 4))  # noqa: E731
W43 = lambda: mk(torch.arange(12, dtype=torch.float32).reshape(4, 3))  # noqa: E731
WT = lambda: mk(torch.arange(6, dtype=torch.float32).reshape(3, 2).t())  # noqa: E731
W13 = lambda: mk(torch.ones(1, 3))  # noqa: E731

# -- non-mutating, non-view ops ---------------------------------------------------
for name, build, outer in [
    ("add.Tensor(W, W)", one(lambda a, b: aten.add.Tensor(a, b), W, W), "aten.add.Tensor"),
    ("add.Tensor(W, P)", one(lambda a, b: aten.add.Tensor(a, b), W, P), "aten.add.Tensor"),
    ("add.Tensor(P, W)", one(lambda a, b: aten.add.Tensor(a, b), P, W), "aten.add.Tensor"),
    ("W + 1", one(lambda a: a + 1, W), "aten.add.Tensor"),
    ("mul.Tensor(W, W)", one(lambda a, b: aten.mul.Tensor(a, b), W, W), "aten.mul.Tensor"),
    ("relu(W)", one(lambda a: aten.relu.default(a), W), "aten.relu.default"),
    ("addmm(W4, W, W3x4)", one(lambda b, a, c: aten.addmm.default(b, a, c), lambda: mk(torch.ones(4)), W, W34),
     "aten.addmm.default"),
    ("matmul(W, W3x4)", one(lambda a, b: aten.matmul.default(a, b), W, W34), "aten.matmul.default"),
    ("cat([W, P])", one(lambda a, b: aten.cat.default([a, b]), W, P), "aten.cat.default"),
    ("cat([W, W])", one(lambda a, b: aten.cat.default([a, b]), W, W), "aten.cat.default"),
    ("cat([P, W], 1)", one(lambda a, b: aten.cat.default([a, b], 1), P, W), "aten.cat.default"),
    ("where.ScalarOther(Wbool, W, 0.0)", one(lambda c, a: aten.where.ScalarOther(c, a, 0.0), WB, W),
     "aten.where.ScalarOther"),
    ("where.self(Wbool, W, P)", one(lambda c, a, b: aten.where.self(c, a, b), WB, W, P), "aten.where.self"),
    ("sum.dim_IntList(W, [1])", one(lambda a: aten.sum.dim_IntList(a, [1]), W), "aten.sum.dim_IntList"),
    ("sum.dim_IntList(W, [0], keepdim)", one(lambda a: aten.sum.dim_IntList(a, [0], True), W),
     "aten.sum.dim_IntList"),
    ("_softmax(W, 1)", one(lambda a: aten._softmax.default(a, 1, False), W), "aten._softmax.default"),
    ("softmax.int(W, -1)", one(lambda a: aten.softmax.int(a, -1), W), "aten.softmax.int"),
    ("clone(W)", one(lambda a: aten.clone.default(a), W), "aten.clone.default"),
    ("max.dim(W, 1)", one(lambda a: aten.max.dim(a, 1), W), "aten.max.dim"),
    ("var_mean.correction(W, [1])", one(lambda a: aten.var_mean.correction(a, [1]), W),
     "aten.var_mean.correction"),
]:
    case(name, build, outer)

# -- view ops, with reapply_views False (the TLS default) and True ----------------
for name, build, outer in [
    ("view(W, [-1])", one(lambda a: aten.view.default(a, [-1]), W), "aten.view.default"),
    ("W.view(3, 2)", one(lambda a: a.view(3, 2), W), "aten.view.default"),
    ("reshape(W, [3, 2]) (a view)", one(lambda a: aten.reshape.default(a, [3, 2]), W), "aten.reshape.default"),
    ("reshape(Wt, [6]) (a copy)", one(lambda a: aten.reshape.default(a, [6]), WT), "aten.reshape.default"),
    ("_unsafe_view(W, [6])", one(lambda a: aten._unsafe_view.default(a, [6]), W), "aten._unsafe_view.default"),
    ("permute(W, [1, 0])", one(lambda a: aten.permute.default(a, [1, 0]), W), "aten.permute.default"),
    ("t(W)", one(lambda a: aten.t.default(a), W), "aten.t.default"),
    ("transpose.int(W, 0, 1)", one(lambda a: aten.transpose.int(a, 0, 1), W), "aten.transpose.int"),
    ("unsqueeze(W, 0)", one(lambda a: aten.unsqueeze.default(a, 0), W), "aten.unsqueeze.default"),
    ("squeeze.dim(W13, 0)", one(lambda a: aten.squeeze.dim(a, 0), W13), "aten.squeeze.dim"),
    ("expand(W1, [2, 4])", one(lambda a: aten.expand.default(a, [2, 4]), W1), "aten.expand.default"),
    ("W[1:]", one(lambda a: a[1:], W), "aten.slice.Tensor"),
    ("W[:, 1]", one(lambda a: a[:, 1], W), "aten.select.int"),
    ("slice.Tensor(W, 1, 0, 2)", one(lambda a: aten.slice.Tensor(a, 1, 0, 2), W), "aten.slice.Tensor"),
    ("select.int(W, 0, 1)", one(lambda a: aten.select.int(a, 0, 1), W), "aten.select.int"),
    ("narrow(W, 1, 1, 2)", one(lambda a: aten.narrow.default(a, 1, 1, 2), W), "aten.narrow.default"),
    ("unbind.int(W)", one(lambda a: aten.unbind.int(a), W), "aten.unbind.int"),
    ("split.Tensor(W43, 2)", one(lambda a: aten.split.Tensor(a, 2), W43), "aten.split.Tensor"),
    ("chunk(W43, 2)", one(lambda a: aten.chunk.default(a, 2), W43), "aten.chunk.default"),
    ("detach(W)", one(lambda a: aten.detach.default(a), W), "aten.detach.default"),
    ("alias(W)", one(lambda a: aten.alias.default(a), W), "aten.alias.default"),
    ("diagonal(W33)", one(lambda a: aten.diagonal.default(a), W33), "aten.diagonal.default"),
    ("view_copy(W, [-1])", one(lambda a: aten.view_copy.default(a, [-1]), W), "aten.view_copy.default"),
    ("permute_copy(W, [1, 0])", one(lambda a: aten.permute_copy.default(a, [1, 0]), W),
     "aten.permute_copy.default"),
    ("t_copy(W)", one(lambda a: aten.t_copy.default(a), W), "aten.t_copy.default"),
]:
    case(name + " reapply=False", build, outer, reapply=False)
    case(name + " reapply=True", build, outer, reapply=True)


def view_of_view():
    f = W()
    v = aten.view.default(f, [-1])
    return [f, v], (lambda: aten.slice.Tensor(v, 0, 1, 4))


def op_on_view():
    f = W()
    v = aten.view.default(f, [-1])
    return [f, v], (lambda: aten.add.Tensor(v, 1))


for r in (False, True):
    case(f"slice(view(W)) reapply={r}", view_of_view, "aten.slice.Tensor", reapply=r)
case("add.Tensor(view(W), 1) reapply=True", op_on_view, "aten.add.Tensor", reapply=True)


# -- factory ops and plain arguments; Functionalize in the TLS include set or not --
def forced(fn):
    """`FunctionalTensorMode`'s guard: Functionalize included, not excluded."""
    def run():
        inc = torch._C._dispatch_tls_local_include_set() | torch._C.DispatchKeySet(FK)
        exc = torch._C._dispatch_tls_local_exclude_set().remove(FK)
        with torch._C._ForceDispatchKeyGuard(inc, exc):
            try:
                return fn()
            finally:
                torch._disable_functionalization()
    return run


NONE = lambda: []  # noqa: E731
for name, build, outer, reapply in [
    ("ones([2])", lambda: ([], lambda: aten.ones.default([2])), "aten.ones.default", None),
    ("ones([2]) Functionalize included", lambda: ([], forced(lambda: aten.ones.default([2]))),
     "aten.ones.default", None),
    ("_op_dk(FK, ones, [2])", lambda: ([], lambda: aten.ones.default._op_dk(FK, [2])), "aten.ones.default", None),
    ("_op_dk(FK, ones, [2]) Functionalize included",
     lambda: ([], forced(lambda: aten.ones.default._op_dk(FK, [2]))), "aten.ones.default", None),
    ("_op_dk(FK, arange, 3) Functionalize included",
     lambda: ([], forced(lambda: aten.arange.default._op_dk(FK, 3))), "aten.arange.default", None),
    ("_op_dk(FK, add, P, P)", one(lambda a, b: aten.add.Tensor._op_dk(FK, a, b), P, P), "aten.add.Tensor", None),
    ("_op_dk(FK, add, P, P) Functionalize included",
     one(lambda a, b: forced(lambda: aten.add.Tensor._op_dk(FK, a, b))(), P, P), "aten.add.Tensor", None),
    ("_op_dk(FK, add, W, W)", one(lambda a, b: aten.add.Tensor._op_dk(FK, a, b), W, W), "aten.add.Tensor", None),
    ("_op_dk(FK, add, W, W) Functionalize included",
     one(lambda a, b: forced(lambda: aten.add.Tensor._op_dk(FK, a, b))(), W, W), "aten.add.Tensor", None),
    ("_op_dk(FK, addmm, W4, W, W3x4) Functionalize included",
     one(lambda b, a, c: forced(lambda: aten.addmm.default._op_dk(FK, b, a, c))(),
         lambda: mk(torch.ones(4)), W, W34), "aten.addmm.default", True),
    ("_op_dk(FK, relu, W) Functionalize included",
     one(lambda a: forced(lambda: aten.relu.default._op_dk(FK, a))(), W), "aten.relu.default", True),
    ("_op_dk(FK, permute, W) Functionalize included",
     one(lambda a: forced(lambda: aten.permute.default._op_dk(FK, a, [1, 0]))(), W), "aten.permute.default", True),
    ("_op_dk(FK, permute, W) reapply=False",
     one(lambda a: aten.permute.default._op_dk(FK, a, [1, 0]), W), "aten.permute.default", False),
    ("_op_dk(FK, permute, P)", one(lambda a: aten.permute.default._op_dk(FK, a, [1, 0]), P),
     "aten.permute.default", None),
    ("_op_dk(FK, split, W43, 2) Functionalize included",
     one(lambda a: forced(lambda: aten.split.Tensor._op_dk(FK, a, 2))(), W43), "aten.split.Tensor", True),
    ("_op_dk(FK, cat, [P, P]) Functionalize included",
     one(lambda a, b: forced(lambda: aten.cat.default._op_dk(FK, [a, b]))(), P, P), "aten.cat.default", None),
    ("_op_dk(FK, cat, [W, P]) Functionalize included",
     one(lambda a, b: forced(lambda: aten.cat.default._op_dk(FK, [a, b]))(), W, P), "aten.cat.default", None),
    ("_op_dk(FK, lift_fresh, P) Functionalize included",
     one(lambda a: forced(lambda: aten.lift_fresh.default._op_dk(FK, a))(), P), "aten.lift_fresh.default", True),
]:
    case(name, build, outer, reapply=reapply)

# -- non-tensor outputs ------------------------------------------------------------
for name, build, outer in [
    ("W.size()", one(lambda a: a.size(), W), None),
    ("W.dim()", one(lambda a: a.dim(), W), None),
    ("W.is_contiguous()", one(lambda a: a.is_contiguous(), W), None),
    ("W0.item()", one(lambda a: a.item(), W0), "aten._local_scalar_dense.default"),
    ("_local_scalar_dense(W0)", one(lambda a: aten._local_scalar_dense.default(a), W0),
     "aten._local_scalar_dense.default"),
    ("is_same_size(W, W)", one(lambda a, b: aten.is_same_size.default(a, b), W, W), "aten.is_same_size.default"),
]:
    case(name, build, outer)

# -- metadata ops: answered from the wrapper's own layout, nothing dispatched below --
for name, build, outer in [
    ("sym_size.int(Wt, 1)", one(lambda a: aten.sym_size.int(a, 1), WT), "aten.sym_size.int"),
    ("size.default(Wt)", one(lambda a: aten.size.default(a), WT), "aten.size.default"),
    ("stride.default(Wt)", one(lambda a: aten.stride.default(a), WT), "aten.stride.default"),
    ("sym_stride.int(Wt, 0)", one(lambda a: aten.sym_stride.int(a, 0), WT), "aten.sym_stride.int"),
    ("dim.default(Wt)", one(lambda a: aten.dim.default(a), WT), "aten.dim.default"),
    ("numel.default(Wt)", one(lambda a: aten.numel.default(a), WT), "aten.numel.default"),
    ("sym_numel.default(Wt)", one(lambda a: aten.sym_numel.default(a), WT), "aten.sym_numel.default"),
    ("is_contiguous.default(Wt)", one(lambda a: aten.is_contiguous.default(a), WT), "aten.is_contiguous.default"),
    ("storage_offset.default(W[1:])", one(lambda a: aten.storage_offset.default(a),
                                          lambda: mk(torch.ones(2, 3)[1:])), "aten.storage_offset.default"),
    ("sym_storage_offset.default(W[1:])", one(lambda a: aten.sym_storage_offset.default(a),
                                              lambda: mk(torch.ones(2, 3)[1:])), "aten.sym_storage_offset.default"),
    ("is_non_overlapping_and_dense(Wt)", one(lambda a: aten.is_non_overlapping_and_dense.default(a), WT),
     "aten.is_non_overlapping_and_dense.default"),
]:
    case(name, build, outer)

# -- a fake value inside the wrapper (what AOT's wrappers hold) --------------------
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

with FakeTensorMode():
    FAKE = torch.empty(2, 3)
case("permute(W(fake), [1, 0]) reapply=True", one(lambda a: aten.permute.default(a, [1, 0]), lambda: mk(FAKE)),
     "aten.permute.default", reapply=True)
case("add.Tensor(W(fake), 1)", one(lambda a: aten.add.Tensor(a, 1), lambda: mk(FAKE)), "aten.add.Tensor")

# -- mutation: functionalization step 3; the shim refuses these ---------------------
for name, build, outer in [
    ("add_(W, 1)", one(lambda a: a.add_(1), W), "aten.add_.Tensor"),
    ("add.out(W, W, out=W)", one(lambda a, b, c: aten.add.out(a, b, out=c), W, W, W), "aten.add.out"),
    ("t_(W)", one(lambda a: a.t_(), W), "aten.t_.default"),
    ("copy_(W, P)", one(lambda a, b: a.copy_(b), W, P), "aten.copy_.default"),
]:
    case(name, build, outer)

# -- the storage a wrapper reports, and _functionalize_unsafe_set --------------------
from torch.multiprocessing.reductions import StorageWeakRef  # noqa: E402

st = {}


def r(name, fn):
    try:
        st[name] = fn()
    except Exception as exc:  # noqa: BLE001
        st[name] = ["raises", type(exc).__name__, str(exc)[:600]]


def same(a, b):
    return StorageWeakRef(a.untyped_storage()) == StorageWeakRef(b.untyped_storage())


old = torch._functionalize_enable_reapply_views(True)
f = mk()
v = aten.view.default(f, [-1])
g = mk()
c = aten.add.Tensor(f, 1)
fv = mk(torch.ones(2, 3)[1:])
torch._functionalize_enable_reapply_views(old)
r("storage nbytes (base)", lambda: f.untyped_storage().nbytes())
r("storage nbytes (view)", lambda: v.untyped_storage().nbytes())
r("storage nbytes (fresh output)", lambda: c.untyped_storage().nbytes())
r("storage nbytes (wrapper of a slice)", lambda: fv.untyped_storage().nbytes())
r("storage device", lambda: str(f.untyped_storage().device))
r("storage data_ptr", lambda: f.untyped_storage().data_ptr())
r("storage base == view", lambda: same(f, v))
r("storage base == other wrapper", lambda: same(f, g))
r("storage base == fresh output", lambda: same(f, c))
r("storage base == its value", lambda: same(f, unwrap(f)))
r("storage twice same", lambda: same(f, f))


def unsafe(dst_mk, src_mk):
    dst, src = dst_mk(), src_mk()
    dv = unwrap(dst) if isf(dst) else None
    before_meta = meta(dst)
    res = torch._functionalize_unsafe_set(dst, src)
    rec = {"returns": res, "meta unchanged": meta(dst) == before_meta, "dst still functional": isf(dst),
           "storage now src's": same(dst, src)}
    if isf(dst):
        rec["value is old value"] = unwrap(dst) is dv
        rec["value"] = unwrap(dst).tolist()
        rec["is_base"] = torch._is_functional_tensor_base(dst)
        rec["storage nbytes"] = dst.untyped_storage().nbytes()
    return rec


def unsafe_view_src():
    b = mk()
    return unsafe(lambda: aten.view.default(mk(), [3, 2]), lambda: b)


r("unsafe_set(W, W(zeros))", lambda: unsafe(W, lambda: mk(torch.zeros(2, 3))))
r("unsafe_set(view(W), W)", unsafe_view_src)
r("unsafe_set(W, W(zeros(3)))", lambda: unsafe(W, lambda: mk(torch.zeros(3))))
r("unsafe_set(W, P)", lambda: unsafe(W, P))
r("unsafe_set(P, W)", lambda: unsafe(P, W))
r("unsafe_set(P, P)", lambda: unsafe(P, lambda: torch.zeros(2, 3)))
r("unsafe_set(W, 1)", lambda: torch._functionalize_unsafe_set(mk(), 1))
out["storage"] = st

del f, v, g, c, fv
gc.collect()
print(json.dumps({"is_shim": IS_SHIM, "out": out}))
'''

#: Upstream's answers, torch 2.13.0 (`.caches/spike-venv`), measured with
#: `_PROBE` on 2026-11-15. `"raises"` rows give the exception type.
UPSTREAM_OPS = {'add.Tensor(W, W)': {'functional': True,
                      'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                      'is_view': False,
                      'base': None,
                      'is_base': True,
                      'multi_output_view': False,
                      'data_mutation': False,
                      'value_type': 'Tensor',
                      'value': [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]],
                      'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                      'value_base': None,
                      'inner': [['aten.add.Tensor', ['T', 'T'], False, True, False]],
                      'outer': 'aten.add.Tensor',
                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                      'inputs_still_functional': [True, True],
                      'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 'add.Tensor(W, P)': {'functional': True,
                      'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                      'is_view': False,
                      'base': None,
                      'is_base': True,
                      'multi_output_view': False,
                      'data_mutation': False,
                      'value_type': 'Tensor',
                      'value': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                      'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                      'value_base': None,
                      'inner': [['aten.add.Tensor', ['T', 'T'], False, True, False]],
                      'outer': 'aten.add.Tensor',
                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                      'inputs_still_functional': [True, False],
                      'inputs_data_mutation_before_after': [[False, False], [None, None]]},
 'add.Tensor(P, W)': {'functional': True,
                      'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                      'is_view': False,
                      'base': None,
                      'is_base': True,
                      'multi_output_view': False,
                      'data_mutation': False,
                      'value_type': 'Tensor',
                      'value': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                      'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                      'value_base': None,
                      'inner': [['aten.add.Tensor', ['T', 'T'], False, True, False]],
                      'outer': 'aten.add.Tensor',
                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                      'inputs_still_functional': [False, True],
                      'inputs_data_mutation_before_after': [[None, None], [False, False]]},
 'W + 1': {'functional': True,
           'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
           'is_view': False,
           'base': None,
           'is_base': True,
           'multi_output_view': False,
           'data_mutation': False,
           'value_type': 'Tensor',
           'value': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
           'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
           'value_base': None,
           'inner': [['aten.add.Tensor', ['T'], False, True, False]],
           'outer': 'aten.add.Tensor',
           'schema': {'mutable': False, 'view': False, 'copy_op': None},
           'inputs_still_functional': [True],
           'inputs_data_mutation_before_after': [[False, False]]},
 'mul.Tensor(W, W)': {'functional': True,
                      'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                      'is_view': False,
                      'base': None,
                      'is_base': True,
                      'multi_output_view': False,
                      'data_mutation': False,
                      'value_type': 'Tensor',
                      'value': [[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]],
                      'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                      'value_base': None,
                      'inner': [['aten.mul.Tensor', ['T', 'T'], False, True, False]],
                      'outer': 'aten.mul.Tensor',
                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                      'inputs_still_functional': [True, True],
                      'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 'relu(W)': {'functional': True,
             'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
             'is_view': False,
             'base': None,
             'is_base': True,
             'multi_output_view': False,
             'data_mutation': False,
             'value_type': 'Tensor',
             'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
             'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
             'value_base': None,
             'inner': [['aten.relu.default', ['T'], False, True, False]],
             'outer': 'aten.relu.default',
             'schema': {'mutable': False, 'view': False, 'copy_op': None},
             'inputs_still_functional': [True],
             'inputs_data_mutation_before_after': [[False, False]]},
 'addmm(W4, W, W3x4)': {'functional': True,
                        'meta': [[2, 4], [4, 1], 0, 'torch.float32', 'cpu'],
                        'is_view': False,
                        'base': None,
                        'is_base': True,
                        'multi_output_view': False,
                        'data_mutation': False,
                        'value_type': 'Tensor',
                        'value': [[21.0, 24.0, 27.0, 30.0], [57.0, 69.0, 81.0, 93.0]],
                        'value_meta': [[2, 4], [4, 1], 0, 'torch.float32', 'cpu'],
                        'value_base': None,
                        'inner': [['aten.addmm.default', ['T', 'T', 'T'], False, True, False]],
                        'outer': 'aten.addmm.default',
                        'schema': {'mutable': False, 'view': False, 'copy_op': None},
                        'inputs_still_functional': [True, True, True],
                        'inputs_data_mutation_before_after': [[False, False],
                                                              [False, False],
                                                              [False, False]]},
 'matmul(W, W3x4)': {'functional': True,
                     'meta': [[2, 4], [4, 1], 0, 'torch.float32', 'cpu'],
                     'is_view': False,
                     'base': None,
                     'is_base': True,
                     'multi_output_view': False,
                     'data_mutation': False,
                     'value_type': 'Tensor',
                     'value': [[20.0, 23.0, 26.0, 29.0], [56.0, 68.0, 80.0, 92.0]],
                     'value_meta': [[2, 4], [4, 1], 0, 'torch.float32', 'cpu'],
                     'value_base': None,
                     'inner': [['aten.mm.default', ['T', 'T'], False, True, False]],
                     'outer': 'aten.matmul.default',
                     'schema': {'mutable': False, 'view': False, 'copy_op': None},
                     'inputs_still_functional': [True, True],
                     'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 'cat([W, P])': {'functional': True,
                 'meta': [[4, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                 'is_view': False,
                 'base': None,
                 'is_base': True,
                 'multi_output_view': False,
                 'data_mutation': False,
                 'value_type': 'Tensor',
                 'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                 'value_meta': [[4, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                 'value_base': None,
                 'inner': [['aten.cat.default', [['T', 'T']], False, True, False]],
                 'outer': 'aten.cat.default',
                 'schema': {'mutable': False, 'view': False, 'copy_op': None},
                 'inputs_still_functional': [True, False],
                 'inputs_data_mutation_before_after': [[False, False], [None, None]]},
 'cat([W, W])': {'functional': True,
                 'meta': [[4, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                 'is_view': False,
                 'base': None,
                 'is_base': True,
                 'multi_output_view': False,
                 'data_mutation': False,
                 'value_type': 'Tensor',
                 'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                 'value_meta': [[4, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                 'value_base': None,
                 'inner': [['aten.cat.default', [['T', 'T']], False, True, False]],
                 'outer': 'aten.cat.default',
                 'schema': {'mutable': False, 'view': False, 'copy_op': None},
                 'inputs_still_functional': [True, True],
                 'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 'cat([P, W], 1)': {'functional': True,
                    'meta': [[2, 6], [6, 1], 0, 'torch.float32', 'cpu'],
                    'is_view': False,
                    'base': None,
                    'is_base': True,
                    'multi_output_view': False,
                    'data_mutation': False,
                    'value_type': 'Tensor',
                    'value': [[1.0, 1.0, 1.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 3.0, 4.0, 5.0]],
                    'value_meta': [[2, 6], [6, 1], 0, 'torch.float32', 'cpu'],
                    'value_base': None,
                    'inner': [['aten.cat.default', [['T', 'T']], False, True, False]],
                    'outer': 'aten.cat.default',
                    'schema': {'mutable': False, 'view': False, 'copy_op': None},
                    'inputs_still_functional': [False, True],
                    'inputs_data_mutation_before_after': [[None, None], [False, False]]},
 'where.ScalarOther(Wbool, W, 0.0)': {'functional': True,
                                      'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                      'is_view': False,
                                      'base': None,
                                      'is_base': True,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [[0.0, 0.0, 2.0], [0.0, 4.0, 0.0]],
                                      'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.scalar_tensor.default', [], False, True, False],
                                                ['aten.where.self', ['T', 'T', 'T'], False, True, False]],
                                      'outer': 'aten.where.ScalarOther',
                                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                      'inputs_still_functional': [True, True],
                                      'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 'where.self(Wbool, W, P)': {'functional': True,
                             'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                             'is_view': False,
                             'base': None,
                             'is_base': True,
                             'multi_output_view': False,
                             'data_mutation': False,
                             'value_type': 'Tensor',
                             'value': [[0.0, 1.0, 2.0], [1.0, 4.0, 1.0]],
                             'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                             'value_base': None,
                             'inner': [['aten.where.self', ['T', 'T', 'T'], False, True, False]],
                             'outer': 'aten.where.self',
                             'schema': {'mutable': False, 'view': False, 'copy_op': None},
                             'inputs_still_functional': [True, True, False],
                             'inputs_data_mutation_before_after': [[False, False],
                                                                   [False, False],
                                                                   [None, None]]},
 'sum.dim_IntList(W, [1])': {'functional': True,
                             'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                             'is_view': False,
                             'base': None,
                             'is_base': True,
                             'multi_output_view': False,
                             'data_mutation': False,
                             'value_type': 'Tensor',
                             'value': [3.0, 12.0],
                             'value_meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                             'value_base': None,
                             'inner': [['aten.sum.dim_IntList', ['T'], False, True, False]],
                             'outer': 'aten.sum.dim_IntList',
                             'schema': {'mutable': False, 'view': False, 'copy_op': None},
                             'inputs_still_functional': [True],
                             'inputs_data_mutation_before_after': [[False, False]]},
 'sum.dim_IntList(W, [0], keepdim)': {'functional': True,
                                      'meta': [[1, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                      'is_view': False,
                                      'base': None,
                                      'is_base': True,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [[3.0, 5.0, 7.0]],
                                      'value_meta': [[1, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.sum.dim_IntList', ['T'], False, True, False]],
                                      'outer': 'aten.sum.dim_IntList',
                                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                      'inputs_still_functional': [True],
                                      'inputs_data_mutation_before_after': [[False, False]]},
 '_softmax(W, 1)': {'functional': True,
                    'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                    'is_view': False,
                    'base': None,
                    'is_base': True,
                    'multi_output_view': False,
                    'data_mutation': False,
                    'value_type': 'Tensor',
                    'value': [[0.09003057330846786, 0.2447284758090973, 0.6652409434318542],
                              [0.09003057330846786, 0.2447284758090973, 0.6652409434318542]],
                    'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                    'value_base': None,
                    'inner': [['aten._softmax.default', ['T'], False, True, False]],
                    'outer': 'aten._softmax.default',
                    'schema': {'mutable': False, 'view': False, 'copy_op': None},
                    'inputs_still_functional': [True],
                    'inputs_data_mutation_before_after': [[False, False]]},
 'softmax.int(W, -1)': {'functional': True,
                        'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                        'is_view': False,
                        'base': None,
                        'is_base': True,
                        'multi_output_view': False,
                        'data_mutation': False,
                        'value_type': 'Tensor',
                        'value': [[0.09003057330846786, 0.2447284758090973, 0.6652409434318542],
                                  [0.09003057330846786, 0.2447284758090973, 0.6652409434318542]],
                        'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                        'value_base': None,
                        'inner': [['aten._softmax.default', ['T'], False, True, False]],
                        'outer': 'aten.softmax.int',
                        'schema': {'mutable': False, 'view': False, 'copy_op': None},
                        'inputs_still_functional': [True],
                        'inputs_data_mutation_before_after': [[False, False]]},
 'clone(W)': {'functional': True,
              'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
              'is_view': False,
              'base': None,
              'is_base': True,
              'multi_output_view': False,
              'data_mutation': False,
              'value_type': 'Tensor',
              'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
              'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
              'value_base': None,
              'inner': [['aten.clone.default', ['T'], False, True, False]],
              'outer': 'aten.clone.default',
              'schema': {'mutable': False, 'view': False, 'copy_op': None},
              'inputs_still_functional': [True],
              'inputs_data_mutation_before_after': [[False, False]]},
 'max.dim(W, 1)': {'seq': 'tuple',
                   'items': [{'functional': True,
                              'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                              'is_view': False,
                              'base': None,
                              'is_base': True,
                              'multi_output_view': False,
                              'data_mutation': False,
                              'value_type': 'Tensor',
                              'value': [2.0, 5.0],
                              'value_meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                              'value_base': None},
                             {'functional': True,
                              'meta': [[2], [1], 0, 'torch.int64', 'cpu'],
                              'is_view': False,
                              'base': None,
                              'is_base': True,
                              'multi_output_view': False,
                              'data_mutation': False,
                              'value_type': 'Tensor',
                              'value': [2, 2],
                              'value_meta': [[2], [1], 0, 'torch.int64', 'cpu'],
                              'value_base': None}],
                   'inner': [['aten.max.dim', ['T'], False, True, False]],
                   'outer': 'aten.max.dim',
                   'schema': {'mutable': False, 'view': False, 'copy_op': None},
                   'inputs_still_functional': [True],
                   'inputs_data_mutation_before_after': [[False, False]]},
 'var_mean.correction(W, [1])': {'seq': 'tuple',
                                 'items': [{'functional': True,
                                            'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                                            'is_view': False,
                                            'base': None,
                                            'is_base': True,
                                            'multi_output_view': False,
                                            'data_mutation': False,
                                            'value_type': 'Tensor',
                                            'value': [1.0, 1.0],
                                            'value_meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                                            'value_base': None},
                                           {'functional': True,
                                            'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                                            'is_view': False,
                                            'base': None,
                                            'is_base': True,
                                            'multi_output_view': False,
                                            'data_mutation': False,
                                            'value_type': 'Tensor',
                                            'value': [1.0, 4.0],
                                            'value_meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                                            'value_base': None}],
                                 'inner': [['aten.var_mean.correction', ['T'], False, True, False]],
                                 'outer': 'aten.var_mean.correction',
                                 'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                 'inputs_still_functional': [True],
                                 'inputs_data_mutation_before_after': [[False, False]]},
 'view(W, [-1]) reapply=False': {'functional': True,
                                 'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                 'is_view': True,
                                 'base': 0,
                                 'is_base': False,
                                 'multi_output_view': False,
                                 'data_mutation': False,
                                 'value_type': 'Tensor',
                                 'value': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                 'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                 'value_base': None,
                                 'inner': [['aten.view_copy.default', ['T'], False, True, False]],
                                 'outer': 'aten.view.default',
                                 'schema': {'mutable': False,
                                            'view': True,
                                            'copy_op': 'aten.view_copy.default'},
                                 'inputs_still_functional': [True],
                                 'inputs_data_mutation_before_after': [[False, False]]},
 'view(W, [-1]) reapply=True': {'functional': True,
                                'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                'is_view': True,
                                'base': 0,
                                'is_base': False,
                                'multi_output_view': False,
                                'data_mutation': False,
                                'value_type': 'Tensor',
                                'value': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                'value_base': None,
                                'inner': [['aten.view.default', ['T'], False, True, True]],
                                'outer': 'aten.view.default',
                                'schema': {'mutable': False,
                                           'view': True,
                                           'copy_op': 'aten.view_copy.default'},
                                'inputs_still_functional': [True],
                                'inputs_data_mutation_before_after': [[False, False]]},
 'W.view(3, 2) reapply=False': {'functional': True,
                                'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                'is_view': True,
                                'base': 0,
                                'is_base': False,
                                'multi_output_view': False,
                                'data_mutation': False,
                                'value_type': 'Tensor',
                                'value': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                                'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                'value_base': None,
                                'inner': [['aten.view_copy.default', ['T'], False, True, False]],
                                'outer': 'aten.view.default',
                                'schema': {'mutable': False,
                                           'view': True,
                                           'copy_op': 'aten.view_copy.default'},
                                'inputs_still_functional': [True],
                                'inputs_data_mutation_before_after': [[False, False]]},
 'W.view(3, 2) reapply=True': {'functional': True,
                               'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                               'is_view': True,
                               'base': 0,
                               'is_base': False,
                               'multi_output_view': False,
                               'data_mutation': False,
                               'value_type': 'Tensor',
                               'value': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                               'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                               'value_base': None,
                               'inner': [['aten.view.default', ['T'], False, True, True]],
                               'outer': 'aten.view.default',
                               'schema': {'mutable': False,
                                          'view': True,
                                          'copy_op': 'aten.view_copy.default'},
                               'inputs_still_functional': [True],
                               'inputs_data_mutation_before_after': [[False, False]]},
 'reshape(W, [3, 2]) (a view) reapply=False': {'functional': True,
                                               'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                               'is_view': True,
                                               'base': 0,
                                               'is_base': False,
                                               'multi_output_view': False,
                                               'data_mutation': False,
                                               'value_type': 'Tensor',
                                               'value': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                                               'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                               'value_base': None,
                                               'inner': [['aten.view_copy.default',
                                                          ['T'],
                                                          False,
                                                          True,
                                                          False]],
                                               'outer': 'aten.reshape.default',
                                               'schema': {'mutable': False, 'view': True, 'copy_op': None},
                                               'inputs_still_functional': [True],
                                               'inputs_data_mutation_before_after': [[False, False]]},
 'reshape(W, [3, 2]) (a view) reapply=True': {'functional': True,
                                              'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                              'is_view': True,
                                              'base': 0,
                                              'is_base': False,
                                              'multi_output_view': False,
                                              'data_mutation': False,
                                              'value_type': 'Tensor',
                                              'value': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                                              'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                              'value_base': None,
                                              'inner': [['aten.view.default', ['T'], False, True, True]],
                                              'outer': 'aten.reshape.default',
                                              'schema': {'mutable': False, 'view': True, 'copy_op': None},
                                              'inputs_still_functional': [True],
                                              'inputs_data_mutation_before_after': [[False, False]]},
 'reshape(Wt, [6]) (a copy) reapply=False': {'functional': True,
                                             'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                             'is_view': False,
                                             'base': None,
                                             'is_base': False,
                                             'multi_output_view': False,
                                             'data_mutation': False,
                                             'value_type': 'Tensor',
                                             'value': [0.0, 2.0, 4.0, 1.0, 3.0, 5.0],
                                             'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                             'value_base': None,
                                             'inner': [['aten.clone.default', ['T'], False, True, False],
                                                       ['aten._unsafe_view.default',
                                                        ['T'],
                                                        False,
                                                        True,
                                                        False]],
                                             'outer': 'aten.reshape.default',
                                             'schema': {'mutable': False, 'view': True, 'copy_op': None},
                                             'inputs_still_functional': [True],
                                             'inputs_data_mutation_before_after': [[False, False]]},
 'reshape(Wt, [6]) (a copy) reapply=True': {'functional': True,
                                            'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                            'is_view': False,
                                            'base': None,
                                            'is_base': False,
                                            'multi_output_view': False,
                                            'data_mutation': False,
                                            'value_type': 'Tensor',
                                            'value': [0.0, 2.0, 4.0, 1.0, 3.0, 5.0],
                                            'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                            'value_base': None,
                                            'inner': [['aten.clone.default', ['T'], False, True, True],
                                                      ['aten._unsafe_view.default',
                                                       ['T'],
                                                       False,
                                                       True,
                                                       True]],
                                            'outer': 'aten.reshape.default',
                                            'schema': {'mutable': False, 'view': True, 'copy_op': None},
                                            'inputs_still_functional': [True],
                                            'inputs_data_mutation_before_after': [[False, False]]},
 '_unsafe_view(W, [6]) reapply=False': {'functional': True,
                                        'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                        'is_view': False,
                                        'base': None,
                                        'is_base': False,
                                        'multi_output_view': False,
                                        'data_mutation': False,
                                        'value_type': 'Tensor',
                                        'value': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                        'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                        'value_base': None,
                                        'inner': [['aten._unsafe_view.default', ['T'], False, True, False]],
                                        'outer': 'aten._unsafe_view.default',
                                        'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                        'inputs_still_functional': [True],
                                        'inputs_data_mutation_before_after': [[False, False]]},
 '_unsafe_view(W, [6]) reapply=True': {'functional': True,
                                       'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                       'is_view': False,
                                       'base': None,
                                       'is_base': False,
                                       'multi_output_view': False,
                                       'data_mutation': False,
                                       'value_type': 'Tensor',
                                       'value': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                       'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                       'value_base': None,
                                       'inner': [['aten._unsafe_view.default', ['T'], False, True, True]],
                                       'outer': 'aten._unsafe_view.default',
                                       'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                       'inputs_still_functional': [True],
                                       'inputs_data_mutation_before_after': [[False, False]]},
 'permute(W, [1, 0]) reapply=False': {'functional': True,
                                      'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                      'is_view': True,
                                      'base': 0,
                                      'is_base': False,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                                      'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.permute_copy.default', ['T'], False, True, False]],
                                      'outer': 'aten.permute.default',
                                      'schema': {'mutable': False,
                                                 'view': True,
                                                 'copy_op': 'aten.permute_copy.default'},
                                      'inputs_still_functional': [True],
                                      'inputs_data_mutation_before_after': [[False, False]]},
 'permute(W, [1, 0]) reapply=True': {'functional': True,
                                     'meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                                     'is_view': True,
                                     'base': 0,
                                     'is_base': False,
                                     'multi_output_view': False,
                                     'data_mutation': False,
                                     'value_type': 'Tensor',
                                     'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                                     'value_meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                                     'value_base': None,
                                     'inner': [['aten.permute.default', ['T'], False, True, True]],
                                     'outer': 'aten.permute.default',
                                     'schema': {'mutable': False,
                                                'view': True,
                                                'copy_op': 'aten.permute_copy.default'},
                                     'inputs_still_functional': [True],
                                     'inputs_data_mutation_before_after': [[False, False]]},
 't(W) reapply=False': {'functional': True,
                        'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                        'is_view': True,
                        'base': 0,
                        'is_base': False,
                        'multi_output_view': False,
                        'data_mutation': False,
                        'value_type': 'Tensor',
                        'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                        'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                        'value_base': None,
                        'inner': [['aten.t_copy.default', ['T'], False, True, False]],
                        'outer': 'aten.t.default',
                        'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.t_copy.default'},
                        'inputs_still_functional': [True],
                        'inputs_data_mutation_before_after': [[False, False]]},
 't(W) reapply=True': {'functional': True,
                       'meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                       'is_view': True,
                       'base': 0,
                       'is_base': False,
                       'multi_output_view': False,
                       'data_mutation': False,
                       'value_type': 'Tensor',
                       'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                       'value_meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                       'value_base': None,
                       'inner': [['aten.t.default', ['T'], False, True, True]],
                       'outer': 'aten.t.default',
                       'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.t_copy.default'},
                       'inputs_still_functional': [True],
                       'inputs_data_mutation_before_after': [[False, False]]},
 'transpose.int(W, 0, 1) reapply=False': {'functional': True,
                                          'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                          'is_view': True,
                                          'base': 0,
                                          'is_base': False,
                                          'multi_output_view': False,
                                          'data_mutation': False,
                                          'value_type': 'Tensor',
                                          'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                                          'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                          'value_base': None,
                                          'inner': [['aten.transpose_copy.int', ['T'], False, True, False]],
                                          'outer': 'aten.transpose.int',
                                          'schema': {'mutable': False,
                                                     'view': True,
                                                     'copy_op': 'aten.transpose_copy.int'},
                                          'inputs_still_functional': [True],
                                          'inputs_data_mutation_before_after': [[False, False]]},
 'transpose.int(W, 0, 1) reapply=True': {'functional': True,
                                         'meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                                         'is_view': True,
                                         'base': 0,
                                         'is_base': False,
                                         'multi_output_view': False,
                                         'data_mutation': False,
                                         'value_type': 'Tensor',
                                         'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                                         'value_meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                                         'value_base': None,
                                         'inner': [['aten.transpose.int', ['T'], False, True, True]],
                                         'outer': 'aten.transpose.int',
                                         'schema': {'mutable': False,
                                                    'view': True,
                                                    'copy_op': 'aten.transpose_copy.int'},
                                         'inputs_still_functional': [True],
                                         'inputs_data_mutation_before_after': [[False, False]]},
 'unsqueeze(W, 0) reapply=False': {'functional': True,
                                   'meta': [[1, 2, 3], [6, 3, 1], 0, 'torch.float32', 'cpu'],
                                   'is_view': True,
                                   'base': 0,
                                   'is_base': False,
                                   'multi_output_view': False,
                                   'data_mutation': False,
                                   'value_type': 'Tensor',
                                   'value': [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]],
                                   'value_meta': [[1, 2, 3], [6, 3, 1], 0, 'torch.float32', 'cpu'],
                                   'value_base': None,
                                   'inner': [['aten.unsqueeze_copy.default', ['T'], False, True, False]],
                                   'outer': 'aten.unsqueeze.default',
                                   'schema': {'mutable': False,
                                              'view': True,
                                              'copy_op': 'aten.unsqueeze_copy.default'},
                                   'inputs_still_functional': [True],
                                   'inputs_data_mutation_before_after': [[False, False]]},
 'unsqueeze(W, 0) reapply=True': {'functional': True,
                                  'meta': [[1, 2, 3], [6, 3, 1], 0, 'torch.float32', 'cpu'],
                                  'is_view': True,
                                  'base': 0,
                                  'is_base': False,
                                  'multi_output_view': False,
                                  'data_mutation': False,
                                  'value_type': 'Tensor',
                                  'value': [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]],
                                  'value_meta': [[1, 2, 3], [6, 3, 1], 0, 'torch.float32', 'cpu'],
                                  'value_base': None,
                                  'inner': [['aten.unsqueeze.default', ['T'], False, True, True]],
                                  'outer': 'aten.unsqueeze.default',
                                  'schema': {'mutable': False,
                                             'view': True,
                                             'copy_op': 'aten.unsqueeze_copy.default'},
                                  'inputs_still_functional': [True],
                                  'inputs_data_mutation_before_after': [[False, False]]},
 'squeeze.dim(W13, 0) reapply=False': {'functional': True,
                                       'meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                       'is_view': True,
                                       'base': 0,
                                       'is_base': False,
                                       'multi_output_view': False,
                                       'data_mutation': False,
                                       'value_type': 'Tensor',
                                       'value': [1.0, 1.0, 1.0],
                                       'value_meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                       'value_base': None,
                                       'inner': [['aten.squeeze_copy.dim', ['T'], False, True, False]],
                                       'outer': 'aten.squeeze.dim',
                                       'schema': {'mutable': False,
                                                  'view': True,
                                                  'copy_op': 'aten.squeeze_copy.dim'},
                                       'inputs_still_functional': [True],
                                       'inputs_data_mutation_before_after': [[False, False]]},
 'squeeze.dim(W13, 0) reapply=True': {'functional': True,
                                      'meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                      'is_view': True,
                                      'base': 0,
                                      'is_base': False,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [1.0, 1.0, 1.0],
                                      'value_meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.squeeze.dim', ['T'], False, True, True]],
                                      'outer': 'aten.squeeze.dim',
                                      'schema': {'mutable': False,
                                                 'view': True,
                                                 'copy_op': 'aten.squeeze_copy.dim'},
                                      'inputs_still_functional': [True],
                                      'inputs_data_mutation_before_after': [[False, False]]},
 'expand(W1, [2, 4]) reapply=False': {'functional': True,
                                      'meta': [[2, 4], [4, 1], 0, 'torch.float32', 'cpu'],
                                      'is_view': True,
                                      'base': 0,
                                      'is_base': False,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]],
                                      'value_meta': [[2, 4], [4, 1], 0, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.expand_copy.default', ['T'], False, True, False]],
                                      'outer': 'aten.expand.default',
                                      'schema': {'mutable': False,
                                                 'view': True,
                                                 'copy_op': 'aten.expand_copy.default'},
                                      'inputs_still_functional': [True],
                                      'inputs_data_mutation_before_after': [[False, False]]},
 'expand(W1, [2, 4]) reapply=True': {'functional': True,
                                     'meta': [[2, 4], [0, 1], 0, 'torch.float32', 'cpu'],
                                     'is_view': True,
                                     'base': 0,
                                     'is_base': False,
                                     'multi_output_view': False,
                                     'data_mutation': False,
                                     'value_type': 'Tensor',
                                     'value': [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]],
                                     'value_meta': [[2, 4], [0, 1], 0, 'torch.float32', 'cpu'],
                                     'value_base': None,
                                     'inner': [['aten.expand.default', ['T'], False, True, True]],
                                     'outer': 'aten.expand.default',
                                     'schema': {'mutable': False,
                                                'view': True,
                                                'copy_op': 'aten.expand_copy.default'},
                                     'inputs_still_functional': [True],
                                     'inputs_data_mutation_before_after': [[False, False]]},
 'W[1:] reapply=False': {'functional': True,
                         'meta': [[1, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                         'is_view': True,
                         'base': 0,
                         'is_base': False,
                         'multi_output_view': False,
                         'data_mutation': False,
                         'value_type': 'Tensor',
                         'value': [[3.0, 4.0, 5.0]],
                         'value_meta': [[1, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                         'value_base': None,
                         'inner': [['aten.slice_copy.Tensor', ['T'], False, True, False]],
                         'outer': 'aten.slice.Tensor',
                         'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.slice_copy.Tensor'},
                         'inputs_still_functional': [True],
                         'inputs_data_mutation_before_after': [[False, False]]},
 'W[1:] reapply=True': {'functional': True,
                        'meta': [[1, 3], [3, 1], 3, 'torch.float32', 'cpu'],
                        'is_view': True,
                        'base': 0,
                        'is_base': False,
                        'multi_output_view': False,
                        'data_mutation': False,
                        'value_type': 'Tensor',
                        'value': [[3.0, 4.0, 5.0]],
                        'value_meta': [[1, 3], [3, 1], 3, 'torch.float32', 'cpu'],
                        'value_base': None,
                        'inner': [['aten.slice.Tensor', ['T'], False, True, True]],
                        'outer': 'aten.slice.Tensor',
                        'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.slice_copy.Tensor'},
                        'inputs_still_functional': [True],
                        'inputs_data_mutation_before_after': [[False, False]]},
 'W[:, 1] reapply=False': {'functional': True,
                           'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                           'is_view': True,
                           'base': 0,
                           'is_base': False,
                           'multi_output_view': False,
                           'data_mutation': False,
                           'value_type': 'Tensor',
                           'value': [1.0, 4.0],
                           'value_meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                           'value_base': None,
                           'inner': [['aten.select_copy.int', ['T'], False, True, False]],
                           'outer': 'aten.select.int',
                           'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.select_copy.int'},
                           'inputs_still_functional': [True],
                           'inputs_data_mutation_before_after': [[False, False]]},
 'W[:, 1] reapply=True': {'functional': True,
                          'meta': [[2], [3], 1, 'torch.float32', 'cpu'],
                          'is_view': True,
                          'base': 0,
                          'is_base': False,
                          'multi_output_view': False,
                          'data_mutation': False,
                          'value_type': 'Tensor',
                          'value': [1.0, 4.0],
                          'value_meta': [[2], [3], 1, 'torch.float32', 'cpu'],
                          'value_base': None,
                          'inner': [['aten.select.int', ['T'], False, True, True]],
                          'outer': 'aten.select.int',
                          'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.select_copy.int'},
                          'inputs_still_functional': [True],
                          'inputs_data_mutation_before_after': [[False, False]]},
 'slice.Tensor(W, 1, 0, 2) reapply=False': {'functional': True,
                                            'meta': [[2, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                            'is_view': True,
                                            'base': 0,
                                            'is_base': False,
                                            'multi_output_view': False,
                                            'data_mutation': False,
                                            'value_type': 'Tensor',
                                            'value': [[0.0, 1.0], [3.0, 4.0]],
                                            'value_meta': [[2, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                            'value_base': None,
                                            'inner': [['aten.slice_copy.Tensor', ['T'], False, True, False]],
                                            'outer': 'aten.slice.Tensor',
                                            'schema': {'mutable': False,
                                                       'view': True,
                                                       'copy_op': 'aten.slice_copy.Tensor'},
                                            'inputs_still_functional': [True],
                                            'inputs_data_mutation_before_after': [[False, False]]},
 'slice.Tensor(W, 1, 0, 2) reapply=True': {'functional': True,
                                           'meta': [[2, 2], [3, 1], 0, 'torch.float32', 'cpu'],
                                           'is_view': True,
                                           'base': 0,
                                           'is_base': False,
                                           'multi_output_view': False,
                                           'data_mutation': False,
                                           'value_type': 'Tensor',
                                           'value': [[0.0, 1.0], [3.0, 4.0]],
                                           'value_meta': [[2, 2], [3, 1], 0, 'torch.float32', 'cpu'],
                                           'value_base': None,
                                           'inner': [['aten.slice.Tensor', ['T'], False, True, True]],
                                           'outer': 'aten.slice.Tensor',
                                           'schema': {'mutable': False,
                                                      'view': True,
                                                      'copy_op': 'aten.slice_copy.Tensor'},
                                           'inputs_still_functional': [True],
                                           'inputs_data_mutation_before_after': [[False, False]]},
 'select.int(W, 0, 1) reapply=False': {'functional': True,
                                       'meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                       'is_view': True,
                                       'base': 0,
                                       'is_base': False,
                                       'multi_output_view': False,
                                       'data_mutation': False,
                                       'value_type': 'Tensor',
                                       'value': [3.0, 4.0, 5.0],
                                       'value_meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                       'value_base': None,
                                       'inner': [['aten.select_copy.int', ['T'], False, True, False]],
                                       'outer': 'aten.select.int',
                                       'schema': {'mutable': False,
                                                  'view': True,
                                                  'copy_op': 'aten.select_copy.int'},
                                       'inputs_still_functional': [True],
                                       'inputs_data_mutation_before_after': [[False, False]]},
 'select.int(W, 0, 1) reapply=True': {'functional': True,
                                      'meta': [[3], [1], 3, 'torch.float32', 'cpu'],
                                      'is_view': True,
                                      'base': 0,
                                      'is_base': False,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [3.0, 4.0, 5.0],
                                      'value_meta': [[3], [1], 3, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.select.int', ['T'], False, True, True]],
                                      'outer': 'aten.select.int',
                                      'schema': {'mutable': False,
                                                 'view': True,
                                                 'copy_op': 'aten.select_copy.int'},
                                      'inputs_still_functional': [True],
                                      'inputs_data_mutation_before_after': [[False, False]]},
 'narrow(W, 1, 1, 2) reapply=False': {'functional': True,
                                      'meta': [[2, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                      'is_view': True,
                                      'base': 0,
                                      'is_base': False,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [[1.0, 2.0], [4.0, 5.0]],
                                      'value_meta': [[2, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.slice_copy.Tensor', ['T'], False, True, False]],
                                      'outer': 'aten.narrow.default',
                                      'schema': {'mutable': False,
                                                 'view': True,
                                                 'copy_op': 'aten.narrow_copy.default'},
                                      'inputs_still_functional': [True],
                                      'inputs_data_mutation_before_after': [[False, False]]},
 'narrow(W, 1, 1, 2) reapply=True': {'functional': True,
                                     'meta': [[2, 2], [3, 1], 1, 'torch.float32', 'cpu'],
                                     'is_view': True,
                                     'base': 0,
                                     'is_base': False,
                                     'multi_output_view': False,
                                     'data_mutation': False,
                                     'value_type': 'Tensor',
                                     'value': [[1.0, 2.0], [4.0, 5.0]],
                                     'value_meta': [[2, 2], [3, 1], 1, 'torch.float32', 'cpu'],
                                     'value_base': None,
                                     'inner': [['aten.slice.Tensor', ['T'], False, True, True]],
                                     'outer': 'aten.narrow.default',
                                     'schema': {'mutable': False,
                                                'view': True,
                                                'copy_op': 'aten.narrow_copy.default'},
                                     'inputs_still_functional': [True],
                                     'inputs_data_mutation_before_after': [[False, False]]},
 'unbind.int(W) reapply=False': {'seq': 'list',
                                 'items': [{'functional': True,
                                            'meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                            'is_view': True,
                                            'base': 0,
                                            'is_base': False,
                                            'multi_output_view': True,
                                            'data_mutation': False,
                                            'value_type': 'Tensor',
                                            'value': [0.0, 1.0, 2.0],
                                            'value_meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                            'value_base': None},
                                           {'functional': True,
                                            'meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                            'is_view': True,
                                            'base': 0,
                                            'is_base': False,
                                            'multi_output_view': True,
                                            'data_mutation': False,
                                            'value_type': 'Tensor',
                                            'value': [3.0, 4.0, 5.0],
                                            'value_meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                            'value_base': None}],
                                 'inner': [['aten.unbind_copy.int', ['T'], False, True, False]],
                                 'outer': 'aten.unbind.int',
                                 'schema': {'mutable': False,
                                            'view': True,
                                            'copy_op': 'aten.unbind_copy.int'},
                                 'inputs_still_functional': [True],
                                 'inputs_data_mutation_before_after': [[False, False]]},
 'unbind.int(W) reapply=True': {'seq': 'list',
                                'items': [{'functional': True,
                                           'meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                           'is_view': True,
                                           'base': 0,
                                           'is_base': False,
                                           'multi_output_view': True,
                                           'data_mutation': False,
                                           'value_type': 'Tensor',
                                           'value': [0.0, 1.0, 2.0],
                                           'value_meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                           'value_base': None},
                                          {'functional': True,
                                           'meta': [[3], [1], 3, 'torch.float32', 'cpu'],
                                           'is_view': True,
                                           'base': 0,
                                           'is_base': False,
                                           'multi_output_view': True,
                                           'data_mutation': False,
                                           'value_type': 'Tensor',
                                           'value': [3.0, 4.0, 5.0],
                                           'value_meta': [[3], [1], 3, 'torch.float32', 'cpu'],
                                           'value_base': None}],
                                'inner': [['aten.unbind.int', ['T'], False, True, True]],
                                'outer': 'aten.unbind.int',
                                'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.unbind_copy.int'},
                                'inputs_still_functional': [True],
                                'inputs_data_mutation_before_after': [[False, False]]},
 'split.Tensor(W43, 2) reapply=False': {'seq': 'list',
                                        'items': [{'functional': True,
                                                   'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                   'is_view': True,
                                                   'base': 0,
                                                   'is_base': False,
                                                   'multi_output_view': True,
                                                   'data_mutation': False,
                                                   'value_type': 'Tensor',
                                                   'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                                   'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                   'value_base': None},
                                                  {'functional': True,
                                                   'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                   'is_view': True,
                                                   'base': 0,
                                                   'is_base': False,
                                                   'multi_output_view': True,
                                                   'data_mutation': False,
                                                   'value_type': 'Tensor',
                                                   'value': [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
                                                   'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                   'value_base': None}],
                                        'inner': [['aten.split_copy.Tensor', ['T'], False, True, False]],
                                        'outer': 'aten.split.Tensor',
                                        'schema': {'mutable': False,
                                                   'view': True,
                                                   'copy_op': 'aten.split_copy.Tensor'},
                                        'inputs_still_functional': [True],
                                        'inputs_data_mutation_before_after': [[False, False]]},
 'split.Tensor(W43, 2) reapply=True': {'seq': 'list',
                                       'items': [{'functional': True,
                                                  'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                  'is_view': True,
                                                  'base': 0,
                                                  'is_base': False,
                                                  'multi_output_view': True,
                                                  'data_mutation': False,
                                                  'value_type': 'Tensor',
                                                  'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                                  'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                  'value_base': None},
                                                 {'functional': True,
                                                  'meta': [[2, 3], [3, 1], 6, 'torch.float32', 'cpu'],
                                                  'is_view': True,
                                                  'base': 0,
                                                  'is_base': False,
                                                  'multi_output_view': True,
                                                  'data_mutation': False,
                                                  'value_type': 'Tensor',
                                                  'value': [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
                                                  'value_meta': [[2, 3], [3, 1], 6, 'torch.float32', 'cpu'],
                                                  'value_base': None}],
                                       'inner': [['aten.split.Tensor', ['T'], False, True, True]],
                                       'outer': 'aten.split.Tensor',
                                       'schema': {'mutable': False,
                                                  'view': True,
                                                  'copy_op': 'aten.split_copy.Tensor'},
                                       'inputs_still_functional': [True],
                                       'inputs_data_mutation_before_after': [[False, False]]},
 'chunk(W43, 2) reapply=False': {'seq': 'list',
                                 'items': [{'functional': True,
                                            'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                            'is_view': True,
                                            'base': 0,
                                            'is_base': False,
                                            'multi_output_view': True,
                                            'data_mutation': False,
                                            'value_type': 'Tensor',
                                            'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                            'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                            'value_base': None},
                                           {'functional': True,
                                            'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                            'is_view': True,
                                            'base': 0,
                                            'is_base': False,
                                            'multi_output_view': True,
                                            'data_mutation': False,
                                            'value_type': 'Tensor',
                                            'value': [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
                                            'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                            'value_base': None}],
                                 'inner': [['aten.split_copy.Tensor', ['T'], False, True, False]],
                                 'outer': 'aten.chunk.default',
                                 'schema': {'mutable': False, 'view': True, 'copy_op': None},
                                 'inputs_still_functional': [True],
                                 'inputs_data_mutation_before_after': [[False, False]]},
 'chunk(W43, 2) reapply=True': {'seq': 'list',
                                'items': [{'functional': True,
                                           'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                           'is_view': True,
                                           'base': 0,
                                           'is_base': False,
                                           'multi_output_view': True,
                                           'data_mutation': False,
                                           'value_type': 'Tensor',
                                           'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                           'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                           'value_base': None},
                                          {'functional': True,
                                           'meta': [[2, 3], [3, 1], 6, 'torch.float32', 'cpu'],
                                           'is_view': True,
                                           'base': 0,
                                           'is_base': False,
                                           'multi_output_view': True,
                                           'data_mutation': False,
                                           'value_type': 'Tensor',
                                           'value': [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
                                           'value_meta': [[2, 3], [3, 1], 6, 'torch.float32', 'cpu'],
                                           'value_base': None}],
                                'inner': [['aten.split.Tensor', ['T'], False, True, True]],
                                'outer': 'aten.chunk.default',
                                'schema': {'mutable': False, 'view': True, 'copy_op': None},
                                'inputs_still_functional': [True],
                                'inputs_data_mutation_before_after': [[False, False]]},
 'detach(W) reapply=False': {'functional': True,
                             'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                             'is_view': False,
                             'base': None,
                             'is_base': False,
                             'multi_output_view': False,
                             'data_mutation': False,
                             'value_type': 'Tensor',
                             'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                             'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                             'value_base': None,
                             'inner': [['aten.detach_copy.default', ['T'], False, True, False]],
                             'outer': 'aten.detach.default',
                             'schema': {'mutable': False,
                                        'view': True,
                                        'copy_op': 'aten.detach_copy.default'},
                             'inputs_still_functional': [True],
                             'inputs_data_mutation_before_after': [[False, False]]},
 'detach(W) reapply=True': {'functional': True,
                            'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                            'is_view': False,
                            'base': None,
                            'is_base': False,
                            'multi_output_view': False,
                            'data_mutation': False,
                            'value_type': 'Tensor',
                            'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                            'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                            'value_base': None,
                            'inner': [['aten.detach.default', ['T'], False, True, True]],
                            'outer': 'aten.detach.default',
                            'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.detach_copy.default'},
                            'inputs_still_functional': [True],
                            'inputs_data_mutation_before_after': [[False, False]]},
 'alias(W) reapply=False': {'functional': True,
                            'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                            'is_view': True,
                            'base': 0,
                            'is_base': False,
                            'multi_output_view': False,
                            'data_mutation': False,
                            'value_type': 'Tensor',
                            'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                            'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                            'value_base': None,
                            'inner': [['aten.alias_copy.default', ['T'], False, True, False]],
                            'outer': 'aten.alias.default',
                            'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.alias_copy.default'},
                            'inputs_still_functional': [True],
                            'inputs_data_mutation_before_after': [[False, False]]},
 'alias(W) reapply=True': {'functional': True,
                           'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                           'is_view': True,
                           'base': 0,
                           'is_base': False,
                           'multi_output_view': False,
                           'data_mutation': False,
                           'value_type': 'Tensor',
                           'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                           'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                           'value_base': None,
                           'inner': [['aten.alias.default', ['T'], False, True, True]],
                           'outer': 'aten.alias.default',
                           'schema': {'mutable': False, 'view': True, 'copy_op': 'aten.alias_copy.default'},
                           'inputs_still_functional': [True],
                           'inputs_data_mutation_before_after': [[False, False]]},
 'diagonal(W33) reapply=False': {'functional': True,
                                 'meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                 'is_view': True,
                                 'base': 0,
                                 'is_base': False,
                                 'multi_output_view': False,
                                 'data_mutation': False,
                                 'value_type': 'Tensor',
                                 'value': [0.0, 4.0, 8.0],
                                 'value_meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                 'value_base': None,
                                 'inner': [['aten.diagonal_copy.default', ['T'], False, True, False]],
                                 'outer': 'aten.diagonal.default',
                                 'schema': {'mutable': False,
                                            'view': True,
                                            'copy_op': 'aten.diagonal_copy.default'},
                                 'inputs_still_functional': [True],
                                 'inputs_data_mutation_before_after': [[False, False]]},
 'diagonal(W33) reapply=True': {'functional': True,
                                'meta': [[3], [4], 0, 'torch.float32', 'cpu'],
                                'is_view': True,
                                'base': 0,
                                'is_base': False,
                                'multi_output_view': False,
                                'data_mutation': False,
                                'value_type': 'Tensor',
                                'value': [0.0, 4.0, 8.0],
                                'value_meta': [[3], [4], 0, 'torch.float32', 'cpu'],
                                'value_base': None,
                                'inner': [['aten.diagonal.default', ['T'], False, True, True]],
                                'outer': 'aten.diagonal.default',
                                'schema': {'mutable': False,
                                           'view': True,
                                           'copy_op': 'aten.diagonal_copy.default'},
                                'inputs_still_functional': [True],
                                'inputs_data_mutation_before_after': [[False, False]]},
 'view_copy(W, [-1]) reapply=False': {'functional': True,
                                      'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                      'is_view': False,
                                      'base': None,
                                      'is_base': True,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                      'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.view_copy.default', ['T'], False, True, False]],
                                      'outer': 'aten.view_copy.default',
                                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                      'inputs_still_functional': [True],
                                      'inputs_data_mutation_before_after': [[False, False]]},
 'view_copy(W, [-1]) reapply=True': {'functional': True,
                                     'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                     'is_view': False,
                                     'base': None,
                                     'is_base': True,
                                     'multi_output_view': False,
                                     'data_mutation': False,
                                     'value_type': 'Tensor',
                                     'value': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                     'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                     'value_base': None,
                                     'inner': [['aten.view_copy.default', ['T'], False, True, True]],
                                     'outer': 'aten.view_copy.default',
                                     'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                     'inputs_still_functional': [True],
                                     'inputs_data_mutation_before_after': [[False, False]]},
 'permute_copy(W, [1, 0]) reapply=False': {'functional': True,
                                           'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                           'is_view': False,
                                           'base': None,
                                           'is_base': True,
                                           'multi_output_view': False,
                                           'data_mutation': False,
                                           'value_type': 'Tensor',
                                           'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                                           'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                           'value_base': None,
                                           'inner': [['aten.permute_copy.default',
                                                      ['T'],
                                                      False,
                                                      True,
                                                      False]],
                                           'outer': 'aten.permute_copy.default',
                                           'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                           'inputs_still_functional': [True],
                                           'inputs_data_mutation_before_after': [[False, False]]},
 'permute_copy(W, [1, 0]) reapply=True': {'functional': True,
                                          'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                          'is_view': False,
                                          'base': None,
                                          'is_base': True,
                                          'multi_output_view': False,
                                          'data_mutation': False,
                                          'value_type': 'Tensor',
                                          'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                                          'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                          'value_base': None,
                                          'inner': [['aten.permute_copy.default', ['T'], False, True, True]],
                                          'outer': 'aten.permute_copy.default',
                                          'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                          'inputs_still_functional': [True],
                                          'inputs_data_mutation_before_after': [[False, False]]},
 't_copy(W) reapply=False': {'functional': True,
                             'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                             'is_view': False,
                             'base': None,
                             'is_base': True,
                             'multi_output_view': False,
                             'data_mutation': False,
                             'value_type': 'Tensor',
                             'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                             'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                             'value_base': None,
                             'inner': [['aten.t_copy.default', ['T'], False, True, False]],
                             'outer': 'aten.t_copy.default',
                             'schema': {'mutable': False, 'view': False, 'copy_op': None},
                             'inputs_still_functional': [True],
                             'inputs_data_mutation_before_after': [[False, False]]},
 't_copy(W) reapply=True': {'functional': True,
                            'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                            'is_view': False,
                            'base': None,
                            'is_base': True,
                            'multi_output_view': False,
                            'data_mutation': False,
                            'value_type': 'Tensor',
                            'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                            'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                            'value_base': None,
                            'inner': [['aten.t_copy.default', ['T'], False, True, True]],
                            'outer': 'aten.t_copy.default',
                            'schema': {'mutable': False, 'view': False, 'copy_op': None},
                            'inputs_still_functional': [True],
                            'inputs_data_mutation_before_after': [[False, False]]},
 'slice(view(W)) reapply=False': {'functional': True,
                                  'meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                  'is_view': True,
                                  'base': 0,
                                  'is_base': False,
                                  'multi_output_view': False,
                                  'data_mutation': False,
                                  'value_type': 'Tensor',
                                  'value': [1.0, 2.0, 3.0],
                                  'value_meta': [[3], [1], 0, 'torch.float32', 'cpu'],
                                  'value_base': None,
                                  'inner': [['aten.slice_copy.Tensor', ['T'], False, True, False]],
                                  'outer': 'aten.slice.Tensor',
                                  'schema': {'mutable': False,
                                             'view': True,
                                             'copy_op': 'aten.slice_copy.Tensor'},
                                  'inputs_still_functional': [True, True],
                                  'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 'slice(view(W)) reapply=True': {'functional': True,
                                 'meta': [[3], [1], 1, 'torch.float32', 'cpu'],
                                 'is_view': True,
                                 'base': 0,
                                 'is_base': False,
                                 'multi_output_view': False,
                                 'data_mutation': False,
                                 'value_type': 'Tensor',
                                 'value': [1.0, 2.0, 3.0],
                                 'value_meta': [[3], [1], 1, 'torch.float32', 'cpu'],
                                 'value_base': None,
                                 'inner': [['aten.slice.Tensor', ['T'], False, True, True]],
                                 'outer': 'aten.slice.Tensor',
                                 'schema': {'mutable': False,
                                            'view': True,
                                            'copy_op': 'aten.slice_copy.Tensor'},
                                 'inputs_still_functional': [True, True],
                                 'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 'add.Tensor(view(W), 1) reapply=True': {'functional': True,
                                         'meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                         'is_view': False,
                                         'base': None,
                                         'is_base': True,
                                         'multi_output_view': False,
                                         'data_mutation': False,
                                         'value_type': 'Tensor',
                                         'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                         'value_meta': [[6], [1], 0, 'torch.float32', 'cpu'],
                                         'value_base': None,
                                         'inner': [['aten.add.Tensor', ['T'], False, True, True]],
                                         'outer': 'aten.add.Tensor',
                                         'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                         'inputs_still_functional': [True, True],
                                         'inputs_data_mutation_before_after': [[False, False],
                                                                               [False, False]]},
 'ones([2])': {'functional': False,
               'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
               'is_view': False,
               'base': None,
               'inner': [['aten.ones.default', [], False, True, False]],
               'outer': 'aten.ones.default',
               'schema': {'mutable': False, 'view': False, 'copy_op': None},
               'inputs_still_functional': [],
               'inputs_data_mutation_before_after': []},
 'ones([2]) Functionalize included': {'functional': True,
                                      'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                                      'is_view': False,
                                      'base': None,
                                      'is_base': True,
                                      'multi_output_view': False,
                                      'data_mutation': False,
                                      'value_type': 'Tensor',
                                      'value': [1.0, 1.0],
                                      'value_meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                                      'value_base': None,
                                      'inner': [['aten.ones.default', [], True, True, False]],
                                      'outer': 'aten.ones.default',
                                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                      'inputs_still_functional': [],
                                      'inputs_data_mutation_before_after': []},
 '_op_dk(FK, ones, [2])': {'functional': True,
                           'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                           'is_view': False,
                           'base': None,
                           'is_base': True,
                           'multi_output_view': False,
                           'data_mutation': False,
                           'value_type': 'Tensor',
                           'value': [1.0, 1.0],
                           'value_meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                           'value_base': None,
                           'inner': [['aten.ones.default', [], False, True, False]],
                           'outer': 'aten.ones.default',
                           'schema': {'mutable': False, 'view': False, 'copy_op': None},
                           'inputs_still_functional': [],
                           'inputs_data_mutation_before_after': []},
 '_op_dk(FK, ones, [2]) Functionalize included': {'functional': True,
                                                  'meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                                                  'is_view': False,
                                                  'base': None,
                                                  'is_base': True,
                                                  'multi_output_view': False,
                                                  'data_mutation': False,
                                                  'value_type': 'Tensor',
                                                  'value': [1.0, 1.0],
                                                  'value_meta': [[2], [1], 0, 'torch.float32', 'cpu'],
                                                  'value_base': None,
                                                  'inner': [['aten.ones.default', [], True, True, False]],
                                                  'outer': 'aten.ones.default',
                                                  'schema': {'mutable': False,
                                                             'view': False,
                                                             'copy_op': None},
                                                  'inputs_still_functional': [],
                                                  'inputs_data_mutation_before_after': []},
 '_op_dk(FK, arange, 3) Functionalize included': {'functional': True,
                                                  'meta': [[3], [1], 0, 'torch.int64', 'cpu'],
                                                  'is_view': False,
                                                  'base': None,
                                                  'is_base': True,
                                                  'multi_output_view': False,
                                                  'data_mutation': False,
                                                  'value_type': 'Tensor',
                                                  'value': [0, 1, 2],
                                                  'value_meta': [[3], [1], 0, 'torch.int64', 'cpu'],
                                                  'value_base': None,
                                                  'inner': [['aten.arange.default', [], True, True, False]],
                                                  'outer': 'aten.arange.default',
                                                  'schema': {'mutable': False,
                                                             'view': False,
                                                             'copy_op': None},
                                                  'inputs_still_functional': [],
                                                  'inputs_data_mutation_before_after': []},
 '_op_dk(FK, add, P, P)': {'functional': False,
                           'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                           'is_view': False,
                           'base': None,
                           'inner': [['aten.add.Tensor', ['T', 'T'], False, True, False]],
                           'outer': 'aten.add.Tensor',
                           'schema': {'mutable': False, 'view': False, 'copy_op': None},
                           'inputs_still_functional': [False, False],
                           'inputs_data_mutation_before_after': [[None, None], [None, None]]},
 '_op_dk(FK, add, P, P) Functionalize included': {'functional': False,
                                                  'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                  'is_view': False,
                                                  'base': None,
                                                  'inner': [['aten.add.Tensor',
                                                             ['T', 'T'],
                                                             True,
                                                             True,
                                                             False]],
                                                  'outer': 'aten.add.Tensor',
                                                  'schema': {'mutable': False,
                                                             'view': False,
                                                             'copy_op': None},
                                                  'inputs_still_functional': [False, False],
                                                  'inputs_data_mutation_before_after': [[None, None],
                                                                                        [None, None]]},
 '_op_dk(FK, add, W, W)': {'functional': True,
                           'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                           'is_view': False,
                           'base': None,
                           'is_base': True,
                           'multi_output_view': False,
                           'data_mutation': False,
                           'value_type': 'Tensor',
                           'value': [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]],
                           'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                           'value_base': None,
                           'inner': [['aten.add.Tensor', ['T', 'T'], False, True, False]],
                           'outer': 'aten.add.Tensor',
                           'schema': {'mutable': False, 'view': False, 'copy_op': None},
                           'inputs_still_functional': [True, True],
                           'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 '_op_dk(FK, add, W, W) Functionalize included': {'functional': True,
                                                  'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                  'is_view': False,
                                                  'base': None,
                                                  'is_base': True,
                                                  'multi_output_view': False,
                                                  'data_mutation': False,
                                                  'value_type': 'Tensor',
                                                  'value': [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]],
                                                  'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                  'value_base': None,
                                                  'inner': [['aten.add.Tensor',
                                                             ['T', 'T'],
                                                             True,
                                                             True,
                                                             False]],
                                                  'outer': 'aten.add.Tensor',
                                                  'schema': {'mutable': False,
                                                             'view': False,
                                                             'copy_op': None},
                                                  'inputs_still_functional': [True, True],
                                                  'inputs_data_mutation_before_after': [[False, False],
                                                                                        [False, False]]},
 '_op_dk(FK, addmm, W4, W, W3x4) Functionalize included': {'functional': True,
                                                           'meta': [[2, 4],
                                                                    [4, 1],
                                                                    0,
                                                                    'torch.float32',
                                                                    'cpu'],
                                                           'is_view': False,
                                                           'base': None,
                                                           'is_base': True,
                                                           'multi_output_view': False,
                                                           'data_mutation': False,
                                                           'value_type': 'Tensor',
                                                           'value': [[21.0, 24.0, 27.0, 30.0],
                                                                     [57.0, 69.0, 81.0, 93.0]],
                                                           'value_meta': [[2, 4],
                                                                          [4, 1],
                                                                          0,
                                                                          'torch.float32',
                                                                          'cpu'],
                                                           'value_base': None,
                                                           'inner': [['aten.addmm.default',
                                                                      ['T', 'T', 'T'],
                                                                      True,
                                                                      True,
                                                                      True]],
                                                           'outer': 'aten.addmm.default',
                                                           'schema': {'mutable': False,
                                                                      'view': False,
                                                                      'copy_op': None},
                                                           'inputs_still_functional': [True, True, True],
                                                           'inputs_data_mutation_before_after': [[False,
                                                                                                  False],
                                                                                                 [False,
                                                                                                  False],
                                                                                                 [False,
                                                                                                  False]]},
 '_op_dk(FK, relu, W) Functionalize included': {'functional': True,
                                                'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                'is_view': False,
                                                'base': None,
                                                'is_base': True,
                                                'multi_output_view': False,
                                                'data_mutation': False,
                                                'value_type': 'Tensor',
                                                'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                                'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                'value_base': None,
                                                'inner': [['aten.relu.default', ['T'], True, True, True]],
                                                'outer': 'aten.relu.default',
                                                'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                                'inputs_still_functional': [True],
                                                'inputs_data_mutation_before_after': [[False, False]]},
 '_op_dk(FK, permute, W) Functionalize included': {'functional': True,
                                                   'meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                                                   'is_view': False,
                                                   'base': None,
                                                   'is_base': False,
                                                   'multi_output_view': False,
                                                   'data_mutation': False,
                                                   'value_type': 'Tensor',
                                                   'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                                                   'value_meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                                                   'value_base': 0,
                                                   'inner': [['aten.permute.default',
                                                              ['T'],
                                                              True,
                                                              True,
                                                              True]],
                                                   'outer': 'aten.permute.default',
                                                   'schema': {'mutable': False,
                                                              'view': True,
                                                              'copy_op': 'aten.permute_copy.default'},
                                                   'inputs_still_functional': [True],
                                                   'inputs_data_mutation_before_after': [[False, False]]},
 '_op_dk(FK, permute, W) reapply=False': {'functional': True,
                                          'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                          'is_view': False,
                                          'base': None,
                                          'is_base': False,
                                          'multi_output_view': False,
                                          'data_mutation': False,
                                          'value_type': 'Tensor',
                                          'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                                          'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
                                          'value_base': None,
                                          'inner': [['aten.permute_copy.default', ['T'], False, True, False]],
                                          'outer': 'aten.permute.default',
                                          'schema': {'mutable': False,
                                                     'view': True,
                                                     'copy_op': 'aten.permute_copy.default'},
                                          'inputs_still_functional': [True],
                                          'inputs_data_mutation_before_after': [[False, False]]},
 '_op_dk(FK, permute, P)': {'functional': False,
                            'meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                            'is_view': True,
                            'base': 0,
                            'inner': [['aten.permute.default', ['T'], False, True, False]],
                            'outer': 'aten.permute.default',
                            'schema': {'mutable': False,
                                       'view': True,
                                       'copy_op': 'aten.permute_copy.default'},
                            'inputs_still_functional': [False],
                            'inputs_data_mutation_before_after': [[None, None]]},
 '_op_dk(FK, split, W43, 2) Functionalize included': {'seq': 'list',
                                                      'items': [{'functional': True,
                                                                 'meta': [[2, 3],
                                                                          [3, 1],
                                                                          0,
                                                                          'torch.float32',
                                                                          'cpu'],
                                                                 'is_view': False,
                                                                 'base': None,
                                                                 'is_base': False,
                                                                 'multi_output_view': True,
                                                                 'data_mutation': False,
                                                                 'value_type': 'Tensor',
                                                                 'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                                                 'value_meta': [[2, 3],
                                                                                [3, 1],
                                                                                0,
                                                                                'torch.float32',
                                                                                'cpu'],
                                                                 'value_base': 0},
                                                                {'functional': True,
                                                                 'meta': [[2, 3],
                                                                          [3, 1],
                                                                          6,
                                                                          'torch.float32',
                                                                          'cpu'],
                                                                 'is_view': False,
                                                                 'base': None,
                                                                 'is_base': False,
                                                                 'multi_output_view': True,
                                                                 'data_mutation': False,
                                                                 'value_type': 'Tensor',
                                                                 'value': [[6.0, 7.0, 8.0],
                                                                           [9.0, 10.0, 11.0]],
                                                                 'value_meta': [[2, 3],
                                                                                [3, 1],
                                                                                6,
                                                                                'torch.float32',
                                                                                'cpu'],
                                                                 'value_base': 0}],
                                                      'inner': [['aten.split.Tensor',
                                                                 ['T'],
                                                                 True,
                                                                 True,
                                                                 True]],
                                                      'outer': 'aten.split.Tensor',
                                                      'schema': {'mutable': False,
                                                                 'view': True,
                                                                 'copy_op': 'aten.split_copy.Tensor'},
                                                      'inputs_still_functional': [True],
                                                      'inputs_data_mutation_before_after': [[False, False]]},
 '_op_dk(FK, cat, [P, P]) Functionalize included': {'functional': False,
                                                    'meta': [[4, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                    'is_view': False,
                                                    'base': None,
                                                    'inner': [['aten.cat.default',
                                                               [['T', 'T']],
                                                               True,
                                                               True,
                                                               False]],
                                                    'outer': 'aten.cat.default',
                                                    'schema': {'mutable': False,
                                                               'view': False,
                                                               'copy_op': None},
                                                    'inputs_still_functional': [False, False],
                                                    'inputs_data_mutation_before_after': [[None, None],
                                                                                          [None, None]]},
 '_op_dk(FK, cat, [W, P]) Functionalize included': {'functional': True,
                                                    'meta': [[4, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                    'is_view': False,
                                                    'base': None,
                                                    'is_base': True,
                                                    'multi_output_view': False,
                                                    'data_mutation': False,
                                                    'value_type': 'Tensor',
                                                    'value': [[0.0, 1.0, 2.0],
                                                              [3.0, 4.0, 5.0],
                                                              [1.0, 1.0, 1.0],
                                                              [1.0, 1.0, 1.0]],
                                                    'value_meta': [[4, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                    'value_base': None,
                                                    'inner': [['aten.cat.default',
                                                               [['T', 'T']],
                                                               True,
                                                               True,
                                                               False]],
                                                    'outer': 'aten.cat.default',
                                                    'schema': {'mutable': False,
                                                               'view': False,
                                                               'copy_op': None},
                                                    'inputs_still_functional': [True, False],
                                                    'inputs_data_mutation_before_after': [[False, False],
                                                                                          [None, None]]},
 '_op_dk(FK, lift_fresh, P) Functionalize included': {'functional': True,
                                                      'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                                                      'is_view': False,
                                                      'base': None,
                                                      'is_base': True,
                                                      'multi_output_view': False,
                                                      'data_mutation': False,
                                                      'value_type': 'Tensor',
                                                      'value': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                                                      'value_meta': [[2, 3],
                                                                     [3, 1],
                                                                     0,
                                                                     'torch.float32',
                                                                     'cpu'],
                                                      'value_base': None,
                                                      'inner': [['aten.lift_fresh.default',
                                                                 ['T'],
                                                                 True,
                                                                 True,
                                                                 True]],
                                                      'outer': 'aten.lift_fresh.default',
                                                      'schema': {'mutable': False,
                                                                 'view': True,
                                                                 'copy_op': 'aten.lift_fresh_copy.default'},
                                                      'inputs_still_functional': [False],
                                                      'inputs_data_mutation_before_after': [[None, None]]},
 'W.size()': {'seq': 'Size',
              'items': [{'value': 2}, {'value': 3}],
              'inner': [],
              'outer': None,
              'schema': None,
              'inputs_still_functional': [True],
              'inputs_data_mutation_before_after': [[False, False]]},
 'W.dim()': {'value': 2,
             'inner': [],
             'outer': None,
             'schema': None,
             'inputs_still_functional': [True],
             'inputs_data_mutation_before_after': [[False, False]]},
 'W.is_contiguous()': {'value': True,
                       'inner': [],
                       'outer': None,
                       'schema': None,
                       'inputs_still_functional': [True],
                       'inputs_data_mutation_before_after': [[False, False]]},
 'W0.item()': {'value': 3.5,
               'inner': [['aten._local_scalar_dense.default', ['T'], False, True, False]],
               'outer': 'aten._local_scalar_dense.default',
               'schema': {'mutable': False, 'view': False, 'copy_op': None},
               'inputs_still_functional': [True],
               'inputs_data_mutation_before_after': [[False, False]]},
 '_local_scalar_dense(W0)': {'value': 3.5,
                             'inner': [['aten._local_scalar_dense.default', ['T'], False, True, False]],
                             'outer': 'aten._local_scalar_dense.default',
                             'schema': {'mutable': False, 'view': False, 'copy_op': None},
                             'inputs_still_functional': [True],
                             'inputs_data_mutation_before_after': [[False, False]]},
 'is_same_size(W, W)': {'value': True,
                        'inner': [['aten.is_same_size.default', ['T', 'T'], False, True, False]],
                        'outer': 'aten.is_same_size.default',
                        'schema': {'mutable': False, 'view': False, 'copy_op': None},
                        'inputs_still_functional': [True, True],
                        'inputs_data_mutation_before_after': [[False, False], [False, False]]},
 'sym_size.int(Wt, 1)': {'value': 3,
                         'inner': [],
                         'outer': 'aten.sym_size.int',
                         'schema': {'mutable': False, 'view': False, 'copy_op': None},
                         'inputs_still_functional': [True],
                         'inputs_data_mutation_before_after': [[False, False]]},
 'size.default(Wt)': {'seq': 'list',
                      'items': [{'value': 2}, {'value': 3}],
                      'inner': [],
                      'outer': 'aten.size.default',
                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                      'inputs_still_functional': [True],
                      'inputs_data_mutation_before_after': [[False, False]]},
 'stride.default(Wt)': {'seq': 'list',
                        'items': [{'value': 1}, {'value': 2}],
                        'inner': [],
                        'outer': 'aten.stride.default',
                        'schema': {'mutable': False, 'view': False, 'copy_op': None},
                        'inputs_still_functional': [True],
                        'inputs_data_mutation_before_after': [[False, False]]},
 'sym_stride.int(Wt, 0)': {'value': 1,
                           'inner': [],
                           'outer': 'aten.sym_stride.int',
                           'schema': {'mutable': False, 'view': False, 'copy_op': None},
                           'inputs_still_functional': [True],
                           'inputs_data_mutation_before_after': [[False, False]]},
 'dim.default(Wt)': {'value': 2,
                     'inner': [],
                     'outer': 'aten.dim.default',
                     'schema': {'mutable': False, 'view': False, 'copy_op': None},
                     'inputs_still_functional': [True],
                     'inputs_data_mutation_before_after': [[False, False]]},
 'numel.default(Wt)': {'value': 6,
                       'inner': [],
                       'outer': 'aten.numel.default',
                       'schema': {'mutable': False, 'view': False, 'copy_op': None},
                       'inputs_still_functional': [True],
                       'inputs_data_mutation_before_after': [[False, False]]},
 'sym_numel.default(Wt)': {'value': 6,
                           'inner': [],
                           'outer': 'aten.sym_numel.default',
                           'schema': {'mutable': False, 'view': False, 'copy_op': None},
                           'inputs_still_functional': [True],
                           'inputs_data_mutation_before_after': [[False, False]]},
 'is_contiguous.default(Wt)': {'value': False,
                               'inner': [],
                               'outer': 'aten.is_contiguous.default',
                               'schema': {'mutable': False, 'view': False, 'copy_op': None},
                               'inputs_still_functional': [True],
                               'inputs_data_mutation_before_after': [[False, False]]},
 'storage_offset.default(W[1:])': {'value': 3,
                                   'inner': [],
                                   'outer': 'aten.storage_offset.default',
                                   'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                   'inputs_still_functional': [True],
                                   'inputs_data_mutation_before_after': [[False, False]]},
 'sym_storage_offset.default(W[1:])': {'value': 3,
                                       'inner': [],
                                       'outer': 'aten.sym_storage_offset.default',
                                       'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                       'inputs_still_functional': [True],
                                       'inputs_data_mutation_before_after': [[False, False]]},
 'is_non_overlapping_and_dense(Wt)': {'value': True,
                                      'inner': [],
                                      'outer': 'aten.is_non_overlapping_and_dense.default',
                                      'schema': {'mutable': False, 'view': False, 'copy_op': None},
                                      'inputs_still_functional': [True],
                                      'inputs_data_mutation_before_after': [[False, False]]},
 'permute(W(fake), [1, 0]) reapply=True': {'functional': True,
                                           'meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                                           'is_view': True,
                                           'base': 0,
                                           'is_base': False,
                                           'multi_output_view': False,
                                           'data_mutation': False,
                                           'value_type': 'FakeTensor',
                                           'value_meta': [[3, 2], [1, 3], 0, 'torch.float32', 'cpu'],
                                           'value_base': None,
                                           'inner': [['aten.permute.default', ['Tfake'], False, True, True]],
                                           'outer': 'aten.permute.default',
                                           'schema': {'mutable': False,
                                                      'view': True,
                                                      'copy_op': 'aten.permute_copy.default'},
                                           'inputs_still_functional': [True],
                                           'inputs_data_mutation_before_after': [[False, False]]},
 'add.Tensor(W(fake), 1)': {'functional': True,
                            'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                            'is_view': False,
                            'base': None,
                            'is_base': True,
                            'multi_output_view': False,
                            'data_mutation': False,
                            'value_type': 'FakeTensor',
                            'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                            'value_base': None,
                            'inner': [['aten.add.Tensor', ['Tfake'], False, True, False],
                                      ['prim.device.default', ['Tfake'], False, False, False]],
                            'outer': 'aten.add.Tensor',
                            'schema': {'mutable': False, 'view': False, 'copy_op': None},
                            'inputs_still_functional': [True],
                            'inputs_data_mutation_before_after': [[False, False]]},
 'add_(W, 1)': {'functional': True,
                'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                'is_view': False,
                'base': None,
                'is_base': True,
                'multi_output_view': False,
                'data_mutation': True,
                'value_type': 'Tensor',
                'value': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                'value_base': None,
                'inner': [['aten.add.Tensor', ['T'], False, True, False]],
                'outer': 'aten.add_.Tensor',
                'schema': {'mutable': True, 'view': False, 'copy_op': None},
                'inputs_still_functional': [True],
                'inputs_data_mutation_before_after': [[False, True]]},
 'add.out(W, W, out=W)': {'functional': True,
                          'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                          'is_view': False,
                          'base': None,
                          'is_base': True,
                          'multi_output_view': False,
                          'data_mutation': True,
                          'value_type': 'Tensor',
                          'value': [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]],
                          'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                          'value_base': None,
                          'inner': [['aten.add.Tensor', ['T', 'T'], False, True, False]],
                          'outer': 'aten.add.out',
                          'schema': {'mutable': True, 'view': False, 'copy_op': None},
                          'inputs_still_functional': [True, True, True],
                          'inputs_data_mutation_before_after': [[False, False],
                                                                [False, False],
                                                                [False, True]]},
 't_(W)': {'functional': True,
           'meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
           'is_view': False,
           'base': None,
           'is_base': False,
           'multi_output_view': False,
           'data_mutation': False,
           'value_type': 'Tensor',
           'value': [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
           'value_meta': [[3, 2], [2, 1], 0, 'torch.float32', 'cpu'],
           'value_base': None,
           'inner': [['aten.t_copy.default', ['T'], False, True, False]],
           'outer': 'aten.t_.default',
           'schema': {'mutable': True, 'view': False, 'copy_op': None},
           'inputs_still_functional': [True],
           'inputs_data_mutation_before_after': [[False, False]]},
 'copy_(W, P)': {'functional': True,
                 'meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                 'is_view': False,
                 'base': None,
                 'is_base': True,
                 'multi_output_view': False,
                 'data_mutation': True,
                 'value_type': 'Tensor',
                 'value': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                 'value_meta': [[2, 3], [3, 1], 0, 'torch.float32', 'cpu'],
                 'value_base': None,
                 'inner': [['aten.copy.default', ['T', 'T'], False, True, False]],
                 'outer': 'aten.copy_.default',
                 'schema': {'mutable': True, 'view': False, 'copy_op': None},
                 'inputs_still_functional': [True, False],
                 'inputs_data_mutation_before_after': [[False, True], [None, None]]},
 'storage': {'storage nbytes (base)': 24,
             'storage nbytes (view)': 24,
             'storage nbytes (fresh output)': 24,
             'storage nbytes (wrapper of a slice)': 24,
             'storage device': 'cpu',
             'storage data_ptr': ['raises',
                                  'RuntimeError',
                                  'Attempted to access the data pointer on an invalid python storage.'],
             'storage base == view': True,
             'storage base == other wrapper': False,
             'storage base == fresh output': False,
             'storage base == its value': False,
             'storage twice same': True,
             'unsafe_set(W, W(zeros))': {'returns': None,
                                         'meta unchanged': True,
                                         'dst still functional': True,
                                         "storage now src's": True,
                                         'value is old value': True,
                                         'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                         'is_base': True,
                                         'storage nbytes': 24},
             'unsafe_set(view(W), W)': {'returns': None,
                                        'meta unchanged': True,
                                        'dst still functional': True,
                                        "storage now src's": True,
                                        'value is old value': True,
                                        'value': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                                        'is_base': False,
                                        'storage nbytes': 24},
             'unsafe_set(W, W(zeros(3)))': {'returns': None,
                                            'meta unchanged': True,
                                            'dst still functional': True,
                                            "storage now src's": True,
                                            'value is old value': True,
                                            'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                            'is_base': True,
                                            'storage nbytes': 12},
             'unsafe_set(W, P)': {'returns': None,
                                  'meta unchanged': True,
                                  'dst still functional': True,
                                  "storage now src's": True,
                                  'value is old value': True,
                                  'value': [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                                  'is_base': True,
                                  'storage nbytes': 24},
             'unsafe_set(P, W)': {'returns': None,
                                  'meta unchanged': True,
                                  'dst still functional': False,
                                  "storage now src's": True},
             'unsafe_set(P, P)': {'returns': None,
                                  'meta unchanged': True,
                                  'dst still functional': False,
                                  "storage now src's": True},
             'unsafe_set(W, 1)': ['raises',
                                  'TypeError',
                                  '_functionalize_unsafe_set(): incompatible function arguments. The '
                                  'following argument types are supported:\n'
                                  '    1. (arg0: torch.Tensor, arg1: torch.Tensor) -> None\n'
                                  '\n'
                                  'Invoked with: _to_functional_tensor(tensor([[0., 1., 2.],\n'
                                  '        [3., 4., 5.]])), 1']}}

_cache = {}


def _shim():
    if "shim" not in _cache:
        _cache["shim"] = _run(_PROBE, True)
        assert _cache["shim"]["is_shim"], "the shim-side probe got upstream torch"
    return _cache["shim"]["out"]


def _upstream():
    if "up" not in _cache:
        _cache["up"] = _run(_PROBE, False) if _have_upstream() else None
    return None if _cache["up"] is None else _cache["up"]["out"]


def _reapply(up_row):
    """The `reapply_views` TLS the kernel's inner op ran with upstream."""
    for entry in up_row.get("inner", []):
        if not entry[0].startswith("prim."):
            return entry[4]
    return None


def _decomposed_upstream(up_row):
    """Upstream's inner ops are not the op the row called (nor its `_copy`):
    upstream decomposed it above Functionalize (CompositeImplicitAutograd)."""
    outer = up_row.get("outer")
    if outer is None:
        return False
    schema = up_row.get("schema") or {}
    allowed = {outer, schema.get("copy_op")}
    ops = [e[0] for e in up_row.get("inner", []) if not e[0].startswith("prim.")]
    return bool(ops) and any(op not in allowed for op in ops)


def _expected_refusal(name, up_row, shim_row):
    """Why the shim must refuse this row, or None if it must compute it.

    Derived from schema properties and the shim's own kernel table, never from
    a hand-picked list of names."""
    schema = (shim_row or {}).get("schema") or {}
    if schema.get("mutable"):
        return ("mutation", "functionalization step 3")
    reapply = _reapply(up_row)
    if schema.get("view") and reapply is False and not schema.get("copy_implemented"):
        return ("no _copy kernel", schema.get("copy_op") or "_copy")
    # The shim has no kernel for the op, and upstream's kernel dispatched
    # exactly that op below itself (so the shim's must too). Not for an op
    # upstream answers without dispatching (the metadata queries: no inner op).
    inner = [e[0] for e in up_row.get("inner", []) if not e[0].startswith("prim.")]
    if up_row.get("outer") and schema and not schema.get("implemented") and inner == [up_row["outer"]]:
        return ("no kernel", up_row["outer"])
    return None


def _comparable(row, decomposed):
    """A row with the two documented exceptions removed (see the docstring)."""
    row = json.loads(json.dumps(row))
    row.pop("schema", None)
    inner = [e for e in row.pop("inner", []) if not e[0].startswith("prim.")]
    if not decomposed:
        # op, argument kinds, Functionalize included, reapply_views.
        row["inner"] = [[e[0], e[1], e[2], e[4]] for e in inner]
    return row


def test_the_recorded_table_is_upstreams_answer():
    """Re-derive `UPSTREAM_OPS` from upstream on every run, so a table that
    stops being upstream's turns this red instead of grading the shim against
    a wish."""
    up = _upstream()
    if up is None:
        _skip.skip("upstream torch not importable without the vendored tree")
        return
    drift = {k: (v, up.get(k)) for k, v in UPSTREAM_OPS.items() if up.get(k) != v}
    assert not drift, f"upstream no longer answers the recorded table: {sorted(drift)}"
    assert set(up) == set(UPSTREAM_OPS), set(up) ^ set(UPSTREAM_OPS)
    # Upstream computes every operator row: the refusals below are the shim's.
    raised = [k for k, v in up.items() if k != "storage" and "raises" in v]
    assert not raised, raised


def test_non_mutating_and_view_ops_agree_with_upstream():
    """Every row the shim must compute (not `_expected_refusal`) agrees with
    upstream field by field: functional-ness, layout, `_is_view`, `_base`,
    base-ness, multi-output-ness, the unwrapped value and its aliasing, the
    inputs' state, and the inner ops `Log` saw with the TLS they ran under."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    shim = _shim()
    assert set(shim) == set(UPSTREAM_OPS), set(shim) ^ set(UPSTREAM_OPS)
    computed, wrong, reported = 0, {}, {}
    for name, up_row in UPSTREAM_OPS.items():
        if name == "storage":
            continue
        got = shim[name]
        if _expected_refusal(name, up_row, got) is not None:
            continue
        computed += 1
        if "raises" in got:
            wrong[name] = ["raised", got["raises"], got["message"]]
            continue
        decomposed = _decomposed_upstream(up_row)
        if decomposed:
            reported[name] = [e[0] for e in got.get("inner", [])]
        a, b = _comparable(up_row, decomposed), _comparable(got, decomposed)
        if a != b:
            wrong[name] = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    print(f"FUNCTIONAL OPS: {computed - len(wrong)} of {computed} computed rows agree; "
          f"inner ops of {len(reported)} upstream-decomposed rows, reported: {reported}")
    assert not wrong, f"{len(wrong)} of {computed} rows differ (upstream, shim): {wrong}"
    assert computed >= 60, computed


def test_the_expected_refusals_refuse_by_name():
    """The rows `_expected_refusal` derives must raise `NotImplementedError`
    naming what is missing: a mutation names functionalization step 3, a view
    with `reapply_views` False names the `_copy` op it needed, an op the shim
    has no kernel for names that op. A refused row must not have dispatched
    anything below the kernel (`Log` saw nothing), or the kernel ran the op
    before refusing it."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    shim = _shim()
    expected, bad = 0, {}
    for name, up_row in UPSTREAM_OPS.items():
        if name == "storage":
            continue
        got = shim[name]
        why = _expected_refusal(name, up_row, got)
        if why is None:
            continue
        expected += 1
        kind, needle = why
        if got.get("raises") != "NotImplementedError" or needle not in got.get("message", ""):
            bad[name] = [kind, needle, got.get("raises"), got.get("message", "")[:300]]
        elif kind == "mutation" and got.get("inner"):
            bad[name] = [kind, "dispatched below the kernel before refusing", got["inner"]]
    print(f"FUNCTIONAL OPS: {expected - len(bad)} of {expected} derived refusals refuse by name")
    assert not bad, bad
    # The derivation must select something of each kind, or it checks nothing.
    kinds = {(_expected_refusal(n, r, shim[n]) or ("",))[0] for n, r in UPSTREAM_OPS.items() if n != "storage"}
    assert {"mutation", "no _copy kernel"} <= kinds, kinds


def test_the_storage_rows_agree_and_a_plain_side_refuses():
    """`untyped_storage()` of a wrapper (size, device, shared by a base and its
    views, refused `data_ptr`) and `_functionalize_unsafe_set` (only the
    storage moves). Rows with a plain side of `unsafe_set` must refuse by name
    on the shim; upstream answers them (`UPSTREAM_OPS["storage"]`)."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    up, got = UPSTREAM_OPS["storage"], _shim()["storage"]
    assert set(up) == set(got), set(up) ^ set(got)
    plain_side = {"unsafe_set(W, P)", "unsafe_set(P, W)", "unsafe_set(P, P)"}
    wrong = {}
    for name, value in up.items():
        mine = got[name]
        if name in plain_side:
            if not (isinstance(mine, list) and mine[:2] == ["raises", "NotImplementedError"]
                    and "_functionalize_unsafe_set" in mine[2]):
                wrong[name] = mine
            continue
        if isinstance(value, list) and value[:1] == ["raises"]:
            # Same exception type; the text is upstream's C++ wording.
            if not (isinstance(mine, list) and mine[:2] == value[:2]):
                wrong[name] = (value, mine)
            continue
        if mine != value:
            wrong[name] = (value, mine)
    assert not wrong, wrong


_COUNTER_PROBE = r'''
import json
import torch
assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"  # AGENTS.md 15.2
aten = torch.ops.aten
FK = torch._C.DispatchKey.Functionalize
C = torch._C._shim_functionalize_counters
out = {}
def delta(name, fn):
    before = C()
    try:
        fn()
        err = None
    except Exception as exc:
        err = type(exc).__name__
    after = C()
    out[name] = [{k: after[k] - before[k] for k in after if after[k] != before[k]}, err]
torch._functionalize_enable_reapply_views(True)
f = torch._to_functional_tensor(torch.arange(6.).reshape(2, 3))
delta("pure: add.Tensor(W, 1)", lambda: aten.add.Tensor(f, 1))
delta("view: permute(W)", lambda: aten.permute.default(f, [1, 0]))
delta("multi-output view: split(W, 1)", lambda: aten.split.Tensor(f, 1))
delta("plain: add.Tensor(P, 1), no wrapper", lambda: aten.add.Tensor(torch.ones(2), 1))
delta("factory: _op_dk(FK, ones)", lambda: aten.ones.default._op_dk(FK, [2]))
delta("plain under _op_dk: add(P, P)", lambda: aten.add.Tensor._op_dk(FK, torch.ones(2), torch.ones(2)))
delta("refused: add_(W, 1)", lambda: f.add_(1))
out["keys"] = sorted(C())
print(json.dumps(out))
'''


def test_the_kernel_counts_what_it_functionalized():
    """AGENTS.md 13.1/16: values alone cannot show which path computed them,
    so the door's kernel counts what it did (`_shim_functionalize_counters`).
    Each call moves exactly the counter of its kind and the total, a plain op
    with no wrapper alive in the call moves nothing, and a refused mutation
    moves only `refused`. If the kernel were bypassed (a wrapper computed by
    some other path), the computing rows would read `{}`."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    env = dict(os.environ, PYTHONPATH=str(VENDOR), TORCH_USE_RTLD_GLOBAL="1")
    proc = subprocess.run([sys.executable, "-c", _COUNTER_PROBE], capture_output=True,
                          text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    want = {
        "pure: add.Tensor(W, 1)": [{"functionalized": 1, "pure": 1}, None],
        "view: permute(W)": [{"functionalized": 1, "view": 1}, None],
        "multi-output view: split(W, 1)": [{"functionalized": 1, "view": 1}, None],
        "plain: add.Tensor(P, 1), no wrapper": [{}, None],
        "factory: _op_dk(FK, ones)": [{"functionalized": 1, "factory": 1}, None],
        "plain under _op_dk: add(P, P)": [{"passthrough": 1}, None],
        "refused: add_(W, 1)": [{"refused": 1}, "NotImplementedError"],
    }
    wrong = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
    assert not wrong, wrong
    assert got["keys"] == sorted(["functionalized", "pure", "view", "factory", "lift",
                                  "metadata", "passthrough", "refused"]), got["keys"]


_STEPS_PROBE = r'''
import json
import torch
assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"  # AGENTS.md 15.2
aten = torch.ops.aten
steps = torch._C._shim_functional_view_steps
torch._functionalize_enable_reapply_views(True)
f = torch._to_functional_tensor(torch.arange(6.).reshape(2, 3))
p = aten.permute.default(f, [1, 0])
s = aten.slice.Tensor(p, 0, 1, 3)
parts = aten.split.Tensor(f, 1)
c = aten.add.Tensor(f, 1)
def canon(v):
    if isinstance(v, torch.Tensor):
        return "Tensor"
    if isinstance(v, (list, tuple)):
        return [canon(x) for x in v]
    return v
out = {
    "base": canon(steps(f)),
    "permute": canon(steps(p)),
    "slice of permute": canon(steps(s)),
    "split[1]": canon(steps(parts[1])),
    "fresh": canon(steps(c)),
}
print(json.dumps(out))
'''


def test_view_steps_are_recorded_for_replay():
    """Shim-only: a view wrapper records the chain of view steps from its
    storage's base (op, non-tensor arguments with the viewed tensor as None,
    the output index of a multi-output view, `reapply_views`), which is what
    steps 3 and 4 replay; a base and a fresh output record none. This is what
    `_is_functional_tensor_base` reads (`steps == []`), so a kernel that shared
    the storage but recorded nothing would also turn the agreement test red."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    env = dict(os.environ, PYTHONPATH=str(VENDOR), TORCH_USE_RTLD_GLOBAL="1")
    proc = subprocess.run([sys.executable, "-c", _STEPS_PROBE], capture_output=True,
                          text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["base"] == [] and got["fresh"] == [], got
    assert [s[0] for s in got["permute"]] == ["aten.permute.default"], got
    assert [s[0] for s in got["slice of permute"]] == ["aten.permute.default", "aten.slice.Tensor"], got
    assert got["split[1]"][-1][0] == "aten.split.Tensor" and got["split[1]"][-1][3] == 1, got
    # The viewed tensor is not kept in a step (it is the storage's base, or
    # the previous step's output); only its non-tensor arguments are.
    assert "Tensor" not in json.dumps(got), got
    assert all(step[4] is True for chain in got.values() for step in chain), got


if __name__ == "__main__":
    sys.exit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        suite="test_functional_ops",
    ) else 0)
