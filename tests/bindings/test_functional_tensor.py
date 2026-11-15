"""Issue #329, step 1 of 4: the functional tensor wrapper and its state queries.

Direction: leader decision 2026-11-15, agent proposal; not owner-confirmed.
Functionalization on the shim is a real wrapper, `Repr::Functional` in
`tensor.rs` (payload in `functional.rs`), with a Functionalize kernel at the
one door in steps 2 to 4 and upstream's Python `FunctionalTensor` /
`FunctionalTensorMode` running unchanged on top.

One probe, `_PROBE`, runs in a subprocess on both sides: on the shim
(`PYTHONPATH` at the vendored tree) and on upstream torch 2.13.0
(`sys.executable` without it). It asserts AGENTS.md section 15.2's shim check
for the side it was asked to run on. It records three tables:

  * `out`: what step 1 answers. Upstream's answers are `UPSTREAM_TABLE`
    below, measured with this probe on 2026-11-15, and re-measured against
    upstream on every run so the table cannot drift into a wish;
  * `pending`: what needs the Functionalize kernel (an operator on a wrapper,
    the mutation and view machinery). Upstream's answers are
    `UPSTREAM_PENDING`; on the shim every row must refuse by name
    (`NotImplementedError`, naming #329), never compute;
  * `after`: the state a refused in-place op leaves behind, which on the shim
    must be exactly the state before it.

Measured on upstream and recorded here because they decide the design: an op
on a wrapper functionalizes with no mode and no functionalize TLS active (the
Functionalize key is on the tensor), so `f + 1` is functional and `f.add_(1)`
sets `has_data_mutation` and leaves the wrapped tensor untouched; a
`TorchDispatchMode` never sees the wrapper (Functionalize sits above Python);
`_op_dk(Functionalize, ...)` on plain tensors is the plain answer, but with
Functionalize in the TLS include set (inside `FunctionalTensorMode`'s
`_ForceDispatchKeyGuard`) even a factory op's output is functional. Two
upstream calls segfault and are not in the probe: `_op_dk(DispatchKey.CPU,
wrapper, wrapper)` and an op on a wrapper with Functionalize TLS-excluded
(exit 139 on 2.13.0).

What these tests cannot find: anything about the Functionalize kernel itself
(steps 2 to 4 have their own tests); whether AOT autograd's metadata pass gets
further on the shim (it stops at the first `_op_dk(Functionalize, ...)`, by
design, until step 2); a Rust caller of `aten_dispatch` that bypasses the
Python door (the tape, capture replay) handing a kernel a wrapper: the
`Where::of`/`visit_for_device` refusal covers that path, and no Python-level
call reaches it in step 1, so no test here can show it failing; `repr` of a
meta or fake wrapper (upstream adds suffixes this probe does not compare);
`backward()` through a tensor `_mirror_autograd_meta_to` marked (upstream
raises "Cannot backprop through mirrored meta"; the shim's tape was not
checked against it).
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

IS_SHIM = hasattr(torch._C, "_aten_implemented")
WANT_SHIM = sys.argv[1] == "shim"
# AGENTS.md 15.2: a probe meant for the shim must not silently import upstream.
assert IS_SHIM == WANT_SHIM, ("shim" if IS_SHIM else "upstream", "probe asked for", sys.argv[1])

FK = torch._C.DispatchKey.Functionalize


def canon(v):
    if isinstance(v, torch.Tensor):
        return "Tensor"
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, torch.Size):
        return ["Size"] + list(v)
    if isinstance(v, (tuple, list)):
        return [canon(x) for x in v]
    if isinstance(v, (torch.dtype, torch.device, torch.layout)):
        return str(v)
    return "object " + type(v).__name__


out = {}
pending = {}


def r(name, fn):
    try:
        v = fn()
    except Exception as exc:  # noqa: BLE001 -- the exception type is the answer
        out[name] = "raises " + type(exc).__name__
        return
    out[name] = canon(v)


def p(name, fn):
    """A row whose upstream answer needs #329 steps 2-4: the shim must refuse."""
    try:
        v = fn()
    except Exception as exc:  # noqa: BLE001
        pending[name] = ["raises", type(exc).__name__, str(exc)[:600]]
        return
    pending[name] = ["value", canon(v)]


def mk(t=None):
    if t is None:
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return torch._to_functional_tensor(t)


t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
f = mk(t)

# -- wrap and unwrap ---------------------------------------------------------
r("is_functional(wrapper)", lambda: torch._is_functional_tensor(f))
r("is_functional(plain)", lambda: torch._is_functional_tensor(t))
r("is_functional(1)", lambda: torch._is_functional_tensor(1))
r("functorch.is_functionaltensor(wrapper)", lambda: torch._C._functorch.is_functionaltensor(f))
r("functorch.is_functionaltensor(plain)", lambda: torch._C._functorch.is_functionaltensor(t))
r("functorch.maybe_get_level(wrapper)", lambda: torch._C._functorch.maybe_get_level(f))
r("type(wrapper) is Tensor", lambda: type(f) is torch.Tensor)
r("wrapper is plain", lambda: f is t)
r("unwrap is plain", lambda: torch._from_functional_tensor(f) is t)
r("unwrap twice is one object", lambda: torch._from_functional_tensor(f) is torch._from_functional_tensor(f))
r("unwrap(plain)", lambda: torch._from_functional_tensor(t))
r("unwrap(1)", lambda: torch._from_functional_tensor(1))
r("wrap(wrapper)", lambda: torch._to_functional_tensor(f))
r("wrap(1)", lambda: torch._to_functional_tensor(1))
r("wrap(None)", lambda: torch._to_functional_tensor(None))
r("wrap(list)", lambda: torch._to_functional_tensor([1.0]))
r("wrap twice gives two wrappers", lambda: torch._to_functional_tensor(t) is torch._to_functional_tensor(t))
r("torch._C has no _to_functional_tensor", lambda: hasattr(torch._C, "_to_functional_tensor"))

# -- the wrapper's metadata, with no mode and no functionalize TLS -------------
for name, fn in [
    ("shape", lambda: f.shape), ("dtype", lambda: f.dtype), ("device", lambda: f.device),
    ("dim()", lambda: f.dim()), ("ndim", lambda: f.ndim), ("numel()", lambda: f.numel()),
    ("size()", lambda: f.size()), ("size(1)", lambda: f.size(1)), ("stride()", lambda: f.stride()),
    ("stride(0)", lambda: f.stride(0)), ("storage_offset()", lambda: f.storage_offset()),
    ("is_contiguous()", lambda: f.is_contiguous()), ("element_size()", lambda: f.element_size()),
    ("requires_grad", lambda: f.requires_grad), ("is_leaf", lambda: f.is_leaf),
    ("grad_fn", lambda: f.grad_fn), ("grad", lambda: f.grad), ("layout", lambda: f.layout),
    ("is_meta", lambda: f.is_meta), ("is_cpu", lambda: f.is_cpu), ("is_cuda", lambda: f.is_cuda),
    ("_is_view()", lambda: f._is_view()), ("_base", lambda: f._base),
    ("data_ptr()", lambda: f.data_ptr()), ("is_floating_point()", lambda: f.is_floating_point()),
    ("is_complex()", lambda: f.is_complex()), ("is_quantized", lambda: f.is_quantized),
    ("is_sparse", lambda: f.is_sparse), ("is_nested", lambda: f.is_nested),
    ("is_neg()", lambda: f.is_neg()), ("_is_zerotensor()", lambda: f._is_zerotensor()),
    ("tolist()", lambda: f.tolist()), ("has_storage", lambda: torch._C._has_storage(f)),
]:
    r("wrapper." + name, fn)
r("wrapper(empty).tolist()", lambda: mk(torch.ones(0)).tolist())
r("wrapper(int64).dtype", lambda: mk(torch.tensor([1, 2])).dtype)
r("wrapper(bool).dtype", lambda: mk(torch.tensor([True])).dtype)
r("wrapper(0-dim).shape", lambda: mk(torch.tensor(2.5)).shape)
r("wrapper(t()).stride()", lambda: mk(torch.ones(2, 3).t()).stride())
r("wrapper(t()).is_contiguous()", lambda: mk(torch.ones(2, 3).t()).is_contiguous())
r("wrapper(view).storage_offset()", lambda: mk(torch.ones(2, 3)[1:]).storage_offset())
r("wrapper(view)._is_view()", lambda: mk(torch.ones(2, 3)[1:])._is_view())
r("wrapper(view)._base", lambda: mk(torch.ones(2, 3)[1:])._base)
r("repr(wrapper) wraps repr(value)",
  lambda: repr(f) == "_to_functional_tensor(" + repr(t) + ")")
r("repr(wrapper(int64)) wraps repr(value)",
  lambda: (lambda v: repr(mk(v)) == "_to_functional_tensor(" + repr(v) + ")")(torch.tensor([1, 2])))

rg = torch.ones(3, requires_grad=True)
frg = mk(rg)
r("wrapper(requires_grad).requires_grad", lambda: frg.requires_grad)
r("wrapper(requires_grad).is_leaf", lambda: frg.is_leaf)
r("unwrap(wrapper(requires_grad)) is value", lambda: torch._from_functional_tensor(frg) is rg)

m = torch.empty(2, 3, device="meta")
fm = mk(m)
r("wrapper(meta).device", lambda: fm.device)
r("wrapper(meta).is_meta", lambda: fm.is_meta)
r("wrapper(meta).shape", lambda: fm.shape)
r("wrapper(meta).data_ptr()", lambda: fm.data_ptr())
r("unwrap(wrapper(meta)) is value", lambda: torch._from_functional_tensor(fm) is m)

from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode():
    fake = torch.empty(2, 3)
ff = mk(fake)
r("wrapper(fake) type", lambda: type(ff).__name__)
r("wrapper(fake).device", lambda: ff.device)
r("wrapper(fake).is_meta", lambda: ff.is_meta)
r("wrapper(fake).shape", lambda: ff.shape)
r("wrapper(fake).data_ptr()", lambda: ff.data_ptr())
r("unwrap(wrapper(fake)) is value", lambda: torch._from_functional_tensor(ff) is fake)
r("unwrap(wrapper(fake)) type", lambda: type(torch._from_functional_tensor(ff)).__name__)

# -- dispatch keys: the wrapped value's, without Python keys, plus Functionalize
DROP = {"Python", "PythonTLSSnapshot"}


def keynames(x):
    ks = torch._C._dispatch_keys(x)
    return sorted(n for n, k in torch._C.DispatchKey.__members__.items() if ks.has(k))


def keys_rule(w, v):
    return sorted(set(keynames(w)) - {"Functionalize"}) == sorted(set(keynames(v)) - DROP)


r("keys(wrapper) has Functionalize", lambda: "Functionalize" in keynames(f))
r("keys(plain) has Functionalize", lambda: "Functionalize" in keynames(t))
r("keys(wrapper) = keys(value) + Functionalize", lambda: keys_rule(f, t))
r("keys(wrapper(meta)) = keys(value) + Functionalize", lambda: keys_rule(fm, m))
r("keys(wrapper(fake)) = keys(value) - Python + Functionalize", lambda: keys_rule(ff, fake))

# -- the state queries on a fresh wrapper ------------------------------------
QUERIES = [
    "_is_functional_tensor_base",
    "_functionalize_has_data_mutation",
    "_functionalize_has_metadata_mutation",
    "_functionalize_was_storage_changed",
    "_functionalize_was_inductor_storage_resized",
    "_functionalize_are_all_mutations_hidden_from_autograd",
    "_functionalize_are_all_mutations_under_no_grad_or_inference_mode",
    "_functionalize_is_multi_output_view",
    "_functionalize_is_symbolic",
    "_functionalize_mutation_counter",
    "_functionalize_storage_changed_counter",
    "_functionalize_inductor_storage_resized_counter",
    "_functionalize_sync",
]
for q in QUERIES:
    fn = getattr(torch, q)
    r(q + "(wrapper)", lambda fn=fn: fn(mk()))
    r(q + "(plain)", lambda fn=fn: fn(t))
    r(q + "(1)", lambda fn=fn: fn(1))
r("_is_functional_tensor_base(wrapper(meta))", lambda: torch._is_functional_tensor_base(fm))
r("_functionalize_has_data_mutation(wrapper(fake))", lambda: torch._functionalize_has_data_mutation(ff))


def marked():
    w = mk()
    torch._functionalize_mark_storage_changed(w)
    return [torch._functionalize_was_storage_changed(w),
            torch._functionalize_storage_changed_counter(w),
            torch._functionalize_has_data_mutation(w)]


r("mark_storage_changed then (changed, counter, data_mutation)", marked)
r("mark_storage_changed(plain)", lambda: torch._functionalize_mark_storage_changed(t))

# -- TLS ---------------------------------------------------------------------
TLS = torch._C._functionalization_reapply_views_tls
r("reapply_views_tls at start", TLS)
r("enable_reapply_views(True)", lambda: torch._functionalize_enable_reapply_views(True))
r("reapply_views_tls after True", TLS)
r("enable_reapply_views(False)", lambda: torch._functionalize_enable_reapply_views(False))
r("reapply_views_tls after False", TLS)
r("enable_reapply_views(1)", lambda: torch._functionalize_enable_reapply_views(1))
r("reapply_views_tls after 1", TLS)
r("enable_reapply_views()", lambda: torch._functionalize_enable_reapply_views())
r("reapply_views_tls after ()", TLS)
r("enable_reapply_views('x')", lambda: torch._functionalize_enable_reapply_views("x"))
torch._functionalize_enable_reapply_views(True)
r("_disable_functionalization()", lambda: torch._disable_functionalization())
r("reapply_views_tls after disable", TLS)
r("Functionalize included after disable", lambda: torch._C._dispatch_tls_is_dispatch_key_included(FK))
r("_disable_functionalization() twice", lambda: torch._disable_functionalization())
r("Functionalize included at rest", lambda: torch._C._dispatch_tls_is_dispatch_key_included(FK))
r("Functionalize excluded at rest", lambda: torch._C._dispatch_tls_is_dispatch_key_excluded(FK))

# -- _mirror_autograd_meta_to -------------------------------------------------


def mirror_leaf():
    dst = mk()
    torch._mirror_autograd_meta_to(torch.ones(3, requires_grad=True), dst)
    return [dst.requires_grad, dst.is_leaf, type(dst.grad_fn).__name__,
            torch._from_functional_tensor(dst).requires_grad]


def mirror_plain():
    dst = torch.ones(3)
    torch._mirror_autograd_meta_to(torch.ones(3), dst)
    return [dst.requires_grad, dst.is_leaf]


r("mirror(requires_grad leaf -> wrapper)", mirror_leaf)
r("mirror(plain -> plain)", mirror_plain)
r("mirror(1, plain)", lambda: torch._mirror_autograd_meta_to(1, torch.ones(1)))

# -- the Functionalize key on plain tensors -----------------------------------
r("_op_dk(Functionalize, plain, plain)", lambda: (lambda o: [torch._is_functional_tensor(o), o.tolist()])(
    torch.ops.aten.add.Tensor._op_dk(FK, torch.ones(2), torch.ones(2))))

# -- what needs the Functionalize kernel (steps 2 to 4): upstream answers, the shim refuses


after = {}


def refused_add_():
    w = mk()
    try:
        w.add_(1)
    finally:
        # Read whether or not `add_` raised: on the shim it must have left
        # the wrapped value and the record exactly as they were.
        after["add_"] = [torch._from_functional_tensor(w).tolist(),
                         torch._functionalize_has_data_mutation(w),
                         torch._functionalize_mutation_counter(w)]
    return [torch._from_functional_tensor(w).tolist(), torch._functionalize_has_data_mutation(w),
            torch._functionalize_mutation_counter(w)]


p("wrapper + 1", lambda: (lambda o: [torch._is_functional_tensor(o), torch._from_functional_tensor(o).tolist()])(mk() + 1))
p("wrapper.add_(1)", refused_add_)
p("wrapper.view(-1)", lambda: torch._is_functional_tensor(mk().view(-1)))
p("wrapper.clone()", lambda: torch._is_functional_tensor(mk().clone()))
p("wrapper.t_()", lambda: torch._functionalize_has_metadata_mutation(mk().t_()))
p("wrapper.detach()", lambda: torch._is_functional_tensor(mk().detach()))
p("wrapper(0-dim).item()", lambda: mk(torch.tensor(3.5)).item())
p("wrapper + plain", lambda: torch._is_functional_tensor(mk() + torch.ones(2, 3)))
p("torch.mm(wrapper, wrapper.t())", lambda: torch._is_functional_tensor(torch.mm(mk(), mk().t())))
p("torch.cat([wrapper, plain])", lambda: torch._is_functional_tensor(torch.cat([mk(), torch.ones(2, 3)])))
p("_op_dk(Functionalize, wrapper, wrapper)",
  lambda: torch._is_functional_tensor(torch.ops.aten.add.Tensor._op_dk(FK, mk(), mk())))
p("wrapper.untyped_storage().nbytes()", lambda: mk().untyped_storage().nbytes())
p("_enable_functionalization()", lambda: (torch._enable_functionalization(reapply_views=True),
                                         torch._disable_functionalization())[1])
p("_functionalize_replace", lambda: torch._functionalize_replace(mk(), torch.zeros(2, 3)))
p("_functionalize_commit_update", lambda: torch._functionalize_commit_update(mk()))
p("_functionalize_unsafe_set", lambda: torch._functionalize_unsafe_set(mk(), mk()))
p("_functionalize_mark_mutation_hidden_from_autograd",
  lambda: torch._functionalize_mark_mutation_hidden_from_autograd(mk()))
p("_freeze_functional_tensor", lambda: torch._freeze_functional_tensor(mk()))
p("_functionalize_get_storage_size(wrapper, True)", lambda: torch._functionalize_get_storage_size(mk(), True))


def force_guard_factory():
    inc = torch._C._dispatch_tls_local_include_set() | torch._C.DispatchKeySet(FK)
    exc = torch._C._dispatch_tls_local_exclude_set()
    with torch._C._ForceDispatchKeyGuard(inc, exc):
        try:
            o = torch.ops.aten.ones.default._op_dk(FK, [2])
        finally:
            torch._disable_functionalization()
    return torch._is_functional_tensor(o)


p("_op_dk(Functionalize, factory) with Functionalize included", force_guard_factory)

# -- a mode does not see a wrapper: Functionalize sits above Python -----------
from torch.utils._python_dispatch import TorchDispatchMode

seen = []


class Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        seen.append([str(func), any(isinstance(a, torch.Tensor) and torch._is_functional_tensor(a)
                                    for a in args)])
        return func(*args, **(kwargs or {}))


w = mk()
with Log():
    p("wrapper * 2 under a logging mode", lambda: torch._is_functional_tensor(w * 2))
out["a mode saw a functional argument"] = any(hit for _, hit in seen)

del f, frg, fm, ff, w
gc.collect()
if IS_SHIM:
    out["shim live wrappers after del"] = torch._C._shim_functional_live()
print(json.dumps({"is_shim": IS_SHIM, "out": out, "pending": pending, "after": after}))
'''

#: Upstream's answers, torch 2.13.0 (`.caches/spike-venv`), measured with
#: `_PROBE` on 2026-11-15. "raises X" is the exception type upstream raises.
UPSTREAM_TABLE = {'is_functional(wrapper)': True,
 'is_functional(plain)': False,
 'is_functional(1)': 'raises TypeError',
 'functorch.is_functionaltensor(wrapper)': True,
 'functorch.is_functionaltensor(plain)': False,
 'functorch.maybe_get_level(wrapper)': -1,
 'type(wrapper) is Tensor': True,
 'wrapper is plain': False,
 'unwrap is plain': True,
 'unwrap twice is one object': True,
 'unwrap(plain)': 'raises RuntimeError',
 'unwrap(1)': 'raises TypeError',
 'wrap(wrapper)': 'raises RuntimeError',
 'wrap(1)': 'raises TypeError',
 'wrap(None)': 'raises TypeError',
 'wrap(list)': 'raises TypeError',
 'wrap twice gives two wrappers': False,
 'torch._C has no _to_functional_tensor': False,
 'wrapper.shape': ['Size', 2, 3],
 'wrapper.dtype': 'torch.float32',
 'wrapper.device': 'cpu',
 'wrapper.dim()': 2,
 'wrapper.ndim': 2,
 'wrapper.numel()': 6,
 'wrapper.size()': ['Size', 2, 3],
 'wrapper.size(1)': 3,
 'wrapper.stride()': [3, 1],
 'wrapper.stride(0)': 3,
 'wrapper.storage_offset()': 0,
 'wrapper.is_contiguous()': True,
 'wrapper.element_size()': 4,
 'wrapper.requires_grad': False,
 'wrapper.is_leaf': True,
 'wrapper.grad_fn': None,
 'wrapper.grad': None,
 'wrapper.layout': 'torch.strided',
 'wrapper.is_meta': False,
 'wrapper.is_cpu': True,
 'wrapper.is_cuda': False,
 'wrapper._is_view()': False,
 'wrapper._base': None,
 'wrapper.data_ptr()': 0,
 'wrapper.is_floating_point()': True,
 'wrapper.is_complex()': False,
 'wrapper.is_quantized': False,
 'wrapper.is_sparse': False,
 'wrapper.is_nested': False,
 'wrapper.is_neg()': False,
 'wrapper._is_zerotensor()': False,
 'wrapper.tolist()': 'raises RuntimeError',
 'wrapper.has_storage': True,
 'wrapper(empty).tolist()': [],
 'wrapper(int64).dtype': 'torch.int64',
 'wrapper(bool).dtype': 'torch.bool',
 'wrapper(0-dim).shape': ['Size'],
 'wrapper(t()).stride()': [1, 3],
 'wrapper(t()).is_contiguous()': False,
 'wrapper(view).storage_offset()': 3,
 'wrapper(view)._is_view()': False,
 'wrapper(view)._base': None,
 'repr(wrapper) wraps repr(value)': True,
 'repr(wrapper(int64)) wraps repr(value)': True,
 'wrapper(requires_grad).requires_grad': False,
 'wrapper(requires_grad).is_leaf': True,
 'unwrap(wrapper(requires_grad)) is value': True,
 'wrapper(meta).device': 'meta',
 'wrapper(meta).is_meta': True,
 'wrapper(meta).shape': ['Size', 2, 3],
 'wrapper(meta).data_ptr()': 0,
 'unwrap(wrapper(meta)) is value': True,
 'wrapper(fake) type': 'Tensor',
 'wrapper(fake).device': 'cpu',
 'wrapper(fake).is_meta': False,
 'wrapper(fake).shape': ['Size', 2, 3],
 'wrapper(fake).data_ptr()': 0,
 'unwrap(wrapper(fake)) is value': True,
 'unwrap(wrapper(fake)) type': 'FakeTensor',
 'keys(wrapper) has Functionalize': True,
 'keys(plain) has Functionalize': False,
 'keys(wrapper) = keys(value) + Functionalize': True,
 'keys(wrapper(meta)) = keys(value) + Functionalize': True,
 'keys(wrapper(fake)) = keys(value) - Python + Functionalize': True,
 '_is_functional_tensor_base(wrapper)': True,
 '_is_functional_tensor_base(plain)': 'raises RuntimeError',
 '_is_functional_tensor_base(1)': 'raises TypeError',
 '_functionalize_has_data_mutation(wrapper)': False,
 '_functionalize_has_data_mutation(plain)': 'raises RuntimeError',
 '_functionalize_has_data_mutation(1)': 'raises TypeError',
 '_functionalize_has_metadata_mutation(wrapper)': False,
 '_functionalize_has_metadata_mutation(plain)': 'raises RuntimeError',
 '_functionalize_has_metadata_mutation(1)': 'raises TypeError',
 '_functionalize_was_storage_changed(wrapper)': False,
 '_functionalize_was_storage_changed(plain)': 'raises RuntimeError',
 '_functionalize_was_storage_changed(1)': 'raises TypeError',
 '_functionalize_was_inductor_storage_resized(wrapper)': False,
 '_functionalize_was_inductor_storage_resized(plain)': 'raises RuntimeError',
 '_functionalize_was_inductor_storage_resized(1)': 'raises TypeError',
 '_functionalize_are_all_mutations_hidden_from_autograd(wrapper)': True,
 '_functionalize_are_all_mutations_hidden_from_autograd(plain)': 'raises RuntimeError',
 '_functionalize_are_all_mutations_hidden_from_autograd(1)': 'raises TypeError',
 '_functionalize_are_all_mutations_under_no_grad_or_inference_mode(wrapper)': True,
 '_functionalize_are_all_mutations_under_no_grad_or_inference_mode(plain)': 'raises RuntimeError',
 '_functionalize_are_all_mutations_under_no_grad_or_inference_mode(1)': 'raises TypeError',
 '_functionalize_is_multi_output_view(wrapper)': False,
 '_functionalize_is_multi_output_view(plain)': 'raises RuntimeError',
 '_functionalize_is_multi_output_view(1)': 'raises TypeError',
 '_functionalize_is_symbolic(wrapper)': False,
 '_functionalize_is_symbolic(plain)': 'raises RuntimeError',
 '_functionalize_is_symbolic(1)': 'raises TypeError',
 '_functionalize_mutation_counter(wrapper)': 0,
 '_functionalize_mutation_counter(plain)': 'raises RuntimeError',
 '_functionalize_mutation_counter(1)': 'raises TypeError',
 '_functionalize_storage_changed_counter(wrapper)': 0,
 '_functionalize_storage_changed_counter(plain)': 'raises RuntimeError',
 '_functionalize_storage_changed_counter(1)': 'raises TypeError',
 '_functionalize_inductor_storage_resized_counter(wrapper)': False,
 '_functionalize_inductor_storage_resized_counter(plain)': 'raises RuntimeError',
 '_functionalize_inductor_storage_resized_counter(1)': 'raises TypeError',
 '_functionalize_sync(wrapper)': None,
 '_functionalize_sync(plain)': 'raises RuntimeError',
 '_functionalize_sync(1)': 'raises TypeError',
 '_is_functional_tensor_base(wrapper(meta))': True,
 '_functionalize_has_data_mutation(wrapper(fake))': False,
 'mark_storage_changed then (changed, counter, data_mutation)': [True, 1, False],
 'mark_storage_changed(plain)': 'raises RuntimeError',
 'reapply_views_tls at start': False,
 'enable_reapply_views(True)': False,
 'reapply_views_tls after True': True,
 'enable_reapply_views(False)': True,
 'reapply_views_tls after False': False,
 'enable_reapply_views(1)': False,
 'reapply_views_tls after 1': True,
 'enable_reapply_views()': True,
 'reapply_views_tls after ()': False,
 "enable_reapply_views('x')": 'raises TypeError',
 '_disable_functionalization()': None,
 'reapply_views_tls after disable': False,
 'Functionalize included after disable': False,
 '_disable_functionalization() twice': None,
 'Functionalize included at rest': False,
 'Functionalize excluded at rest': False,
 'mirror(requires_grad leaf -> wrapper)': [True, False, 'Error', False],
 'mirror(plain -> plain)': [False, True],
 'mirror(1, plain)': 'raises TypeError',
 '_op_dk(Functionalize, plain, plain)': [False, [2.0, 2.0]],
 'a mode saw a functional argument': False}

#: Upstream's answers to the rows that need the Functionalize kernel (#329
#: steps 2 to 4), measured the same way. The shim refuses every one by name.
UPSTREAM_PENDING = {'wrapper + 1': [True, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
 'wrapper.add_(1)': [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], True, 1],
 'wrapper.view(-1)': True,
 'wrapper.clone()': True,
 'wrapper.t_()': True,
 'wrapper.detach()': True,
 'wrapper(0-dim).item()': 3.5,
 'wrapper + plain': True,
 'torch.mm(wrapper, wrapper.t())': True,
 'torch.cat([wrapper, plain])': True,
 '_op_dk(Functionalize, wrapper, wrapper)': True,
 'wrapper.untyped_storage().nbytes()': 24,
 '_enable_functionalization()': None,
 '_functionalize_replace': None,
 '_functionalize_commit_update': None,
 '_functionalize_unsafe_set': None,
 '_functionalize_mark_mutation_hidden_from_autograd': None,
 '_freeze_functional_tensor': None,
 '_functionalize_get_storage_size(wrapper, True)': 24,
 '_op_dk(Functionalize, factory) with Functionalize included': True,
 'wrapper * 2 under a logging mode': True}

#: What a wrapper looks like after `add_(1)` upstream: the wrapped tensor's
#: new value (the original `t` is untouched, a separate measurement), the
#: data-mutation flag and the mutation counter.
UPSTREAM_AFTER = {'add_': [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], True, 1]}

_SHIM_ONLY_ROWS = {"shim live wrappers after del"}

_cache = {}


def _shim():
    if "shim" not in _cache:
        _cache["shim"] = _run(_PROBE, True)
        assert _cache["shim"]["is_shim"], "the shim-side probe got upstream torch"
    return _cache["shim"]


def _upstream():
    if "up" not in _cache:
        _cache["up"] = _run(_PROBE, False) if _have_upstream() else None
    return _cache["up"]


def test_the_recorded_tables_are_upstreams_answers():
    """Re-derive `UPSTREAM_TABLE`, `UPSTREAM_PENDING` and `UPSTREAM_AFTER`
    from upstream on every run, so a table that stops being upstream's turns
    this red instead of silently grading the shim against a wish."""
    up = _upstream()
    if up is None:
        _skip.skip("upstream torch not importable without the vendored tree")
        return
    assert not up["is_shim"]
    drift = {k: (v, up["out"].get(k)) for k, v in UPSTREAM_TABLE.items() if up["out"].get(k) != v}
    assert not drift, f"upstream no longer answers the recorded table: {drift}"
    assert set(up["out"]) == set(UPSTREAM_TABLE), "the probe and the table name different rows"
    got = {k: v[1] for k, v in up["pending"].items()}
    assert got == UPSTREAM_PENDING, {k: (UPSTREAM_PENDING.get(k), got.get(k))
                                     for k in set(got) | set(UPSTREAM_PENDING)
                                     if got.get(k) != UPSTREAM_PENDING.get(k)}
    assert all(v[0] == "value" for v in up["pending"].values()), up["pending"]
    assert up["after"] == UPSTREAM_AFTER, up["after"]


def test_the_wrapper_and_its_state_queries_answer_as_upstream_measured():
    """Every step-1 row: wrap and unwrap (identity included), the predicates,
    the wrapper's own metadata with no mode active, the dispatch-key rule, the
    thirteen state queries on a fresh wrapper and their refusals of a plain
    tensor and of a non-tensor, the TLS setters, `_mirror_autograd_meta_to`,
    and `_op_dk(Functionalize, ...)` on plain tensors."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _shim()["out"]
    wrong = {k: (v, got.get(k)) for k, v in UPSTREAM_TABLE.items() if got.get(k) != v}
    assert not wrong, f"{len(wrong)} of {len(UPSTREAM_TABLE)} rows differ (upstream, shim): {wrong}"
    assert set(got) - _SHIM_ONLY_ROWS == set(UPSTREAM_TABLE), "the probe and the table name different rows"
    print(f"FUNCTIONAL: {len(UPSTREAM_TABLE)} measured rows agree")


def test_every_op_on_a_wrapper_refuses_by_name_until_the_functionalize_kernel():
    """The invariant step 1 exists to keep: no kernel sees through a wrapper.

    Upstream answers each of these rows (`UPSTREAM_PENDING`); the shim must
    refuse every one with `NotImplementedError` naming #329, and an operator
    row must also say it refused because of a functional tensor. A row that
    returns a value on the shim computed on the wrapped tensor, or answered
    the machinery's question without the machinery: both are the silent
    answer AGENTS.md section 18 forbids."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    pending = _shim()["pending"]
    assert set(pending) == set(UPSTREAM_PENDING), (set(pending) ^ set(UPSTREAM_PENDING))
    bad = {}
    for name, (kind, *rest) in pending.items():
        if kind != "raises" or rest[0] != "NotImplementedError" or "#329" not in rest[1]:
            bad[name] = [kind] + rest
    assert not bad, f"{len(bad)} of {len(pending)} rows did not refuse by name: {bad}"
    for name in ("wrapper + 1", "wrapper.add_(1)", "wrapper.view(-1)", "torch.mm(wrapper, wrapper.t())",
                 "torch.cat([wrapper, plain])", "_op_dk(Functionalize, wrapper, wrapper)",
                 "wrapper * 2 under a logging mode"):
        assert "functional tensor" in pending[name][2], (name, pending[name][2])
    # An in-place op names step 3, a non-mutating one step 2.
    assert "step 3" in pending["wrapper.add_(1)"][2], pending["wrapper.add_(1)"][2]
    assert "step 2" in pending["wrapper + 1"][2], pending["wrapper + 1"][2]
    print(f"FUNCTIONAL: {len(pending)} of {len(UPSTREAM_PENDING)} kernel-dependent rows refuse by name")


def test_a_refused_inplace_op_leaves_the_value_and_the_record_untouched():
    """`f.add_(1)` refuses on the shim. Refusing is not enough if the kernel
    ran first: the wrapped value must still be `arange(6)` and the record
    must still say no mutation. Upstream's state after the same call is
    `UPSTREAM_AFTER` (mutated, flagged, counted), which is what step 3 must
    produce."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    after = _shim()["after"]["add_"]
    assert after == [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], False, 0], after
    assert UPSTREAM_AFTER["add_"] != after, "upstream's row must differ, or this checks nothing"


def test_a_mode_never_sees_a_wrapper():
    """Upstream's Functionalize key sits above Python, so a dispatch mode sees
    what the Functionalize kernel unwrapped (`a mode saw a functional
    argument` is False upstream). On the shim the door refuses before the
    mode consult; if that check moved below the modes, the logging mode would
    be handed the wrapper and the row would read True."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _shim()
    assert got["out"]["a mode saw a functional argument"] is False
    assert got["pending"]["wrapper * 2 under a logging mode"][0] == "raises"


def test_the_door_gate_closes_when_the_last_wrapper_dies():
    """The door walks arguments only while a wrapper payload is alive. After
    the probe drops every wrapper it made, the count is 0 again: a payload
    that never dies would leave every later dispatch paying the walk, and a
    count that never rose would mean the walk never ran."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    assert _shim()["out"]["shim live wrappers after del"] == 0


_TENSOR_PROBE = r'''
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"  # AGENTS.md 15.2
out = {}
def r(name, fn):
    try:
        fn()
    except Exception as exc:
        out[name] = [type(exc).__name__, str(exc)[:400]]
    else:
        out[name] = ["returned"]
t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
live0 = torch._C._shim_functional_live()
f = torch._to_functional_tensor(t)
out["live rises"] = torch._C._shim_functional_live() - live0
# `tensor()` refuses the variant. These two reach it directly, with no door
# in between; `numpy()` reaches the door first (through `detach`).
r("tensor() via _shim_numpy_parts", lambda: f._shim_numpy_parts())
r("tensor() via _shim_tensor_bytes", lambda: torch._C._shim_tensor_bytes(f))
r("numpy()", lambda: f.numpy())
r("plain.set_(wrapper)", lambda: torch.zeros(2, 3).set_(f))
r("wrapper.set_(plain)", lambda: f.set_(torch.zeros(2, 3)))
def set_data():
    p = torch.zeros(2, 3)
    p.data = f
r("plain.data = wrapper", set_data)
r("wrapper.untyped_storage()", lambda: f.untyped_storage())
r("_enable_functionalization()", lambda: torch._enable_functionalization())
out["included after refused enable"] = torch._C._dispatch_tls_is_dispatch_key_included(
    torch._C.DispatchKey.Functionalize)
out["value unchanged"] = torch._from_functional_tensor(f).tolist()
out["still functional"] = torch._is_functional_tensor(f)
print(json.dumps(out))
'''


def test_tensor_refuses_the_functional_variant_and_storage_swaps_refuse():
    """Shim-only, because these are the shim's structural refusals rather
    than upstream behaviour: `tensor()` refuses `Repr::Functional` (reached
    directly through two Rust readers; `numpy()`, which upstream answers by
    reading the wrapper's unallocated memory, an array of garbage, measured,
    refuses at the door), and `set_` / `.data =`
    with a wrapper on either side refuse instead of replacing the wrapper's
    representation or sharing its payload with a plain tensor."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    env = dict(os.environ, PYTHONPATH=str(VENDOR), TORCH_USE_RTLD_GLOBAL="1")
    proc = subprocess.run([sys.executable, "-c", _TENSOR_PROBE], capture_output=True,
                          text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got["live rises"] == 1, got
    for name in ("tensor() via _shim_numpy_parts", "tensor() via _shim_tensor_bytes"):
        kind, text = got[name]
        assert kind == "NotImplementedError", (name, got[name])
        assert "functional tensor (torch._to_functional_tensor)" in text, (name, text)
    kind, text = got["numpy()"]
    assert kind == "NotImplementedError" and "functional tensor" in text, got["numpy()"]
    for name in ("plain.set_(wrapper)", "wrapper.set_(plain)", "plain.data = wrapper",
                 "wrapper.untyped_storage()", "_enable_functionalization()"):
        kind, *rest = got[name]
        assert kind == "NotImplementedError" and "#329" in rest[0], (name, got[name])
    assert got["included after refused enable"] is False
    assert got["value unchanged"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert got["still functional"] is True


if __name__ == "__main__":
    sys.exit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        suite="test_functional_tensor",
    ) else 0)
