"""`torch.Tensor._make_wrapper_subclass` on the shim (issue #340).

Leader decision 2026-11-15, agent proposal; not owner-confirmed. The leader's
two conditions: an argument upstream accepts that the shim cannot represent
refuses BY NAME, and a wrapper tensor never falls to the dense kernels
silently.

Upstream's `FunctionalTensor.__new__`
(`torch/_subclasses/functional_tensor.py:198`) builds itself with this call, so
`ExportedProgram.run_decompositions()` and executorch's `to_edge` stopped there
while the name was a generated surface stub. A wrapper is a tensor with sizes,
strides, a dtype and a device and no storage; every operator on it is answered
by its class's `__torch_dispatch__`.

**These tests were written before the implementation and were not run by the
round that wrote them** (leader rule, 2026-11-15: local work stops at compile
checks). Their first red and first green are CI's to observe. The upstream
half was run: `EXPECTED` below is the output of `_TABLE` on upstream 2.13.0.

What was measured on upstream 2.13.0, in short (`EXPECTED` is the whole of it):

    signature   (cls, size, strides=None, storage_offset=None, memory_format=None,
                 dtype=None, layout=strided, device=None, pin_memory=False,
                 requires_grad=False, dispatch_sizes_strides_policy=None,
                 dispatch_device=False, dispatch_layout=False,
                 _extra_dispatch_keys=None, storage_size=None)
                2 to 15 positional; cls must be a type, a Tensor subclass, and
                define __torch_dispatch__ ("Plain must define __torch_dispatch__")
    defaults    float32 (the default dtype), strided, cpu, contiguous strides,
                offset 0, requires_grad False, is_leaf True, _is_view() False,
                data_ptr() 0 (offset * itemsize from a null base)
    memory_format   parsed and never read: (2, 3, 4, 5) with channels_last
                    answers strides (60, 20, 5, 1)
    device      the label as given, index kept: 'cpu:0', 'meta:3', 'cuda:1'
    keys        the device's keys plus Python and PythonTLSSnapshot
    queries     shape, dtype, device, stride, numel, dim, is_contiguous and the
                rest answer without calling the class
    operators   the class's __torch_dispatch__(func, types, args, kwargs), read
                off the instance, a subclass before its superclass; an active
                TorchDispatchMode sees the call first; an in-place operator
                returns its receiver whatever the class returned
    declining   every handler returning NotImplemented is a TypeError:
                "Multiple dispatch failed for 'torch._ops.aten.view.default'; ..."
    no_dispatch()   the backend kernel runs on a null data pointer: `w + 1`
                    kills the process, `w.view(6)` returns a plain Tensor

What makes each test able to fail:

* `test_the_pasted_table_is_what_upstream_answers_today` runs `_TABLE` on
  upstream and compares it with the pasted `EXPECTED`. A different torch, or an
  edit to either, turns it red: the table cannot go stale silently.
* `test_construction_and_every_argument_agree_with_upstream`: ignoring an
  argument (strides, storage_offset, dtype, device, requires_grad), reading
  `memory_format`, normalising a device index away, dropping the `Python`
  key from a wrapper's dispatch keys, giving a meta wrapper an Autocast key,
  or wording a bad argument differently changes a row. Putting the surface
  stub back turns every row into `NotImplementedError`.
* `test_a_wrapper_answers_metadata_as_upstream_does`: each query is a row, and
  each row also records which operators reached the class while it was asked.
  A query that dispatches, or one that reads storage (`tensor()` refuses on a
  wrapper), changes its row.
* `test_the_dispatch_protocol_agrees_with_upstream`: calling the class and not
  the instance (an instance method then receives the operator as `self`),
  skipping tensors inside a list, asking a superclass before its subclass,
  stopping at the first class that declines, asking the class before the mode,
  or returning the class's object from an in-place operator each change a row.
* `test_a_class_that_declines_gets_upstreams_typeerror`: falling through to
  the native path after `NotImplemented` makes these rows return or raise
  something else.
* `test_each_argument_the_shim_cannot_represent_refuses_naming_it`: accepting
  any of the sixteen (silently ignoring a policy, a storage size, an extra
  key, a sparse layout) turns its row from a refusal into `ok`. The upstream
  half asserts upstream still accepts each one, so a refusal upstream also
  gives cannot hide in this list.
* `test_a_wrapper_never_falls_to_a_dense_kernel` is the leader's condition.
  Thirty-four calls in six groups (pure, view, in-place, factory-like,
  sequence, Python-level), each tried three ways: the class declines, the
  class no longer defines `__torch_dispatch__`, and `no_dispatch()` is held.
  A call that returns anything is a failure. Each call is also run on a dense
  tensor first, and must succeed there, so a refusal cannot be "this operator
  is not implemented". Removing the refusal at the native door
  (`aten.rs::visit_for_device`) leaves `tensor()` as the backstop, and the
  test still fails: the direct groups require the door's own refusal and
  `torch._C._shim_wrapper_door_refusals()` rising by exactly one per call.
  Returning `Ok(dense)` from `tensor()` for `Repr::Wrapper` makes rows return.
* `test_the_shims_own_refusals_on_a_wrapper_name_themselves`: answering
  `untyped_storage()` with a meta or empty storage turns the row to `returned`.

What this suite cannot find:

* Anything about functionalization. `torch._to_functional_tensor` is a
  raising stub on develop, so `FunctionalTensor.__new__` still stops one line
  before it reaches the call these tests cover.
* A wrapper reached by a route that never passes the dispatch door and never
  calls `tensor()`. The thirty-four calls are a sample of the operator
  surface, not the surface.
* Whether upstream's `has()` answers for alias keys (`Dense`, `Autograd`)
  agree: only nineteen runtime keys are asked.
* `torch._C._dispatch_keys` on a tensor that is not a wrapper. Develop's
  fill (`torchnative/_dynamo_fills.py`) answers those and this round left its
  answers as they are. They differ from upstream, measured the same day:
  upstream's cpu set has `AutocastCPU`, its mps set `AutocastMPS`, and a
  `_make_subclass` tensor whose class overrides `__torch_dispatch__` has
  `Python` and `PythonTLSSnapshot`. No row here asks, so no row is red for it.
* What `no_dispatch()` does upstream beyond the two cases quoted above: the
  shim refuses every operator there, which is narrower than upstream for the
  operators that only read metadata (`view`, `zeros_like`).
* The message of a declined binary operator (`n + 1`): only its type is
  compared, because Python's own operator protocol rewords it upstream.
* Subclasses that are not wrappers. For those, a declining
  `__torch_dispatch__` still falls through to the native path, as it did
  before this change; upstream raises there too.
"""

import json
import os
import pathlib
import subprocess
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

import _skip  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")


def _available():
    """Is there a vendored tree for the shim side to import?"""
    return os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py"))


class _NoUpstream(Exception):
    """This interpreter has no upstream torch to measure against."""


def _run(script, shim, timeout=900):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        if not shim and "No module named 'torch'" in proc.stderr:
            raise _NoUpstream()
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # The side each probe landed on is asserted, not assumed: an empty vendored
    # tree makes the "shim" side silently import upstream (AGENTS.md 15.2).
    assert out["is_shim"] is shim, f"probe meant for shim={shim} ran on the other side"
    return out


_PREAMBLE = r"""
import json, re, sys, warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils._python_dispatch import TorchDispatchMode

IS_SHIM = hasattr(torch._C, "_aten_implemented")
mk = torch.Tensor._make_wrapper_subclass
LOG = []
LAST = [None]
# Runtime keys only. Upstream's `has()` also answers True for alias and
# functionality keys (`Dense`, `Autograd`, `CompositeImplicitAutograd`) that a
# tensor's set implies; those are not asked here.
KEY_NAMES = (
    "CPU", "CUDA", "MPS", "Meta", "SparseCPU", "Python", "PythonTLSSnapshot",
    "ADInplaceOrView", "AutogradCPU", "AutogradCUDA", "AutogradMPS",
    "AutogradMeta", "AutocastCPU", "AutocastCUDA", "AutocastMPS", "Conjugate",
    "Negative", "ZeroTensor", "Functionalize",
)


# Made before any mode is entered: a handler that built its answer with a
# factory would send that factory to whichever mode is active around it.
POOL = [torch.full((1,), 7.0) for _ in range(96)]


def fresh():
    # A new tensor per call: upstream refuses to make one object the result of
    # two view ops ("already had autograd metadata").
    LAST[0] = POOL.pop()
    return LAST[0]


def kind(a):
    if isinstance(a, torch.Tensor):
        return type(a).__name__
    if isinstance(a, (list, tuple)):
        return ["seq"] + [kind(x) for x in a]
    return type(a).__name__


def names(types):
    return [t.__name__ for t in types]


class W(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        LOG.append(["W", str(func), names(types), [kind(a) for a in args]])
        return fresh()


class I(torch.Tensor):
    # An instance method, as FunctionalTensor's is.
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        LOG.append(["I", type(self).__name__, str(func), names(types)])
        return fresh()


class N(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        LOG.append(["N", str(func)])
        return NotImplemented


class N2(N):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        LOG.append(["N2", str(func)])
        return NotImplemented


class A(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        LOG.append(["A", str(func), names(types)])
        return fresh()


class B(A):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        LOG.append(["B", str(func), names(types)])
        return fresh()


class Plain(torch.Tensor):
    pass


class M(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        LOG.append(["mode", str(func), names(types)])
        return func(*args, **(kwargs or {}))


class MN(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        LOG.append(["mode-declines", str(func)])
        return NotImplemented


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return "<" + type(value).__name__ + ">"
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def ans(f, message=True):
    try:
        return ["ok", jsonable(f())]
    except BaseException as exc:  # noqa: BLE001
        if not message:
            return [type(exc).__name__]
        return [type(exc).__name__, re.sub(r"0x[0-9a-fA-F]+", "0x", str(exc))]


def keys_of(t):
    ks = torch._C._dispatch_keys(t)
    return [n for n in KEY_NAMES if ks.has(getattr(torch._C.DispatchKey, n))]


def facts(w):
    return {
        "shape": list(w.shape),
        "stride": list(w.stride()),
        "storage_offset": w.storage_offset(),
        "dtype": str(w.dtype),
        "layout": str(w.layout),
        "device": str(w.device),
        "requires_grad": w.requires_grad,
        "is_leaf": w.is_leaf,
        "is_contiguous": w.is_contiguous(),
        "keys": keys_of(w),
        "type": type(w).__name__,
    }


def built(**kw):
    def run():
        size = kw.pop("size", (2, 3))
        return facts(mk(W, size, **kw))
    return ans(run)


out = {"is_shim": IS_SHIM}
"""

# The table: what each argument does, what a wrapper answers, who is called.
_TABLE = _PREAMBLE + r"""
# --- signature: what each argument changes ---------------------------------
DK = torch._C.DispatchKey
construct = {
    "defaults": built(),
    "size=list": built(size=[2, 3]),
    "size=torch.Size": built(size=torch.Size([2, 3])),
    "size=()": built(size=()),
    "size=(0,)": built(size=(0,)),
    "size=(0,3) strides=(5,7)": built(size=(0, 3), strides=(5, 7)),
    "size=(2,3,4,5)": built(size=(2, 3, 4, 5)),
    "strides=(1,2)": built(strides=(1, 2)),
    "strides=(1,2) storage_offset=5": built(strides=(1, 2), storage_offset=5),
    "strides=(100,7)": built(strides=(100, 7)),
    "strides=(0,0)": built(strides=(0, 0)),
    "memory_format=channels_last 4d": built(size=(2, 3, 4, 5), memory_format=torch.channels_last),
    "memory_format=contiguous_format": built(memory_format=torch.contiguous_format),
    "memory_format=preserve_format": built(memory_format=torch.preserve_format),
    "dtype=int64": built(dtype=torch.int64),
    "dtype=bool": built(dtype=torch.bool),
    "dtype=float64": built(dtype=torch.float64),
    "dtype=complex64": built(dtype=torch.complex64),
    "layout=strided": built(layout=torch.strided),
    "device='cpu'": built(device="cpu"),
    "device=torch.device('cpu')": built(device=torch.device("cpu")),
    "device='cpu:0'": built(device="cpu:0"),
    "device='meta'": built(device="meta"),
    "device='meta:3'": built(device="meta:3"),
    "device='mps'": built(device="mps"),
    "device='cuda:1'": built(device="cuda:1"),
    "pin_memory=False": built(pin_memory=False),
    "requires_grad=True": built(requires_grad=True),
    "requires_grad=True complex64": built(requires_grad=True, dtype=torch.complex64),
    "dispatch_sizes_strides_policy=None": built(dispatch_sizes_strides_policy=None),
    "dispatch_device=False dispatch_layout=False": built(dispatch_device=False, dispatch_layout=False),
    "_extra_dispatch_keys=None": built(_extra_dispatch_keys=None),
    "_extra_dispatch_keys={CPU}": built(_extra_dispatch_keys=torch._C.DispatchKeySet(DK.CPU)),
    "storage_size=None": built(storage_size=None),
    "cls and size by keyword": ans(lambda: facts(mk(cls=W, size=(2, 3)))),
    "fifteen positional": ans(lambda: facts(mk(
        W, (2, 3), (3, 1), 0, None, torch.float64, torch.strided,
        torch.device("cpu"), False, True, None, False, False, None, None))),
}


def default_dtype_follows():
    torch.set_default_dtype(torch.float64)
    try:
        return str(mk(W, (1,)).dtype)
    finally:
        torch.set_default_dtype(torch.float32)


def channels_last_strides():
    w = mk(W, (2, 3, 4, 5), (60, 1, 15, 3))
    return [w.is_contiguous(), w.is_contiguous(memory_format=torch.channels_last)]


construct["dtype=None under set_default_dtype(float64)"] = ans(default_dtype_follows)
construct["channels-last strides, asked both ways"] = ans(channels_last_strides)
out["construct"] = construct

# --- signature: what upstream refuses, in its words -------------------------
out["refused_by_upstream"] = {
    "no arguments": ans(lambda: mk()),
    "cls only": ans(lambda: mk(W)),
    "sixteen positional": ans(lambda: mk(
        W, (2, 3), (3, 1), 0, None, torch.float64, torch.strided,
        torch.device("cpu"), False, True, None, False, False, None, None, None)),
    "unexpected keyword": ans(lambda: mk(W, (2, 3), bogus=None)),
    "size twice": ans(lambda: mk(W, (2,), size=(2,))),
    "cls=None": ans(lambda: mk(None, (2,))),
    "cls=3": ans(lambda: mk(3, (2,))),
    "cls=int": ans(lambda: mk(int, (2,))),
    "cls without __torch_dispatch__": ans(lambda: mk(Plain, (2,))),
    "cls=torch.Tensor": ans(lambda: mk(torch.Tensor, (2,))),
    "size=3": ans(lambda: mk(W, 3)),
    "size='ab'": ans(lambda: mk(W, "ab")),
    "size=None": ans(lambda: mk(W, None)),
    "size=(1.5,)": ans(lambda: mk(W, (1.5,))),
    "size=(True, 3)": ans(lambda: mk(W, (True, 3))),
    "strides of another rank": ans(lambda: mk(W, (2, 3), strides=(1,))),
    "strides='x'": ans(lambda: mk(W, (2,), strides="x")),
    "strides=(1.5,)": ans(lambda: mk(W, (2,), strides=(1.5,))),
    "storage_offset=5 without strides": ans(lambda: mk(W, (2, 3), storage_offset=5)),
    "storage_offset=0 without strides": ans(lambda: mk(W, (2, 3), storage_offset=0)),
    "storage_offset=1.5": ans(lambda: mk(W, (2,), strides=(1,), storage_offset=1.5)),
    "memory_format='x'": ans(lambda: mk(W, (2,), memory_format="x")),
    "dtype='float32'": ans(lambda: mk(W, (2,), dtype="float32")),
    "layout=None": ans(lambda: mk(W, (2,), layout=None)),
    "layout='strided'": ans(lambda: mk(W, (2,), layout="strided")),
    "device=1.5": ans(lambda: mk(W, (2,), device=1.5)),
    "pin_memory=None": ans(lambda: mk(W, (2,), pin_memory=None)),
    "requires_grad=1": ans(lambda: mk(W, (2,), requires_grad=1)),
    "requires_grad=True int64": ans(lambda: mk(W, (2,), requires_grad=True, dtype=torch.int64)),
    "requires_grad=True bool": ans(lambda: mk(W, (2,), requires_grad=True, dtype=torch.bool)),
    "dispatch_sizes_strides_policy='bogus'": ans(lambda: mk(W, (2,), dispatch_sizes_strides_policy="bogus")),
    "dispatch_sizes_strides_policy=3": ans(lambda: mk(W, (2,), dispatch_sizes_strides_policy=3)),
    "dispatch_device=None": ans(lambda: mk(W, (2,), dispatch_device=None)),
    "dispatch_layout=None": ans(lambda: mk(W, (2,), dispatch_layout=None)),
    "_extra_dispatch_keys=3": ans(lambda: mk(W, (2,), _extra_dispatch_keys=3)),
    "storage_size='x'": ans(lambda: mk(W, (2,), storage_size="x")),
}

# --- what a wrapper answers with no mode active -----------------------------
w = mk(W, (2, 3))
queries = {}
for name, f, message in (
    ("tuple(w.shape)", lambda: list(w.shape), True),
    ("type(w.shape).__name__", lambda: type(w.shape).__name__, True),
    ("w.size()", lambda: list(w.size()), True),
    ("w.size(0)", lambda: w.size(0), True),
    ("w.size(-1)", lambda: w.size(-1), True),
    ("str(w.dtype)", lambda: str(w.dtype), True),
    ("str(w.device)", lambda: str(w.device), True),
    ("str(w.layout)", lambda: str(w.layout), True),
    ("w.requires_grad", lambda: w.requires_grad, True),
    ("w.is_leaf", lambda: w.is_leaf, True),
    ("w.grad_fn is None", lambda: w.grad_fn is None, True),
    ("w.grad is None", lambda: w.grad is None, True),
    ("w.stride()", lambda: list(w.stride()), True),
    ("w.stride(0)", lambda: w.stride(0), True),
    ("w.storage_offset()", lambda: w.storage_offset(), True),
    ("w.numel()", lambda: w.numel(), True),
    ("w.dim()", lambda: w.dim(), True),
    ("w.ndim", lambda: w.ndim, True),
    ("w.element_size()", lambda: w.element_size(), True),
    ("w._is_view()", lambda: w._is_view(), True),
    ("w._base is None", lambda: w._base is None, True),
    ("w.data_ptr()", lambda: w.data_ptr(), True),
    ("w.is_contiguous()", lambda: w.is_contiguous(), True),
    ("w.is_meta", lambda: w.is_meta, True),
    ("w.is_cpu", lambda: w.is_cpu, True),
    ("w.is_sparse", lambda: w.is_sparse, True),
    ("w.is_quantized", lambda: w.is_quantized, True),
    ("w.is_nested", lambda: w.is_nested, True),
    ("w.is_conj()", lambda: w.is_conj(), True),
    ("w.is_neg()", lambda: w.is_neg(), True),
    ("w._is_zerotensor()", lambda: w._is_zerotensor(), True),
    ("w.is_inference()", lambda: w.is_inference(), True),
    ("w.get_device()", lambda: w.get_device(), True),
    ("torch._C._has_storage(w)", lambda: torch._C._has_storage(w), True),
    ("torch._is_functional_tensor(w)", lambda: torch._is_functional_tensor(w), True),
    ("type(w).__name__", lambda: type(w).__name__, True),
    ("isinstance(w, torch.Tensor)", lambda: isinstance(w, torch.Tensor), True),
    ("isinstance(w, W)", lambda: isinstance(w, W), True),
    ("len(w)", lambda: len(w), True),
    ("w.tolist()", lambda: w.tolist(), True),
    ("w.item()", lambda: w.item(), True),
    ("bool(w)", lambda: bool(w), True),
    ("dispatch keys", lambda: keys_of(w), True),
    ("has the Python key", lambda: torch._C._dispatch_keys(w).has(DK.Python), True),
    ("an attribute sticks", lambda: (setattr(w, "elem", 5), w.elem)[1], True),
):
    LOG.clear()
    queries[name] = ans(f, message) + [{"dispatched": [entry[1] for entry in LOG]}]


def after_set_throw():
    t = mk(W, (2, 3))
    torch._C._set_throw_on_mutable_data_ptr(t)
    return t.data_ptr()


def scalar_goes_to_the_class():
    LOG.clear()
    try:
        mk(W, ()).item()
    except BaseException:  # noqa: BLE001
        pass
    return [entry[1] for entry in LOG]


def requires_grad_after():
    t = mk(W, (2, 3))
    t.requires_grad_(True)
    return [t.requires_grad, t.is_leaf, t.grad_fn is None]


queries["data_ptr() after _set_throw_on_mutable_data_ptr"] = ans(after_set_throw, False)
queries["item() of a 0-d wrapper asks the class"] = ans(scalar_goes_to_the_class)
queries["requires_grad_(True) afterwards"] = ans(requires_grad_after)
queries["strided wrapper: _is_view(), data_ptr()"] = ans(
    lambda: [mk(W, (2, 3), (1, 2), 5)._is_view(), mk(W, (2, 3), (1, 2), 5).data_ptr()])
out["queries"] = queries

# --- the dispatch protocol --------------------------------------------------
d = torch.ones(2, 3)
dc = torch.ones(2, 3)
wi = mk(I, (2, 3))
n = mk(N, (2, 3))
n2 = mk(N2, (2, 3))
a = mk(A, (2, 3))
b = mk(B, (2, 3))


def call(f, message=True, receiver=None):
    LOG.clear()
    LAST[0] = None
    try:
        result = f()
        answer = LAST[0]
        if receiver is not None and result is receiver:
            what = "the receiver"
        elif result is answer or (
            isinstance(result, torch.Tensor)
            and answer is not None
            and type(result) is type(answer)
            and tuple(result.shape) == tuple(answer.shape)
            and result.data_ptr() == answer.data_ptr()
        ):
            # The object itself, or (upstream, for the `*_like` factories) a
            # second Python object over the same tensor.
            what = "the class's answer"
        else:
            what = "something else"
        outcome = ["ok", what]
    except BaseException as exc:  # noqa: BLE001
        outcome = [type(exc).__name__]
        if message:
            outcome.append(re.sub(r"0x[0-9a-fA-F]+", "0x", str(exc)))
    return outcome + [[list(entry) for entry in LOG]]


def under(mode, f):
    def run():
        with mode:
            return f()
    return run


out["dispatch"] = {
    "torch.ops.aten.add.Tensor(w, d)": call(lambda: torch.ops.aten.add.Tensor(w, d)),
    "torch.add(w, d)": call(lambda: torch.add(w, d)),
    "torch.add(d, w)": call(lambda: torch.add(d, w)),
    "w + 1": call(lambda: w + 1),
    "w.add_(1)": call(lambda: w.add_(1), receiver=w),
    "w.view(6)": call(lambda: w.view(6)),
    "w.t()": call(lambda: w.t()),
    "w.sum()": call(lambda: w.sum()),
    "w.clone()": call(lambda: w.clone()),
    "w == d": call(lambda: w == d),
    "torch.zeros_like(w)": call(lambda: torch.zeros_like(w)),
    "torch.cat([w, d])": call(lambda: torch.cat([w, d])),
    "torch.stack([d, w])": call(lambda: torch.stack([d, w])),
    "dc.copy_(w)": call(lambda: dc.copy_(w), receiver=dc),
    "instance method: torch.add(wi, d)": call(lambda: torch.add(wi, d)),
    "instance method: torch.add(d, wi)": call(lambda: torch.add(d, wi)),
    "two classes: torch.add(w, wi)": call(lambda: torch.add(w, wi)),
    "two classes: torch.add(wi, w)": call(lambda: torch.add(wi, w)),
    "subclass first: torch.add(a, b)": call(lambda: torch.add(a, b)),
    "subclass first: torch.add(b, a)": call(lambda: torch.add(b, a)),
    "a declining class, then the next: torch.add(n, w)": call(lambda: torch.add(n, w)),
    "a mode sees it first: torch.add(w, d)": call(under(M(), lambda: torch.add(w, d))),
    "a mode alone: torch.add(d, d)": call(under(M(), lambda: torch.add(d, d))),
    "a mode declines, the class answers: torch.add(w, d)": call(
        under(MN(), lambda: torch.add(w, d))),
    "requires_grad=True: torch.add(wg, d)": call(
        lambda: torch.add(mk(W, (2, 3), requires_grad=True), d)),
}

# --- a class that answers NotImplemented ------------------------------------
out["declined"] = {
    "n.view(6)": call(lambda: n.view(6)),
    "n.add_(1)": call(lambda: n.add_(1), receiver=n),
    "torch.add(n, d)": call(lambda: torch.add(n, d)),
    "torch.zeros_like(n)": call(lambda: torch.zeros_like(n)),
    "torch.cat([n, d])": call(lambda: torch.cat([n, d])),
    "torch.add(n, n2)": call(lambda: torch.add(n, n2)),
    "torch.add(n2, n)": call(lambda: torch.add(n2, n)),
    "under a declining mode: n.view(6)": call(under(MN(), lambda: n.view(6))),
    "n + 1 (type only)": call(lambda: n + 1, False)[:1],
}

print(json.dumps(out, sort_keys=True))
"""

# The arguments upstream accepts and the shim refuses by name.
_REFUSED = _PREAMBLE + r"""
# --- arguments upstream accepts and this shim cannot represent ---------------
def symint():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    return ShapeEnv().create_unbacked_symint()


def tried(f):
    try:
        f()
        return ["ok"]
    except BaseException as exc:  # noqa: BLE001
        return [type(exc).__name__, str(exc)]


DK = torch._C.DispatchKey
out["refused"] = [
    ["size", "a SymInt", tried(lambda: mk(W, (symint(), 3)))],
    ["size", "a negative extent", tried(lambda: mk(W, (-1, 0)))],
    ["strides", "a SymInt", tried(lambda: mk(W, (2, 3), (symint(), 1)))],
    ["strides", "a negative stride", tried(lambda: mk(W, (2, 3), (-3, 1)))],
    ["storage_offset", "a SymInt", tried(lambda: mk(W, (2, 3), (3, 1), symint()))],
    ["storage_offset", "a negative offset", tried(lambda: mk(W, (2, 3), (3, 1), -1))],
    ["layout", "torch.sparse_coo", tried(lambda: mk(W, (2, 3), layout=torch.sparse_coo))],
    ["device", "a bare index", tried(lambda: mk(W, (2, 3), device=0))],
    ["pin_memory", "True", tried(lambda: mk(W, (2, 3), pin_memory=True))],
    ["dispatch_sizes_strides_policy", "'sizes'",
     tried(lambda: mk(W, (2, 3), dispatch_sizes_strides_policy="sizes"))],
    ["dispatch_sizes_strides_policy", "'strides'",
     tried(lambda: mk(W, (2, 3), dispatch_sizes_strides_policy="strides"))],
    ["dispatch_device", "True", tried(lambda: mk(W, (2, 3), dispatch_device=True))],
    ["dispatch_layout", "True", tried(lambda: mk(W, (2, 3), dispatch_layout=True))],
    ["_extra_dispatch_keys", "ZeroTensor",
     tried(lambda: mk(W, (2, 3), _extra_dispatch_keys=torch._C.DispatchKeySet(DK.ZeroTensor)))],
    ["_extra_dispatch_keys", "Conjugate",
     tried(lambda: mk(W, (2, 3), _extra_dispatch_keys=torch._C.DispatchKeySet(DK.Conjugate)))],
    ["storage_size", "100", tried(lambda: mk(W, (2, 3), storage_size=100))],
]
print(json.dumps(out, sort_keys=True))
"""

# The leader's condition: shim only, because upstream has no door counter.
_LEADER = _PREAMBLE + r"""
# --- a wrapper never reaches a kernel (shim only) ----------------------------
assert IS_SHIM
import torch.nn.functional as F

refusals = torch._C._shim_wrapper_door_refusals
wt = torch.ones(4, 3)

OPS = [
    # [group, label, callable(x, d)]: x is the wrapper (or its dense control).
    ["pure", "torch.ops.aten.add.Tensor(x, d)", lambda x, d: torch.ops.aten.add.Tensor(x, d)],
    ["pure", "torch.add(d, x)", lambda x, d: torch.add(d, x)],
    ["pure", "torch.mul(x, 2.0)", lambda x, d: torch.mul(x, 2.0)],
    ["pure", "x.sum()", lambda x, d: x.sum()],
    ["pure", "x.clone()", lambda x, d: x.clone()],
    ["pure", "torch.exp(x)", lambda x, d: torch.exp(x)],
    ["view", "x.view(6)", lambda x, d: x.view(6)],
    ["view", "x.t()", lambda x, d: x.t()],
    ["view", "x.reshape(3, 2)", lambda x, d: x.reshape(3, 2)],
    ["view", "x.unsqueeze(0)", lambda x, d: x.unsqueeze(0)],
    ["view", "x.permute(1, 0)", lambda x, d: x.permute(1, 0)],
    ["in-place", "x.add_(1)", lambda x, d: x.add_(1)],
    ["in-place", "x.zero_()", lambda x, d: x.zero_()],
    ["in-place", "d.copy_(x)", lambda x, d: d.copy_(x)],
    ["in-place", "d.add_(x)", lambda x, d: d.add_(x)],
    ["factory-like", "torch.zeros_like(x)", lambda x, d: torch.zeros_like(x)],
    ["factory-like", "torch.ones_like(x)", lambda x, d: torch.ones_like(x)],
    ["factory-like", "torch.empty_like(x)", lambda x, d: torch.empty_like(x)],
    ["factory-like", "torch.full_like(x, 3.0)", lambda x, d: torch.full_like(x, 3.0)],
    ["factory-like", "x.new_zeros(2)", lambda x, d: x.new_zeros(2)],
    ["sequence", "torch.cat([x, d])", lambda x, d: torch.cat([x, d])],
    ["sequence", "torch.stack([d, x])", lambda x, d: torch.stack([d, x])],
    ["python-level", "F.linear(x, wt)", lambda x, d: F.linear(x, wt)],
    ["python-level", "F.layer_norm(x, (3,))", lambda x, d: F.layer_norm(x, (3,))],
    ["python-level", "torch.softmax(x, -1)", lambda x, d: torch.softmax(x, -1)],
    ["python-level", "F.gelu(x)", lambda x, d: F.gelu(x)],
    ["python-level", "F.relu(x)", lambda x, d: F.relu(x)],
    ["python-level", "x @ wt.t()", lambda x, d: x @ wt.t()],
    ["python-level", "x.mean()", lambda x, d: x.mean()],
    ["python-level", "x.pow(2)", lambda x, d: x.pow(2)],
    ["python-level", "x.to(torch.float64)", lambda x, d: x.to(torch.float64)],
    ["python-level", "x[0]", lambda x, d: x[0]],
    ["python-level", "torch.where(d > 0, x, d)", lambda x, d: torch.where(d > 0, x, d)],
    ["python-level", "repr(x)", lambda x, d: repr(x)],
]


class Gone(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        LOG.append(["Gone", str(func)])
        return fresh()


declining = mk(N, (2, 3))
answering = mk(W, (2, 3))
orphan = mk(Gone, (2, 3))
# The class stops answering after the wrapper exists: construction itself
# refuses a class with no `__torch_dispatch__`, so this is the only way to hold
# a wrapper whose class says nothing at all.
del Gone.__torch_dispatch__


def attempt(f, x):
    d = torch.ones(2, 3)
    LOG.clear()
    before = refusals()
    try:
        result = f(x, d)
        outcome = ["returned", type(result).__name__]
    except BaseException as exc:  # noqa: BLE001
        outcome = [type(exc).__name__, str(exc)]
    return {
        "outcome": outcome,
        "door_refusals": refusals() - before,
        "asked": [entry[0] for entry in LOG],
    }


rows = []
for group, label, f in OPS:
    row = {"group": group, "op": label}
    row["control"] = attempt(f, torch.ones(2, 3))
    row["class declines"] = attempt(f, declining)
    row["class no longer answers"] = attempt(f, orphan)
    with torch._C._DisableTorchDispatch():
        row["no_dispatch()"] = attempt(f, answering)
    rows.append(row)
out["rows"] = rows


def storage_question():
    try:
        answering.untyped_storage()
        return ["returned"]
    except BaseException as exc:  # noqa: BLE001
        return [type(exc).__name__, str(exc)]


out["untyped_storage"] = storage_question()
out["is_wrapper"] = [
    torch._C._shim_is_wrapper(answering),
    torch._C._shim_is_wrapper(torch.ones(2)),
    torch._C._shim_is_wrapper(torch.ones(2, device="meta")),
    torch._C._shim_is_wrapper(torch.Tensor._make_subclass(N, torch.ones(2))),
]
print(json.dumps(out, sort_keys=True))
"""

#: Measured on upstream torch 2.13.0 (the spike-venv wheel) on 2026-11-15 by
#: running `_TABLE` with no PYTHONPATH. Pasted, not written:
#: `test_the_pasted_table_is_what_upstream_answers_today` runs the same probe
#: on upstream on every run and fails if this copy has gone stale.
EXPECTED = {'construct': {'_extra_dispatch_keys=None': ['ok',
                                             {'device': 'cpu',
                                              'dtype': 'torch.float32',
                                              'is_contiguous': True,
                                              'is_leaf': True,
                                              'keys': ['CPU',
                                                       'Python',
                                                       'PythonTLSSnapshot',
                                                       'ADInplaceOrView',
                                                       'AutogradCPU',
                                                       'AutocastCPU'],
                                              'layout': 'torch.strided',
                                              'requires_grad': False,
                                              'shape': [2, 3],
                                              'storage_offset': 0,
                                              'stride': [3, 1],
                                              'type': 'W'}],
               '_extra_dispatch_keys={CPU}': ['ok',
                                              {'device': 'cpu',
                                               'dtype': 'torch.float32',
                                               'is_contiguous': True,
                                               'is_leaf': True,
                                               'keys': ['CPU',
                                                        'Python',
                                                        'PythonTLSSnapshot',
                                                        'ADInplaceOrView',
                                                        'AutogradCPU',
                                                        'AutocastCPU'],
                                               'layout': 'torch.strided',
                                               'requires_grad': False,
                                               'shape': [2, 3],
                                               'storage_offset': 0,
                                               'stride': [3, 1],
                                               'type': 'W'}],
               'channels-last strides, asked both ways': ['ok', [False, True]],
               'cls and size by keyword': ['ok',
                                           {'device': 'cpu',
                                            'dtype': 'torch.float32',
                                            'is_contiguous': True,
                                            'is_leaf': True,
                                            'keys': ['CPU',
                                                     'Python',
                                                     'PythonTLSSnapshot',
                                                     'ADInplaceOrView',
                                                     'AutogradCPU',
                                                     'AutocastCPU'],
                                            'layout': 'torch.strided',
                                            'requires_grad': False,
                                            'shape': [2, 3],
                                            'storage_offset': 0,
                                            'stride': [3, 1],
                                            'type': 'W'}],
               'defaults': ['ok',
                            {'device': 'cpu',
                             'dtype': 'torch.float32',
                             'is_contiguous': True,
                             'is_leaf': True,
                             'keys': ['CPU',
                                      'Python',
                                      'PythonTLSSnapshot',
                                      'ADInplaceOrView',
                                      'AutogradCPU',
                                      'AutocastCPU'],
                             'layout': 'torch.strided',
                             'requires_grad': False,
                             'shape': [2, 3],
                             'storage_offset': 0,
                             'stride': [3, 1],
                             'type': 'W'}],
               "device='cpu'": ['ok',
                                {'device': 'cpu',
                                 'dtype': 'torch.float32',
                                 'is_contiguous': True,
                                 'is_leaf': True,
                                 'keys': ['CPU',
                                          'Python',
                                          'PythonTLSSnapshot',
                                          'ADInplaceOrView',
                                          'AutogradCPU',
                                          'AutocastCPU'],
                                 'layout': 'torch.strided',
                                 'requires_grad': False,
                                 'shape': [2, 3],
                                 'storage_offset': 0,
                                 'stride': [3, 1],
                                 'type': 'W'}],
               "device='cpu:0'": ['ok',
                                  {'device': 'cpu:0',
                                   'dtype': 'torch.float32',
                                   'is_contiguous': True,
                                   'is_leaf': True,
                                   'keys': ['CPU',
                                            'Python',
                                            'PythonTLSSnapshot',
                                            'ADInplaceOrView',
                                            'AutogradCPU',
                                            'AutocastCPU'],
                                   'layout': 'torch.strided',
                                   'requires_grad': False,
                                   'shape': [2, 3],
                                   'storage_offset': 0,
                                   'stride': [3, 1],
                                   'type': 'W'}],
               "device='cuda:1'": ['ok',
                                   {'device': 'cuda:1',
                                    'dtype': 'torch.float32',
                                    'is_contiguous': True,
                                    'is_leaf': True,
                                    'keys': ['CUDA',
                                             'Python',
                                             'PythonTLSSnapshot',
                                             'ADInplaceOrView',
                                             'AutogradCUDA',
                                             'AutocastCUDA'],
                                    'layout': 'torch.strided',
                                    'requires_grad': False,
                                    'shape': [2, 3],
                                    'storage_offset': 0,
                                    'stride': [3, 1],
                                    'type': 'W'}],
               "device='meta'": ['ok',
                                 {'device': 'meta',
                                  'dtype': 'torch.float32',
                                  'is_contiguous': True,
                                  'is_leaf': True,
                                  'keys': ['Meta',
                                           'Python',
                                           'PythonTLSSnapshot',
                                           'ADInplaceOrView',
                                           'AutogradMeta'],
                                  'layout': 'torch.strided',
                                  'requires_grad': False,
                                  'shape': [2, 3],
                                  'storage_offset': 0,
                                  'stride': [3, 1],
                                  'type': 'W'}],
               "device='meta:3'": ['ok',
                                   {'device': 'meta:3',
                                    'dtype': 'torch.float32',
                                    'is_contiguous': True,
                                    'is_leaf': True,
                                    'keys': ['Meta',
                                             'Python',
                                             'PythonTLSSnapshot',
                                             'ADInplaceOrView',
                                             'AutogradMeta'],
                                    'layout': 'torch.strided',
                                    'requires_grad': False,
                                    'shape': [2, 3],
                                    'storage_offset': 0,
                                    'stride': [3, 1],
                                    'type': 'W'}],
               "device='mps'": ['ok',
                                {'device': 'mps',
                                 'dtype': 'torch.float32',
                                 'is_contiguous': True,
                                 'is_leaf': True,
                                 'keys': ['MPS',
                                          'Python',
                                          'PythonTLSSnapshot',
                                          'ADInplaceOrView',
                                          'AutogradMPS',
                                          'AutocastMPS'],
                                 'layout': 'torch.strided',
                                 'requires_grad': False,
                                 'shape': [2, 3],
                                 'storage_offset': 0,
                                 'stride': [3, 1],
                                 'type': 'W'}],
               "device=torch.device('cpu')": ['ok',
                                              {'device': 'cpu',
                                               'dtype': 'torch.float32',
                                               'is_contiguous': True,
                                               'is_leaf': True,
                                               'keys': ['CPU',
                                                        'Python',
                                                        'PythonTLSSnapshot',
                                                        'ADInplaceOrView',
                                                        'AutogradCPU',
                                                        'AutocastCPU'],
                                               'layout': 'torch.strided',
                                               'requires_grad': False,
                                               'shape': [2, 3],
                                               'storage_offset': 0,
                                               'stride': [3, 1],
                                               'type': 'W'}],
               'dispatch_device=False dispatch_layout=False': ['ok',
                                                               {'device': 'cpu',
                                                                'dtype': 'torch.float32',
                                                                'is_contiguous': True,
                                                                'is_leaf': True,
                                                                'keys': ['CPU',
                                                                         'Python',
                                                                         'PythonTLSSnapshot',
                                                                         'ADInplaceOrView',
                                                                         'AutogradCPU',
                                                                         'AutocastCPU'],
                                                                'layout': 'torch.strided',
                                                                'requires_grad': False,
                                                                'shape': [2, 3],
                                                                'storage_offset': 0,
                                                                'stride': [3, 1],
                                                                'type': 'W'}],
               'dispatch_sizes_strides_policy=None': ['ok',
                                                      {'device': 'cpu',
                                                       'dtype': 'torch.float32',
                                                       'is_contiguous': True,
                                                       'is_leaf': True,
                                                       'keys': ['CPU',
                                                                'Python',
                                                                'PythonTLSSnapshot',
                                                                'ADInplaceOrView',
                                                                'AutogradCPU',
                                                                'AutocastCPU'],
                                                       'layout': 'torch.strided',
                                                       'requires_grad': False,
                                                       'shape': [2, 3],
                                                       'storage_offset': 0,
                                                       'stride': [3, 1],
                                                       'type': 'W'}],
               'dtype=None under set_default_dtype(float64)': ['ok', 'torch.float64'],
               'dtype=bool': ['ok',
                              {'device': 'cpu',
                               'dtype': 'torch.bool',
                               'is_contiguous': True,
                               'is_leaf': True,
                               'keys': ['CPU',
                                        'Python',
                                        'PythonTLSSnapshot',
                                        'ADInplaceOrView',
                                        'AutogradCPU',
                                        'AutocastCPU'],
                               'layout': 'torch.strided',
                               'requires_grad': False,
                               'shape': [2, 3],
                               'storage_offset': 0,
                               'stride': [3, 1],
                               'type': 'W'}],
               'dtype=complex64': ['ok',
                                   {'device': 'cpu',
                                    'dtype': 'torch.complex64',
                                    'is_contiguous': True,
                                    'is_leaf': True,
                                    'keys': ['CPU',
                                             'Python',
                                             'PythonTLSSnapshot',
                                             'ADInplaceOrView',
                                             'AutogradCPU',
                                             'AutocastCPU'],
                                    'layout': 'torch.strided',
                                    'requires_grad': False,
                                    'shape': [2, 3],
                                    'storage_offset': 0,
                                    'stride': [3, 1],
                                    'type': 'W'}],
               'dtype=float64': ['ok',
                                 {'device': 'cpu',
                                  'dtype': 'torch.float64',
                                  'is_contiguous': True,
                                  'is_leaf': True,
                                  'keys': ['CPU',
                                           'Python',
                                           'PythonTLSSnapshot',
                                           'ADInplaceOrView',
                                           'AutogradCPU',
                                           'AutocastCPU'],
                                  'layout': 'torch.strided',
                                  'requires_grad': False,
                                  'shape': [2, 3],
                                  'storage_offset': 0,
                                  'stride': [3, 1],
                                  'type': 'W'}],
               'dtype=int64': ['ok',
                               {'device': 'cpu',
                                'dtype': 'torch.int64',
                                'is_contiguous': True,
                                'is_leaf': True,
                                'keys': ['CPU',
                                         'Python',
                                         'PythonTLSSnapshot',
                                         'ADInplaceOrView',
                                         'AutogradCPU',
                                         'AutocastCPU'],
                                'layout': 'torch.strided',
                                'requires_grad': False,
                                'shape': [2, 3],
                                'storage_offset': 0,
                                'stride': [3, 1],
                                'type': 'W'}],
               'fifteen positional': ['ok',
                                      {'device': 'cpu',
                                       'dtype': 'torch.float64',
                                       'is_contiguous': True,
                                       'is_leaf': True,
                                       'keys': ['CPU',
                                                'Python',
                                                'PythonTLSSnapshot',
                                                'ADInplaceOrView',
                                                'AutogradCPU',
                                                'AutocastCPU'],
                                       'layout': 'torch.strided',
                                       'requires_grad': True,
                                       'shape': [2, 3],
                                       'storage_offset': 0,
                                       'stride': [3, 1],
                                       'type': 'W'}],
               'layout=strided': ['ok',
                                  {'device': 'cpu',
                                   'dtype': 'torch.float32',
                                   'is_contiguous': True,
                                   'is_leaf': True,
                                   'keys': ['CPU',
                                            'Python',
                                            'PythonTLSSnapshot',
                                            'ADInplaceOrView',
                                            'AutogradCPU',
                                            'AutocastCPU'],
                                   'layout': 'torch.strided',
                                   'requires_grad': False,
                                   'shape': [2, 3],
                                   'storage_offset': 0,
                                   'stride': [3, 1],
                                   'type': 'W'}],
               'memory_format=channels_last 4d': ['ok',
                                                  {'device': 'cpu',
                                                   'dtype': 'torch.float32',
                                                   'is_contiguous': True,
                                                   'is_leaf': True,
                                                   'keys': ['CPU',
                                                            'Python',
                                                            'PythonTLSSnapshot',
                                                            'ADInplaceOrView',
                                                            'AutogradCPU',
                                                            'AutocastCPU'],
                                                   'layout': 'torch.strided',
                                                   'requires_grad': False,
                                                   'shape': [2, 3, 4, 5],
                                                   'storage_offset': 0,
                                                   'stride': [60, 20, 5, 1],
                                                   'type': 'W'}],
               'memory_format=contiguous_format': ['ok',
                                                   {'device': 'cpu',
                                                    'dtype': 'torch.float32',
                                                    'is_contiguous': True,
                                                    'is_leaf': True,
                                                    'keys': ['CPU',
                                                             'Python',
                                                             'PythonTLSSnapshot',
                                                             'ADInplaceOrView',
                                                             'AutogradCPU',
                                                             'AutocastCPU'],
                                                    'layout': 'torch.strided',
                                                    'requires_grad': False,
                                                    'shape': [2, 3],
                                                    'storage_offset': 0,
                                                    'stride': [3, 1],
                                                    'type': 'W'}],
               'memory_format=preserve_format': ['ok',
                                                 {'device': 'cpu',
                                                  'dtype': 'torch.float32',
                                                  'is_contiguous': True,
                                                  'is_leaf': True,
                                                  'keys': ['CPU',
                                                           'Python',
                                                           'PythonTLSSnapshot',
                                                           'ADInplaceOrView',
                                                           'AutogradCPU',
                                                           'AutocastCPU'],
                                                  'layout': 'torch.strided',
                                                  'requires_grad': False,
                                                  'shape': [2, 3],
                                                  'storage_offset': 0,
                                                  'stride': [3, 1],
                                                  'type': 'W'}],
               'pin_memory=False': ['ok',
                                    {'device': 'cpu',
                                     'dtype': 'torch.float32',
                                     'is_contiguous': True,
                                     'is_leaf': True,
                                     'keys': ['CPU',
                                              'Python',
                                              'PythonTLSSnapshot',
                                              'ADInplaceOrView',
                                              'AutogradCPU',
                                              'AutocastCPU'],
                                     'layout': 'torch.strided',
                                     'requires_grad': False,
                                     'shape': [2, 3],
                                     'storage_offset': 0,
                                     'stride': [3, 1],
                                     'type': 'W'}],
               'requires_grad=True': ['ok',
                                      {'device': 'cpu',
                                       'dtype': 'torch.float32',
                                       'is_contiguous': True,
                                       'is_leaf': True,
                                       'keys': ['CPU',
                                                'Python',
                                                'PythonTLSSnapshot',
                                                'ADInplaceOrView',
                                                'AutogradCPU',
                                                'AutocastCPU'],
                                       'layout': 'torch.strided',
                                       'requires_grad': True,
                                       'shape': [2, 3],
                                       'storage_offset': 0,
                                       'stride': [3, 1],
                                       'type': 'W'}],
               'requires_grad=True complex64': ['ok',
                                                {'device': 'cpu',
                                                 'dtype': 'torch.complex64',
                                                 'is_contiguous': True,
                                                 'is_leaf': True,
                                                 'keys': ['CPU',
                                                          'Python',
                                                          'PythonTLSSnapshot',
                                                          'ADInplaceOrView',
                                                          'AutogradCPU',
                                                          'AutocastCPU'],
                                                 'layout': 'torch.strided',
                                                 'requires_grad': True,
                                                 'shape': [2, 3],
                                                 'storage_offset': 0,
                                                 'stride': [3, 1],
                                                 'type': 'W'}],
               'size=()': ['ok',
                           {'device': 'cpu',
                            'dtype': 'torch.float32',
                            'is_contiguous': True,
                            'is_leaf': True,
                            'keys': ['CPU',
                                     'Python',
                                     'PythonTLSSnapshot',
                                     'ADInplaceOrView',
                                     'AutogradCPU',
                                     'AutocastCPU'],
                            'layout': 'torch.strided',
                            'requires_grad': False,
                            'shape': [],
                            'storage_offset': 0,
                            'stride': [],
                            'type': 'W'}],
               'size=(0,)': ['ok',
                             {'device': 'cpu',
                              'dtype': 'torch.float32',
                              'is_contiguous': True,
                              'is_leaf': True,
                              'keys': ['CPU',
                                       'Python',
                                       'PythonTLSSnapshot',
                                       'ADInplaceOrView',
                                       'AutogradCPU',
                                       'AutocastCPU'],
                              'layout': 'torch.strided',
                              'requires_grad': False,
                              'shape': [0],
                              'storage_offset': 0,
                              'stride': [1],
                              'type': 'W'}],
               'size=(0,3) strides=(5,7)': ['ok',
                                            {'device': 'cpu',
                                             'dtype': 'torch.float32',
                                             'is_contiguous': True,
                                             'is_leaf': True,
                                             'keys': ['CPU',
                                                      'Python',
                                                      'PythonTLSSnapshot',
                                                      'ADInplaceOrView',
                                                      'AutogradCPU',
                                                      'AutocastCPU'],
                                             'layout': 'torch.strided',
                                             'requires_grad': False,
                                             'shape': [0, 3],
                                             'storage_offset': 0,
                                             'stride': [5, 7],
                                             'type': 'W'}],
               'size=(2,3,4,5)': ['ok',
                                  {'device': 'cpu',
                                   'dtype': 'torch.float32',
                                   'is_contiguous': True,
                                   'is_leaf': True,
                                   'keys': ['CPU',
                                            'Python',
                                            'PythonTLSSnapshot',
                                            'ADInplaceOrView',
                                            'AutogradCPU',
                                            'AutocastCPU'],
                                   'layout': 'torch.strided',
                                   'requires_grad': False,
                                   'shape': [2, 3, 4, 5],
                                   'storage_offset': 0,
                                   'stride': [60, 20, 5, 1],
                                   'type': 'W'}],
               'size=list': ['ok',
                             {'device': 'cpu',
                              'dtype': 'torch.float32',
                              'is_contiguous': True,
                              'is_leaf': True,
                              'keys': ['CPU',
                                       'Python',
                                       'PythonTLSSnapshot',
                                       'ADInplaceOrView',
                                       'AutogradCPU',
                                       'AutocastCPU'],
                              'layout': 'torch.strided',
                              'requires_grad': False,
                              'shape': [2, 3],
                              'storage_offset': 0,
                              'stride': [3, 1],
                              'type': 'W'}],
               'size=torch.Size': ['ok',
                                   {'device': 'cpu',
                                    'dtype': 'torch.float32',
                                    'is_contiguous': True,
                                    'is_leaf': True,
                                    'keys': ['CPU',
                                             'Python',
                                             'PythonTLSSnapshot',
                                             'ADInplaceOrView',
                                             'AutogradCPU',
                                             'AutocastCPU'],
                                    'layout': 'torch.strided',
                                    'requires_grad': False,
                                    'shape': [2, 3],
                                    'storage_offset': 0,
                                    'stride': [3, 1],
                                    'type': 'W'}],
               'storage_size=None': ['ok',
                                     {'device': 'cpu',
                                      'dtype': 'torch.float32',
                                      'is_contiguous': True,
                                      'is_leaf': True,
                                      'keys': ['CPU',
                                               'Python',
                                               'PythonTLSSnapshot',
                                               'ADInplaceOrView',
                                               'AutogradCPU',
                                               'AutocastCPU'],
                                      'layout': 'torch.strided',
                                      'requires_grad': False,
                                      'shape': [2, 3],
                                      'storage_offset': 0,
                                      'stride': [3, 1],
                                      'type': 'W'}],
               'strides=(0,0)': ['ok',
                                 {'device': 'cpu',
                                  'dtype': 'torch.float32',
                                  'is_contiguous': False,
                                  'is_leaf': True,
                                  'keys': ['CPU',
                                           'Python',
                                           'PythonTLSSnapshot',
                                           'ADInplaceOrView',
                                           'AutogradCPU',
                                           'AutocastCPU'],
                                  'layout': 'torch.strided',
                                  'requires_grad': False,
                                  'shape': [2, 3],
                                  'storage_offset': 0,
                                  'stride': [0, 0],
                                  'type': 'W'}],
               'strides=(1,2)': ['ok',
                                 {'device': 'cpu',
                                  'dtype': 'torch.float32',
                                  'is_contiguous': False,
                                  'is_leaf': True,
                                  'keys': ['CPU',
                                           'Python',
                                           'PythonTLSSnapshot',
                                           'ADInplaceOrView',
                                           'AutogradCPU',
                                           'AutocastCPU'],
                                  'layout': 'torch.strided',
                                  'requires_grad': False,
                                  'shape': [2, 3],
                                  'storage_offset': 0,
                                  'stride': [1, 2],
                                  'type': 'W'}],
               'strides=(1,2) storage_offset=5': ['ok',
                                                  {'device': 'cpu',
                                                   'dtype': 'torch.float32',
                                                   'is_contiguous': False,
                                                   'is_leaf': True,
                                                   'keys': ['CPU',
                                                            'Python',
                                                            'PythonTLSSnapshot',
                                                            'ADInplaceOrView',
                                                            'AutogradCPU',
                                                            'AutocastCPU'],
                                                   'layout': 'torch.strided',
                                                   'requires_grad': False,
                                                   'shape': [2, 3],
                                                   'storage_offset': 5,
                                                   'stride': [1, 2],
                                                   'type': 'W'}],
               'strides=(100,7)': ['ok',
                                   {'device': 'cpu',
                                    'dtype': 'torch.float32',
                                    'is_contiguous': False,
                                    'is_leaf': True,
                                    'keys': ['CPU',
                                             'Python',
                                             'PythonTLSSnapshot',
                                             'ADInplaceOrView',
                                             'AutogradCPU',
                                             'AutocastCPU'],
                                    'layout': 'torch.strided',
                                    'requires_grad': False,
                                    'shape': [2, 3],
                                    'storage_offset': 0,
                                    'stride': [100, 7],
                                    'type': 'W'}]},
 'declined': {'n + 1 (type only)': ['TypeError'],
              'n.add_(1)': ['TypeError',
                            "Multiple dispatch failed for 'torch._ops.aten.add_.Tensor'; all "
                            '__torch_dispatch__ handlers returned NotImplemented:\n'
                            '\n'
                            "  - tensor subclass <class '__main__.N'>\n"
                            '\n'
                            'For more information, try re-running with TORCH_LOGS=not_implemented',
                            [['N', 'aten.add_.Tensor']]],
              'n.view(6)': ['TypeError',
                            "Multiple dispatch failed for 'torch._ops.aten.view.default'; all "
                            '__torch_dispatch__ handlers returned NotImplemented:\n'
                            '\n'
                            "  - tensor subclass <class '__main__.N'>\n"
                            '\n'
                            'For more information, try re-running with TORCH_LOGS=not_implemented',
                            [['N', 'aten.view.default']]],
              'torch.add(n, d)': ['TypeError',
                                  "Multiple dispatch failed for 'torch._ops.aten.add.Tensor'; all "
                                  '__torch_dispatch__ handlers returned NotImplemented:\n'
                                  '\n'
                                  "  - tensor subclass <class '__main__.N'>\n"
                                  '\n'
                                  'For more information, try re-running with '
                                  'TORCH_LOGS=not_implemented',
                                  [['N', 'aten.add.Tensor']]],
              'torch.add(n, n2)': ['TypeError',
                                   "Multiple dispatch failed for 'torch._ops.aten.add.Tensor'; all "
                                   '__torch_dispatch__ handlers returned NotImplemented:\n'
                                   '\n'
                                   "  - tensor subclass <class '__main__.N2'>\n"
                                   "  - tensor subclass <class '__main__.N'>\n"
                                   '\n'
                                   'For more information, try re-running with '
                                   'TORCH_LOGS=not_implemented',
                                   [['N2', 'aten.add.Tensor'], ['N', 'aten.add.Tensor']]],
              'torch.add(n2, n)': ['TypeError',
                                   "Multiple dispatch failed for 'torch._ops.aten.add.Tensor'; all "
                                   '__torch_dispatch__ handlers returned NotImplemented:\n'
                                   '\n'
                                   "  - tensor subclass <class '__main__.N2'>\n"
                                   "  - tensor subclass <class '__main__.N'>\n"
                                   '\n'
                                   'For more information, try re-running with '
                                   'TORCH_LOGS=not_implemented',
                                   [['N2', 'aten.add.Tensor'], ['N', 'aten.add.Tensor']]],
              'torch.cat([n, d])': ['TypeError',
                                    "Multiple dispatch failed for 'torch._ops.aten.cat.default'; "
                                    'all __torch_dispatch__ handlers returned NotImplemented:\n'
                                    '\n'
                                    "  - tensor subclass <class '__main__.N'>\n"
                                    '\n'
                                    'For more information, try re-running with '
                                    'TORCH_LOGS=not_implemented',
                                    [['N', 'aten.cat.default']]],
              'torch.zeros_like(n)': ['TypeError',
                                      'Multiple dispatch failed for '
                                      "'torch._ops.aten.zeros_like.default'; all "
                                      '__torch_dispatch__ handlers returned NotImplemented:\n'
                                      '\n'
                                      "  - tensor subclass <class '__main__.N'>\n"
                                      '\n'
                                      'For more information, try re-running with '
                                      'TORCH_LOGS=not_implemented',
                                      [['N', 'aten.zeros_like.default']]],
              'under a declining mode: n.view(6)': ['TypeError',
                                                    'Multiple dispatch failed for '
                                                    "'torch._ops.aten.view.default'; all "
                                                    '__torch_dispatch__ handlers returned '
                                                    'NotImplemented:\n'
                                                    '\n'
                                                    '  - mode object <__main__.MN object at 0x>\n'
                                                    "  - tensor subclass <class '__main__.N'>\n"
                                                    '\n'
                                                    'For more information, try re-running with '
                                                    'TORCH_LOGS=not_implemented',
                                                    [['mode-declines', 'aten.view.default'],
                                                     ['N', 'aten.view.default']]]},
 'dispatch': {'a declining class, then the next: torch.add(n, w)': ['ok',
                                                                    "the class's answer",
                                                                    [['N', 'aten.add.Tensor'],
                                                                     ['W',
                                                                      'aten.add.Tensor',
                                                                      ['N', 'W'],
                                                                      ['N', 'W']]]],
              'a mode alone: torch.add(d, d)': ['ok',
                                                'something else',
                                                [['mode', 'aten.add.Tensor', []]]],
              'a mode declines, the class answers: torch.add(w, d)': ['ok',
                                                                      "the class's answer",
                                                                      [['mode-declines',
                                                                        'aten.add.Tensor'],
                                                                       ['W',
                                                                        'aten.add.Tensor',
                                                                        ['W'],
                                                                        ['W', 'Tensor']]]],
              'a mode sees it first: torch.add(w, d)': ['ok',
                                                        "the class's answer",
                                                        [['mode', 'aten.add.Tensor', ['W']],
                                                         ['W',
                                                          'aten.add.Tensor',
                                                          ['W'],
                                                          ['W', 'Tensor']]]],
              'dc.copy_(w)': ['ok',
                              'the receiver',
                              [['W', 'aten.copy_.default', ['W'], ['Tensor', 'W']]]],
              'instance method: torch.add(d, wi)': ['ok',
                                                    "the class's answer",
                                                    [['I', 'I', 'aten.add.Tensor', ['I']]]],
              'instance method: torch.add(wi, d)': ['ok',
                                                    "the class's answer",
                                                    [['I', 'I', 'aten.add.Tensor', ['I']]]],
              'requires_grad=True: torch.add(wg, d)': ['ok',
                                                       "the class's answer",
                                                       [['W',
                                                         'aten.add.Tensor',
                                                         ['W'],
                                                         ['W', 'Tensor']]]],
              'subclass first: torch.add(a, b)': ['ok',
                                                  "the class's answer",
                                                  [['B', 'aten.add.Tensor', ['B', 'A']]]],
              'subclass first: torch.add(b, a)': ['ok',
                                                  "the class's answer",
                                                  [['B', 'aten.add.Tensor', ['B', 'A']]]],
              'torch.add(d, w)': ['ok',
                                  "the class's answer",
                                  [['W', 'aten.add.Tensor', ['W'], ['Tensor', 'W']]]],
              'torch.add(w, d)': ['ok',
                                  "the class's answer",
                                  [['W', 'aten.add.Tensor', ['W'], ['W', 'Tensor']]]],
              'torch.cat([w, d])': ['ok',
                                    "the class's answer",
                                    [['W', 'aten.cat.default', ['W'], [['seq', 'W', 'Tensor']]]]],
              'torch.ops.aten.add.Tensor(w, d)': ['ok',
                                                  "the class's answer",
                                                  [['W',
                                                    'aten.add.Tensor',
                                                    ['W'],
                                                    ['W', 'Tensor']]]],
              'torch.stack([d, w])': ['ok',
                                      "the class's answer",
                                      [['W',
                                        'aten.stack.default',
                                        ['W'],
                                        [['seq', 'Tensor', 'W']]]]],
              'torch.zeros_like(w)': ['ok',
                                      "the class's answer",
                                      [['W', 'aten.zeros_like.default', ['W'], ['W']]]],
              'two classes: torch.add(w, wi)': ['ok',
                                                "the class's answer",
                                                [['W', 'aten.add.Tensor', ['W', 'I'], ['W', 'I']]]],
              'two classes: torch.add(wi, w)': ['ok',
                                                "the class's answer",
                                                [['I', 'I', 'aten.add.Tensor', ['I', 'W']]]],
              'w + 1': ['ok',
                        "the class's answer",
                        [['W', 'aten.add.Tensor', ['W'], ['W', 'int']]]],
              'w == d': ['ok',
                         "the class's answer",
                         [['W', 'aten.eq.Tensor', ['W'], ['W', 'Tensor']]]],
              'w.add_(1)': ['ok', 'the receiver', [['W', 'aten.add_.Tensor', ['W'], ['W', 'int']]]],
              'w.clone()': ['ok',
                            "the class's answer",
                            [['W', 'aten.clone.default', ['W'], ['W']]]],
              'w.sum()': ['ok', "the class's answer", [['W', 'aten.sum.default', ['W'], ['W']]]],
              'w.t()': ['ok', "the class's answer", [['W', 'aten.t.default', ['W'], ['W']]]],
              'w.view(6)': ['ok',
                            "the class's answer",
                            [['W', 'aten.view.default', ['W'], ['W', ['seq', 'int']]]]]},
 'queries': {'an attribute sticks': ['ok', 5, {'dispatched': []}],
             'bool(w)': ['RuntimeError',
                         'Boolean value of Tensor with more than one value is ambiguous',
                         {'dispatched': []}],
             'data_ptr() after _set_throw_on_mutable_data_ptr': ['RuntimeError'],
             'dispatch keys': ['ok',
                               ['CPU',
                                'Python',
                                'PythonTLSSnapshot',
                                'ADInplaceOrView',
                                'AutogradCPU',
                                'AutocastCPU'],
                               {'dispatched': []}],
             'has the Python key': ['ok', True, {'dispatched': []}],
             'isinstance(w, W)': ['ok', True, {'dispatched': []}],
             'isinstance(w, torch.Tensor)': ['ok', True, {'dispatched': []}],
             'item() of a 0-d wrapper asks the class': ['ok', ['aten._local_scalar_dense.default']],
             'len(w)': ['ok', 2, {'dispatched': []}],
             'requires_grad_(True) afterwards': ['ok', [True, True, True]],
             'str(w.device)': ['ok', 'cpu', {'dispatched': []}],
             'str(w.dtype)': ['ok', 'torch.float32', {'dispatched': []}],
             'str(w.layout)': ['ok', 'torch.strided', {'dispatched': []}],
             'strided wrapper: _is_view(), data_ptr()': ['ok', [False, 20]],
             'torch._C._has_storage(w)': ['ok', True, {'dispatched': []}],
             'torch._is_functional_tensor(w)': ['ok', False, {'dispatched': []}],
             'tuple(w.shape)': ['ok', [2, 3], {'dispatched': []}],
             'type(w).__name__': ['ok', 'W', {'dispatched': []}],
             'type(w.shape).__name__': ['ok', 'Size', {'dispatched': []}],
             'w._base is None': ['ok', True, {'dispatched': []}],
             'w._is_view()': ['ok', False, {'dispatched': []}],
             'w._is_zerotensor()': ['ok', False, {'dispatched': []}],
             'w.data_ptr()': ['ok', 0, {'dispatched': []}],
             'w.dim()': ['ok', 2, {'dispatched': []}],
             'w.element_size()': ['ok', 4, {'dispatched': []}],
             'w.get_device()': ['ok', -1, {'dispatched': []}],
             'w.grad is None': ['ok', True, {'dispatched': []}],
             'w.grad_fn is None': ['ok', True, {'dispatched': []}],
             'w.is_conj()': ['ok', False, {'dispatched': []}],
             'w.is_contiguous()': ['ok', True, {'dispatched': []}],
             'w.is_cpu': ['ok', True, {'dispatched': []}],
             'w.is_inference()': ['ok', False, {'dispatched': []}],
             'w.is_leaf': ['ok', True, {'dispatched': []}],
             'w.is_meta': ['ok', False, {'dispatched': []}],
             'w.is_neg()': ['ok', False, {'dispatched': []}],
             'w.is_nested': ['ok', False, {'dispatched': []}],
             'w.is_quantized': ['ok', False, {'dispatched': []}],
             'w.is_sparse': ['ok', False, {'dispatched': []}],
             'w.item()': ['RuntimeError',
                          'a Tensor with 6 elements cannot be converted to Scalar',
                          {'dispatched': []}],
             'w.ndim': ['ok', 2, {'dispatched': []}],
             'w.numel()': ['ok', 6, {'dispatched': []}],
             'w.requires_grad': ['ok', False, {'dispatched': []}],
             'w.size()': ['ok', [2, 3], {'dispatched': []}],
             'w.size(-1)': ['ok', 3, {'dispatched': []}],
             'w.size(0)': ['ok', 2, {'dispatched': []}],
             'w.storage_offset()': ['ok', 0, {'dispatched': []}],
             'w.stride()': ['ok', [3, 1], {'dispatched': []}],
             'w.stride(0)': ['ok', 3, {'dispatched': []}],
             'w.tolist()': ['RuntimeError',
                            '.tolist() is not supported for tensor subclasses, got W',
                            {'dispatched': []}]},
 'refused_by_upstream': {'_extra_dispatch_keys=3': ['TypeError',
                                                    '_make_wrapper_subclass(): argument '
                                                    "'_extra_dispatch_keys' must be "
                                                    'DispatchKeySet, not int'],
                         'cls only': ['TypeError',
                                      '_make_wrapper_subclass() missing 1 required positional '
                                      'arguments: "size"'],
                         'cls without __torch_dispatch__': ['TypeError',
                                                            'Plain must define __torch_dispatch__'],
                         'cls=3': ['TypeError', 'cls must be a type (got int)'],
                         'cls=None': ['TypeError', 'cls must be a type (got NoneType)'],
                         'cls=int': ['TypeError',
                                     'Creating a Tensor subclass from a class that does not '
                                     'inherit from Tensor is not possible. Make sure your class '
                                     'inherits from Tensor.'],
                         'cls=torch.Tensor': ['TypeError', 'Tensor must define __torch_dispatch__'],
                         'device=1.5': ['TypeError',
                                        "_make_wrapper_subclass(): argument 'device' must be "
                                        'torch.device, not float'],
                         'dispatch_device=None': ['TypeError',
                                                  '_make_wrapper_subclass(): argument '
                                                  "'dispatch_device' must be bool, not NoneType"],
                         'dispatch_layout=None': ['TypeError',
                                                  '_make_wrapper_subclass(): argument '
                                                  "'dispatch_layout' must be bool, not NoneType"],
                         "dispatch_sizes_strides_policy='bogus'": ['ValueError',
                                                                   'Unknown sizes_strides_policy: '
                                                                   "bogus; expected 'strides' or "
                                                                   "'sizes'"],
                         'dispatch_sizes_strides_policy=3': ['TypeError',
                                                             '_make_wrapper_subclass(): argument '
                                                             "'dispatch_sizes_strides_policy' must "
                                                             'be str, not int'],
                         "dtype='float32'": ['TypeError',
                                             "_make_wrapper_subclass(): argument 'dtype' must be "
                                             'torch.dtype, not str'],
                         "layout='strided'": ['TypeError',
                                              "_make_wrapper_subclass(): argument 'layout' must be "
                                              'torch.layout, not str'],
                         'layout=None': ['TypeError',
                                         "_make_wrapper_subclass(): argument 'layout' must be "
                                         'torch.layout, not NoneType'],
                         "memory_format='x'": ['TypeError',
                                               "_make_wrapper_subclass(): argument 'memory_format' "
                                               'must be torch.memory_format, not str'],
                         'no arguments': ['TypeError',
                                          '_make_wrapper_subclass() missing 2 required positional '
                                          'argument: "cls", "size"'],
                         'pin_memory=None': ['TypeError',
                                             "_make_wrapper_subclass(): argument 'pin_memory' must "
                                             'be bool, not NoneType'],
                         'requires_grad=1': ['TypeError',
                                             "_make_wrapper_subclass(): argument 'requires_grad' "
                                             'must be bool, not int'],
                         'requires_grad=True bool': ['RuntimeError',
                                                     'Only Tensors of floating point and complex '
                                                     'dtype can require gradients'],
                         'requires_grad=True int64': ['RuntimeError',
                                                      'Only Tensors of floating point and complex '
                                                      'dtype can require gradients'],
                         'sixteen positional': ['TypeError',
                                                '_make_wrapper_subclass() takes from 2 to 15 '
                                                'positional arguments but 16 were given'],
                         'size twice': ['TypeError',
                                        '_make_wrapper_subclass() got multiple values for argument '
                                        "'size'"],
                         "size='ab'": ['TypeError',
                                       "_make_wrapper_subclass(): argument 'size' (position 2) "
                                       'must be tuple of ints, not str'],
                         'size=(1.5,)': ['TypeError',
                                         "_make_wrapper_subclass(): argument 'size' (position 2) "
                                         'must be tuple of ints, but found element of type float '
                                         'at pos 0'],
                         'size=(True, 3)': ['TypeError',
                                            "_make_wrapper_subclass(): argument 'size' (position "
                                            '2) must be tuple of ints, but found element of type '
                                            'bool at pos 0'],
                         'size=3': ['TypeError',
                                    "_make_wrapper_subclass(): argument 'size' (position 2) must "
                                    'be tuple of ints, not int'],
                         'size=None': ['TypeError',
                                       "_make_wrapper_subclass(): argument 'size' (position 2) "
                                       'must be tuple of ints, not NoneType'],
                         'storage_offset=0 without strides': ['RuntimeError',
                                                              'setting storage offset without '
                                                              'stride not supported'],
                         'storage_offset=1.5': ['TypeError',
                                                '_make_wrapper_subclass(): argument '
                                                "'storage_offset' must be int, not float"],
                         'storage_offset=5 without strides': ['RuntimeError',
                                                              'setting storage offset without '
                                                              'stride not supported'],
                         "storage_size='x'": ['TypeError',
                                              "_make_wrapper_subclass(): argument 'storage_size' "
                                              'must be int, not str'],
                         'strides of another rank': ['RuntimeError',
                                                     'dimensionality of sizes (2) must match '
                                                     'dimensionality of strides (1)'],
                         "strides='x'": ['TypeError',
                                         "_make_wrapper_subclass(): argument 'strides' must be "
                                         'tuple of ints, not str'],
                         'strides=(1.5,)': ['TypeError',
                                            "_make_wrapper_subclass(): argument 'strides' must be "
                                            'tuple of ints, not tuple'],
                         'unexpected keyword': ['TypeError',
                                                '_make_wrapper_subclass() got an unexpected '
                                                "keyword argument 'bogus'"]}}

_SECTIONS = ("construct", "refused_by_upstream", "queries", "dispatch", "declined")

_CACHE = {}


def _cached(key, script, shim):
    if (key, shim) not in _CACHE:
        _CACHE[(key, shim)] = _run(script, shim)
    return _CACHE[(key, shim)]


def _shim_table():
    """The table as the shim answers it, or None after registering a skip."""
    if not _available():
        _skip.skip("no vendored torch tree")
        return None
    return _cached("table", _TABLE, True)


def _differences(section, got):
    want = EXPECTED[section]
    lines = []
    for key in sorted(set(want) | set(got)):
        if key not in got:
            lines.append(f"{key}: missing on the shim; upstream {want[key]!r}")
        elif key not in want:
            lines.append(f"{key}: not in the pasted table; the shim {got[key]!r}")
        elif want[key] != got[key]:
            lines.append(f"{key}:\n      upstream {want[key]!r}\n      shim     {got[key]!r}")
    return lines


def _assert_section_agrees(section):
    table = _shim_table()
    if table is None:
        return None
    lines = _differences(section, table[section])
    assert not lines, (
        f"{len(lines)} of {len(EXPECTED[section])} rows of {section!r} differ from "
        f"upstream 2.13.0:\n  " + "\n  ".join(lines)
    )
    return len(EXPECTED[section])


def test_the_pasted_table_is_what_upstream_answers_today():
    """`EXPECTED` is a paste. This is what keeps it from being a stale one."""
    try:
        upstream = _cached("table", _TABLE, False)
    except _NoUpstream:
        _skip.skip("no upstream torch in this interpreter to re-derive the table from")
        return
    for section in _SECTIONS:
        lines = []
        want, got = EXPECTED[section], upstream[section]
        for key in sorted(set(want) | set(got)):
            if want.get(key) != got.get(key):
                lines.append(f"{key}:\n      pasted   {want.get(key)!r}\n      upstream {got.get(key)!r}")
        assert not lines, (
            f"the pasted table's {section!r} is not what upstream answers:\n  "
            + "\n  ".join(lines)
        )
    assert set(upstream) - {"is_shim"} == set(_SECTIONS), sorted(upstream)
    rows = sum(len(EXPECTED[s]) for s in _SECTIONS)
    assert rows >= 157, f"the table shrank to {rows} rows"


def test_construction_and_every_argument_agree_with_upstream():
    """What each argument changes, and what upstream says to a bad one."""
    built = _assert_section_agrees("construct")
    if built is None:
        return
    refused = _assert_section_agrees("refused_by_upstream")
    assert built >= 38 and refused >= 36, (built, refused)


def test_a_wrapper_answers_metadata_as_upstream_does():
    """Every query, and that none of them called the class to answer."""
    rows = _assert_section_agrees("queries")
    if rows is None:
        return
    # A property of the pasted table itself, so that an edit to it cannot
    # quietly record a query that dispatches.
    dispatched = {
        name: row[-1]["dispatched"]
        for name, row in EXPECTED["queries"].items()
        if isinstance(row[-1], dict) and row[-1]["dispatched"]
    }
    assert not dispatched, f"the pasted table has a query that dispatches: {dispatched}"
    assert rows >= 48, rows


def test_the_dispatch_protocol_agrees_with_upstream():
    """Which handler is called, with what, in what order, and what comes back."""
    rows = _assert_section_agrees("dispatch")
    if rows is None:
        return
    assert rows >= 25, rows


def test_a_class_that_declines_gets_upstreams_typeerror():
    """`NotImplemented` from every handler is upstream's `TypeError`."""
    rows = _assert_section_agrees("declined")
    if rows is None:
        return
    assert rows >= 9, rows


_REFUSED_ARGUMENTS = {
    "size", "strides", "storage_offset", "layout", "device", "pin_memory",
    "dispatch_sizes_strides_policy", "dispatch_device", "dispatch_layout",
    "_extra_dispatch_keys", "storage_size",
}


def test_each_argument_the_shim_cannot_represent_refuses_naming_it():
    """Sixteen values upstream accepts; each refuses here naming its argument."""
    if not _available():
        _skip.skip("no vendored torch tree")
        return
    shim = _cached("refused", _REFUSED, True)["refused"]
    assert len(shim) == 16, len(shim)
    assert {argument for argument, _, _ in shim} == _REFUSED_ARGUMENTS
    wrong = []
    for argument, what, outcome in shim:
        named = (
            outcome[0] == "NotImplementedError"
            and f"argument '{argument}'" in outcome[1]
            and "_make_wrapper_subclass" in outcome[1]
        )
        if not named:
            wrong.append(f"{argument} ({what}): {outcome!r}")
    assert not wrong, (
        "these did not refuse naming their argument:\n  " + "\n  ".join(wrong)
    )
    # The other half: each one is something upstream ACCEPTS. A value upstream
    # refuses too belongs in the table above, in upstream's words.
    try:
        upstream = _cached("refused", _REFUSED, False)["refused"]
    except _NoUpstream:
        _skip.skip("the shim half passed; no upstream torch to confirm it accepts these")
        return
    # A bare device index means "the current accelerator" upstream, so where
    # the host has none upstream refuses it too (CI run 43670318805, Linux:
    # "Cannot access accelerator device when none is available."). There
    # the shim's refusal is parity; on a host with an accelerator (the Mac,
    # where the table was measured) upstream accepts it and the row counts.
    host_has_no_accelerator = "Cannot access accelerator device when none is available."
    rejected = [
        f"{a} ({w}): {o!r}" for a, w, o in upstream
        if o != ["ok"] and not (len(o) > 1 and host_has_no_accelerator in str(o[1]))
    ]
    assert not rejected, (
        "upstream no longer accepts these, so refusing them is parity and not a "
        "narrowing:\n  " + "\n  ".join(rejected)
    )


_DIRECT_GROUPS = ("pure", "view", "in-place", "factory-like", "sequence")
_DOOR = "refuses to run a kernel on a wrapper tensor"


def test_a_wrapper_never_falls_to_a_dense_kernel():
    """The leader's condition: refuse by name, never compute as if dense.

    Upstream measured on 2.13.0 for the same thirty-four calls with the
    declining class: every one raises `TypeError`, none returns.
    """
    if not _available():
        _skip.skip("no vendored torch tree")
        return
    rows = _cached("leader", _LEADER, True)["rows"]
    assert len(rows) == 34, len(rows)
    assert {row["group"] for row in rows} == set(_DIRECT_GROUPS) | {"python-level"}
    problems = []
    door_refusals = 0
    for row in rows:
        op, group = row["op"], row["group"]
        direct = group in _DIRECT_GROUPS

        control = row["control"]
        if control["outcome"][0] != "returned":
            problems.append(
                f"{op}: does not run on a dense tensor ({control['outcome']!r}), so "
                f"its refusal on a wrapper proves nothing")
        if control["door_refusals"] != 0:
            problems.append(f"{op}: the door counted a refusal for a dense tensor")

        got = row["class declines"]
        kind, message = got["outcome"][0], got["outcome"][-1]
        upstreams = (
            kind == "TypeError"
            and "Multiple dispatch failed for 'torch._ops.aten." in message
            and "<class '__main__.N'>" in message
        )
        operator = kind == "TypeError" and "unsupported operand" in message
        backstop = kind == "NotImplementedError" and "wrapper tensor" in message
        if kind == "returned":
            problems.append(f"{op}: RETURNED a {message} although the class declined")
        elif not (upstreams or (not direct and (operator or backstop))):
            problems.append(f"{op}: the class declined and the call raised {got['outcome']!r}")
        if not backstop and "N" not in got["asked"]:
            problems.append(f"{op}: the declining class was never asked")
        if got["door_refusals"] != 0:
            problems.append(f"{op}: a declined call went on to the native door")

        for variant, owner in (("class no longer answers", "Gone"), ("no_dispatch()", "W")):
            got = row[variant]
            kind, message = got["outcome"][0], got["outcome"][-1]
            at_the_door = kind == "NotImplementedError" and _DOOR in message
            backstop = (
                kind == "NotImplementedError"
                and "wrapper tensor" in message
                and not at_the_door
            )
            if kind == "returned":
                problems.append(f"{op} [{variant}]: RETURNED a {message}")
            elif at_the_door:
                if not message.startswith("aten.") or f"class {owner}" not in message:
                    problems.append(
                        f"{op} [{variant}]: the refusal does not name the operator "
                        f"and the class: {message!r}")
                if got["door_refusals"] != 1:
                    problems.append(
                        f"{op} [{variant}]: one refusal, and the counter moved by "
                        f"{got['door_refusals']}")
                door_refusals += 1
            elif backstop and not direct:
                if got["door_refusals"] != 0:
                    problems.append(f"{op} [{variant}]: counted a refusal it did not word")
            else:
                problems.append(f"{op} [{variant}]: raised {got['outcome']!r}")
            if got["asked"]:
                problems.append(f"{op} [{variant}]: the class was asked: {got['asked']}")
    assert not problems, (
        f"{len(problems)} problems over {len(rows)} calls:\n  " + "\n  ".join(problems)
    )
    direct_rows = sum(1 for row in rows if row["group"] in _DIRECT_GROUPS)
    assert direct_rows == 22, direct_rows
    assert door_refusals >= 2 * direct_rows, (door_refusals, direct_rows)


def test_the_shims_own_refusals_on_a_wrapper_name_themselves():
    """`untyped_storage()` refuses by name, and `_shim_is_wrapper` tells the arms apart."""
    if not _available():
        _skip.skip("no vendored torch tree")
        return
    shim = _cached("leader", _LEADER, True)
    outcome = shim["untyped_storage"]
    assert outcome[0] == "NotImplementedError", outcome
    assert "TensorBase.untyped_storage" in outcome[1] and "wrapper tensor" in outcome[1], outcome
    # A wrapper; a dense tensor; a meta tensor; a `_make_subclass` tensor whose
    # class overrides `__torch_dispatch__` (a FakeTensor's shape). Only the first.
    assert shim["is_wrapper"] == [True, False, False, False], shim["is_wrapper"]


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_wrapper_subclass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
