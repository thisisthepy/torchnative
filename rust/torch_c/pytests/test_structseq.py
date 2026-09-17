"""The wall nine of the ten exportable architectures share -- and what it
actually is.

`docs/graph/VARMEAN.md` §4 localised the stop to one line and named it a
result *type*:

    proxy_tensor.py:714  return val.__class__([extract_val(x) for x in val])
    TypeError: return_types_native_layer_norm.__new__() missing 2 required
               positional arguments: 'out1' and 'out2'

and §4.2 named the next measurement -- `_cached_dispatch_impl`'s cache -- as
this round's first task.  That measurement was made, and it moves the cause
two levels below where the traceback points.  `docs/graph/STRUCTSEQ.md` is the
record; the short form, because these tests are shaped by it:

* Upstream's *first*, uncached `aten.native_layer_norm.default` under
  `FakeTensorMode` returns the **same `_out_wrapper` NamedTuple this shim
  returns**.  There is no structseq involved, and upstream has no
  `torch.return_types.native_layer_norm` at all.  The plain `tuple`
  `extract_val` sees upstream is built by `_output_from_cache_entry`, which
  ends `return tuple(outputs)`.
* This shim never reaches that line, because `_make_cache_entry` raises
  `_BypassDispatchCache("non-FakeTensor output")` and writes a *negative*
  cache entry.  So the NamedTuple is returned on every call instead of on the
  first.
* It raises that because `native_layer_norm`'s fake outputs are **bare `meta`
  tensors, not `FakeTensor`s** -- and they are bare because the shim's
  dispatcher door is mode-driven only.  With every mode popped, which is
  exactly the state a `fake_impls` handler runs in, a `FakeTensor` argument
  did not reach its own `__torch_dispatch__`.  `fx - fx` came back `meta`.

So the fix is not a result type and is not `native_layer_norm`.  It is the
`Python` dispatch key that upstream carries on the tensor and this shim did
not carry at all.  `test_a_fake_tensor_argument_reaches_its_subclass_with_every_mode_popped`
is that guarantee; everything below it is a consequence, and is asserted
separately so that a regression says which level it happened at.

Every expected answer is **upstream torch's own, computed in a separate
subprocess from the same probe text**.  Nothing here is a table written beside
the implementation.

What these tests cannot see, stated so nobody reads more into them:

* They judge the ops the probes name.  An op not in a probe is not judged.
* `test_the_multi_value_result_classes_match_upstream_after_the_first_call`
  judges the class of a *repeat* dispatch, which is the one `extract_val`
  meets.  It deliberately does not assert anything about the first call,
  because on that call upstream returns the NamedTuple too -- asserting a
  plain tuple there would be asserting something upstream does not do.
* Nothing here claims the ten architectures export.  `export_sweep.py` is the
  only thing that reports that number, and `docs/graph/STRUCTSEQ.md` §5 is
  where it is reported.
"""

import importlib.util
import json
import math
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

#: float32 machine epsilon -- the export test's floor, `docs/numerics/AGREE.md`.
_EPS32 = 1.1920929e-07


def _golden_tolerances():
    """`tools/golden/dtypes.py::TOLERANCES`, imported rather than restated.

    Same reason as `test_varmean.py`: a tolerance written here could be
    widened without anything noticing; one read out of the golden harness
    cannot, because widening it there reddens the harness's own self-test.
    """
    path = os.path.join(_REPO_ROOT, "tools", "golden", "dtypes.py")
    spec = importlib.util.spec_from_file_location("_golden_dtypes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {name: (t.atol, t.rtol) for name, t in module.TOLERANCES.items()}


_TOL = _golden_tolerances()


def _available():
    return os.path.isfile(_VENDOR_SHIM)


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
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # The side each probe landed on is asserted, not assumed: an empty vendored
    # tree makes the "shim" side silently import upstream (CLAUDE.md §3).
    assert out["is_shim"] is shim, f"probe meant for shim={shim} ran on the other side"
    return out


# ---------------------------------------------------------------------------
# Probe 1 -- the root cause.  Every torch-dispatch mode popped, which is the
# state a `fake_impls` handler and a `_refs` body actually run in, and a
# `FakeTensor` argument handed to ordinary ops.
# ---------------------------------------------------------------------------

_SUBCLASS_PROBE = r"""
import json
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
import torch.utils._python_dispatch as pd

out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "cases": {}}
aten = torch.ops.aten

x = torch.randn(4, 6)
w = torch.randn(6)
with FakeTensorMode() as m:
    fx = m.from_tensor(x)
    fw = m.from_tensor(w)


def describe(v):
    items = list(v) if isinstance(v, (tuple, list)) else [v]
    return {
        "cls": type(v).__name__,
        "elems": [
            {"cls": type(e).__name__,
             "is_fake": isinstance(e, FakeTensor),
             "device": str(e.device),
             "shape": list(e.shape),
             "dtype": str(e.dtype).split(".")[-1],
             "stride": list(e.stride())}
            for e in items
        ],
    }


def case(name, fn):
    try:
        out["cases"][name] = describe(fn())
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "message": str(exc)[:300]}


# `_disable_current_modes()` pops the mode stack and the infra slots, which is
# precisely what `FakeTensorMode.dispatch` has already done by the time a
# `fake_impls` handler or a `_refs` body runs.  Upstream still reaches
# `FakeTensor.__torch_dispatch__` here, through the `Python` dispatch key that
# lives on the tensor rather than on the stack.
with pd._disable_current_modes():
    out["stack_depth"] = torch._C._len_torch_dispatch_stack()
    case("sub", lambda: fx - fx)
    case("rsqrt", lambda: torch.rsqrt(fx))
    case("add.Tensor", lambda: aten.add.Tensor(fx, fx))
    case("mul.Tensor", lambda: aten.mul.Tensor(fx, fw))
    case("var_mean.dim", lambda: aten.var_mean.dim(fx, [1], False, True))
    case("contiguous", lambda: aten.contiguous.default(fx))
    case("native_layer_norm", lambda: aten.native_layer_norm.default(fx, [6], fw, fw, 1e-5))

    # The recursion guard.  `in_kernel_invocation_manager` wraps the real meta
    # call in `torch._C._DisableTorchDispatch()` precisely so that FakeTensor
    # arguments do NOT re-enter their own subclass -- without this, the
    # subclass key would recurse until the stack blew.
    #
    # **Upstream cannot be asked what it answers here: it segfaults.**  On
    # torch 2.13.0 in this venv, the first operator applied to a `FakeTensor`
    # inside `torch._C._DisableTorchDispatch()` with the mode stack popped
    # takes SIGSEGV (exit 139), before any output is flushed.  So this block
    # runs on the shim side only, and the test that reads it compares the
    # shim against *itself* rather than inventing an upstream answer.
    if out["is_shim"]:
        with torch._C._DisableTorchDispatch():
            out["suppressed_depth"] = torch._C._len_torch_dispatch_stack()
            case("no_dispatch/sub", lambda: fx - fx)
            case("no_dispatch/mul.Tensor", lambda: aten.mul.Tensor(fx, fw))

print(json.dumps(out))
"""


def _compare_cases(shim, up, keys, what):
    """Element-for-element comparison of two probe results, per named case."""
    problems = []
    for key in keys:
        s, u = shim["cases"].get(key), up["cases"].get(key)
        assert s is not None and u is not None, f"{key} missing from a probe"
        if "raised" in u or "raised" in s:
            if s.get("raised") != u.get("raised"):
                problems.append(f"{key}: shim {s.get('raised', s)} vs upstream {u.get('raised', u)}")
            continue
        if s["elems"] != u["elems"]:
            for i, (se, ue) in enumerate(zip(s["elems"], u["elems"])):
                if se != ue:
                    problems.append(f"{key}[{i}]: shim {se} vs upstream {ue}")
            if len(s["elems"]) != len(u["elems"]):
                problems.append(f"{key}: arity {len(s['elems'])} vs {len(u['elems'])}")
    assert not problems, what + "\n  " + "\n  ".join(problems)


def test_a_fake_tensor_argument_reaches_its_subclass_with_every_mode_popped():
    """THE ROOT CAUSE.  A tensor subclass carries its own dispatch key.

    With no torch-dispatch mode anywhere on the stack, upstream still routes an
    op with a `FakeTensor` argument into `FakeTensor.__torch_dispatch__`, and
    the result is a `FakeTensor` on the tensor's *fake* device.  This shim's
    door consulted the mode stack and nothing else, so the same call fell
    through to the dense path and produced a bare `meta` tensor.

    That is the whole of `docs/graph/VARMEAN.md` §4's wall, two levels down:
    bare meta outputs make `_make_cache_entry` bypass, the bypass keeps the
    `_out_wrapper` NamedTuple, and the NamedTuple is what `extract_val` cannot
    rebuild.

    `sub` and `add.Tensor` are in here next to `native_layer_norm` on purpose.
    If only the named op were repaired, these two would stay wrong and the
    next architecture would find the same wall under a different name.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_SUBCLASS_PROBE, shim=True)
    up = _run(_SUBCLASS_PROBE, shim=False)

    # The probe has to actually be in the state it claims to be testing.
    assert up["stack_depth"] == 0, up["stack_depth"]
    assert shim["stack_depth"] == 0, shim["stack_depth"]

    # Upstream's own answer is the specification, and it is asserted to be the
    # interesting one rather than taken on trust: if upstream also returned
    # bare meta tensors here, this test would be comparing two wrongs.
    for key in ("sub", "rsqrt", "add.Tensor", "mul.Tensor"):
        assert all(e["is_fake"] for e in up["cases"][key]["elems"]), (
            f"upstream did not produce FakeTensors for {key} -- "
            "the premise of this test is wrong")

    _compare_cases(
        shim, up,
        ("sub", "rsqrt", "add.Tensor", "mul.Tensor", "var_mean.dim", "contiguous"),
        "with every dispatch mode popped, a FakeTensor argument did not reach "
        "its subclass:")
    print("SUBCLASS: %d ops routed to __torch_dispatch__ with an empty mode stack"
          % 6)


def test_no_dispatch_still_suppresses_subclass_dispatch():
    """The recursion guard, asserted rather than assumed.

    `in_kernel_invocation_manager` runs the real meta kernel inside
    `torch._C._DisableTorchDispatch()` with `FakeTensor` arguments still in
    hand.  A subclass key that ignored that guard would re-enter
    `FakeTensor.__torch_dispatch__` from inside the call that
    `FakeTensor.__torch_dispatch__` made, and recurse until the stack blew.

    This is the test that makes the repair narrow.  Widening the door to
    "always consult the subclass" passes the test above and fails this one.

    **Upstream is not the oracle here, and that is measured, not assumed.**
    Asking torch 2.13.0 the same question segfaults it: with the mode stack
    popped, the first operator applied to a `FakeTensor` inside
    `torch._C._DisableTorchDispatch()` takes SIGSEGV before flushing any
    output.  `docs/graph/STRUCTSEQ.md` §4 records the reproducer.  So the
    comparison is the shim against itself, across the guard -- which is the
    property that actually matters and needs no second side:

      * inside the guard the subclass must NOT be consulted, and
      * outside it the same op on the same tensor must be,

    so a door that never reads the guard, and a door that always suppresses,
    each fail a different half.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_SUBCLASS_PROBE, shim=True)
    assert shim["suppressed_depth"] == 0, shim["suppressed_depth"]
    for key in ("sub", "mul.Tensor"):
        inside = shim["cases"]["no_dispatch/" + key]
        outside = shim["cases"][key]
        assert "raised" not in inside, inside
        assert all(not e["is_fake"] for e in inside["elems"]), (
            f"no_dispatch/{key} reached the subclass anyway: {inside['elems']}")
        assert all(e["is_fake"] for e in outside["elems"]), (
            f"{key} outside the guard did not reach the subclass: {outside['elems']}")
        assert inside["elems"] != outside["elems"], (
            f"{key} answers the same inside and outside no_dispatch() -- "
            "this test cannot distinguish a guard that is read from one "
            "that is not")
    print("NO_DISPATCH: suppression distinguishable on 2 ops")


def test_native_layer_norms_fake_outputs_are_fake_tensors_on_the_inputs_device():
    """The first consequence: the op VARMEAN §4 named.

    All three outputs -- normalized, mean, rstd -- must be `FakeTensor`s on the
    input's fake device, with upstream's shape, dtype and stride.  Before the
    repair all three were bare `meta` tensors, which is what made
    `_make_cache_entry` bypass.

    Stride is compared because it is free here and because
    `docs/graph/STRIDE.md` is about this shim getting shapes right and strides
    wrong.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_SUBCLASS_PROBE, shim=True)
    up = _run(_SUBCLASS_PROBE, shim=False)
    s = shim["cases"]["native_layer_norm"]
    u = up["cases"]["native_layer_norm"]
    assert "raised" not in u, u
    assert "raised" not in s, s
    assert len(u["elems"]) == 3, u
    assert all(e["is_fake"] for e in u["elems"]), u
    assert all(e["is_fake"] for e in s["elems"]), (
        "native_layer_norm's fake outputs are not FakeTensors: %s" % s["elems"])
    assert s["elems"] == u["elems"], (s["elems"], u["elems"])
    print("LAYER_NORM: 3/3 outputs FakeTensor on %s, strides %s"
          % (s["elems"][0]["device"], [e["stride"] for e in s["elems"]]))


# ---------------------------------------------------------------------------
# Probe 2 -- the cache, and the result class `extract_val` actually meets.
# ---------------------------------------------------------------------------

_CACHE_PROBE = r"""
import json
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "ops": {}}
aten = torch.ops.aten

x = torch.randn(4, 6)
w = torch.randn(6)
b = torch.randn(6)

CASES = [
    ("native_layer_norm", lambda fx, fw, fb: aten.native_layer_norm.default(fx, [6], fw, fb, 1e-5)),
    ("native_batch_norm", lambda fx, fw, fb: aten.native_batch_norm.default(fx, fw, fb, None, None, True, 0.1, 1e-5)),
    ("native_group_norm", lambda fx, fw, fb: aten.native_group_norm.default(fx, fw, fb, 4, 6, 1, 2, 1e-5)),
    ("var_mean.dim", lambda fx, fw, fb: aten.var_mean.dim(fx, [1], False, True)),
    ("max.dim", lambda fx, fw, fb: aten.max.dim(fx, 1)),
    ("min.dim", lambda fx, fw, fb: aten.min.dim(fx, 1)),
    ("sort", lambda fx, fw, fb: aten.sort.default(fx, 1)),
    ("topk", lambda fx, fw, fb: aten.topk.default(fx, 2)),
]

for name, fn in CASES:
    # A fresh mode per op, and the process-wide cache counters sampled around
    # the pair, so one op's bypass cannot be read off another's.
    before = dict(FakeTensorMode.cache_bypasses)
    try:
        with FakeTensorMode() as m:
            fx, fw, fb = m.from_tensor(x), m.from_tensor(w), m.from_tensor(b)
            first = fn(fx, fw, fb)
            second = fn(fx, fw, fb)
        after = dict(FakeTensorMode.cache_bypasses)
        # `extract_val`'s own construction: proxy_tensor.py:714 rebuilds a
        # sequence result by handing its class one iterable.
        try:
            type(second)(list(second))
            rebuildable = True
            rebuild_error = None
        except Exception as exc:
            rebuildable = False
            rebuild_error = type(exc).__name__
        out["ops"][name] = {
            "first_cls": type(first).__name__,
            "second_cls": type(second).__name__,
            "rebuildable": rebuildable,
            "rebuild_error": rebuild_error,
            "new_bypasses": {k: after[k] - before.get(k, 0)
                             for k in after if after[k] - before.get(k, 0) > 0},
        }
    except Exception as exc:
        out["ops"][name] = {"raised": type(exc).__name__, "message": str(exc)[:300]}

print(json.dumps(out))
"""

_MULTI_VALUE_OPS = ("native_layer_norm", "native_batch_norm", "native_group_norm",
                    "var_mean.dim", "max.dim", "min.dim", "sort", "topk")


def test_the_fake_tensor_cache_makes_an_entry_instead_of_a_bypass():
    """The second consequence: a cacheable result.

    `_make_cache_entry` raises `_BypassDispatchCache("non-FakeTensor output")`
    when any element of a tuple result is not a `FakeTensor`, and writes a
    *negative* entry under that key -- so the op is never cached again for
    those arguments, and every later call re-runs the ref and returns its
    NamedTuple.  This asserts that no such bypass is recorded for any of the
    eight multi-value ops, and that upstream records none either.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_CACHE_PROBE, shim=True)
    up = _run(_CACHE_PROBE, shim=False)
    problems = []
    for name in _MULTI_VALUE_OPS:
        s, u = shim["ops"][name], up["ops"][name]
        if "raised" in u:
            continue
        assert "raised" not in s, f"{name}: shim raised {s}"
        if u["new_bypasses"]:
            # Upstream itself bypasses for this op; nothing to hold the shim to.
            continue
        if s["new_bypasses"]:
            problems.append(f"{name}: shim bypassed the cache {s['new_bypasses']}, "
                            f"upstream did not")
    assert not problems, "\n  ".join(problems)
    print("CACHE: 0 bypasses across %d multi-value ops" % len(_MULTI_VALUE_OPS))


def test_the_multi_value_result_classes_match_upstream_after_the_first_call():
    """The third consequence, and the answer to "one op, or every multi-value
    return?".

    It is neither.  The spelling of this shim's multi-value returns was never
    the divergence: upstream's own *first* dispatch returns the same
    `_out_wrapper` NamedTuple, and `torch.return_types.native_layer_norm` does
    not exist on either side.  What `extract_val` meets is the *repeat*
    dispatch, and there upstream returns a plain `tuple` because
    `_output_from_cache_entry` ends `return tuple(outputs)`.

    So this asserts the repeat class against upstream's for all eight ops, and
    asserts nothing about the first -- see this module's docstring.  A round
    that rewrote every `return_types_*` into a structseq would have made the
    shim differ from upstream in a second place while fixing neither.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_CACHE_PROBE, shim=True)
    up = _run(_CACHE_PROBE, shim=False)
    problems = []
    for name in _MULTI_VALUE_OPS:
        s, u = shim["ops"][name], up["ops"][name]
        if "raised" in u:
            continue
        assert "raised" not in s, f"{name}: shim raised {s}"
        if s["second_cls"] != u["second_cls"]:
            problems.append(f"{name}: repeat dispatch is {s['second_cls']}, "
                            f"upstream {u['second_cls']}")
        if u["rebuildable"] and not s["rebuildable"]:
            problems.append(f"{name}: extract_val cannot rebuild {s['second_cls']} "
                            f"({s['rebuild_error']}); upstream's "
                            f"{u['second_cls']} rebuilds")
    assert not problems, "\n  ".join(problems)
    print("RESULT CLASS: %d multi-value ops agree with upstream on repeat dispatch"
          % len(_MULTI_VALUE_OPS))


# ---------------------------------------------------------------------------
# Probe 2b -- the re-boxing rule itself, with no FakeTensor anywhere.
#
# The tests above reach the rule through `FakeTensorMode`, which means a
# regression in either could be blamed on the other.  This probe drives it
# directly: an ordinary `TorchDispatchMode` that deliberately returns a
# `namedtuple` (for the tuple rule) or a `tuple` subclass (for the list rule),
# on dense tensors, with the answer read by the caller.
# ---------------------------------------------------------------------------

_RESHAPE_PROBE = r"""
import collections
import json
import torch
from torch.utils._python_dispatch import TorchDispatchMode

out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "ops": {}}
aten = torch.ops.aten

NAMED = collections.namedtuple("NAMED", "a b c")


class Subtuple(tuple):
    pass


# Answers every op with a *named* sequence of the real answer.  The point is
# that the object handed back is deliberately the wrong box: upstream's
# `OpOverload.__call__` re-boxes it per the schema and the caller never sees
# `NAMED` or `Subtuple`, whereas a door that passes the object through does.
class Renaming(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        with torch._C._DisableTorchDispatch():
            r = func(*args, **(kwargs or {}))
        if isinstance(r, torch.Tensor):
            return r
        if isinstance(r, (tuple, list)):
            if len(r) == 3:
                return NAMED(*r)
            return Subtuple(r)
        return r


x = torch.randn(4, 6)
w = torch.randn(6)
b = torch.randn(6)

CASES = [
    ("native_layer_norm", lambda: aten.native_layer_norm.default(x, [6], w, b, 1e-5)),
    ("max.dim", lambda: aten.max.dim(x, 1)),
    ("min.dim", lambda: aten.min.dim(x, 1)),
    ("sort", lambda: aten.sort.default(x, 1)),
    ("topk", lambda: aten.topk.default(x, 2)),
    ("var_mean.correction", lambda: aten.var_mean.correction(x, [1], correction=0, keepdim=True)),
    ("split.Tensor", lambda: aten.split.Tensor(x, 2)),
    ("unbind.int", lambda: aten.unbind.int(x, 0)),
    ("relu", lambda: aten.relu.default(x)),
]

for name, fn in CASES:
    try:
        with Renaming():
            r = fn()
        try:
            type(r)(list(r)) if isinstance(r, (tuple, list)) else None
            rebuildable = True
        except Exception:
            rebuildable = False
        out["ops"][name] = {
            "cls": type(r).__name__,
            "len": len(r) if isinstance(r, (tuple, list)) else None,
            "rebuildable": rebuildable,
        }
    except Exception as exc:
        out["ops"][name] = {"raised": type(exc).__name__, "message": str(exc)[:300]}

print(json.dumps(out))
"""

_RESHAPE_OPS = ("native_layer_norm", "max.dim", "min.dim", "sort", "topk",
                "var_mean.correction", "split.Tensor", "unbind.int", "relu")


def test_the_dispatcher_reboxes_a_modes_answer_into_the_schemas_shape():
    """THE REPAIR, asserted where it lives and with no FakeTensor in sight.

    Upstream's `OpOverload.__call__` converts whatever `__torch_dispatch__`
    returned to IValues per the schema and boxes them back out, so the *mode*
    does not get to choose the result class.  A schema with more than one
    return yields a plain `tuple`; a schema with one return of list type
    (`split.Tensor`, `unbind.int`) yields a plain `list`.  Both halves are
    upstream's measured answer here, not a rule read off its source.

    This is the test that says the repair belongs in the dispatcher rather
    than in six operators.  A round that taught `native_layer_norm` to return
    a structseq would leave every other row of this probe wrong.

    `relu` is in the list as the control: one Tensor return, nothing to
    re-box, and a repair that started converting single results would redden
    it.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_RESHAPE_PROBE, shim=True)
    up = _run(_RESHAPE_PROBE, shim=False)
    problems = []
    for name in _RESHAPE_OPS:
        s, u = shim["ops"][name], up["ops"][name]
        if "raised" in u:
            continue
        if "raised" in s:
            problems.append(f"{name}: shim raised {s}")
            continue
        if (s["cls"], s["len"]) != (u["cls"], u["len"]):
            problems.append(f"{name}: shim {s['cls']}(len={s['len']}) vs "
                            f"upstream {u['cls']}(len={u['len']})")
        if u["rebuildable"] and not s["rebuildable"]:
            problems.append(f"{name}: extract_val could not rebuild {s['cls']}")
    assert not problems, "\n  ".join(problems)

    # The probe has to be doing something, or it proves nothing: upstream must
    # be re-boxing at least one of these away from the class the mode returned.
    assert any(up["ops"][n]["cls"] not in ("NAMED", "Subtuple")
               and up["ops"][n]["len"] is not None for n in _RESHAPE_OPS), (
        "upstream returned the mode's own class everywhere -- this probe "
        "cannot see re-boxing at all")
    boxes = {n: shim["ops"][n]["cls"] for n in _RESHAPE_OPS}
    print("RESHAPE: %s" % boxes)


# ---------------------------------------------------------------------------
# Probe 3 -- THE BAR.  VARMEAN §4's four-line module, not a paraphrase.
# ---------------------------------------------------------------------------

_EXPORT_PROBE = r"""
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}


class M(torch.nn.Module):
    # `docs/graph/VARMEAN.md` §4's reproducer, verbatim in shape: one
    # `nn.LayerNorm` and nothing else, so that a failure is this wall and not
    # a neighbour's.
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.ln = torch.nn.LayerNorm(6)
        with torch.no_grad():
            self.ln.weight.copy_(torch.randn(6))
            self.ln.bias.copy_(torch.randn(6))

    def forward(self, x):
        return self.ln(x)


torch.manual_seed(0)
mod = M().eval()
x = torch.randn(4, 6)
with torch.no_grad():
    eager = mod(x)
eager = eager if isinstance(eager, (tuple, list)) else (eager,)
out["eager"] = [t.flatten().tolist() for t in eager]
out["shapes"] = [list(t.shape) for t in eager]
if not out["is_shim"]:
    exact = M().double().eval()
    with torch.no_grad():
        ref = exact(x.double())
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    out["oracle"] = [(e.double() - f).abs().flatten().tolist()
                     for e, f in zip(eager, ref)]
stage = "export"
try:
    ep = torch.export.export(mod, (x,))
    out["targets"] = sorted({str(n.target) for n in ep.graph.nodes
                             if n.op == "call_function"})
    stage = "replay"
    with torch.no_grad():
        replayed = ep.module()(x)
    replayed = replayed if isinstance(replayed, (tuple, list)) else (replayed,)
    out["replayed"] = [t.flatten().tolist() for t in replayed]
    out["stage"] = "done"
except Exception:
    import traceback
    out["stage"] = stage
    out["error"] = traceback.format_exc()[-3000:]
print(json.dumps(out))
"""


def test_a_four_line_layer_norm_module_exports_replays_and_agrees():
    """THE BAR.  Three verdicts, kept separate, and none of them is "returned".

    `docs/graph/COMPILE.md` records the trap this guards against: a
    superficial repair that makes the machinery hand back something which
    looks finished and computes nothing.  So it is not enough that
    `torch.export.export()` stopped raising -- `ep.module()(x)` must run, its
    outputs must agree element-wise with the module's own eager outputs, and
    the shim's eager outputs must agree with upstream's, so that a kernel
    wrong in the same way on both of this shim's paths is caught too.

    The tolerance is derived, `docs/numerics/AGREE.md` §2's method: the p90 of
    upstream's own float32-vs-float64 relative error on these very outputs,
    floored at 8 ulp.  There is no constant here for anyone to widen.

    2-D input on purpose, `test_varmean.py`'s reason: a 3-D activation reaches
    a `view` inside fake mode and stops at `FakeTensorDeviceMismatchError`,
    which `docs/graph/STRIDE.md` §6 analysed and deferred.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_EXPORT_PROBE, shim=True)
    up = _run(_EXPORT_PROBE, shim=False)
    assert up["stage"] == "done", up.get("error")
    assert shim["stage"] == "done", (
        f"shim stopped at stage {shim['stage']}:\n{shim.get('error')}")
    assert shim["shapes"] == up["shapes"], (shim["shapes"], up["shapes"])

    worst_rel = 0.0
    for i, (eager_s, replay_s, eager_u, oracle) in enumerate(
            zip(shim["eager"], shim["replayed"], up["eager"], up["oracle"])):
        scale = max(abs(v) for v in eager_u)
        rel = sorted(e / scale for e in oracle)
        tol = max(rel[int(0.9 * (len(rel) - 1))], 8 * _EPS32) * scale
        replay = max(abs(a - b) for a, b in zip(replay_s, eager_s))
        cross = max(abs(a - b) for a, b in zip(eager_s, eager_u))
        assert replay <= tol, f"output {i}: replay disagrees with eager: {replay} > {tol}"
        assert cross <= tol, f"output {i}: shim eager disagrees with upstream: {cross} > {tol}"
        worst_rel = max(worst_rel, replay / scale, cross / scale)

    # The graph has to contain the work.  An `ExportedProgram` holding no
    # operators would replay and agree trivially -- docs/graph/EXPORT.md §4.2
    # is the argument that such a graph does not look wrong.
    targets = shim["targets"]
    assert len(targets) >= 2, f"the exported graph holds only {targets}"
    assert any("layer_norm" in t or "rsqrt" in t or "sqrt" in t for t in targets), targets
    print(f"EXPORT: exported+replayed+agreed, worst relative {worst_rel:.3e}; "
          f"{len(targets)} distinct call targets in the graph")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
