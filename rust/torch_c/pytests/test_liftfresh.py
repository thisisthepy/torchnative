"""`aten.lift_fresh_copy.default`, and the three walls standing behind it.

`docs/graph/STRUCTSEQ.md` §5 landed `torch.export` at **7 of 10** under the
replay-and-agree bar and recorded that the three that remain -- `big_bird`,
`blip`, `canine` -- all stop at the same named operator, which
`docs/graph/VARMEAN.md` §3 already had on the board for 7 of the forty.  It was
never behind the layer-norm wall; it stood beside it.

**Closing that operator alone did not move the count**, and that is the finding
this file is shaped around.  It was the first of four walls in a single code
path -- fake mode's constant propagation -- each unreachable until the one in
front of it fell: the operator, then `UntypedStorage._weak_ref`, then
`torch._C._SchemaInfo.is_mutable`, then an alias-annotation parser defect that
predates this round.  `docs/graph/LIFTFRESH.md` is the whole chain; the tests
below are grouped in the order the sweep found them.  The count went **7 of 10
-> 9 of 10**; the tenth (`canine`) stops in `docs/graph/STRIDE.md` §3's
territory and is not in this chain.

Measured before anything was written (both sides, separate subprocesses):

* upstream's schema is `aten::lift_fresh_copy(Tensor self) -> Tensor` and the
  result **does not share storage with its input** -- it is a copy, not the
  alias `lift_fresh` returns;
* it lays its output out **contiguously**, which is where it parts company with
  `aten.clone.default`: on a transposed `(4, 3)` input of stride `(1, 4)`,
  `lift_fresh_copy` answers stride `(3, 1)` and `clone` answers `(1, 4)`.  An
  implementation that reused `clone` would be wrong exactly there, and
  `test_lift_fresh_copy_is_contiguous_where_clone_preserves` is that
  measurement;
* the shim had **no kernel at all, dense or meta** -- so this is not
  `docs/graph/EXPORT6.md` §5's "already in `_aten_implemented()`, merely lacked
  a meta kernel" shape, which is the first thing the brief asked to check.  It
  needs **both**, and for two different reasons: the meta kernel so the op can
  be traced under `FakeTensorMode`, the dense kernel so the `ExportedProgram`
  can be **replayed** on the real constant afterwards.

`aten.lift_fresh.default` is deliberately NOT changed.  It is already
implemented as the identity it is upstream, and it does not appear in an
exported graph at all: functionalisation rewrites it to `lift_fresh_copy`
before the graph is built.  `test_export_emits_lift_fresh_copy_and_not_lift_fresh`
is that measurement, so the choice is recorded as a fact rather than a
preference.

Nothing here asserts "export() returned".  `docs/graph/COMPILE.md`'s trap --- a
superficial repair that hands back something which looks finished and computes
nothing --- is guarded by asserting the three verdicts separately (exported /
replayed / **agreed**) and by requiring the operator to actually be in the
graph.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

#: float32 machine epsilon.  Every tolerance below is a multiple of this and of
#: nothing else; `docs/numerics/AGREE.md` is the method.
_EPS32 = 1.1920929e-07


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _run(script, shim, timeout=1800):
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
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# the dense kernel: a copy, laid out contiguously
# ---------------------------------------------------------------------------

_DENSE_PROBE = r"""
import json
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}


def record(name, fn):
    try:
        out["cases"][name] = fn()
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "msg": str(exc)[:160]}


def describe(t):
    return {"shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "device": str(t.device),
            "vals": [float(v) for v in t.flatten().tolist()]}


lfc = torch.ops.aten.lift_fresh_copy.default

# contiguous, several dtypes -- the values and the dtype must survive exactly
for name in ["float32", "float64", "int64", "int32", "bool"]:
    dt = getattr(torch, name)
    base = (torch.arange(6, dtype=torch.float32).reshape(2, 3) % 2).to(dt)
    record("dtype_" + name, (lambda b: lambda: describe(lfc(b)))(base))

# 0-d, the shape `torch.tensor(7.0)` produces
record("zero_dim", lambda: describe(lfc(torch.tensor(7.0))))
# empty
record("empty", lambda: describe(lfc(torch.empty(0, dtype=torch.float32))))

# a copy, not an alias -- the property that separates it from `lift_fresh`
def not_an_alias():
    src = torch.tensor([1.0, 2.0, 3.0])
    got = lfc(src)
    got_ptr_differs = got.data_ptr() != src.data_ptr()
    got.fill_(9.0)
    return {"ptr_differs": got_ptr_differs,
            "source_unchanged": src.flatten().tolist() == [1.0, 2.0, 3.0]}


record("copy_not_alias", not_an_alias)

# THE distinguishing layout fact, against `clone` in the same breath
def transposed():
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    return {"input": describe(base),
            "lift_fresh_copy": describe(lfc(base)),
            "clone": describe(torch.ops.aten.clone.default(base))}


record("transposed", transposed)
print(json.dumps(out))
"""


def test_lift_fresh_copy_is_a_dense_copy_agreeing_with_upstream():
    """Shape, stride, dtype, device AND values, against upstream's own answer
    in a second subprocess -- never against a table written beside the
    implementation.

    `bool` and the integer dtypes are in the sweep because a kernel that
    routed through a float buffer would still get `float32` right.  The 0-d
    case is the shape `torch.tensor(7.0)` actually produces, and the empty
    case is where `PreserveFormat` and a contiguous rule are measurably
    different elsewhere in this tree.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_DENSE_PROBE, shim=True), _run(_DENSE_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for name in sorted(up["cases"]):
        u, s = up["cases"][name], shim["cases"][name]
        assert "raised" not in u, f"upstream itself refuses {name}: {u}"
        assert "raised" not in s, f"{name}: the shim refuses -- {s}"
        assert s == u, f"{name}: shim {s} != upstream {u}"
    assert up["cases"]["copy_not_alias"]["ptr_differs"] is True, (
        "upstream changed: lift_fresh_copy is supposed to be a copy")
    print(f"LFCDENSE: {len(up['cases'])} dense cases identical to upstream, "
          f"values included")


def test_lift_fresh_copy_is_contiguous_where_clone_preserves():
    """The one fact that makes this a kernel of its own rather than an alias
    of `clone`, asserted as a *difference* so it cannot be satisfied by both
    answers being the same wrong thing.

    On a transposed `(4, 3)` view of stride `(1, 4)`:

        lift_fresh_copy  ->  stride (3, 1)     a fresh contiguous buffer
        clone            ->  stride (1, 4)     the input's layout, preserved

    A `lift_fresh_copy` implemented by calling `clone` passes every value
    check in this file and fails this one.  It is the reason the meta rule is
    `AlwaysContiguous` and not `PreserveFormat`.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_DENSE_PROBE, shim=True), _run(_DENSE_PROBE, shim=False)
    u, s = up["cases"]["transposed"], shim["cases"]["transposed"]
    assert "raised" not in s, f"the shim refuses the transposed case: {s}"
    assert u["input"]["stride"] != u["lift_fresh_copy"]["stride"], (
        "upstream changed: the input was supposed to be non-contiguous")
    assert u["lift_fresh_copy"]["stride"] != u["clone"]["stride"], (
        "upstream changed: lift_fresh_copy and clone were supposed to differ "
        f"here -- {u}")
    assert s["lift_fresh_copy"] == u["lift_fresh_copy"], (
        f"shim {s['lift_fresh_copy']} != upstream {u['lift_fresh_copy']}")
    assert s["lift_fresh_copy"]["stride"] != s["clone"]["stride"], (
        "the shim answers the same layout for lift_fresh_copy and clone, so "
        "it is spelling one as the other: "
        f"{s['lift_fresh_copy']['stride']} vs {s['clone']['stride']}")
    print("LFCLAYOUT: lift_fresh_copy stride "
          f"{s['lift_fresh_copy']['stride']} vs clone {s['clone']['stride']}, "
          "both matching upstream")


# ---------------------------------------------------------------------------
# the meta kernel: what fake-mode tracing actually asks for
# ---------------------------------------------------------------------------

_META_PROBE = r"""
import json
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}


def record(name, fn):
    try:
        t = fn()
        out["cases"][name] = {"shape": list(t.shape), "stride": list(t.stride()),
                              "dtype": str(t.dtype), "device": str(t.device)}
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "msg": str(exc)[:160]}


lfc = torch.ops.aten.lift_fresh_copy.default

record("meta_contiguous",
       lambda: lfc(torch.empty(2, 3, device="meta", dtype=torch.float32)))
record("meta_zero_dim",
       lambda: lfc(torch.empty((), device="meta", dtype=torch.float32)))
record("meta_int64",
       lambda: lfc(torch.empty(4, device="meta", dtype=torch.int64)))
record("meta_bool",
       lambda: lfc(torch.empty(2, 2, device="meta", dtype=torch.bool)))
record("meta_empty",
       lambda: lfc(torch.empty(0, device="meta", dtype=torch.float32)))
# non-contiguous meta input: the gated `Unverified` rule refuses here, so this
# case is the one that proves the op has a *verified* stride rule.
record("meta_transposed",
       lambda: lfc(torch.empty(3, 4, device="meta", dtype=torch.float32).t()))
record("meta_sliced",
       lambda: lfc(torch.empty(8, device="meta", dtype=torch.float32)[1:5]))

# and the same op reached the way export reaches it: through fake mode.  The
# source is contiguous because `torch.tensor([...])` is -- the non-contiguous
# fake case is a pre-existing gap of this shim that has nothing to do with this
# operator, and `test_a_noncontiguous_fake_tensor_is_not_this_operators_gap`
# holds it down separately rather than letting it hide here.
from torch._subclasses.fake_tensor import FakeTensorMode


def fake():
    with FakeTensorMode() as mode:
        src = mode.from_tensor(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        return lfc(src)


record("under_fake_tensor_mode", fake)
print(json.dumps(out))
"""


def test_lift_fresh_copy_meta_kernel_agrees_with_upstream():
    """A meta tensor carries shape, dtype, device and stride and no storage,
    so those four are the whole of the answer and all four are compared.

    `meta_transposed` and `meta_sliced` are the load-bearing rows: before this
    round the shim refused non-contiguous meta inputs for this op with *the
    gated refusal* ("has not been shown to produce upstream's output stride"),
    which is a different failure from having no kernel and would have been
    missed by a probe that only ever passed contiguous inputs.

    `under_fake_tensor_mode` is the path `torch.export` actually takes; it is
    separate because a meta kernel that works on a bare meta tensor and not
    through `FakeTensorMode` would leave the export exactly where it was.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_META_PROBE, shim=True), _run(_META_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for name in sorted(up["cases"]):
        u, s = up["cases"][name], shim["cases"][name]
        assert "raised" not in u, f"upstream itself refuses {name}: {u}"
        assert "raised" not in s, f"{name}: the shim refuses -- {s}"
        assert s == u, f"{name}: shim {s} != upstream {u}"
    assert up["cases"]["meta_transposed"]["stride"] == [3, 1], (
        "upstream changed: the meta rule was measured as always-contiguous, "
        f"got {up['cases']['meta_transposed']}")
    print(f"LFCMETA: {len(up['cases'])} meta cases identical to upstream, "
          "stride included")


_FAKE_VIEW_PROBE = r"""
import json
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}


def record(name, op):
    try:
        with FakeTensorMode() as mode:
            src = mode.from_tensor(
                torch.arange(12, dtype=torch.float32).reshape(3, 4).t())
            t = op(src)
        out["cases"][name] = {"shape": list(t.shape), "stride": list(t.stride())}
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "msg": str(exc)[:80]}


record("clone", torch.ops.aten.clone.default)
record("contiguous", torch.ops.aten.contiguous.default)
record("lift_fresh_copy", torch.ops.aten.lift_fresh_copy.default)
print(json.dumps(out))
"""


def test_a_noncontiguous_fake_tensor_is_not_this_operators_gap():
    """A gap this round found, measured, and deliberately did NOT close --
    recorded here so it is a known quantity rather than an absence.

    A `FakeTensor` that is a *view* (non-contiguous strides) reaches
    `TensorBase._base`, which this shim does not implement, and it does so for
    **every** copy-shaped operator: `clone` and `contiguous` fail identically
    to `lift_fresh_copy`.  So it is not this operator's gap, and closing it is
    not this operator's job -- it is the view-under-fake-mode area of
    `docs/graph/STRIDE.md`.

    The test is written as "all three behave the same" rather than skipped,
    because that is a claim that can go red: if `lift_fresh_copy` ever becomes
    the odd one out in either direction -- refusing when the others succeed,
    or succeeding when the others refuse -- this fails and says so.  A test
    that simply omitted the case would be CLAUDE.md §5.5's shape.

    `torch.export` does not reach this: the constant a Python literal produces
    is contiguous, which is why THE BAR below passes with this gap open.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_FAKE_VIEW_PROBE, shim=True), _run(_FAKE_VIEW_PROBE, shim=False)
    for name in ("clone", "contiguous", "lift_fresh_copy"):
        assert "raised" not in up["cases"][name], (
            f"upstream itself refuses {name}: {up['cases'][name]}")
    refused = {n for n in ("clone", "contiguous", "lift_fresh_copy")
               if "raised" in shim["cases"][n]}
    assert refused in ({"clone", "contiguous", "lift_fresh_copy"}, set()), (
        "lift_fresh_copy no longer behaves like the other copy-shaped ops on a "
        f"non-contiguous FakeTensor -- refused: {sorted(refused)}, "
        f"cases: {shim['cases']}")
    if refused:
        print("LFCFAKEVIEW: the non-contiguous FakeTensor gap is still open and "
              "still shared by clone/contiguous/lift_fresh_copy alike")
    else:
        print("LFCFAKEVIEW: the non-contiguous FakeTensor gap has closed for "
              f"all three: {shim['cases']}")


# ---------------------------------------------------------------------------
# THE BAR
# ---------------------------------------------------------------------------

_EXPORT_PROBE = r"""
import json
import traceback
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "stage": "start"}


# A Python literal inside forward -- the construction that makes
# functionalisation emit `lift_fresh_copy`.
class M(torch.nn.Module):
    def forward(self, x):
        weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bias = torch.tensor([0.5, -0.25, 0.125])
        return (x * weights) + bias


try:
    torch.manual_seed(0)
    module = M()
    example = (torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7.0,)

    eager = module(*example)
    out["eager"] = [float(v) for v in eager.flatten().tolist()]
    out["shape"] = list(eager.shape)

    # upstream's own float32-vs-float64 relative error on these very outputs,
    # which is what the tolerance is derived from.  docs/numerics/AGREE.md §2.
    wide = M().double()(*(a.double() for a in example))
    out["oracle"] = [abs(float(a) - float(b)) for a, b in
                     zip(eager.flatten().tolist(), wide.flatten().tolist())]

    out["stage"] = "exported"
    ep = torch.export.export(module, example)
    out["targets"] = sorted({str(n.target) for n in ep.graph.nodes
                             if n.op == "call_function"})

    out["stage"] = "replayed"
    replayed = ep.module()(*example)
    out["replayed"] = [float(v) for v in replayed.flatten().tolist()]
    out["replayed_shape"] = list(replayed.shape)
    out["stage"] = "done"
except Exception:
    out["error"] = traceback.format_exc()[-3000:]
print(json.dumps(out))
"""


def test_a_module_with_a_tensor_literal_exports_replays_and_agrees():
    """THE BAR, and it is three verdicts kept apart, none of which is
    "export() returned without raising".

    `ep.module()(x)` must run; its outputs must agree element-wise with the
    module's own eager outputs; and the shim's eager outputs must agree with
    *upstream's*, so that a kernel wrong in the same way on both of the shim's
    paths is still caught.

    The tolerance is derived, `docs/numerics/AGREE.md` §2's method: the p90 of
    upstream's own float32-vs-float64 relative error on these very outputs,
    floored at 8 ulp.  There is no constant here for anyone to widen.

    The graph assertion is `docs/graph/COMPILE.md`'s: an `ExportedProgram`
    holding no operators would replay and agree trivially, so
    `lift_fresh_copy` is required to be **in** the graph.  That also makes the
    test specific -- if functionalisation ever stopped emitting it, this would
    go red rather than quietly passing on a graph that no longer exercises the
    operator this round exists for.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_EXPORT_PROBE, shim=True), _run(_EXPORT_PROBE, shim=False)
    assert up["stage"] == "done", up.get("error")
    assert shim["stage"] == "done", (
        f"shim stopped at stage {shim['stage']}:\n{shim.get('error')}")
    assert shim["shape"] == up["shape"] == shim["replayed_shape"], (
        shim["shape"], up["shape"], shim["replayed_shape"])

    scale = max(abs(v) for v in up["eager"])
    rel = sorted(e / scale for e in up["oracle"])
    tol = max(rel[int(0.9 * (len(rel) - 1))], 8 * _EPS32) * scale
    replay = max(abs(a - b) for a, b in zip(shim["replayed"], shim["eager"]))
    cross = max(abs(a - b) for a, b in zip(shim["eager"], up["eager"]))
    assert replay <= tol, f"replay disagrees with eager: {replay} > {tol}"
    assert cross <= tol, f"shim eager disagrees with upstream: {cross} > {tol}"

    targets = shim["targets"]
    assert any("lift_fresh_copy" in t for t in targets), (
        f"the exported graph does not contain lift_fresh_copy: {targets}")
    assert len(targets) >= 3, f"the exported graph holds only {targets}"
    print("LFCEXPORT: exported+replayed+agreed, worst relative "
          f"{max(replay, cross) / scale:.3e}; {len(targets)} distinct call "
          f"targets including lift_fresh_copy")


def test_export_emits_lift_fresh_copy_and_not_lift_fresh():
    """Why `aten.lift_fresh.default` was left alone, recorded as a
    measurement rather than as a preference.

    Functionalisation rewrites `lift_fresh` into `lift_fresh_copy` before the
    exported graph is built, so `lift_fresh` never appears in one.  Adding or
    altering it would have unblocked no architecture.  Asserted on *upstream's*
    graph as well as the shim's, so it is upstream's rule being recorded, not
    this shim's behaviour being described.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_EXPORT_PROBE, shim=True), _run(_EXPORT_PROBE, shim=False)
    assert up["stage"] == "done", up.get("error")
    assert shim["stage"] == "done", (
        f"shim stopped at stage {shim['stage']}:\n{shim.get('error')}")
    for side, got in (("upstream", up["targets"]), ("shim", shim["targets"])):
        assert any("lift_fresh_copy" in t for t in got), (side, got)
        assert not any(t.endswith("lift_fresh.default") for t in got), (
            f"{side}: lift_fresh survived into the exported graph: {got}")

    # The two graphs are not required to be identical, and the difference is
    # named rather than tolerated: upstream also emits `aten.detach_.default`
    # for the literal and this shim does not.  That is a separate divergence
    # with no bearing on the `lift_fresh` family, so it is asserted to be
    # *exactly* that one target -- any other drift fails here.
    only_upstream = set(up["targets"]) - set(shim["targets"])
    only_shim = set(shim["targets"]) - set(up["targets"])
    assert only_shim == set(), f"the shim emits targets upstream does not: {only_shim}"
    assert only_upstream <= {"aten.detach_.default"}, (
        f"unexpected targets missing from the shim's graph: {only_upstream}")
    print("LFCFAMILY: lift_fresh absent and lift_fresh_copy present on both "
          f"sides; shim targets {shim['targets']}, upstream-only "
          f"{sorted(only_upstream)}")


# ---------------------------------------------------------------------------
# the wall that was standing directly behind it, in the same code path
# ---------------------------------------------------------------------------

_WEAKREF_PROBE = r"""
import json
import torch
from torch.multiprocessing.reductions import StorageWeakRef

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}


def record(name, fn):
    try:
        out["cases"][name] = fn()
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "msg": str(exc)[:160]}


base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
other = torch.arange(12, dtype=torch.float32).reshape(3, 4)

# `_weak_ref` is an integer identity for the storage ...
record("is_int", lambda: isinstance(base.untyped_storage()._weak_ref(), int))
# ... stable across calls on the same storage ...
record("stable", lambda: base.untyped_storage()._weak_ref()
       == base.untyped_storage()._weak_ref())
# ... shared by a view of that storage ...
record("view_shares", lambda: base.untyped_storage()._weak_ref()
       == base[1:, 1:].untyped_storage()._weak_ref())
# ... and different for a different storage.
record("distinct", lambda: base.untyped_storage()._weak_ref()
       != other.untyped_storage()._weak_ref())
# It equals `_cdata`, which is this shim's storage identity.
record("equals_cdata", lambda: base.untyped_storage()._weak_ref()
       == base.untyped_storage()._cdata)

# What the fake-tensor constant map actually does with it: use StorageWeakRef
# as a dict key.  Hash and equality both read `cdata`, so this is the whole
# contract that path depends on.
def as_dict_key():
    a, b = StorageWeakRef(base._typed_storage()), StorageWeakRef(base._typed_storage())
    c = StorageWeakRef(other._typed_storage())
    d = {a: "base"}
    return {"same_storage_hits": d.get(b), "other_storage_misses": d.get(c),
            "eq": a == b, "neq": a != c, "hash_eq": hash(a) == hash(b)}


record("as_dict_key", as_dict_key)

# `_free_weak_ref` has to exist and be callable -- `StorageWeakRef.__del__`
# calls it, and an exception there is only "ignored", never fixed.
def free_is_callable():
    torch.Storage._free_weak_ref(base.untyped_storage()._weak_ref())
    return True


record("free_weak_ref_callable", free_is_callable)
print(json.dumps(out))
"""


def test_storage_weak_ref_is_an_identity_the_constant_map_can_key_on():
    """The wall standing directly *behind* `lift_fresh_copy`, in the same code
    path, and the reason closing the operator alone did not move the count.

    Fake mode's constant-propagation path runs the real kernel on the real
    constant and then registers it: `_dispatch_impl` ->
    `from_real_tensor(make_constant=True)` -> `add_constant_storage_mapping`
    -> `StorageWeakRef(...)` -> `storage._weak_ref()`.  So this is not reached
    until `lift_fresh_copy` succeeds -- it was behind the operator, not beside
    it, and the sweep only revealed it once the operator landed.

    `_weak_ref` was inherited from `_StorageBase`, whose body is
    `raise NotImplementedError` -- upstream's own abstract stub, identical in
    the vendored tree and in the installed wheel, overridden upstream by the C
    `UntypedStorage` and not overridden here.

    **What the caller needs is an identity, not a weak reference.**
    `fake_tensor.py` never calls `_expired`; it tracks liveness with Python
    `weakref.ref` on the *tensors* and uses `StorageWeakRef` purely as a dict
    key, whose `__hash__` and `__eq__` both read `cdata` and nothing else.
    That is exactly what this shim's `_cdata` already is, so `_weak_ref`
    answers it -- asserted here against `_cdata`, across a view (which must
    share) and a second storage (which must not), so an implementation
    returning a constant would fail `distinct` and one returning a fresh
    number each call would fail `stable`.

    `_expired` is deliberately left refusing --
    `test_expired_still_refuses_rather_than_guessing` says why.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_WEAKREF_PROBE, shim=True), _run(_WEAKREF_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for name in sorted(up["cases"]):
        u, s = up["cases"][name], shim["cases"][name]
        assert not (isinstance(u, dict) and "raised" in u), (
            f"upstream itself refuses {name}: {u}")
        assert not (isinstance(s, dict) and "raised" in s), (
            f"{name}: the shim refuses -- {s}")
    # The raw integers differ between processes and sides -- it is the
    # *relations* that have to agree, and every case above is a relation.
    for name in ("is_int", "stable", "view_shares", "distinct", "equals_cdata",
                 "free_weak_ref_callable"):
        assert shim["cases"][name] == up["cases"][name] is True, (
            f"{name}: shim {shim['cases'][name]} != upstream {up['cases'][name]}")
    assert shim["cases"]["as_dict_key"] == up["cases"]["as_dict_key"], (
        shim["cases"]["as_dict_key"], up["cases"]["as_dict_key"])
    assert shim["cases"]["as_dict_key"]["same_storage_hits"] == "base"
    assert shim["cases"]["as_dict_key"]["other_storage_misses"] is None
    print("LFCWEAKREF: _weak_ref is a stable storage identity, shared by views, "
          "distinct across storages, and usable as a StorageWeakRef dict key -- "
          "every relation matching upstream")


_EXPIRED_PROBE = r"""
import json
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}
s = torch.arange(4, dtype=torch.float32).untyped_storage()
try:
    out["cases"]["expired"] = {"value": bool(torch.Storage._expired(s._weak_ref()))}
except Exception as exc:
    out["cases"]["expired"] = {"raised": type(exc).__name__}
print(json.dumps(out))
"""


def test_expired_still_refuses_rather_than_guessing():
    """A limitation kept explicit instead of papered over.

    `_weak_ref` above hands back a storage *identity*; nothing is retained, so
    this shim genuinely cannot say whether that storage is still alive.
    Answering `False` ("still alive") would be the cheap way to make
    `_expired` return something, and it would be a claim that is wrong the
    moment it matters -- CLAUDE.md §5.5's shape, a check that cannot fail.

    Nothing on the `torch.export` path calls it: `fake_tensor.py` tracks
    liveness with `weakref.ref` on the tensors instead.  So it refuses, and
    this test holds the refusal in place, so that if someone later makes
    `_expired` answer, they have to come here and justify it rather than
    silently acquire a wrong answer.  Upstream answering it is asserted too,
    so the gap is recorded as a real difference.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_EXPIRED_PROBE, shim=True), _run(_EXPIRED_PROBE, shim=False)
    assert "raised" not in up["cases"]["expired"], (
        f"upstream itself refuses _expired: {up['cases']['expired']}")
    assert up["cases"]["expired"]["value"] is False, (
        "upstream changed: a live storage was supposed to be unexpired")
    assert shim["cases"]["expired"].get("raised") == "NotImplementedError", (
        "the shim now answers _expired -- if that is deliberate it needs a "
        "liveness mechanism behind it and this test rewritten, not deleted: "
        f"{shim['cases']['expired']}")
    print("LFCEXPIRED: _expired still refuses (no liveness is tracked); "
          "upstream answers False for a live storage")


# ---------------------------------------------------------------------------
# and the third wall in the same chain
# ---------------------------------------------------------------------------

_NATIVE_FUNCTIONS = os.path.join(
    _VENDOR_DIR, "torchgen", "packaged", "ATen", "native", "native_functions.yaml"
)

#: For every `- func:` entry of the vendored `native_functions.yaml`, the whole
#: of what `_SchemaInfo` is asked on the export path: whether the op mutates
#: anything, whether each named argument is the mutated one, and whether an
#: argument exists.  Identical text on both sides, so the only thing that can
#: differ is the answer.
_SCHEMA_INFO_PROBE = r"""
import json
import torch

path = %(path)r
rows = []
for line in open(path, encoding="utf-8"):
    if not line.startswith("- func:"):
        continue
    text = "aten::" + line[len("- func:"):].strip()
    try:
        schema = torch._C.parse_schema(text)
    except Exception:
        continue
    try:
        info = torch._C._SchemaInfo(schema)
        names = [a.name for a in schema.arguments]
        rows.append([text.split("(", 1)[0],
                     bool(info.is_mutable()),
                     [bool(info.is_mutable(n)) for n in names],
                     [bool(info.has_argument(n)) for n in names],
                     bool(info.has_argument("definitely_not_an_argument"))])
    except Exception as exc:
        rows.append([text.split("(", 1)[0], "RAISED", type(exc).__name__,
                     str(exc)[:80], None])
print(json.dumps({"is_shim": "torchnative" in (torch.__file__ or ""),
                  "rows": rows}))
"""


def test_schema_info_answers_mutability_for_every_schema_upstream_does():
    """The third wall in the same chain, and the one that was actually still
    standing between 7 of 10 and 10 of 10.

    Fake mode's constant bookkeeping calls it right after the storage map:
    `invalidate_written_to_constants` -> `get_schema_info(func)` ->
    `schema_info.is_mutable()`, so that an op which *writes* to a traced
    constant can invalidate every alias of it.  `torch._C._SchemaInfo` was a
    generated placeholder here and every call raised.

    It is a derivation, not new information: the answer is already in the
    schema this shim parses.  `is_mutable()` is "does any argument carry a
    write alias annotation" -- `Tensor(a!)` -- and `is_mutable(name)` is that
    question for one argument.  Measured first: this shim's
    `schema.arguments[i].alias_info.is_write` already agrees with upstream's,
    which is why the derivation is sound and why it is written in Python over
    the parsed schema rather than in Rust over the text.

    **The oracle is upstream, over all 2584 `- func:` entries of the vendored
    `native_functions.yaml`** -- the same file the shim parses -- not a table
    written beside the implementation, which would be the same author agreeing
    with himself.  A single wrong answer for a single argument of a single op
    fails.  `has_argument` is checked for a name that is in no schema too, so
    an implementation returning `True` unconditionally cannot pass.
    """
    if not _available() or not os.path.isfile(_NATIVE_FUNCTIONS):
        print("SKIP no vendored shim")
        return
    script = _SCHEMA_INFO_PROBE % {"path": _NATIVE_FUNCTIONS}
    shim, up = _run(script, shim=True), _run(script, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    assert len(up["rows"]) > 2000, f"only {len(up['rows'])} schemas parsed upstream"
    assert len(shim["rows"]) == len(up["rows"]), (
        f"the shim parsed {len(shim['rows'])} schemas, upstream {len(up['rows'])}")

    mutable = 0
    for s_row, u_row in zip(shim["rows"], up["rows"]):
        assert u_row[1] != "RAISED", f"upstream itself refuses {u_row[0]}: {u_row}"
        assert s_row[1] != "RAISED", f"{s_row[0]}: the shim refuses -- {s_row[2:]}"
        assert s_row == u_row, f"{u_row[0]}: shim {s_row[1:]} != upstream {u_row[1:]}"
        mutable += bool(u_row[1])

    # The sweep has to contain both answers, or "agrees with upstream" would be
    # satisfied by a constant.  Asserted rather than assumed.
    assert mutable > 0, "no schema in the sweep was mutable"
    assert mutable < len(up["rows"]), "every schema in the sweep was mutable"
    print(f"LFCSCHEMAINFO: {len(up['rows'])} schemas, is_mutable/has_argument "
          f"identical to upstream for every argument of every one; "
          f"{mutable} mutable, {len(up['rows']) - mutable} not")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
