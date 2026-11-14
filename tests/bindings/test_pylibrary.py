"""Python `torch.library` kernels for custom namespaces run through `_aten_dispatch`.

Issue #227. Before it, `torch._C._dispatch_library` recorded every Python
registration into `_shim_registrations` and dropped it: `Library("ns", "DEF")`,
`.define(...)`, `.impl(..., "CompositeExplicitAutograd")` all appeared to
succeed, and calling the op raised `aten op not implemented in torch._C shim:
ns.f.default`. `_dispatch_has_kernel_for_dispatch_key` was the constant
`False`, so ExecuTorch's `exir/dialects/backend/_ops.py:69-73` assertion (a
backend op must have a `CompositeExplicitAutograd`, `Meta` or
`CompositeImplicitAutograd` kernel) could never hold, and
`torch._C._is_opaque_type_registered` (`torch/utils/_pytree.py:623`, through
`torch/_library/opaque_object.py`) was a raising stub.

What this suite asserts, every value recomputed in a **separate subprocess**
on each side (the shim and upstream torch), with no expected-value table
written beside the implementation:

* define + impl at `CompositeExplicitAutograd`, `CompositeImplicitAutograd`,
  `CPU` and `Meta`, called through `torch.ops.<ns>.<op>`, agree with upstream;
* a meta tensor runs the `Meta` kernel (a log of which kernel ran is part of
  the answer) and gets its output shape from it; so does a `FakeTensor`, and
  `register_fake` / `custom_op(...).register_fake` work the same way;
* `CompositeImplicitAutograd` decomposes above a `TorchDispatchMode` (the mode
  sees the inner ops, never the custom op) while a `CompositeExplicitAutograd`
  op is seen as itself, as upstream orders them;
* `_dispatch_has_kernel_for_dispatch_key` / `_for_any_dispatch_key` / the
  computed-kernel predicate answer as upstream does for these ops, and still
  answer `False` for aten names (docs/design/REGISTRATIONS.md section 3.4);
* a key with no kernel refuses with a `NotImplementedError` naming the op and
  the key (AGENTS.md section 18);
* `_is_opaque_type_registered` answers from the registry
  `register_opaque_type` writes to;
* the door is what runs the kernel, and dropping the registrations again is
  caught: see the two nullification tests at the end.

What this suite cannot find: whether exir as a whole imports (it is not in the
gate's venv), the `Autograd` key (recorded and dropped, so a custom op's
autograd formula does not run; asserted as a named gap below rather than
hidden), and registrations in the `aten`/`prims` namespaces, which stay
recorded and dropped because the native table owns those ops.
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
_VENDOR_SHIM = str(_REPO_ROOT / "torchnative" / "python" / "torch" / "_C.abi3.so")


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _run(argv, shim, timeout=1800):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, *argv],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # Which side the probe landed on is asserted, not assumed: an empty
    # vendored tree makes the "shim" side silently import upstream
    # (AGENTS.md section 15.2), and every comparison would be upstream
    # against itself.
    assert out["is_shim"] is shim, f"probe meant for shim={shim} ran on the other side"
    return out


# ---------------------------------------------------------------------------
# The probe. Identical text on both sides; only the answers can differ.
# ---------------------------------------------------------------------------
_PROBE = r"""
import enum, json, sys, warnings
warnings.filterwarnings("ignore")
import torch
from torch.library import Library

C = torch._C
IS_SHIM = hasattr(C, "_aten_implemented")
if IS_SHIM and "--drop" in sys.argv:
    # The pre-#227 recorder, verbatim in effect: `impl` records and drops.
    def _dropped(self, name, dispatch_key, fn, with_keyset=False):
        C._shim_registrations.append(("impl", self.ns, name, dispatch_key))
    C._dispatch_library.impl = _dropped
if IS_SHIM and "--disarm" in sys.argv:
    # The Python registry still fills, but the door is never told.
    C._shim_arm_python_kernel = lambda key: None

DK = C.DispatchKey
r = {}
ran = []


def rec(name, fn):
    try:
        r[name] = {"value": fn()}
    except Exception as exc:
        r[name] = {"raised": type(exc).__name__, "msg": str(exc)[:600]}


def desc(t):
    return {"device": t.device.type, "shape": list(t.shape), "type": type(t).__name__}


def logged(fn):
    del ran[:]
    out = fn()
    return {"out": out, "ran": list(ran)}


lib = Library("tnpl", "DEF")

lib.define("f(Tensor x) -> Tensor")
lib.impl("f", lambda x: x * 2 + 1, "CompositeExplicitAutograd")

lib.define("g(Tensor x) -> Tensor")
def g_cpu(x):
    ran.append("g.CPU")
    return torch.cat([x, x], 1) + 10
def g_meta(x):
    ran.append("g.Meta")
    return x.new_empty((x.shape[0], 2 * x.shape[1]))
lib.impl("g", g_cpu, "CPU")
lib.impl("g", g_meta, "Meta")

lib.define("h(Tensor x) -> Tensor")
lib.impl("h", lambda x: x.neg() * 3, "CompositeImplicitAutograd")

lib.define("m(Tensor x) -> Tensor")
lib.impl("m", lambda x: torch.empty_like(x), "Meta")

lib.define("z(Tensor x) -> Tensor")

lib.define("w(Tensor x) -> Tensor")
def w_cpu(keyset, x):
    ran.append("w:" + type(keyset).__name__)
    return x - 1
lib.impl("w", w_cpu, "CPU", with_keyset=True)

# exir's `executorch_prim` shape: Scalar in, Scalar out, no tensor at all,
# registered the way `bind_pattern_to_op` registers it (after asking).
lib.define("s(Scalar a, Scalar b) -> Scalar")
s_before = [torch.ops.tnpl.s.default.has_kernel_for_dispatch_key(k) for k in (
    DK.CompositeExplicitAutograd, DK.CompositeImplicitAutograd, DK.Meta)]
lib.impl("s", lambda a, b: a + b, "CompositeImplicitAutograd")

lib.define("k(Tensor x) -> Tensor")
lib.impl("k", lambda x: x * 5, "CPU")
# Registrations that go through more of `torch.library` than `Library.impl`
# are recorded as results of their own, so one that raises shows up by name
# instead of taking the whole probe down.
rec("setup_register_fake", lambda: torch.library.register_fake("tnpl::k")(
    lambda x: torch.empty_like(x)) is not None)

torch.library.define("tnpl::dv", "(Tensor x) -> Tensor", lib=lib)
rec("setup_device_typed_impl", lambda: torch.library.impl(
    "tnpl::dv", "cpu", lambda x: x * 4, lib=lib) is not None)

lib.define("d(Tensor x) -> Tensor")
lib.impl("d", lambda x: x + 100, "CPU")

# The decorator form `exir/passes/dim_order_ops_registry.py` and
# `_device_copy_ops_registry.py` use, on an out variant: a keyword-only
# `out` argument that must reach the kernel as a keyword.
lib.define("e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
torch.library.impl(lib, "e.out", "CompositeImplicitAutograd")(
    lambda self, *, out: out.copy_(self * 3))


def _co(x: torch.Tensor) -> torch.Tensor:
    return x + 7


def make_custom_op():
    made = torch.library.custom_op("tnpl::co", mutates_args=())(_co)
    made.register_fake(lambda x: torch.empty_like(x))
    return made


co = None
rec("setup_custom_op", lambda: globals().__setitem__("co", make_custom_op()) is None)


x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
meta = torch.empty(3, 4, device="meta")
ops = torch.ops.tnpl

rec("f_call", lambda: ops.f(x).tolist())
rec("f_default", lambda: ops.f.default(x).tolist())
rec("f_meta", lambda: desc(ops.f(meta)))
rec("g_cpu", lambda: logged(lambda: ops.g(x).tolist()))
rec("g_meta", lambda: logged(lambda: desc(ops.g(meta))))
rec("h_call", lambda: ops.h(x).tolist())
rec("m_meta", lambda: desc(ops.m(meta)))
rec("m_cpu", lambda: ops.m(x).tolist())
rec("z_cpu", lambda: ops.z(x).tolist())
rec("w_call", lambda: logged(lambda: ops.w(x).tolist()))
rec("s_before", lambda: s_before)
rec("s_call", lambda: ops.s(2, 3))
rec("s_backend_assert", lambda: any(torch.ops.tnpl.s.default.has_kernel_for_dispatch_key(k) for k in (
    DK.Meta, DK.CompositeExplicitAutograd, DK.CompositeImplicitAutograd)))
rec("k_cpu", lambda: ops.k(x).tolist())
rec("k_meta", lambda: desc(ops.k(meta)))
rec("dv_call", lambda: ops.dv(x).tolist())
rec("e_out", lambda: (lambda o: [ops.e.out(x, out=o) is o, o.tolist()])(torch.zeros(2, 2)))
rec("co_call", lambda: co(x).tolist())
rec("h_decompose", lambda: ops.h.default.decompose(x).tolist())
rec("f_decompose", lambda: ops.f.default.decompose(x) is NotImplemented)
rec("g_op_dk_meta", lambda: logged(lambda: list(ops.g.default._op_dk(DK.Meta, x).shape)))

frag = Library("tnpl", "FRAGMENT")
frag.impl("d", lambda x: x + 200, "CPU")
rec("d_newer", lambda: ops.d(x).tolist())
frag._destroy()
rec("d_restored", lambda: ops.d(x).tolist())

from torch.utils._python_dispatch import TorchDispatchMode


class Log(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        return func(*args, **(kwargs or {}))


def under_log(fn):
    with Log() as log:
        fn()
    return log.seen


rec("f_mode", lambda: under_log(lambda: ops.f(x)))
rec("h_mode", lambda: (lambda seen: {
    "custom_op_seen": any("tnpl" in s for s in seen), "inner_ops": bool(seen)})(
    under_log(lambda: ops.h(x))))

from torch._subclasses.fake_tensor import FakeTensorMode


def under_fake(fn):
    with FakeTensorMode():
        fx = torch.empty(3, 4)
        del ran[:]
        out = fn(fx)
        return {"shape": list(out.shape), "type": type(out).__name__, "ran": list(ran)}


rec("g_fake", lambda: under_fake(ops.g))
rec("k_fake", lambda: under_fake(ops.k))
rec("co_fake", lambda: under_fake(co))

KEYS = ("CompositeExplicitAutograd", "CompositeImplicitAutograd", "Meta", "CPU")
NAMES = ("tnpl::f", "tnpl::g", "tnpl::h", "tnpl::m", "tnpl::z", "tnpl::k", "tnpl::co", "tnpl::dv")
rec("has_kernel", lambda: {n: {k: bool(C._dispatch_has_kernel_for_dispatch_key(n, k)) for k in KEYS}
                           for n in NAMES})
rec("has_kernel_enum", lambda: {n: bool(C._dispatch_has_kernel_for_dispatch_key(n, DK.Meta))
                                for n in NAMES})
rec("any_kernel", lambda: {n: bool(C._dispatch_has_kernel_for_any_dispatch_key(
    n, C.DispatchKeySet(DK.CPU))) for n in NAMES})
rec("overload_method", lambda: {
    "g_meta": torch.ops.tnpl.g.default.has_kernel_for_dispatch_key(DK.Meta),
    "f_meta": torch.ops.tnpl.f.default.has_kernel_for_dispatch_key(DK.Meta),
    "f_cea": torch.ops.tnpl.f.default.has_kernel_for_dispatch_key(DK.CompositeExplicitAutograd)})
rec("computed", lambda: {
    "f_meta": C._dispatch_has_computed_kernel_for_dispatch_key("tnpl::f", "Meta"),
    "h_cpu": C._dispatch_has_computed_kernel_for_dispatch_key("tnpl::h", "CPU"),
    "m_cpu": C._dispatch_has_computed_kernel_for_dispatch_key("tnpl::m", "CPU"),
    "m_meta": C._dispatch_has_computed_kernel_for_dispatch_key("tnpl::m", "Meta")})
rec("computed_z_meta", lambda: C._dispatch_has_computed_kernel_for_dispatch_key("tnpl::z", "Meta"))
rec("co_autograd", lambda: bool(C._dispatch_has_kernel_for_dispatch_key("tnpl::co", "Autograd")))
rec("aten_kept", lambda: [
    bool(C._dispatch_has_kernel_for_dispatch_key("aten::add.Tensor", "CPU")),
    bool(C._dispatch_has_kernel_for_dispatch_key("aten::sqrt", "CompositeImplicitAutograd")),
    bool(C._dispatch_has_kernel_for_any_dispatch_key("aten::add.Tensor", C.DispatchKeySet(DK.CPU)))])

rec("key_for_device", lambda: [C._dispatch_key_for_device(d) for d in ("cpu", "cuda", "meta", "mps")])

from torch._library.opaque_object import is_opaque_type, register_opaque_type


class Color(enum.Enum):
    RED = 1


class Tag:
    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        return isinstance(other, Tag) and other.v == self.v

    def __hash__(self):
        return hash(self.v)

    def __fx_repr__(self):
        return (f"Tag({self.v})", {"Tag": Tag})


def opaque_cycle():
    name = f"{Tag.__module__}.{Tag.__qualname__}"
    before = C._is_opaque_type_registered(name)
    register_opaque_type(Tag, typ="value")
    return [before, C._is_opaque_type_registered(name), is_opaque_type(Tag)]


rec("opaque_enum", lambda: C._is_opaque_type_registered("enum.Enum"))
rec("opaque_absent", lambda: C._is_opaque_type_registered("tnpl.NotAType"))
rec("opaque_enum_subclass", lambda: is_opaque_type(Color))
rec("opaque_registered", opaque_cycle)

if IS_SHIM:
    rec("shim_live", lambda: [list(p) for p in C._shim_python_kernels() if p[0].startswith("tnpl.")])
    rec("shim_armed", lambda: [k for k in C._shim_armed_python_kernels() if k.startswith("tnpl.")])
    rec("shim_dropped", lambda: [[str(v) for v in e] for e in C._shim_registrations
                                 if e[0] == "impl" and e[1] == "tnpl"])
    rec("shim_aten_dropped", lambda: sum(1 for e in C._shim_registrations
                                         if e[0] == "impl" and e[1] == "aten"))
    rec("shim_not_advertised", lambda: [k for k in list(C._aten_implemented())
                                        + list(C._aten_all_implemented()) if k.startswith("tnpl.")])
    rec("shim_arm_aten", lambda: C._shim_arm_python_kernel("aten.add.Tensor"))

print(json.dumps({"is_shim": IS_SHIM, "r": r}))
"""


_CACHE = {}


def _probe(shim, *flags):
    key = (shim, flags)
    if key not in _CACHE:
        _CACHE[key] = _run(list(flags), shim)["r"]
    return _CACHE[key]


def _value(results, name):
    entry = results[name]
    assert "value" in entry, f"{name} raised {entry.get('raised')}: {entry.get('msg')}"
    return entry["value"]


#: Results compared value for value against upstream.
_AGREE = (
    "f_call", "f_default", "f_meta", "g_cpu", "g_meta", "h_call", "m_meta",
    "w_call", "s_before", "s_call", "s_backend_assert", "k_cpu", "k_meta",
    "dv_call", "e_out", "co_call", "setup_register_fake", "setup_device_typed_impl", "setup_custom_op", "h_decompose", "f_decompose", "g_op_dk_meta",
    "d_newer", "d_restored", "f_mode", "h_mode", "g_fake", "k_fake", "co_fake",
    "has_kernel", "has_kernel_enum", "any_kernel", "overload_method", "computed",
    "key_for_device", "opaque_enum", "opaque_absent", "opaque_enum_subclass",
    "opaque_registered",
)


def _disagreements(shim, up, names=_AGREE):
    out = []
    for name in names:
        if shim.get(name) != up.get(name):
            out.append(f"{name}: shim {shim.get(name)!r} != upstream {up.get(name)!r}")
    return out


# ---------------------------------------------------------------------------
# 1. Each key, called through `torch.ops`, against upstream.
# ---------------------------------------------------------------------------
def test_upstream_answers_the_probe_without_raising():
    """The oracle itself: every compared result is a value on upstream.

    If upstream raised on one, the comparison below would be comparing two
    errors, which can agree while both sides are broken.
    """
    up = _probe(False)
    raised = {n: up[n] for n in _AGREE if "value" not in up[n]}
    assert not raised, f"upstream raised on {sorted(raised)}: {raised}"
    assert _value(up, "f_call") == [[3.0, 5.0], [7.0, 9.0]], up["f_call"]


def test_kernels_at_every_key_agree_with_upstream_through_torch_ops():
    """define + impl at CEA, CIA, CPU and Meta, called through `torch.ops`.

    One comparison over every result in `_AGREE`, so a single wrong answer
    names itself rather than hiding behind an aggregate.
    """
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim, up = _probe(True), _probe(False)
    wrong = _disagreements(shim, up)
    assert not wrong, "the shim disagrees with upstream on:\n" + "\n".join(wrong)
    print(f"PYLIB: {len(_AGREE)} results agree with upstream")


def test_a_meta_tensor_runs_the_meta_kernel_and_takes_its_shape():
    """`g` has a CPU kernel and a Meta kernel with a different output rule
    written out; a meta input must run the Meta one (the log says which ran)
    and come back with its shape, on a meta tensor and on a FakeTensor."""
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim = _probe(True)
    meta = _value(shim, "g_meta")
    assert meta["ran"] == ["g.Meta"], meta
    assert meta["out"] == {"device": "meta", "shape": [3, 8], "type": "Tensor"}, meta
    fake = _value(shim, "g_fake")
    assert fake == {"shape": [3, 8], "type": "FakeTensor", "ran": ["g.Meta"]}, fake
    assert _value(shim, "g_cpu")["ran"] == ["g.CPU"], shim["g_cpu"]
    # `f` has only a CompositeExplicitAutograd kernel, which covers Meta.
    assert _value(shim, "f_meta") == {"device": "meta", "shape": [3, 4], "type": "Tensor"}


def test_register_fake_and_custom_op_reach_fake_and_meta_tensors():
    """`torch.library.register_fake` and `custom_op(...).register_fake` both
    register through `_dispatch_library` (a `Meta` impl plus the fake
    registry); both shapes propagate."""
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim = _probe(True)
    assert _value(shim, "k_meta") == {"device": "meta", "shape": [3, 4], "type": "Tensor"}
    assert _value(shim, "k_fake")["shape"] == [3, 4], shim["k_fake"]
    assert _value(shim, "co_fake")["type"] == "FakeTensor", shim["co_fake"]
    assert _value(shim, "co_call") == [[8.0, 9.0], [10.0, 11.0]], shim["co_call"]


def test_composite_implicit_decomposes_above_a_mode_and_explicit_does_not():
    """Upstream's order: CIA runs at the autograd level, above the `Python`
    key, so a `TorchDispatchMode` sees what it decomposes into; a CEA op is a
    backend kernel and the mode sees the op itself."""
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim = _probe(True)
    assert _value(shim, "f_mode") == ["tnpl.f.default"], shim["f_mode"]
    assert _value(shim, "h_mode") == {"custom_op_seen": False, "inner_ops": True}, shim["h_mode"]


# ---------------------------------------------------------------------------
# 2. The has-kernel queries, both ways.
# ---------------------------------------------------------------------------
def test_has_kernel_queries_answer_truthfully_for_python_kernels():
    """True where a kernel was registered at exactly that key, False where it
    was not, in both spellings of the key; `_for_any_dispatch_key` excludes
    alias keys; the computed predicate counts the composites that cover a
    backend key. The exir assertion (`s_backend_assert`) holds."""
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim = _probe(True)
    table = _value(shim, "has_kernel")
    assert table["tnpl::f"] == {"CompositeExplicitAutograd": True, "CompositeImplicitAutograd": False,
                                "Meta": False, "CPU": False}, table["tnpl::f"]
    assert table["tnpl::g"]["CPU"] and table["tnpl::g"]["Meta"], table["tnpl::g"]
    assert not any(table["tnpl::z"].values()), table["tnpl::z"]
    assert _value(shim, "any_kernel") == {
        "tnpl::f": False, "tnpl::g": True, "tnpl::h": False, "tnpl::m": False,
        "tnpl::z": False, "tnpl::k": True, "tnpl::co": False, "tnpl::dv": True}, shim["any_kernel"]
    assert _value(shim, "s_before") == [False, False, False], shim["s_before"]
    assert _value(shim, "s_backend_assert") is True
    computed = _value(shim, "computed")
    assert computed == {"f_meta": True, "h_cpu": True, "m_cpu": False, "m_meta": True}, computed
    # The one computed answer that differs, by name: upstream says True for an
    # op with no kernel at all at `Meta`, because its C++ build registers a
    # `Meta` backend fallback (docs/graph/BFALLBACK.md). This shim registers
    # none, so it says False rather than promise a kernel.
    assert _value(_probe(False), "computed_z_meta") is True
    assert _value(shim, "computed_z_meta") is False


def test_aten_names_keep_the_constant_false_and_custom_ops_stay_off_the_golden_list():
    """REGISTRATIONS.md section 3.4: for aten the narrow predicate must stay
    `False` and never validate, or `activate_meta()` dies; that is unchanged.
    And a Python kernel is not a native one: no `tnpl.*` key may appear in
    `_aten_implemented()` (which would demand a golden case builder) or in
    `_aten_all_implemented()`; arming an aten key is refused by name."""
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim = _probe(True)
    assert _value(shim, "aten_kept") == [False, False, False], shim["aten_kept"]
    assert _value(shim, "shim_not_advertised") == [], shim["shim_not_advertised"]
    arm = shim["shim_arm_aten"]
    assert arm.get("raised") == "ValueError" and "aten.add.Tensor" in arm["msg"], arm


# ---------------------------------------------------------------------------
# 3. Refusals.
# ---------------------------------------------------------------------------
def test_a_key_with_no_kernel_refuses_by_name():
    """`m` has only a Meta kernel; a CPU call refuses naming the op and the
    key, as upstream's `Could not run 'tnpl::m' with arguments from the 'CPU'
    backend` does. `z` has no kernel at all and still refuses, naming itself.
    Same exception type as upstream in both cases."""
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim, up = _probe(True), _probe(False)
    for name in ("m_cpu", "z_cpu"):
        assert shim[name].get("raised") == up[name].get("raised") == "NotImplementedError", (
            name, shim[name], up[name])
    assert "tnpl::m" in shim["m_cpu"]["msg"] and "'CPU'" in shim["m_cpu"]["msg"], shim["m_cpu"]
    assert "Meta" in shim["m_cpu"]["msg"], shim["m_cpu"]
    assert "tnpl" in shim["z_cpu"]["msg"] and "z" in shim["z_cpu"]["msg"], shim["z_cpu"]


def test_the_autograd_key_is_a_named_gap_not_a_claimed_kernel():
    """`custom_op` registers an `Autograd` kernel. The shim has no autograd
    dispatch level for Python kernels, so that registration is recorded in
    `_shim_registrations` and dropped, and the predicate says `False` for it
    where upstream says `True`. Asserted so the day it is wired, this reddens
    and the docs move with it."""
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim, up = _probe(True), _probe(False)
    assert _value(up, "co_autograd") is True
    assert _value(shim, "co_autograd") is False
    dropped = _value(shim, "shim_dropped")
    assert ["impl", "tnpl", "co", "Autograd"] in dropped, dropped
    live = _value(shim, "shim_live")
    assert ["tnpl.co.default", "CompositeExplicitAutograd"] in live, live
    # Live kernels are no longer listed as dropped, so the list still means
    # "what does not run"; the 1000-odd aten Meta registrations still are.
    assert not any(e[3] in ("CompositeExplicitAutograd", "CPU", "Meta",
                            "CompositeImplicitAutograd") for e in dropped), dropped
    assert _value(shim, "shim_aten_dropped") > 1000, shim["shim_aten_dropped"]


# ---------------------------------------------------------------------------
# 4. Opaque types.
# ---------------------------------------------------------------------------
def test_is_opaque_type_registered_reads_the_registry_register_opaque_type_writes():
    """`import torch` registers `enum.Enum` (`opaque_object.py:272`); an
    unregistered name answers False; registering a value type flips its
    answer from False to True; an Enum subclass is opaque. All four agree with
    upstream (they are in `_AGREE`); asserted directly here so the meaning does
    not depend on the oracle."""
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    shim = _probe(True)
    assert _value(shim, "opaque_enum") is True
    assert _value(shim, "opaque_absent") is False
    assert _value(shim, "opaque_enum_subclass") is True
    assert _value(shim, "opaque_registered") == [False, True, True], shim["opaque_registered"]


# ---------------------------------------------------------------------------
# 5. Nullification (AGENTS.md section 17.5).
# ---------------------------------------------------------------------------
def test_dropping_the_registrations_again_is_caught():
    """Nullification: `--drop` replaces `_DispatchLibrary.impl` with the
    pre-#227 recorder (append to `_shim_registrations`, keep nothing). The
    same probe must then disagree with upstream on the kernel results and the
    has-kernel table; if it did not, the agreement test above could not fail.

    Source-level form of the same mutant: make `impl` in `_install_library`
    (`torchnative/rust/torch_c/src/bootstrap.py`) only append to
    `registrations`; `f_call` then raises `aten op not implemented in torch._C
    shim: tnpl.f.default`, which is the measured state issue #227 opened on.
    """
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    dropped, up = _probe(True, "--drop"), _probe(False)
    wrong = _disagreements(dropped, up)
    for name in ("f_call", "g_meta", "h_call", "has_kernel", "s_backend_assert"):
        assert any(w.startswith(name + ":") for w in wrong), (
            f"dropping the registrations left {name} agreeing with upstream: "
            f"{dropped.get(name)!r}")
    assert dropped["f_call"].get("raised") == "NotImplementedError", dropped["f_call"]
    assert "tnpl.f.default" in dropped["f_call"]["msg"], dropped["f_call"]
    print(f"PYLIB: --drop disagrees with upstream on {len(wrong)} of {len(_AGREE)} results")


def test_the_door_is_what_runs_the_kernel():
    """Nullification of the door's side only: `--disarm` keeps the Python
    registry intact but replaces `_shim_arm_python_kernel` with a no-op, so
    `_aten_dispatch` is never told the op has a kernel. Every call must then
    refuse at the native table. If a kernel still ran, something other than
    the one door (a Python-level shortcut in `_op_callable`, say) would be
    calling it, which AGENTS.md section 11 forbids.
    """
    if not _available():
        _skip.skip(f"no vendored shim at {_VENDOR_SHIM}")
        return
    disarmed = _probe(True, "--disarm")
    for name in ("f_call", "g_cpu", "h_call", "w_call", "s_call"):
        entry = disarmed[name]
        assert entry.get("raised") == "NotImplementedError", (name, entry)
        assert "aten op not implemented in torch._C shim" in entry["msg"], (name, entry)
    assert _value(disarmed, "shim_armed") == [], disarmed["shim_armed"]
    # The registry itself still answers, which is what isolates the door.
    assert _value(disarmed, "has_kernel")["tnpl::f"]["CompositeExplicitAutograd"] is True
    armed = _value(_probe(True), "shim_armed")
    assert "tnpl.f.default" in armed and "tnpl.g.default" in armed, armed


def _main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    return _skip.run_tests(tests, suite="test_pylibrary")


if __name__ == "__main__":
    sys.exit(_main())
