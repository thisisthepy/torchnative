"""`torch.var_mean` -- the pair-returning op `torch.export` stops at.

`docs/graph/STRIDE.md` §5 measured where the forty architectures stop and found
the number that matters is narrower than it looks: **ten** of the forty are
ones upstream torch can export at all, and those ten stop at exactly two
places -- `torch.var_mean` (6) and `torch._C._select_conv_backend` (4).  This
file is the first of the two.

Nothing here asserts "export() returned".  `docs/graph/COMPILE.md` records the
trap: a superficial fix can make the machinery silently hand back something
that looks finished and computes nothing.  So the bar is the one
`export_sweep.py` reports -- **exported, replayed, and agreeing element-wise
with the unexported module** -- and the last test in this file is that bar on
a module whose only new requirement is `var_mean`.

Every expected answer below is **upstream torch's own, computed in a separate
subprocess** from the same probe text.  No table of expected numbers is
written beside the implementation, and the numeric tolerances are imported
from `tools/golden/dtypes.py` rather than restated, so that widening one here
is not possible without widening the golden harness's.

What this file cannot see, stated so nobody reads more into it:

* It judges the overloads and shapes the probes build.  A spelling not in the
  probe is not judged.
* `test_var_mean_resolves_the_overload_var_resolves_and_that_is_not_upstreams`
  pins a **disagreement**, not an agreement.  It is here because the
  disagreement is pre-existing and shared with `torch.var` (docs/graph/EXPORT5.md
  §9), and a silent change to it should redden something.
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

    A tolerance written here could be widened without anything noticing; one
    read out of the golden harness cannot, because widening it there reddens
    the harness's own self-test.
    """
    path = os.path.join(_REPO_ROOT, "tools", "golden", "dtypes.py")
    spec = importlib.util.spec_from_file_location("_golden_dtypes", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `@dataclass` looks its own class's module up
    # in `sys.modules`, and an unregistered one makes it raise.
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


def _close(x, y, dtype_name):
    atol, rtol = _TOL.get(dtype_name, (0.0, 0.0))
    xf, yf = float(x), float(y)
    if math.isnan(xf) or math.isnan(yf):
        return math.isnan(xf) and math.isnan(yf)
    if math.isinf(xf) or math.isinf(yf):
        return xf == yf
    return math.isclose(xf, yf, rel_tol=rtol, abs_tol=atol)


# ---------------------------------------------------------------------------
# The value probe.  Identical text on both sides; only the answers can differ.
# ---------------------------------------------------------------------------

_VALUE_PROBE = r"""
import json
import torch

aten = torch.ops.aten
out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "cases": {}}

# Values that are not a symmetric progression, so a wrong reduction axis is a
# wrong number rather than the same number by symmetry.
SIX = [1.5, -0.25, 3.0, 0.75, -2.5, 4.25]
TWENTY_FOUR = [round((i * i % 17) / 3.0 - 2.0, 6) for i in range(24)]


def make(flat, shape, dtype_name):
    t = torch.tensor(flat, dtype=torch.float64).reshape(shape)
    return t.to(getattr(torch, dtype_name))


def describe(pair):
    var, mean = pair[0], pair[1]
    return {
        "var": {"shape": list(var.shape), "dtype": str(var.dtype).split(".")[-1],
                "values": [float(v) for v in var.flatten().tolist()],
                "stride": list(var.stride()), "type": type(var).__name__},
        "mean": {"shape": list(mean.shape), "dtype": str(mean.dtype).split(".")[-1],
                 "values": [float(v) for v in mean.flatten().tolist()],
                 "stride": list(mean.stride()), "type": type(mean).__name__},
    }


def case(name, fn):
    try:
        out["cases"][name] = describe(fn())
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "message": str(exc)}


# --- aten.var_mean.correction: the overload upstream's binding always uses ---
for dtype_name in ("float64", "float32", "float16", "bfloat16"):
    for flat, shape, label in ((SIX, (2, 3), "2x3"), (TWENTY_FOUR, (2, 3, 4), "2x3x4"),
                               ([1.5, 3.5], (2,), "n2")):
        x = make(flat, shape, dtype_name)
        for dim in (None, [0], [-1], [0, 1], []):
            for correction in (None, 0, 1, 2):
                for keepdim in (False, True):
                    name = (f"correction/{dtype_name}/{label}/dim={dim}/"
                            f"c={correction}/keep={keepdim}")
                    case(name, lambda x=x, d=dim, c=correction, k=keepdim:
                         aten.var_mean.correction(x, d, correction=c, keepdim=k))

# --- aten.var_mean.dim: `unbiased`, the spelling `_refs.native_layer_norm`
# --- and `_refs.native_group_norm` actually use ------------------------------
for dtype_name in ("float64", "float32", "float16", "bfloat16"):
    x = make(TWENTY_FOUR, (2, 3, 4), dtype_name)
    for dim in ([1], [0, 2], [2], []):
        for unbiased in (True, False):
            for keepdim in (False, True):
                name = f"dim/{dtype_name}/dim={dim}/unbiased={unbiased}/keep={keepdim}"
                case(name, lambda x=x, d=dim, u=unbiased, k=keepdim:
                     aten.var_mean.dim(x, d, u, k))

# --- aten.var_mean.default: no `dim` at all ---------------------------------
for dtype_name in ("float64", "float32", "float16", "bfloat16"):
    for flat, shape, label in ((SIX, (2, 3), "2x3"), ([1.5, 3.5], (2,), "n2")):
        x = make(flat, shape, dtype_name)
        for unbiased in (True, False):
            name = f"default/{dtype_name}/{label}/unbiased={unbiased}"
            case(name, lambda x=x, u=unbiased: aten.var_mean.default(x, u))

# --- the clamped denominator: `max(0, n - correction)`, inf vs nan ----------
# `var` records five measured rows for this (aten.rs::var_reduce); the pair
# has to answer them identically or it is not sharing that arithmetic.
for flat, shape, correction, note in (
    ([3.0], (), None, "n-c == 0, m2 == 0 -> nan"),
    ([], (0,), None, "n-c == -1, m2 == 0 -> nan"),
    ([1.0, 3.0], (2,), 2, "n-c == 0, m2 > 0 -> inf"),
    ([1.0, 3.0], (2,), 3, "n-c == -1, m2 > 0 -> inf"),
    ([2.0, 2.0], (2,), 2, "n-c == 0, m2 == 0 -> nan"),
):
    x = make(flat, shape, "float32")
    case(f"denominator/{note}",
         lambda x=x, c=correction: aten.var_mean.correction(x, None, correction=c,
                                                            keepdim=False))

# --- a fractional correction: `correction` is a Scalar, not an int ----------
x = make(SIX, (2, 3), "float64")
case("fractional/correction=0.5",
     lambda: aten.var_mean.correction(x, None, correction=0.5, keepdim=False))

# --- refusals ---------------------------------------------------------------
for dtype_name in ("int64", "int32", "bool"):
    t = torch.tensor([1, 0, 1, 1], dtype=getattr(torch, dtype_name))
    case(f"refuse/{dtype_name}/correction",
         lambda t=t: aten.var_mean.correction(t, None, correction=1, keepdim=False))
    case(f"refuse/{dtype_name}/default", lambda t=t: aten.var_mean.default(t, True))
x = make(SIX, (2, 3), "float32")
case("refuse/duplicate-dim",
     lambda: aten.var_mean.correction(x, [0, 0], correction=1, keepdim=False))
case("refuse/out-of-range-dim",
     lambda: aten.var_mean.correction(x, [5], correction=1, keepdim=False))

# --- non-contiguous input: the values must follow the layout, not the bytes --
base = make(TWENTY_FOUR, (2, 3, 4), "float64")
case("noncontig/transpose",
     lambda: aten.var_mean.correction(base.transpose(0, 2), [1], correction=1,
                                      keepdim=False))
case("noncontig/slice",
     lambda: aten.var_mean.correction(base[:, 1:, ::2], None, correction=1,
                                      keepdim=False))

print(json.dumps(out))
"""


def _values_agree(name, shim_case, up_case):
    if "raised" in up_case or "raised" in shim_case:
        return (shim_case.get("raised"), shim_case.get("message")), \
               (up_case.get("raised"), up_case.get("message"))
    for half in ("var", "mean"):
        s, u = shim_case[half], up_case[half]
        assert s["shape"] == u["shape"], f"{name}: {half} shape {s['shape']} != {u['shape']}"
        assert s["dtype"] == u["dtype"], f"{name}: {half} dtype {s['dtype']} != {u['dtype']}"
        assert s["type"] == u["type"], (
            f"{name}: {half} is a {s['type']} here and a {u['type']} upstream -- "
            f"an unpromoted TensorBase inside the tuple, docs/graph/STRIDE.md §8")
        assert len(s["values"]) == len(u["values"]), f"{name}: {half} length"
        for i, (a, b) in enumerate(zip(s["values"], u["values"])):
            assert _close(a, b, u["dtype"]), (
                f"{name}: {half}[{i}] = {a!r} here, {b!r} upstream "
                f"(tolerance {_TOL.get(u['dtype'])})")
    return None, None


def test_var_mean_agrees_with_upstream_element_wise_on_both_halves():
    """Both halves of the pair, over dtype x shape x dim x correction x keepdim.

    `var` alone would pass a kernel that computed the variance correctly and
    handed back the wrong second element entirely, so `mean` is compared as a
    result and not as a by-product.  The dtypes, shapes and the result's
    Python *type* are compared too -- `docs/graph/STRIDE.md` §8 found three
    meta arms returning bare `TensorBase` objects inside a tuple, which no
    value comparison can see.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_VALUE_PROBE, shim=True)
    up = _run(_VALUE_PROBE, shim=False)
    assert set(shim["cases"]) == set(up["cases"])
    compared, refusals = 0, 0
    for name in sorted(up["cases"]):
        s_err, u_err = _values_agree(name, shim["cases"][name], up["cases"][name])
        if u_err is None and s_err is None:
            compared += 1
        else:
            refusals += 1
    assert compared >= 250, f"only {compared} cases actually computed a pair"
    print(f"VALUES: {compared} var_mean results agree with upstream element-wise "
          f"on both halves; {refusals} cases raised on one side or both")


def test_var_mean_refuses_exactly_what_upstream_refuses_and_says_what_upstream_says():
    """The refusals, by name.

    Separated from the agreement test because a kernel that raises on
    everything would make that one vacuous: it compares only the cases that
    produced a pair.  Here the refusing cases are the subject, and upstream's
    own message text is the expectation.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_VALUE_PROBE, shim=True)
    up = _run(_VALUE_PROBE, shim=False)
    mismatched, checked = [], 0
    for name in sorted(up["cases"]):
        s, u = shim["cases"][name], up["cases"][name]
        if "raised" not in u and "raised" not in s:
            continue
        checked += 1
        if ("raised" in u) != ("raised" in s):
            mismatched.append((name, s.get("raised", "no error"), u.get("raised", "no error")))
        elif u["message"] not in s["message"] and s["message"] not in u["message"]:
            mismatched.append((name, s["message"], u["message"]))
    assert not mismatched, f"{len(mismatched)} refusals differ from upstream: {mismatched}"
    assert checked >= 8, f"only {checked} refusal cases -- the probe lost its refusals"
    print(f"REFUSALS: {checked} cases raise on both sides with upstream's own wording")


# ---------------------------------------------------------------------------
# The correction default, on its own, because getting it wrong is silent
# ---------------------------------------------------------------------------

_DEFAULT_PROBE = r"""
import json
import torch
out = {"is_shim": hasattr(torch._C, "_aten_implemented")}
x = torch.tensor([1.0, 3.0])
r = {}
r["python_binding"] = [float(t) for t in torch.var_mean(x)]
r["correction_none"] = [float(t) for t in torch.ops.aten.var_mean.correction(
    x, None, correction=None, keepdim=False)]
r["correction_zero"] = [float(t) for t in torch.ops.aten.var_mean.correction(
    x, None, correction=0, keepdim=False)]
r["unbiased_true"] = [float(t) for t in torch.ops.aten.var_mean.default(x, True)]
r["unbiased_false"] = [float(t) for t in torch.ops.aten.var_mean.default(x, False)]
out["r"] = r
print(json.dumps(out))
"""


def test_var_mean_correction_defaults_to_one_not_zero():
    """`correction=None` means **1**, and at n=2 the two answers differ by 2x.

    The trap `aten.rs::var_correction` already records for `var`: the default
    is not 0.  Two elements is where it is loudest -- the biased variance of
    `[1, 3]` is 1.0 and the unbiased one is 2.0 -- so a kernel that defaulted
    to 0 would still look plausible at n=1000 and is caught here.

    The expectation is upstream's, read in a subprocess; the literals in the
    docstring are commentary.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_DEFAULT_PROBE, shim=True)["r"]
    up = _run(_DEFAULT_PROBE, shim=False)["r"]
    assert shim == up, f"shim={shim} upstream={up}"
    # And that upstream's own answers really do separate the two conventions,
    # so this test cannot pass by both sides being wrong in the same way.
    assert up["correction_none"] != up["correction_zero"], up
    assert up["unbiased_true"] == up["correction_none"], up
    assert up["unbiased_false"] == up["correction_zero"], up
    print(f"CORRECTION: default is correction=1 on both sides: {up}")


# ---------------------------------------------------------------------------
# The meta kernel -- what `torch.export` actually traces through
# ---------------------------------------------------------------------------

_META_PROBE = r"""
import json
import torch

aten = torch.ops.aten
out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "cases": {}}


def lay(t):
    return {"shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype).split(".")[-1], "type": type(t).__name__,
            "offset": t.storage_offset()}


def case(name, fn):
    try:
        pair = fn()
        out["cases"][name] = [lay(pair[0]), lay(pair[1])]
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "message": str(exc)}


builds = {
    "contig": lambda: torch.empty(3, 4, 5, device="meta"),
    "transposed": lambda: torch.empty(4, 3, 5, device="meta").transpose(0, 1),
    "permuted": lambda: torch.empty(5, 3, 4, device="meta").permute(1, 2, 0),
    "sliced": lambda: torch.empty(6, 4, 5, device="meta")[1:4],
    # Channels-last built by PERMUTING an NHWC tensor, not by
    # `.to(memory_format=...)`: the dense side of this shim cannot re-lay a
    # tensor and refuses that call by name (docs/graph/STRIDE.md §4), so a
    # probe written the obvious way would compare a refusal against a layout.
    # A permuted NHWC tensor is channels-last on both sides with no
    # memory-format request anywhere -- which is §4's own finding.
    "channels_last": lambda: torch.empty(2, 4, 5, 3, device="meta").permute(0, 3, 1, 2),
}
for label, build in builds.items():
    for dim in (None, [1], [0, -1], []):
        for keepdim in (False, True):
            case(f"correction/{label}/dim={dim}/keep={keepdim}",
                 lambda b=build, d=dim, k=keepdim:
                 aten.var_mean.correction(b(), d, correction=1, keepdim=k))
            case(f"dim/{label}/dim={dim}/keep={keepdim}",
                 lambda b=build, d=dim, k=keepdim:
                 aten.var_mean.dim(b(), d, False, k))
    case(f"default/{label}", lambda b=build: aten.var_mean.default(b(), True))

for dtype_name in ("float64", "float32", "float16", "bfloat16", "int64", "bool"):
    case(f"dtype/{dtype_name}",
         lambda d=dtype_name: aten.var_mean.correction(
             torch.empty(2, 3, device="meta", dtype=getattr(torch, d)),
             None, correction=1, keepdim=False))

print(json.dumps(out))
"""


def test_var_mean_on_meta_answers_upstreams_shape_dtype_and_stride():
    """The meta arm, compared exactly -- strides are integers, no tolerance.

    `docs/graph/STRIDE.md` §3 makes every meta kernel a claim about the
    output's *stride*, and the claim here is `AlwaysContiguous`: measured
    against upstream across five input layouts including channels-last.  It is
    a measurement and not a reading, which is why the channels-last and
    permuted inputs are in the probe -- `sort`, `index.Tensor` and
    `_weight_norm_interface` looked like members of that class and are not.

    Both halves are checked, and so is each half's Python type: a bare
    `TensorBase` inside the tuple is invisible to a shape comparison.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_META_PROBE, shim=True)
    up = _run(_META_PROBE, shim=False)
    assert set(shim["cases"]) == set(up["cases"])
    wrong, compared, refused = [], 0, 0
    for name in sorted(up["cases"]):
        s, u = shim["cases"][name], up["cases"][name]
        if isinstance(u, dict):
            refused += 1
            if not isinstance(s, dict):
                wrong.append((name, "answered", f"upstream raised {u['raised']}"))
            continue
        if isinstance(s, dict):
            wrong.append((name, f"raised {s.get('raised')}: {s.get('message')}", u))
            continue
        compared += 1
        if s != u:
            wrong.append((name, s, u))
    assert not wrong, f"{len(wrong)} meta answers differ from upstream: {wrong[:6]}"
    assert compared >= 40, f"only {compared} meta cases answered"
    print(f"META: {compared} var_mean meta results identical to upstream "
          f"(shape, stride, offset, dtype, type); {refused} refused on both sides")


def test_var_mean_on_meta_refuses_with_the_dense_kernels_wording():
    """A second pinned divergence, and this one is upstream disagreeing with
    itself.

    Upstream's CPU kernel refuses an integral input with `var_mean only
    support floating point and complex dtypes`.  Upstream's *meta* kernel
    refuses the same input with `mean(): could not infer output dtype...`,
    because its meta arm is the `_refs` decomposition and the refusal surfaces
    from the `mean` inside it.  `docs/devices/META.md` §7.1 settles which one
    this shim follows -- the dense kernel's, called rather than restated, so a
    meta answer cannot promise what the dense path would refuse -- and
    `_weight_norm_interface` already carries four divergences of the same
    shape.

    Pinned rather than left unstated: both sides refusing is all the meta test
    above checks, and a shim that started answering here, or started echoing
    upstream's meta wording, would slip past it.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_META_PROBE, shim=True)["cases"]
    up = _run(_META_PROBE, shim=False)["cases"]
    checked = 0
    for name in ("dtype/int64", "dtype/bool"):
        s, u = shim[name], up[name]
        assert isinstance(s, dict) and "raised" in s, f"{name}: shim answered {s}"
        assert isinstance(u, dict) and "raised" in u, f"{name}: upstream answered {u}"
        assert "var_mean only support floating point" in s["message"], s
        assert "could not infer output dtype" in u["message"], (
            f"upstream's meta wording changed: {u['message']}")
        checked += 1
    assert checked == 2
    print("META REFUSAL: this shim uses the dense kernel's wording; upstream's "
          "meta arm uses `mean()`'s (META.md §7.1)")


# ---------------------------------------------------------------------------
# The overload the Python binding resolves to -- a pinned DISAGREEMENT
# ---------------------------------------------------------------------------

_OVERLOAD_PROBE = r"""
import json
import torch
from torch.utils._python_dispatch import TorchDispatchMode

out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "seen": {}}
x = torch.tensor([1.0, 3.0])


class Logger(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


calls = {
    "bare": lambda: torch.var_mean(x),
    "dim": lambda: torch.var_mean(x, dim=0),
    "unbiased_positional": lambda: torch.var_mean(x, False),
    "dim_unbiased_keepdim": lambda: torch.var_mean(x, 0, False, True),
    "correction": lambda: torch.var_mean(x, dim=0, correction=0),
}
for name, call in calls.items():
    log = Logger()
    try:
        with log:
            call()
        out["seen"][name] = [o for o in log.ops if "var_mean" in o]
    except Exception as exc:
        out["seen"][name] = {"raised": type(exc).__name__, "message": str(exc)}
print(json.dumps(out))
"""


def test_var_mean_resolves_the_overload_var_resolves_and_that_is_not_upstreams():
    """A pinned disagreement, not an agreement.

    Upstream's `PythonArgParser` sends **every** spelling of `torch.var_mean`
    to `aten.var_mean.correction`, translating the deprecated `unbiased` form
    on the way.  This shim's resolver takes the first schema in
    `overloads.json` that binds, so `torch.var_mean(x)` reaches
    `aten.var_mean.default` and `torch.var_mean(x, dim=0)` reaches
    `aten.var_mean.dim`.  `torch.var` has had exactly this disagreement since
    it landed (`docs/graph/EXPORT5.md` §9), and `var_mean` inherits the table
    shape rather than inventing a second convention.

    It is pinned here because it is a **numerically invisible** difference:
    the values agree, so the value tests above cannot see it, and a later
    change to the table -- in either direction -- should redden something
    rather than pass silently.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_OVERLOAD_PROBE, shim=True)["seen"]
    up = _run(_OVERLOAD_PROBE, shim=False)["seen"]
    assert all(v == ["aten.var_mean.correction"] for v in up.values()), (
        f"upstream no longer funnels every spelling to .correction: {up}")
    assert shim == {
        "bare": ["aten.var_mean.default"],
        "dim": ["aten.var_mean.dim"],
        "unbiased_positional": ["aten.var_mean.default"],
        "dim_unbiased_keepdim": ["aten.var_mean.dim"],
        "correction": ["aten.var_mean.correction"],
    }, shim
    # And the same shape for `torch.var`, so this is inherited and not new.
    print(f"OVERLOADS: shim resolves {shim}; upstream funnels all five to .correction")


# ---------------------------------------------------------------------------
# The bar: exported, replayed, agreed
# ---------------------------------------------------------------------------

_EXPORT_PROBE = r"""
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}


class Normed(torch.nn.Module):
    # `var_mean` in both of the spellings the vendored tree itself uses:
    # `_refs.native_layer_norm` calls it with `unbiased=False` (-> var_mean.dim
    # here) and `_decomp.decompositions._batch_norm_no_update` with
    # `correction=0` (-> var_mean.correction).
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.g = torch.nn.Parameter(torch.randn(6))
        self.b = torch.nn.Parameter(torch.randn(6))

    def forward(self, x):
        v1, m1 = torch.var_mean(x, dim=-1, unbiased=False, keepdim=True)
        a = (x - m1) * torch.rsqrt(v1 + 1e-5) * self.g + self.b
        v2, m2 = torch.var_mean(a, dim=0, correction=0, keepdim=True)
        b = (a - m2) / torch.sqrt(v2 + 1e-5)
        v3, m3 = torch.var_mean(b)
        return b * v3 + m3, v1.flatten(), m2.flatten()


torch.manual_seed(0)
mod = Normed().eval()
x = torch.randn(5, 6)
with torch.no_grad():
    eager = mod(x)
out["eager"] = [t.flatten().tolist() for t in eager]
out["shapes"] = [list(t.shape) for t in eager]
if not out["is_shim"]:
    with torch.no_grad():
        exact = Normed().double().eval()(x.double())
    out["oracle"] = [(e.double() - f).abs().flatten().tolist()
                     for e, f in zip(eager, exact)]
stage = "export"
try:
    ep = torch.export.export(mod, (x,))
    out["targets"] = sorted({str(n.target) for n in ep.graph.nodes
                             if n.op == "call_function"})
    stage = "replay"
    with torch.no_grad():
        replayed = ep.module()(x)
    out["replayed"] = [t.flatten().tolist() for t in replayed]
    out["stage"] = "done"
except Exception:
    import traceback
    out["stage"] = stage
    out["error"] = traceback.format_exc()[-3000:]
print(json.dumps(out))
"""


def test_a_module_that_uses_var_mean_exports_replays_and_agrees():
    """THE BAR.  Three verdicts, kept separate, and none of them is "returned".

    `torch.export.export()` returns an `ExportedProgram`; `ep.module()(x)`
    runs; and its outputs agree element-wise with the module's own eager
    outputs -- and the shim's eager outputs agree with upstream's, so a kernel
    that is wrong in the same way on both of this shim's paths is caught too.

    The tolerance is derived, `docs/numerics/AGREE.md` §2's method: the p90 of
    upstream's own float32-vs-float64 relative error on these very outputs,
    floored at 8 ulp.  It is not a constant anyone can widen.

    The module stays 2-D on purpose.  A 3-D activation would reach a `view`
    inside fake mode and stop at `FakeTensorDeviceMismatchError`, which is a
    dispatcher question (`docs/graph/STRIDE.md` §6) and not this round's.
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
    assert len(targets) >= 6, f"the exported graph holds only {targets}"
    assert any("sqrt" in t for t in targets), targets
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
