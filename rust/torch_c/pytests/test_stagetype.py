"""SPEC S6.5: adaptation stage 0 and stage 1 are separated by **type**, and a
backward-free build refuses stage 1 **at import time** (INTENT §4).

**What "a backward-free build" means here, because it did not exist before.**
No `torch._C` built from this crate lacks the tape -- `tape::register` is
unconditional in `rust/torch_c/src/lib.rs` -- so there is no artefact to probe
for an absent backward, and inventing a build flag to manufacture one would be
a build system nobody asked for. The smallest honest form is a configuration:
`TORCHNATIVE_BACKWARD=off` in the environment when `torchnative.adapt` is
imported. `adapt.BACKWARD` reports what was read. Every check below runs in a
**fresh subprocess** with that variable set or explicitly removed, because the
configuration is read once, at import, and an in-process test would be
reading whichever value this interpreter happened to import with.

**What each test holds down, and the mutant that turns it red** (AGENTS.md
§17.5; each was made and watched go red -- see the round report):

* the two stages are distinct, inspectable types, and `adapt.Tent` is the
  stage-1 type from the stage-1 module
      -> `Tent` re-parented onto `adapt.Method`
      -> `__all__` listing the stage-1 names in a backward-free build (caught
         by the import test, which asserts what `__all__` advertises)
* importing the stage-1 module, by any spelling, is refused at import, naming
  the module and the reason -- and the same imports *succeed* with the
  variable removed, so the refusal cannot be an unrelated import failure
      -> the module-level guard in `adapt/gradient.py` deleted
* a stage-1 or stage-2 method a *user* defines is refused when its class
  statement runs -- at the import of the user's module -- naming the class
      -> `Method.__init_subclass__`'s guard deleted
* stage 0 still adapts and reverts in that configuration, and never loads the
  stage-1 module
      -> `adapt/__init__.py` importing `gradient` eagerly
* a stage declared by attribute rather than by type is refused at `wrap`
      -> the stage-1 type check in `Adapted.__init__` deleted
      -> the stage-0 type check reverted to `hasattr(method, "select_modules")`
* a value of the variable that is neither on nor off is refused by name
      -> the parser reading every unknown value as "on"
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")
_ENV = "TORCHNATIVE_BACKWARD"


def _available():
    return os.path.isfile(_VENDOR_SHIM) and os.path.isfile(
        os.path.join(_VENDOR_DIR, "torch", "__init__.py"))


# Shared preamble: the shim, not upstream (CLAUDE.md §3), and a refusal
# recorder that keeps the exception's type and full text.
_PRELUDE = r'''
import json, sys, importlib.util
import torch
assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"
import torch.nn as nn
out = {}

def attempt(key, fn):
    try:
        fn()
    except BaseException as e:
        out[key] = "%s: %s" % (type(e).__name__, e)
    else:
        out[key] = "ACCEPTED"

def user_module(name, source):
    """Import `source` as module `name` -- what `import name` does with a file."""
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        exec(compile(source, "<%s>" % name, "exec"), mod.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return mod

STAGE1_SRC = """
from torchnative import adapt
class MyEntropy(adapt.Method):
    stage = adapt.STAGE_NARROW_BACKWARD
    def select(self, model): return []
    def objective(self, outputs): return outputs.sum()
"""
STAGE2_SRC = STAGE1_SRC.replace("STAGE_NARROW_BACKWARD", "STAGE_FULL_AUTOGRAD")

C = 4
def bn_model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(C, C), nn.BatchNorm1d(C), nn.Linear(C, C))
    return m.eval()

def batch(seed):
    return torch.tensor([[((seed * 7 + i * 3 + j) % 11) / 3.0 - 1.5 for j in range(C)]
                         for i in range(6)])
'''

_BACKWARD_FREE_SCRIPT = _PRELUDE + r'''
from torchnative import adapt
out["BACKWARD"] = adapt.BACKWARD
out["all"] = sorted(adapt.__all__)
out["gradient_loaded_by_adapt"] = "torchnative.adapt.gradient" in sys.modules

def from_adapt_import_tent():
    from torchnative.adapt import Tent  # noqa: F401
def from_adapt_import_gradient_method():
    from torchnative.adapt import GradientMethod  # noqa: F401
def import_gradient_module():
    import torchnative.adapt.gradient  # noqa: F401
def from_gradient_import_tent():
    from torchnative.adapt.gradient import Tent  # noqa: F401

attempt("from_adapt_import_tent", from_adapt_import_tent)
attempt("from_adapt_import_gradient_method", from_adapt_import_gradient_method)
attempt("import_gradient_module", import_gradient_module)
attempt("from_gradient_import_tent", from_gradient_import_tent)
attempt("attribute_tent", lambda: adapt.Tent)

attempt("user_stage1", lambda: user_module("user_stage1", STAGE1_SRC))
attempt("user_stage2", lambda: user_module("user_stage2", STAGE2_SRC))

# Stage 0, end to end, in the same process that refused stage 1.
model = bn_model()
base_mean = model[1].running_mean.tolist()
w = adapt.wrap(model, method=adapt.BatchNormStats())
w.online()
for s in range(3):
    w(batch(s))
out["stage0_moved"] = max(abs(a - b) for a, b in
                          zip(model[1].running_mean.tolist(), base_mean))
out["stage0_count"] = int(model[1].num_batches_tracked.item())
w.revert()
out["stage0_reverted"] = model[1].running_mean.tolist() == base_mean
out["stage0_type"] = type(w.adapted).__name__

# A stage declared by attribute: the type says stage 0, the attribute says 1.
m = adapt.BatchNormStats()
m.stage = adapt.STAGE_NARROW_BACKWARD
attempt("attribute_stage1_wrap", lambda: adapt.wrap(bn_model(), method=m))

out["gradient_loaded_at_end"] = "torchnative.adapt.gradient" in sys.modules
json.dump(out, sys.stdout)
'''

_BACKWARD_SCRIPT = _PRELUDE + r'''
from torchnative import adapt
out["BACKWARD"] = adapt.BACKWARD
out["all"] = sorted(adapt.__all__)
out["gradient_loaded_by_adapt"] = "torchnative.adapt.gradient" in sys.modules
from torchnative.adapt import Tent, GradientMethod, StatisticsMethod, BatchNormStats, Method
import torchnative.adapt.gradient as gradient
out["tent_is_gradient_tent"] = Tent is gradient.Tent and adapt.Tent is gradient.Tent
out["tent_module"] = Tent.__module__
out["gradient_method_module"] = GradientMethod.__module__
out["statistics_method_module"] = StatisticsMethod.__module__
out["tent_is_gradient"] = issubclass(Tent, GradientMethod)
out["tent_is_statistics"] = issubclass(Tent, StatisticsMethod)
out["bns_is_statistics"] = issubclass(BatchNormStats, StatisticsMethod)
out["bns_is_gradient"] = issubclass(BatchNormStats, GradientMethod)
out["bases_disjoint"] = (not issubclass(GradientMethod, StatisticsMethod)
                         and not issubclass(StatisticsMethod, GradientMethod))
out["both_methods"] = issubclass(GradientMethod, Method) and issubclass(StatisticsMethod, Method)
out["stages"] = [StatisticsMethod.stage, GradientMethod.stage, Tent.stage, BatchNormStats.stage]

# A user's stage-1 method as a GradientMethod subclass is accepted and wraps.
attempt("user_gradient_subclass", lambda: user_module(
    "user_grad", STAGE1_SRC.replace("adapt.Method", "adapt.GradientMethod")))
mod = sys.modules.get("user_grad")
if mod is not None:
    attempt("user_gradient_wrap", lambda: adapt.wrap(bn_model(), method=mod.MyEntropy()))

# A bare `Method` declaring stage 1 by attribute: definable, not wrappable.
attempt("user_stage1_define", lambda: user_module("user_stage1", STAGE1_SRC))
bare = sys.modules.get("user_stage1")
if bare is not None:
    attempt("bare_stage1_wrap", lambda: adapt.wrap(bn_model(), method=bare.MyEntropy()))
# And the other direction: a Tent told it is stage 0 is still a gradient type.
t = Tent()
t.stage = adapt.STAGE_FORWARD_ONLY
attempt("tent_as_stage0_wrap", lambda: adapt.wrap(bn_model(), method=t))
attempt("tent_wrap", lambda: adapt.wrap(bn_model(), method=Tent()))
# A duck: declares stage 0 and has a working `select_modules`, but is not the
# stage-0 type. Before S6.5 `hasattr(method, "select_modules")` admitted it.
class DuckStage0(Method):
    stage = adapt.STAGE_FORWARD_ONLY
    def select_modules(self, model): return ["1"]
attempt("duck_stage0_wrap", lambda: adapt.wrap(bn_model(), method=DuckStage0()))
json.dump(out, sys.stdout)
'''

_STAGE0_ONLY_SCRIPT = _PRELUDE + r'''
from torchnative import adapt
model = bn_model()
w = adapt.wrap(model, method=adapt.BatchNormStats()).online()
w(batch(0))
out["gradient_loaded"] = "torchnative.adapt.gradient" in sys.modules
json.dump(out, sys.stdout)
'''

_BAD_VALUE_SCRIPT = r'''
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"
out = {}
try:
    import torchnative.adapt  # noqa: F401
except BaseException as e:
    out["import"] = "%s: %s" % (type(e).__name__, e)
else:
    out["import"] = "ACCEPTED"
json.dump(out, sys.stdout)
'''


def _run(script, backward):
    """`backward`: None removes the variable, a string sets it to that."""
    env = dict(os.environ)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["PYTHONPATH"] = _VENDOR_DIR
    if backward is None:
        env.pop(_ENV, None)
    else:
        env[_ENV] = backward
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "stagetype subprocess (%s=%r) exited %d\n--- stdout ---\n%s\n"
            "--- stderr ---\n%s" % (_ENV, backward, proc.returncode,
                                    proc.stdout, proc.stderr))
    return json.loads(proc.stdout)


_CACHE = {}


def _cached(key, script, backward):
    if key not in _CACHE:
        _CACHE[key] = _run(script, backward)
    return _CACHE[key]


def _free():
    return _cached("free", _BACKWARD_FREE_SCRIPT, "off")


def _with():
    return _cached("with", _BACKWARD_SCRIPT, None)


_MODULE_REFUSAL = "ImportError: torchnative.adapt.gradient: this module holds the stage-1"


def _skip():
    if _available():
        return False
    print("SKIP stagetype: no vendored tree or no shim at %s" % _VENDOR_SHIM)
    return True


# -- the tests -------------------------------------------------------------


def test_stage_0_and_stage_1_are_distinct_types():
    """The distinction is a type a test can inspect, not a convention.

    `StatisticsMethod` (stage 0) lives in `torchnative.adapt`; `GradientMethod`
    (stage 1) lives in `torchnative.adapt.gradient`; neither is a subclass of
    the other; the shipped method of each stage is a subclass of its own base
    and not of the other's. `adapt.Tent` still resolves -- it is the stage-1
    module's `Tent`, reached lazily -- so existing callers are unchanged.
    """
    if _skip():
        return
    r = _with()
    assert r["BACKWARD"] is True, r["BACKWARD"]
    assert r["tent_is_gradient_tent"] is True, r
    assert r["tent_module"] == "torchnative.adapt.gradient", r["tent_module"]
    assert r["gradient_method_module"] == "torchnative.adapt.gradient", r
    assert r["statistics_method_module"] == "torchnative.adapt", r
    assert r["tent_is_gradient"] is True and r["tent_is_statistics"] is False, r
    assert r["bns_is_statistics"] is True and r["bns_is_gradient"] is False, r
    assert r["bases_disjoint"] is True and r["both_methods"] is True, r
    assert r["stages"] == [0, 1, 1, 0], r["stages"]
    assert {"GradientMethod", "Tent"} <= set(r["all"]), r["all"]
    print("ok   stagetype: StatisticsMethod and GradientMethod are disjoint types; "
          "Tent is a GradientMethod, BatchNormStats a StatisticsMethod")


def test_a_backward_free_build_refuses_the_stage_1_module_at_import_by_name():
    """Every spelling of importing a stage-1 method fails *at the import*.

    The refusal is asserted on its text -- it must name the module, say that
    stage 1 needs a backward, and name the configuration that says there is
    none -- and the very same imports are asserted to *succeed* with the
    variable removed (`test_stage_0_and_stage_1_are_distinct_types` runs them).
    Without that control this test would stay green over a stage-1 module that
    failed to import for any unrelated reason.
    """
    if _skip():
        return
    r, ctrl = _free(), _with()
    assert r["BACKWARD"] is False, r["BACKWARD"]
    # The control: the same names import when the build has a backward.
    assert ctrl["tent_is_gradient_tent"] is True, ctrl
    for key in ("from_adapt_import_tent", "from_adapt_import_gradient_method",
                "import_gradient_module", "from_gradient_import_tent",
                "attribute_tent"):
        got = r[key]
        assert got.startswith(_MODULE_REFUSAL), (key, got)
        assert "needs a backward" in got, (key, got)
        assert "TORCHNATIVE_BACKWARD='off'" in got, (key, got)
        assert "BatchNormStats" in got, (key, got)  # names what *is* available
    # And the stage-1 names are not advertised by the stage-0 package.
    assert "Tent" not in r["all"] and "GradientMethod" not in r["all"], r["all"]
    assert {"BatchNormStats", "StatisticsMethod", "wrap"} <= set(r["all"]), r["all"]
    print("ok   stagetype: a backward-free build refuses torchnative.adapt.gradient "
          "at import, five spellings, by name and reason")


def test_a_stage_1_method_a_user_defines_is_refused_when_its_class_is_created():
    """A gradient method need not come from this library to be refused.

    A user module that subclasses `adapt.Method` and declares stage 1 (or 2) is
    refused when its class statement runs, which is when that module is
    imported -- not at `wrap`, and not at the first step. The refusal names the
    class by module and qualified name, so a deployment that pulled it in
    transitively is told which file did.
    """
    if _skip():
        return
    r = _free()
    for key, stage in (("user_stage1", "stage 1"), ("user_stage2", "stage 2")):
        got = r[key]
        assert got.startswith("ImportError: torchnative.adapt: %s.MyEntropy declares %s"
                              % (key, stage)), (key, got)
        assert "needs a backward" in got, (key, got)
        assert "TORCHNATIVE_BACKWARD='off'" in got, (key, got)
    # The control: the same source defines cleanly with a backward.
    assert _with()["user_stage1_define"] == "ACCEPTED", _with()["user_stage1_define"]
    print("ok   stagetype: a user's stage-1/2 class is refused at class creation")


def test_stage_0_still_adapts_and_reverts_in_a_backward_free_build():
    """The point of stage 0: a device with no backward can still adapt.

    In the same process that refused stage 1, `BatchNormStats` moves the
    running statistics, counts its three batches, and `revert` puts them back
    exactly -- and the stage-1 module is never loaded, before or after. That
    last part is also asserted in the ordinary configuration: a stage-0 user
    does not execute stage-1 code in either build.
    """
    if _skip():
        return
    r = _free()
    assert r["gradient_loaded_by_adapt"] is False, r
    assert r["stage0_type"] == "BufferSnapshot", r["stage0_type"]
    assert r["stage0_moved"] > 1e-3, r["stage0_moved"]
    assert r["stage0_count"] == 3, r["stage0_count"]
    assert r["stage0_reverted"] is True, r
    assert r["gradient_loaded_at_end"] is False, r
    assert _with()["gradient_loaded_by_adapt"] is False, _with()
    plain = _cached("stage0_only", _STAGE0_ONLY_SCRIPT, None)
    assert plain["gradient_loaded"] is False, plain
    print("ok   stagetype: stage 0 adapts (moved %.3g over 3 batches) and reverts "
          "with no backward, never loading the stage-1 module" % r["stage0_moved"])


def test_a_stage_declared_by_attribute_and_not_by_type_is_refused_at_wrap():
    """The type decides the stage; an attribute that disagrees is refused.

    Before S6.5 `Adapted` read `method.stage` and nothing else, so the split
    was a convention: anything that said "stage 1" was run as stage 1. Now a
    stage-1 method has to *be* a `GradientMethod` and a stage-0 method a
    `StatisticsMethod`. This is the backstop behind the import-time refusal,
    for the one road it cannot see -- a stage set on an instance after its
    class was created -- and it holds in both configurations.
    """
    if _skip():
        return
    w, f = _with(), _free()
    assert w["user_gradient_subclass"] == "ACCEPTED", w["user_gradient_subclass"]
    assert w["user_gradient_wrap"] == "ACCEPTED", w["user_gradient_wrap"]
    assert w["tent_wrap"] == "ACCEPTED", w["tent_wrap"]
    got = w["bare_stage1_wrap"]
    assert got.startswith("TypeError: torchnative.adapt: 'MyEntropy' declares stage 1"), got
    assert "GradientMethod" in got, got
    got = w["duck_stage0_wrap"]
    assert got.startswith("NotImplementedError: torchnative.adapt: 'DuckStage0' "
                          "declares stage 0"), got
    assert "not a StatisticsMethod" in got, got
    got = w["tent_as_stage0_wrap"]
    assert got.startswith("NotImplementedError:") and "stage 0" in got, got
    assert "StatisticsMethod" in got, got
    got = f["attribute_stage1_wrap"]
    assert got.startswith("TypeError: torchnative.adapt: 'BatchNormStats' declares stage 1"), got
    assert "TORCHNATIVE_BACKWARD='off'" in got, got
    print("ok   stagetype: a stage set by attribute against its type is refused at wrap")


def test_an_unrecognised_backward_setting_is_refused_by_name():
    """`TORCHNATIVE_BACKWARD=maybe` is neither, and is not guessed at.

    Reading an unknown value as "on" would put stage 1 into a deployment that
    meant to exclude it; reading it as "off" would refuse a working build for
    a typo. Either is a silent choice, so `import torchnative.adapt` refuses,
    naming the variable, the value, and the spellings it accepts.
    """
    if _skip():
        return
    r = _cached("bad", _BAD_VALUE_SCRIPT, "maybe")
    got = r["import"]
    assert got.startswith("ValueError: torchnative.adapt: TORCHNATIVE_BACKWARD='maybe'"), got
    assert "'off'" in got and "'on'" in got, got
    for spelling in ("0", "OFF", " false "):
        assert _run(_BAD_VALUE_SCRIPT, spelling)["import"] == "ACCEPTED", spelling
    print("ok   stagetype: an unrecognised TORCHNATIVE_BACKWARD value refuses by name")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
