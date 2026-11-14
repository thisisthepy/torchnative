"""A frame Dynamo produces no code for, on a compiled device, refuses by name (#303, SPEC S1.4).

`model.to(torchnative.device.npu)` attaches the hook-free compile engine
(`torchnative.compile.attach_to_module`). `test_compile_refuse.py` (#255) covers
the path where capture raises. This suite covers the path with no exception:
`convert_frame` returns a `ConvertFrameReturn` whose `guarded_code` is None.
Until #303 `Engine.run` answered that by running the original function eagerly
and counting `stats["skip"]`, so an attached forward past Dynamo's recompile
limit (`convert_frame.py`: `exceeds_recompile_limit` sets the frame to
`FrameAction.RUN_ONLY`) ran on the host without a word.

Now:

* the engine records why a frame produced no code, in `Engine.skips`
  (`SkipRecord`: Dynamo's `skip_reason`, the frame action, the suppressed
  cause, and for the recompile limit its config name and value);
* for an attached module's root forward, no code raises `CompileRefused`
  naming the forward, the device resolution and that reason; with
  `torch._dynamo.config.suppress_errors = True` it warns
  (`CompileFallbackWarning`) and runs eagerly, and `stats["fail"]` counts it;
* a plain `Engine.wrap` / `torchnative.compile(fn)` keeps upstream's eager run,
  with the reason recorded;
* a nested frame (a resume function after a graph break) that Dynamo skips is
  recorded and not refused, and a root forward with no tensor ops at all
  (Dynamo's "No ops traced") is recorded and not refused: there is nothing to
  run on the device.

The recompile limit is reached the way `test_compile_nopep523.py` reaches it:
a Python float argument with `specialize_float = True`, so each value is its
own graph.

Nullification this file is meant to catch: restoring the silent eager return
in `Engine.run` (`stats["skip"] += 1; return fn(*args, **kwargs)` with no
record and no root check) turns
`test_an_attached_forward_past_the_recompile_limit_refuses_naming_the_limit`
red (no `CompileRefused`), and `test_suppress_errors_warns_and_agrees_with_eager`
and `test_a_plain_compiled_function_past_the_limit_runs_and_records_why` red
(no warning, no record).
"""

import os
import pathlib
import sys
import warnings

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip  # noqa: E402


def _imports():
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        return None
    import torch
    import torch.nn as nn

    assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"
    import torchnative.compile as C

    C._ensure_shim_ready()  # before the config below, which it would otherwise reset
    return torch, nn, C


class _Resolution:
    """Stands in for `NpuResolution`: only `backend`, `unit` and repr are read."""

    backend = "openvino"
    unit = "Intel NPU"

    def __repr__(self):
        return "<NpuResolution windows: openvino -> Intel NPU>"


class _Limit:
    """`recompile_limit = n`, floats specialized, restored on exit."""

    def __init__(self, n, suppress=False):
        self.n = n
        self.suppress = suppress

    def __enter__(self):
        import torch._dynamo.config as dc

        self.dc = dc
        self.saved = (
            dc.recompile_limit,
            dc.accumulated_recompile_limit,
            dc.specialize_float,
            dc.suppress_errors,
        )
        dc.recompile_limit = self.n
        dc.accumulated_recompile_limit = 64
        dc.specialize_float = True
        dc.suppress_errors = self.suppress
        return self

    def __exit__(self, *exc):
        dc = self.dc
        (
            dc.recompile_limit,
            dc.accumulated_recompile_limit,
            dc.specialize_float,
            dc.suppress_errors,
        ) = self.saved
        return False


def _close(a, b):
    """A captured graph against the eager forward: the same ops, so within 1e-5."""
    return float((a.detach().float() - b.detach().float()).abs().max().item()) <= 1e-5


class _Counter:
    def __init__(self):
        self.graphs = 0

    def __call__(self, gm, example_inputs):
        self.graphs += 1
        return gm.forward


def _attached(torch, nn, C, resolution=None):
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.lin = nn.Linear(4, 4)

        def forward(self, x, k):
            return self.lin(x) * k + 1.0

    model = Scaled()
    c = _Counter()
    eng = C.Engine(c)
    C.attach_to_module(model, backend=eng, resolution=resolution)
    return model, eng, c


def test_an_attached_forward_past_the_recompile_limit_refuses_naming_the_limit():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, C = mods
    model, eng, c = _attached(torch, nn, C, resolution=_Resolution())
    x = torch.ones(2, 4)
    with _Limit(2), torch.no_grad():
        model(x, 1.0)
        model(x, 2.0)
        assert c.graphs == 2, f"two float values must be two graphs, got {c.graphs}"
        try:
            model(x, 3.0)
        except C.CompileRefused as exc:
            text = str(exc)
            assert "Scaled.forward" in text, text
            assert "Intel NPU" in text and "openvino" in text, text
            assert "recompile_limit" in text and "(2)" in text, text
            assert "suppress_errors" in text, text
        else:
            raise AssertionError(
                "an attached forward past the recompile limit ran eagerly without a word"
            )
        assert eng.stats["fail"] == 1, eng.stats
        assert c.graphs == 2, f"nothing may compile past the limit, got {c.graphs}"
        rec = [r for r in eng.skips if r.function.endswith("Scaled.forward")]
        assert len(rec) == 1, [str(r) for r in eng.skips]
        assert rec[0].limit_type == "recompile_limit" and rec[0].limit == 2, rec[0]
        assert rec[0].action == "RUN_ONLY", rec[0]

        # Upstream's RUN_ONLY: the frame is not translated again, a cached graph
        # still serves its own guards, and a miss refuses again (not a silent run).
        t = eng.stats["translate"]
        model(x, 1.0)
        try:
            model(x, 4.0)
        except C.CompileRefused as exc:
            assert "recompile_limit" in str(exc), exc
        else:
            raise AssertionError("the second miss past the limit ran eagerly without a word")
        assert eng.stats["translate"] == t, (t, eng.stats)
        assert eng.stats["fail"] == 2, eng.stats
    print("ok   compile_skip: an attached forward past the recompile limit refuses, naming the limit")


def test_suppress_errors_warns_and_agrees_with_eager():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, C = mods
    model, eng, c = _attached(torch, nn, C, resolution=_Resolution())
    x = torch.ones(2, 4)
    with _Limit(2, suppress=True), torch.no_grad():
        model(x, 1.0)
        model(x, 2.0)
        expected = type(model).forward(model, x, 3.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = model(x, 3.0)
    assert torch.equal(got, expected), "the eager result under suppress_errors is not the forward's"
    named = [w for w in caught if issubclass(w.category, C.CompileFallbackWarning)]
    assert len(named) == 1, [str(w.message) for w in caught]
    text = str(named[0].message)
    assert "Scaled.forward" in text and "Intel NPU" in text, text
    assert "recompile_limit" in text and "(2)" in text, text
    assert eng.stats["fail"] == 1, eng.stats
    print("ok   compile_skip: suppress_errors warns naming the limit and agrees with eager")


def test_a_plain_compiled_function_past_the_limit_runs_and_records_why():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, C = mods
    c = _Counter()
    eng = C.Engine(c)

    def f(x, k):
        return x * k + 1.0

    run = eng.wrap(f)
    x = torch.ones(4)
    with _Limit(2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for k in (1.0, 2.0, 3.0, 4.0, 1.0):
            got = run(x, k)
            assert _close(got, f(x, k)), (k, got)
    assert c.graphs == 2, c.graphs
    assert eng.stats["fail"] == 0, eng.stats
    assert eng.stats["skip"] >= 2, eng.stats
    assert not [w for w in caught if issubclass(w.category, C.CompileFallbackWarning)]
    rec = [r for r in eng.skips if r.function.endswith(f.__qualname__)]
    assert len(rec) == 1, [str(r) for r in eng.skips]
    assert rec[0].limit_type == "recompile_limit" and rec[0].limit == 2, rec[0]
    assert "recompile_limit" in str(rec[0]) and "(2)" in str(rec[0]), str(rec[0])
    print("ok   compile_skip: a plain compiled function past the limit runs eagerly and records why")


def test_below_the_limit_nothing_is_refused():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, C = mods
    model, eng, c = _attached(torch, nn, C, resolution=_Resolution())
    x = torch.ones(2, 4)
    with _Limit(3), torch.no_grad():
        for k in (1.0, 2.0, 1.0, 2.0, 2.0):
            got = model(x, k)
            assert _close(got, type(model).forward(model, x, k)), k
    assert c.graphs == 2, c.graphs
    assert eng.stats["fail"] == 0 and eng.stats["hit"] == 3, eng.stats
    assert eng.skips == [], [str(r) for r in eng.skips]
    print("ok   compile_skip: below the recompile limit an attached forward compiles and nothing refuses")


def test_a_nested_frame_dynamo_skips_is_recorded_not_refused():
    """A resume function with no ops after a graph break is Dynamo's to skip.

    The root forward is captured (a graph reaches the backend, and the graph
    break gives it a resume function the engine wraps). The resume function
    only returns, so Dynamo skips it ("No ops traced"). That is a nested frame:
    the engine records it and the root check, which only the root frame's own
    translation can trigger, does not fire.
    """
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, C = mods

    class Breaks(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            y = self.lin(x) * 2
            torch._dynamo.graph_break()
            return y

    model = Breaks()
    x = torch.ones(2, 4)
    with torch.no_grad():
        expected = type(model).forward(model, x)
        c = _Counter()
        eng = C.Engine(c)
        C.attach_to_module(model, backend=eng, resolution=_Resolution())
        got = model(x)  # a CompileRefused here is the confusion this test exists for
    assert _close(got, expected), (got, expected)
    assert eng.stats["fail"] == 0, eng.stats
    assert c.graphs >= 1, "the root forward captured nothing"
    assert eng.stats["resume_wrapped"] >= 1, eng.stats
    assert not any(r.function.endswith("Breaks.forward") for r in eng.skips), [str(r) for r in eng.skips]
    assert any("resume_in" in r.function for r in eng.skips), [str(r) for r in eng.skips]
    print("ok   compile_skip: a nested resume frame Dynamo skips is recorded, the root is not refused")


def test_a_root_forward_with_no_tensor_ops_is_recorded_not_refused():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, C = mods

    class Identity(nn.Module):
        def forward(self, x):
            return x

    model = Identity()
    eng = C.Engine(_Counter())
    C.attach_to_module(model, backend=eng, resolution=_Resolution())
    x = torch.ones(2, 4)
    assert model(x) is x
    assert model(x) is x  # the second call takes the skipped-code arm
    assert eng.stats["fail"] == 0, eng.stats
    rec = [r for r in eng.skips if r.function.endswith("Identity.forward")]
    assert len(rec) == 1 and "No ops traced" in rec[0].reason, [str(r) for r in eng.skips]
    print("ok   compile_skip: a root forward with no tensor ops is recorded and runs as itself")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        _skip._state["current"] = name
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            if name in _skip._state["skipped"]:
                print(f"SKIP test_compile_skip: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
