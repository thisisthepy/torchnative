"""A failed capture on a compiled device refuses by name (#255, SPEC S1.4).

`model.to(torchnative.device.npu)` lowers the module and then attaches the
hook-free compile engine (`torchnative.compile.attach_to_module`). Until #255
the attached forward caught every exception whose class name contained
`Dynamo`, `Unsupported` or `Internal`, counted it in `stats["fail"]` and reran
the original forward eagerly. The answer was right and nothing said the graph
had not been captured: a silent fallback, which S1.4 forbids.

Now, as upstream `torch.compile` does:

* by default the failure raises `torchnative.compile.CompileRefused`, naming
  the forward that failed, the compiled device's resolution when there is one,
  and the cause (chained);
* with `torch._dynamo.config.suppress_errors = True` (upstream's own opt-in)
  it warns, naming the same things, and runs eagerly; `stats["fail"]` counts it.

The failure is injected at `Engine.run`, because the property under test is what
`attach_to_module` does with a Dynamo-class failure, and a real one cannot be
produced on demand. Graph breaks are not exceptions (Dynamo resumes) and are
covered by `test_compile_nopep523.py`.

Nullification this file is meant to catch: restoring the old catch-all
(`stats["fail"] += 1; return orig_fwd_fn(...)` with no raise or warning) turns
`test_a_dynamo_failure_refuses_by_name` and
`test_suppress_errors_warns_naming_the_function` red.
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
    import torch._dynamo.exc as dexc
    import torchnative.compile as C

    return torch, nn, dexc, C


class _Resolution:
    """Stands in for `NpuResolution`: only `backend`, `unit` and repr are read."""

    backend = "openvino"
    unit = "Intel NPU"

    def __repr__(self):
        return "<NpuResolution windows: openvino -> Intel NPU>"


def _attached(torch, nn, C, raising, resolution=None):
    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x) * 2

    model = Small()
    eng = C.Engine()

    def boom(fn, args, kwargs):
        raise raising

    eng.run = boom
    C.attach_to_module(model, backend=eng, resolution=resolution)
    return model, eng


def test_a_dynamo_failure_refuses_by_name():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, dexc, C = mods
    cause = dexc.InternalTorchDynamoError("injected capture failure")
    model, eng = _attached(torch, nn, C, cause, resolution=_Resolution())
    try:
        model(torch.ones(2, 4))
    except C.CompileRefused as exc:
        text = str(exc)
        assert exc.__cause__ is cause, exc.__cause__
        assert "Small.forward" in text, text
        assert "Intel NPU" in text and "openvino" in text, text
        assert "InternalTorchDynamoError" in text and "injected capture failure" in text, text
        assert "suppress_errors" in text, text
    else:
        raise AssertionError("a Dynamo capture failure on a compiled device ran eagerly without a word")
    assert eng.stats["fail"] == 1, eng.stats
    print("ok   compile_refuse: a capture failure on a compiled device refuses by name")


def test_suppress_errors_warns_naming_the_function():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, dexc, C = mods
    import torch._dynamo.config as cfg

    cause = dexc.Unsupported("injected unsupported construct") if hasattr(dexc, "Unsupported") else dexc.InternalTorchDynamoError("x")
    model, eng = _attached(torch, nn, C, cause, resolution=_Resolution())
    x = torch.ones(2, 4)
    expected = type(model).forward(model, x)
    saved = cfg.suppress_errors
    cfg.suppress_errors = True
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = model(x)
    finally:
        cfg.suppress_errors = saved
    assert torch.equal(got, expected), "the eager result under suppress_errors is not the forward's"
    named = [w for w in caught if issubclass(w.category, C.CompileFallbackWarning)]
    assert len(named) == 1, [str(w.message) for w in caught]
    text = str(named[0].message)
    assert "Small.forward" in text and "Intel NPU" in text and type(cause).__name__ in text, text
    assert eng.stats["fail"] == 1, eng.stats
    print("ok   compile_refuse: suppress_errors runs eagerly and says so, naming the forward")


def test_without_a_resolution_it_still_refuses():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, dexc, C = mods
    model, _eng = _attached(torch, nn, C, dexc.InternalTorchDynamoError("plain compile"))
    try:
        model(torch.ones(2, 4))
    except C.CompileRefused as exc:
        assert "Small.forward" in str(exc), exc
    else:
        raise AssertionError("torchnative.compile(module) swallowed a capture failure")
    print("ok   compile_refuse: torchnative.compile(module) refuses the same way")


def test_a_non_dynamo_error_propagates_unchanged():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, _dexc, C = mods
    cause = ValueError("a user error inside forward")
    model, eng = _attached(torch, nn, C, cause, resolution=_Resolution())
    try:
        model(torch.ones(2, 4))
    except ValueError as exc:
        assert exc is cause, exc
    else:
        raise AssertionError("a ValueError from forward was swallowed")
    assert eng.stats["fail"] == 0, eng.stats
    print("ok   compile_refuse: an error that is not a capture failure propagates as itself")


def test_a_lowered_leaf_with_host_side_work_compiles_without_refusal_and_runs_as_itself():
    """A device leaf is opaque to capture: Dynamo graph-breaks around it (#255).

    The lowered leaves (`_NPULinear`, `_NPUEmbedding`, the CoreML leaves) are
    the device's own execution: their forward marshals the activation to bytes
    on the host (`.tolist()`, `struct.pack`) and calls a vendor runtime. Before
    #255 Dynamo traced *into* `_NPUEmbedding.forward`, died on `.tolist()`
    (`no meta kernel for aten.unbind.int`), and the attached forward reran
    eagerly without a word; with the refusal in place it raised CompileRefused.

    `torchnative.compile._device_leaf_forward` is what the leaves use, so this
    module uses it too. Asserted: no refusal and no failure; the answer is the
    eager one; the leaf ran as itself, on real values (its host-side record is
    the bytes of the real ids); the rest of the forward was captured (a graph
    reached the backend); and the graph break Dynamo took names the device
    leaf, so it came from the marker and not from some other construct.

    Nullification this catches: dropping `_device_leaf_forward` from the leaf
    (Dynamo traces into it, and the break, if it comes at all, does not name
    the leaf) or making `_device_leaf_forward` return its argument unwrapped.
    """
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, nn, _dexc, C = mods
    import struct

    from torch._dynamo.utils import counters

    records = []

    class HostLeaf(nn.Module):
        def forward(self, x):
            ids = [int(v) for v in x.detach().reshape(-1).tolist()]
            raw = struct.pack(f"<{len(ids)}q", *ids)
            records.append(raw)
            vals = [float(v) * 0.5 for v in struct.unpack(f"<{len(ids)}q", raw)]
            return torch.tensor(vals, dtype=torch.float32).reshape(*x.shape)

    HostLeaf.forward = C._device_leaf_forward(HostLeaf.forward, unit="test device")

    class WithLeaf(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.leaf = HostLeaf()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            # Captured work on both sides of the leaf, so a graph reaches the
            # backend whichever side Dynamo compiles first.
            h = self.leaf(x + 0)
            return self.lin(h) * 2

    model = WithLeaf()
    x = torch.tensor([[1, 4, 7, 2], [3, 5, 8, 6]], dtype=torch.int64)
    expected = type(model).forward(model, x)
    del records[:]

    graphs = []

    def recording(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    eng = C.Engine(recording)
    counters.clear()
    C.attach_to_module(model, backend=eng, resolution=_Resolution())
    got = model(x)  # a CompileRefused here is the failure this test exists for

    assert eng.stats["fail"] == 0, eng.stats
    assert torch.equal(got, expected), (got, expected)
    want = struct.pack("<8q", 1, 4, 7, 2, 3, 5, 8, 6)
    assert records == [want], records
    assert graphs, "nothing around the leaf was captured"
    breaks = list(counters["graph_break"]) + list(counters["unimplemented"])
    assert any("torchnative device leaf" in str(k) and "test device" in str(k) for k in breaks), breaks
    print("ok   compile_refuse: a lowered leaf with host-side .tolist() compiles "
          "without refusal, graph-breaks by name and runs as itself")


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
                print(f"SKIP test_compile_refuse: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
