"""A backend name that torchnative cannot honour refuses by name (#274, SPEC S1.4).

Until #274, `torchnative.compile.Engine` mapped the backend names `"eager"`,
`"inductor"` and `"npu"` to the same callable, `lambda gm, ex: gm.forward`. So
`torchnative.compile(model, backend="npu")` ran the captured graph eagerly on
the host while the name said NPU, and `backend="inductor"` ran it eagerly while
the name said Inductor. Every answer was right, and nothing said where the
graph had run: a silent fallback.

Now:

* `"npu"` refuses by name. There is no NPU graph compiler. The NPU is reached
  through `model.to(torchnative.device.npu)`, which lowers the device leaves.
* `"inductor"` refuses by name. The shim has no Inductor.
* `None` and `"eager"` run the graph eagerly. That is upstream's own meaning of
  `backend="eager"`, so the name is true.
* Any other string resolves through upstream's
  `torch._dynamo.backends.registry.lookup_backend`, as `torch.compile` does,
  so an unknown name raises upstream's `InvalidBackend`.
* `.to(torchnative.device.npu)` attaches with `backend="eager"` explicitly and
  records on `model.torchnative_compile["graph_runs_on"]` that the graph
  between the lowered leaves runs on the host.

Nullification this file is meant to catch: restoring
`backend in ("eager", "inductor", "npu")` turns
`test_npu_backend_refuses_by_name` and `test_inductor_backend_refuses_by_name`
red; passing an unknown string through unresolved turns
`test_an_unknown_backend_name_raises_upstreams_invalid_backend` red.
"""

import os
import pathlib
import sys

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

    assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"
    import torchnative.compile as C

    return torch, C


def _refuses(C, name):
    try:
        C.Engine(name)
    except NotImplementedError as exc:
        return str(exc)
    raise AssertionError(f"Engine({name!r}) accepted a backend it cannot honour")


def test_npu_backend_refuses_by_name():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    _torch, C = mods
    text = _refuses(C, "npu")
    assert "'npu'" in text, text
    assert "torchnative.device.npu" in text, text
    assert "eagerly" in text, text
    print("ok   compile_backend: backend='npu' refuses by name and points at .to(torchnative.device.npu)")


def test_inductor_backend_refuses_by_name():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    _torch, C = mods
    text = _refuses(C, "inductor")
    assert "'inductor'" in text and "Inductor" in text, text
    assert "eager" in text, text
    print("ok   compile_backend: backend='inductor' refuses by name")


def test_eager_and_none_still_run_and_agree():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, C = mods

    def f(x):
        return x * 2 + 1

    x = torch.arange(6.0).reshape(2, 3)
    for name in (None, "eager"):
        eng = C.Engine(name)
        got = eng.wrap(f)(x)
        assert torch.equal(got, f(x)), (name, got)
        assert eng.stats["fail"] == 0, (name, eng.stats)
    print("ok   compile_backend: None and 'eager' still run the graph and agree")


def test_an_unknown_backend_name_raises_upstreams_invalid_backend():
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    _torch, C = mods
    from torch._dynamo.exc import InvalidBackend

    try:
        C.Engine("no_such_backend_274")
    except InvalidBackend as exc:
        assert "no_such_backend_274" in str(exc), exc
    else:
        raise AssertionError("an unknown backend name was accepted")
    print("ok   compile_backend: an unknown name raises upstream's InvalidBackend")


def test_to_npu_records_that_the_graph_runs_on_the_host():
    """`attach_to_module`, as `.to(device.npu)` calls it, names where the graph runs."""
    mods = _imports()
    if mods is None:
        _skip.skip("   (skipped: no vendored torch tree)")
        return
    torch, C = mods
    import torch.nn as nn

    from torchnative.device import _module_to

    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x) * 2

    model = Small()
    _module_to._attach_compile(model, resolution=None)
    record = model.torchnative_compile
    assert record["backend"] == "eager", record
    assert "host" in record["graph_runs_on"], record
    print("ok   compile_backend: .to(npu)'s attach records that the graph between leaves runs on the host")


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
                print(f"SKIP test_compile_backend: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
