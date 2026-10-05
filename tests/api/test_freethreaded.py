"""`torch._C` on a free-threaded interpreter (CPython 3.15t, issue #51).

The abi3 wheel cannot load into 3.15t, so the extension also builds without
abi3 (`cargo build --no-default-features` against that interpreter). This suite
loads that build in a 3.15t subprocess and checks two things:

  * **It is the shim and it computes.** `_aten_implemented` is present, and
    `torch.ones(2) + 1` gives upstream's answer.
  * **It keeps the GIL.** pyo3 0.29 declares a module GIL-free unless told
    otherwise. The first 3.15t build took it at its word, and
    `sys._is_gil_enabled()` read False, before any of the shim's globals had
    been audited for free threading. `lib.rs` now asks for the GIL
    (`gil_used = true`), so the interpreter re-enables it on import and warns.
    When the audit and its thread tests land, this test flips with them.

Inputs, both required or every test skips by name:
  TORCHNATIVE_FT_PYTHON  a 3.15t interpreter (with numpy, sympy, typing_extensions)
  TORCHNATIVE_FT_STAGE   a vendored tree whose `torch/` holds the 3.15t `_C`

What this cannot see: thread safety. One thread imports and adds; nothing here
races the globals.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_support"))
import _skip  # noqa: E402

_PROBE = r"""
import json, sys, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import torch
print(json.dumps({
    "ft_build": bool(__import__("sysconfig").get_config_var("Py_GIL_DISABLED")),
    "shim": hasattr(torch._C, "_aten_implemented"),
    "gil": sys._is_gil_enabled(),
    "warned": [str(w.message) for w in caught if "GIL" in str(w.message)],
    "value": (torch.ones(2) + 1).tolist(),
}))
"""


def _probe():
    py = os.environ.get("TORCHNATIVE_FT_PYTHON")
    stage = os.environ.get("TORCHNATIVE_FT_STAGE")
    if not py or not stage:
        raise _skip.Skip("TORCHNATIVE_FT_PYTHON / TORCHNATIVE_FT_STAGE not set (issue #51)")
    env = dict(os.environ, PYTHONPATH=stage, TORCH_USE_RTLD_GLOBAL="1")
    env.pop("PYTHON_GIL", None)
    proc = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, env=env, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"3.15t probe exited {proc.returncode}\n{proc.stderr[-2000:]}")
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_ft_interpreter_loads_the_shim_and_computes():
    r = _probe()
    assert r["ft_build"], "TORCHNATIVE_FT_PYTHON is not a free-threaded build"
    assert r["shim"], "imported upstream torch, not the shim"
    assert r["value"] == [2.0, 2.0], r["value"]


def test_module_keeps_the_gil_until_the_audit_lands():
    r = _probe()
    assert r["gil"] is True, "torch._C declared itself GIL-free before the #51 audit"
    assert any("torch._C" in w for w in r["warned"]), r["warned"]


# The thread hammer (#51, #76). `PYTHON_GIL=0` overrides the module's
# `gil_used = true`, so the threads really run at once: each case below failed,
# or was unsound, while the state it touches was process-global.
_HAMMER = r"""
import json, sys, threading
import torch
assert hasattr(torch._C, "_aten_implemented")
assert not sys._is_gil_enabled(), "PYTHON_GIL=0 did not take"
N, ROUNDS = 8, 40
bar = threading.Barrier(N)
errors, result = [], {}

def guard(fn):
    def run(i):
        try:
            fn(i)
        except BaseException as e:
            errors.append(f"{fn.__name__}[{i}]: {type(e).__name__}: {e}")
            bar.abort()
    return run

def ref_draws(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(64, generator=g).tolist() for _ in range(ROUNDS)]

@guard
def rng(i):
    want = ref_draws(1000 + i)
    bar.wait()
    g = torch.Generator().manual_seed(1000 + i)
    for r in range(ROUNDS):
        churn = torch.Generator().manual_seed(r)   # freed every round
        torch.rand(8, generator=churn)
        del churn
        got = torch.randn(64, generator=g).tolist()
        assert got == want[r], f"seed {1000 + i} round {r} drew another stream"

@guard
def grad(i):
    bar.wait()
    for _ in range(ROUNDS):
        if i % 2:
            with torch.no_grad():
                assert not torch.is_grad_enabled()
                y = torch.ones(4, requires_grad=True) * 2
                assert y.grad_fn is None
        else:
            assert torch.is_grad_enabled(), "another thread's no_grad applied here"
            y = torch.ones(4, requires_grad=True) * 2
            assert y.grad_fn is not None, "grad_fn dropped: another thread's no_grad"

@guard
def mode(i):
    bar.wait()
    for _ in range(ROUNDS):
        if i % 2:
            with torch.device("meta"):
                assert torch.empty(2).device.type == "meta"
        else:
            d = torch.empty(2).device.type
            assert d == "cpu", f"another thread's device mode applied here: {d}"

for name, fn in [("rng", rng), ("grad", grad), ("mode", mode)]:
    bar.reset()
    ts = [threading.Thread(target=fn, args=(i,)) for i in range(N)]
    [t.start() for t in ts]
    [t.join(120) for t in ts]
    result[name] = [e for e in errors if e.startswith(name + "[")]
print(json.dumps(result))
"""


def _hammer():
    py = os.environ.get("TORCHNATIVE_FT_PYTHON")
    stage = os.environ.get("TORCHNATIVE_FT_STAGE")
    if not py or not stage:
        raise _skip.Skip("TORCHNATIVE_FT_PYTHON / TORCHNATIVE_FT_STAGE not set (issue #51)")
    env = dict(os.environ, PYTHONPATH=stage, TORCH_USE_RTLD_GLOBAL="1", PYTHON_GIL="0")
    proc = subprocess.run([py, "-c", _HAMMER], capture_output=True, text=True, env=env, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"hammer exited {proc.returncode}\n{proc.stderr[-3000:]}")
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


_HAMMER_RESULT = {}


def _case(name):
    if not _HAMMER_RESULT:
        _HAMMER_RESULT.update(_hammer())
    errs = _HAMMER_RESULT[name]
    assert not errs, "\n".join(errs[:5])


def test_hammer_rng_streams_stay_their_own_under_churn():
    _case("rng")


def test_hammer_grad_mode_is_per_thread_without_the_gil():
    _case("grad")


def test_hammer_device_mode_is_per_thread_without_the_gil():
    _case("mode")


if __name__ == "__main__":
    raise SystemExit(_skip.run_tests(
        [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)],
        "api/test_freethreaded",
    ))
