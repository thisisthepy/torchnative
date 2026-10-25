"""`torch._C` on a free-threaded interpreter (CPython 3.15t, issue #51).

The abi3 wheel cannot load into 3.15t, so the extension also builds without
abi3 (`cargo build --no-default-features` against that interpreter). This suite
loads that build in a 3.15t subprocess and checks two things:

  * **It is the shim and it computes.** `_aten_implemented` is present, and
    `torch.ones(2) + 1` gives upstream's answer.
  * **It leaves the GIL disabled.** With the stage-1 and stage-2 free-threading
    audits complete (#51, #76, #114), `lib.rs` declares `gil_used = false`
    (`Py_mod_gil = Py_MOD_GIL_NOT_USED`), so importing `torch` on a 3.15t
    interpreter without setting `PYTHON_GIL` leaves `sys._is_gil_enabled()`
    as `False` and emits no GIL warning.

Inputs, both required or every test skips by name:
  TORCHNATIVE_FT_PYTHON  a 3.15t interpreter (with numpy, sympy, typing_extensions)
  TORCHNATIVE_FT_STAGE   a vendored tree whose `torch/` holds the 3.15t `_C`
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


def test_module_leaves_the_gil_disabled_on_import():
    r = _probe()
    assert r["gil"] is False, f"importing torch re-enabled the GIL: {r}"
    assert not any("torch._C" in w for w in r["warned"]), r["warned"]


# The thread hammer (#51, #76, #114). Without `PYTHON_GIL` set in the environment,
# `torch._C`'s `Py_mod_gil = Py_MOD_GIL_NOT_USED` declaration keeps the GIL disabled
# so the worker threads run concurrently without any environment override.
_HAMMER = r"""
import json, sys, threading
import torch
assert hasattr(torch._C, "_aten_implemented")
assert not sys._is_gil_enabled(), "GIL was enabled during free-threaded hammer"
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

# #114 R23 (audit H6): one shared tensor; half the threads compute on it, half
# toggle its setters. With `&mut self` setters every compute that met a setter
# raised `Already mutably borrowed`; upstream raises nothing here.
SHARED = torch.full((64,), 2.0)

@guard
def setters(i):
    bar.wait()
    for r in range(ROUNDS * 25):
        if i % 2:
            SHARED.requires_grad = bool(r % 2)
            SHARED._backward_hooks = None if r % 3 else {}
        else:
            got = (SHARED + 1).sum().item()
            assert got == 192.0, got

# #114 Group 1: bootstrap thread-local state (inference_mode, autocast, vmap)
# and R20 STRIDED_BARRIER vs in-place write.
BASE_R20 = torch.zeros(16)

@guard
def tls_modes(i):
    bar.wait()
    for r in range(ROUNDS):
        if i % 2:
            with torch.inference_mode():
                assert torch.is_inference_mode_enabled(), "inference_mode not enabled inside block"
                torch._C.set_autocast_dtype("cpu", torch.float16)
                lvl = torch._C.autocast_increment_nesting()
                assert lvl == 1, f"autocast nesting leaked across threads: {lvl}"
                assert torch._C.get_autocast_dtype("cpu") == torch.float16
                assert torch._C.autocast_decrement_nesting() == 0
                torch._C.set_autocast_dtype("cpu", torch.bfloat16)
        else:
            assert not torch.is_inference_mode_enabled(), "another thread's inference_mode applied here"
            assert torch._C.get_autocast_dtype("cpu") == torch.bfloat16, "another thread's autocast dtype leaked here"
            lvl = torch._C.autocast_increment_nesting()
            assert lvl == 1, f"another thread's autocast nesting leaked here: {lvl}"
            assert torch._C.autocast_decrement_nesting() == 0
        idx = torch.arange(8)
        out = torch.vmap(lambda x: x + i)(idx).tolist()
        assert out == [i + k for k in range(8)], f"vmap corrupted across threads: {out}"
        if i % 2:
            v = BASE_R20.as_strided((4,), (2,))
            del v
        else:
            try:
                BASE_R20.fill_(1.0)
            except RuntimeError as e:
                assert "as_strided" in str(e), str(e)

for name, fn in [("rng", rng), ("grad", grad), ("mode", mode), ("setters", setters), ("tls_modes", tls_modes)]:
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
    env = dict(os.environ, PYTHONPATH=stage, TORCH_USE_RTLD_GLOBAL="1")
    env.pop("PYTHON_GIL", None)
    proc = subprocess.run([py, "-c", _HAMMER], capture_output=True, text=True, env=env, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"hammer exited {proc.returncode}\n{proc.stderr[-3000:]}")
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


_HAMMER_RESULT = {}


def _case(name):
    if not _HAMMER_RESULT:
        try:
            _HAMMER_RESULT.update(_hammer())
        except _skip.Skip as e:
            # Kept a skip: re-raising it as RuntimeError turned a runner
            # without a 3.15t interpreter into five FAILs.
            _HAMMER_RESULT["__skip__"] = e
        except Exception as e:
            _HAMMER_RESULT["__err__"] = str(e)
    if "__skip__" in _HAMMER_RESULT:
        raise _HAMMER_RESULT["__skip__"]
    if "__err__" in _HAMMER_RESULT:
        raise RuntimeError(_HAMMER_RESULT["__err__"])
    errs = _HAMMER_RESULT[name]
    assert not errs, "\n".join(errs[:5])


def test_hammer_rng_streams_stay_their_own_under_churn():
    _case("rng")


def test_hammer_grad_mode_is_per_thread_without_the_gil():
    _case("grad")


def test_hammer_device_mode_is_per_thread_without_the_gil():
    _case("mode")


def test_hammer_shared_tensor_setters_do_not_block_its_readers():
    _case("setters")


def test_hammer_tls_modes_and_vmap_stay_per_thread():
    _case("tls_modes")



if __name__ == "__main__":
    raise SystemExit(_skip.run_tests(
        [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)],
        "api/test_freethreaded",
    ))
