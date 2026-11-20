"""Issue #415 -- `torch.isnan`, `torch.isposinf`, `torch.isneginf` and the `Tensor` methods.

These stopped with "overload resolution has no table entry for this op". Each
spelling now resolves to its own `aten.<name>.default`, as upstream does.
Complex `isnan` is not implemented (refuses by name, not probed); complex
`isposinf` and `isneginf` raise `RuntimeError` upstream and here.

Every case goes through the spelling a user writes and is compared against a
**live upstream torch in a separate process** (`env -u PYTHONPATH -u
TORCH_USE_RTLD_GLOBAL`), shape, dtype and values included. Same harness as
`test_bind5.py`.
"""

import json
import os
import subprocess
import sys

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

_PROBE_SCRIPT = r"""
import json
import torch

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def rec(name, fn):
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
        return
    out[name] = {"ok": r.tolist(), "shape": list(r.shape), "dtype": str(r.dtype)}


inf, nan = float("inf"), float("nan")
for op in ("isnan", "isposinf", "isneginf"):
    f = getattr(torch, op)
    for dn in ("float64", "float32", "float16", "bfloat16"):
        dt = getattr(torch, dn)
        t = torch.tensor([inf, -inf, nan, 0.0, -0.0, 1.5, 65000.0], dtype=dt)
        rec(f"{op}_fn_{dn}", lambda f=f, t=t: f(t))
        rec(f"{op}_method_{dn}", lambda op=op, t=t: getattr(t, op)())
        rec(f"{op}_zero_d_{dn}", lambda f=f, dt=dt: f(torch.tensor(nan, dtype=dt)))
        rec(f"{op}_empty_{dn}", lambda f=f, dt=dt: f(torch.empty(0, dtype=dt)))
        rec(f"{op}_empty2d_{dn}", lambda f=f, dt=dt: f(torch.empty(2, 0, dtype=dt)))
        m = torch.tensor([[inf, 1.0, -inf], [2.0, nan, 3.0]], dtype=dt)
        rec(f"{op}_transposed_{dn}", lambda f=f, m=m: f(m.t()))
        rec(f"{op}_offset_{dn}", lambda f=f, m=m: f(m[:, 1:]))
        rec(f"{op}_stepped_{dn}", lambda op=op, m=m: getattr(m[:, ::2], op)())
    for dn in ("bool", "int64", "int32", "int16", "int8", "uint8"):
        dt = getattr(torch, dn)
        t = torch.tensor([0, 1, 1, 0], dtype=dt)
        rec(f"{op}_fn_{dn}", lambda f=f, t=t: f(t))
        rec(f"{op}_method_{dn}", lambda op=op, t=t: getattr(t, op)())
        rec(f"{op}_empty_{dn}", lambda f=f, dt=dt: f(torch.empty(0, dtype=dt)))
        rec(f"{op}_zero_d_{dn}", lambda f=f, t=t: f(t[0]))
        rec(f"{op}_offset_{dn}", lambda f=f, t=t: f(t[1:]))
# complex: isposinf and isneginf raise upstream; both sides must raise
for op in ("isposinf", "isneginf"):
    rec(f"{op}_complex64", lambda op=op: getattr(torch, op)(torch.tensor([1 + 1j])))
print(json.dumps(out))
"""

_cache = {}


def _run(side):
    if side in _cache:
        return _cache[side]
    env = dict(os.environ)
    if side == "shim":
        if not os.path.isfile(_VENDOR_SHIM):
            _cache[side] = None
            return None
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"the {side} probe failed to run:\n{proc.stderr[-3000:]}"
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, f"the {side} probe imported the wrong torch ({data['_marker']})"
    _cache[side] = data
    return data


def _names():
    up = _run("upstream")
    return sorted(k for k in up if not k.startswith("_"))


_COMPLEX = ("isposinf_complex64", "isneginf_complex64")


def test_upstream_answers_every_case():
    up = _run("upstream")
    for name in _names():
        if name in _COMPLEX:
            assert up[name].get("raised") == "RuntimeError", f"{name}: {up[name]}"
            continue
        assert "raised" not in up[name], f"{name}: upstream itself refused: {up[name]}"


def test_every_case_agrees_with_upstream():
    shim = _run("shim")
    if shim is None:
        return
    up = _run("upstream")
    for name in _names():
        got, want = shim[name], up[name]
        if name in _COMPLEX:
            assert got.get("raised") == want["raised"] == "RuntimeError", f"{name}: {got} vs {want}"
            continue
        assert "raised" not in got, f"{name}: the shim refused where upstream computed: {got}"
        assert got["dtype"] == want["dtype"] == "torch.bool", f"{name}: {got['dtype']} vs {want['dtype']}"
        assert got["shape"] == want["shape"], f"{name}: shape {got['shape']} != {want['shape']}"
        assert got["ok"] == want["ok"], f"{name}: {got['ok']} != {want['ok']}"


def test_the_specials_pick_the_right_values():
    shim = _run("shim")
    if shim is None:
        return
    want = {
        "isnan": [False, False, True, False, False, False, False],
        "isposinf": [True, False, False, False, False, False, False],
        "isneginf": [False, True, False, False, False, False, False],
    }
    for op, expect in want.items():
        for dn in ("float64", "float32", "float16", "bfloat16"):
            assert shim[f"{op}_fn_{dn}"]["ok"] == expect, (op, dn)
            assert shim[f"{op}_method_{dn}"]["ok"] == expect, (op, dn)
