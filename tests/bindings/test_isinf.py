"""Issue #403 -- `torch.isinf` and `Tensor.isinf`.

`transformers`' `TypicalLogitsWarper` calls `torch.isinf`, which stopped with
"overload resolution has no table entry for this op". Both spellings now
resolve to `aten.isinf.default`, as upstream does for `isinf(Tensor self)`.

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
for dn in ("float64", "float32", "float16", "bfloat16"):
    dt = getattr(torch, dn)
    t = torch.tensor([inf, -inf, nan, 0.0, -0.0, 1.5, 65000.0], dtype=dt)
    rec(f"fn_{dn}", lambda t=t: torch.isinf(t))
    rec(f"method_{dn}", lambda t=t: t.isinf())
    rec(f"zero_d_{dn}", lambda dt=dt: torch.isinf(torch.tensor(-inf, dtype=dt)))
    rec(f"empty_{dn}", lambda dt=dt: torch.isinf(torch.empty(0, dtype=dt)))
    rec(f"empty2d_{dn}", lambda dt=dt: torch.isinf(torch.empty(2, 0, dtype=dt)))
    m = torch.tensor([[inf, 1.0, -inf], [2.0, nan, 3.0]], dtype=dt)
    rec(f"transposed_{dn}", lambda m=m: torch.isinf(m.t()))
    rec(f"offset_{dn}", lambda m=m: torch.isinf(m[:, 1:]))
    rec(f"stepped_{dn}", lambda m=m: m[:, ::2].isinf())
for dn in ("bool", "int64", "int32", "int16", "int8", "uint8"):
    dt = getattr(torch, dn)
    t = torch.tensor([0, 1, 1, 0], dtype=dt)
    rec(f"fn_{dn}", lambda t=t: torch.isinf(t))
    rec(f"method_{dn}", lambda t=t: t.isinf())
    rec(f"empty_{dn}", lambda dt=dt: torch.isinf(torch.empty(0, dtype=dt)))
    rec(f"zero_d_{dn}", lambda t=t: torch.isinf(t[0]))
    rec(f"offset_{dn}", lambda t=t: torch.isinf(t[1:]))
# the `TypicalLogitsWarper` shape: a mask over a score matrix with -inf filled in
scores = torch.tensor([[1.0, -inf, 2.0], [-inf, -inf, 0.5]])
rec("typical_warper_shape", lambda: torch.isinf(scores).any(dim=-1))
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


def test_upstream_answers_every_case():
    up = _run("upstream")
    for name in _names():
        assert "raised" not in up[name], f"{name}: upstream itself refused: {up[name]}"


def test_every_case_agrees_with_upstream():
    shim = _run("shim")
    if shim is None:
        return
    up = _run("upstream")
    for name in _names():
        got, want = shim[name], up[name]
        assert "raised" not in got, f"{name}: the shim refused where upstream computed: {got}"
        assert got["dtype"] == want["dtype"] == "torch.bool", f"{name}: {got['dtype']} vs {want['dtype']}"
        assert got["shape"] == want["shape"], f"{name}: shape {got['shape']} != {want['shape']}"
        assert got["ok"] == want["ok"], f"{name}: {got['ok']} != {want['ok']}"


def test_the_specials_are_true_for_both_infinities_only():
    shim = _run("shim")
    if shim is None:
        return
    for dn in ("float64", "float32", "float16", "bfloat16"):
        assert shim[f"fn_{dn}"]["ok"] == [True, True, False, False, False, False, False]
        assert shim[f"method_{dn}"]["ok"] == shim[f"fn_{dn}"]["ok"]
