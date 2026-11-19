"""`nll_loss` gradients for every reduction, with and without class weights (#249).

The tape's `nll_loss_forward` rule reshaped the incoming gradient to `[1]`
whatever the reduction, so `reduction="none"` (one gradient per row) and any
class-weighted loss over more than one row (`grad * weight[target]` is one value
per row) could not be differentiated. Found while building #237's
`PseudoLabel`, which wanted a per-sample cross entropy.

**The grade is agrees.** The shim's gradients are compared with upstream
autograd's on the same inputs. The tolerance is derived (AGENTS.md section 16):
upstream re-runs each case in float64, and the shim may differ from upstream's
float32 by at most `agree_sweep.ORACLE_FACTOR` times upstream's own float32
error, floored at 8 ulp of the scale.

**Why the per-row coefficients.** `reduction="none"` is reduced here with a
different coefficient per row before the gradient is taken. Summing it plainly
would make every row's incoming gradient 1, and a rule that broadcast a scalar
to every row would then agree by accident.

Each case has an ignored row (`ignore_index=-100`), so the masking branch runs
under every reduction.

What this file cannot find: `nll_loss2d`, inputs that are not 2-D (the rule
refuses those by name), and `label_smoothing`.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip  # noqa: E402
from agree_sweep import F32_EPS, ORACLE_FACTOR  # noqa: E402

_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

try:
    import torch as _upstream_torch
except Exception:  # pragma: no cover - reported as a skip below
    _upstream_torch = None


def _available():
    return _upstream_torch is not None and os.path.isfile(_VENDOR_SHIM)


# Executed verbatim on both sides.
_CASES_SRC = r'''
import torch.nn.functional as F
R, C = 5, 4
TARGET = [2, 0, -100, 3, 1]

def _det(n, seed):
    return [((seed * 1103515245 + i * 12345) % 2000 - 1000) / 400.0 for i in range(n)]

def cases(dtype):
    out = {}
    target = torch.tensor(TARGET)
    weight = torch.tensor([0.5, 2.0, 1.25, 3.0], dtype=dtype)
    coeff = torch.tensor([1.0, -2.0, 3.0, 0.5, 4.0], dtype=dtype)
    for fn in ("cross_entropy", "nll_loss"):
        for reduction in ("none", "sum", "mean"):
            for weighted in (False, True):
                x = torch.tensor(_det(R * C, 7), dtype=dtype).reshape(R, C).requires_grad_(True)
                inp = x if fn == "cross_entropy" else x.log_softmax(-1)
                loss = getattr(F, fn)(inp, target, weight=weight if weighted else None,
                                      reduction=reduction)
                objective = (loss * coeff).sum() if reduction == "none" else loss
                try:
                    (g,) = torch.autograd.grad(objective, [x])
                    value = g.reshape(-1).tolist()
                except Exception as e:
                    value = "%s: %s" % (type(e).__name__, str(e)[:300])
                out["%s/%s/%s" % (fn, reduction, "weighted" if weighted else "plain")] = value
    return out
'''

_SHIM_SCRIPT = (
    "import json, sys\nimport torch\n"
    "assert hasattr(torch._C, '_aten_implemented'), 'not the shim'\n"
    + _CASES_SRC
    + "json.dump(cases(torch.float32), sys.stdout)\n"
)


def _shim():
    env = dict(os.environ)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["PYTHONPATH"] = _VENDOR_DIR
    proc = subprocess.run([sys.executable, "-c", _SHIM_SCRIPT], capture_output=True,
                          text=True, env=env, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError("nllreduce subprocess exited %d\n%s\n%s"
                           % (proc.returncode, proc.stdout, proc.stderr))
    return json.loads(proc.stdout)


def _oracle(dtype):
    ns = {"torch": _upstream_torch}
    exec(_CASES_SRC, ns)
    return ns["cases"](dtype)


_CACHE = {}


def _results():
    if not _CACHE:
        _CACHE["shim"] = _shim()
        _CACHE["o32"] = _oracle(_upstream_torch.float32)
        _CACHE["o64"] = _oracle(_upstream_torch.float64)
    return _CACHE["shim"], _CACHE["o32"], _CACHE["o64"]


def _check(reduction):
    s, a, b = _results()
    keys = sorted(k for k in a if k.split("/")[1] == reduction)
    assert len(keys) == 4, keys
    worst = []
    for k in keys:
        assert isinstance(a[k], list) and isinstance(b[k], list), (k, a[k], b[k])
        assert isinstance(s[k], list), "%s: %s" % (k, s[k])
        scale = max(abs(v) for v in b[k])
        assert scale > 1e-6, (k, scale)
        rel = max(abs(x - y) for x, y in zip(s[k], a[k])) / scale
        oracle_rel = max(abs(x - y) for x, y in zip(a[k], b[k])) / scale
        tol = ORACLE_FACTOR * max(oracle_rel, 8 * F32_EPS)
        assert rel <= tol, "%s: rel %.3g > tol %.3g" % (k, rel, tol)
        # The ignored row's gradient is exactly zero on both sides.
        assert s[k][2 * 4:3 * 4] == [0.0] * 4, (k, s[k][8:12])
        worst.append(rel)
    return max(worst)


def test_reduction_none_gradients_agree_with_upstream():
    if not _available():
        _skip.skip("no vendored shim or no upstream torch")
        return
    w = _check("none")
    print(f"ok   nllreduce: reduction='none' (plain and weighted) agrees, worst rel {w:.3g}")


def test_reduction_sum_gradients_agree_with_upstream():
    if not _available():
        _skip.skip("no vendored shim or no upstream torch")
        return
    w = _check("sum")
    print(f"ok   nllreduce: reduction='sum' (plain and weighted) agrees, worst rel {w:.3g}")


def test_reduction_mean_gradients_agree_with_upstream():
    if not _available():
        _skip.skip("no vendored shim or no upstream torch")
        return
    w = _check("mean")
    print(f"ok   nllreduce: reduction='mean' (plain and weighted) agrees, worst rel {w:.3g}")


if __name__ == "__main__":
    failures = _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")],
        suite="test_nllreduce",
    )
    raise SystemExit(1 if failures else 0)
