"""Tensor basics on torchnative's WASM build, checked against upstream torch (examples/README.md).

The web example does not run the common task: SmolLM2-135M's download and memory do not fit
a browser tab's wasm32 heap. It runs these cases instead, each one a small, deterministic
use of the API a page would make: creation, elementwise maths, reductions, matmul,
indexing, the seeded random generators, `nn` layers and autograd.

`reference.py` runs the same cases with upstream torch on a host and writes
`reference.json`. `run.mjs` (Node) and `index.html` (a browser) run them inside Pyodide
with torchnative's `pyemscripten` wheel and call `main()`, which compares each case:

- `exact` cases (integers, booleans, shapes, dtypes, and the seeded generators) must equal
  upstream's value exactly.
- `float` cases must lie within a tolerance derived from upstream itself: per element,
  the larger of upstream's own float32-vs-float64 difference and 8 float32 ulps of the
  value (AGENTS.md §16). Nothing here chooses a tolerance by hand.

`main()` prints one `BASICS` line per case and returns non-zero if any case disagrees or
raises. It refuses to run on a `torch` that is not torchnative's.
"""

import json
import math
import sys

# (name, kind, fn). `fn(torch, dtype)` returns a nested list, a number, a bool or a string.
# A `float` case is computed in `dtype` so the reference can run it in float64 too; an
# `exact` case ignores `dtype`.


def _creation(torch, dtype):
    x = torch.arange(12, dtype=dtype).reshape(3, 4)
    return [list(x.shape), str(torch.zeros(2, 3).dtype), torch.full((2,), 7).tolist(),
            torch.eye(3, dtype=torch.int64).tolist(), x.tolist()]


def _elementwise(torch, dtype):
    x = torch.linspace(-2.0, 2.0, 9, dtype=dtype)
    return [(x + 1).tolist(), (x * x).tolist(), torch.exp(x).tolist(), torch.sin(x).tolist(),
            torch.tanh(x).tolist(), x.clamp(-1.0, 0.5).tolist(), torch.relu(x).tolist()]


def _reductions(torch, dtype):
    x = torch.tensor([[3.0, -1.0, 4.0], [1.5, 9.0, -2.5]], dtype=dtype)
    return [x.sum().item(), x.mean(dim=1).tolist(), x.amax(dim=0).tolist(),
            x.std().item(), torch.linalg.vector_norm(x).item()]


def _argmax(torch, dtype):
    x = torch.tensor([[3.0, -1.0, 4.0], [1.5, 9.0, -2.5]])
    values, indices = x.max(dim=1)
    return [values.tolist(), indices.tolist(), x.argmax().item(), x.argsort(dim=1).tolist()]


def _matmul(torch, dtype):
    a = torch.arange(6, dtype=dtype).reshape(2, 3) / 7
    b = torch.arange(12, dtype=dtype).reshape(3, 4) - 5
    return [(a @ b).tolist(), torch.bmm(a.unsqueeze(0), b.unsqueeze(0)).tolist()]


def _indexing(torch, dtype):
    x = torch.arange(24).reshape(2, 3, 4)
    y = x.clone()
    y[:, 1, :] = -1
    return [x[1, 2].tolist(), x[:, :, ::2].tolist(), x.transpose(0, 2).shape[0],
            torch.cat([x[0], x[1]], dim=0).shape[0], torch.stack([x[0], x[1]]).shape[0],
            x[x % 5 == 0].tolist(), y.sum().item(), x.flip(2)[0, 0].tolist()]


def _integers(torch, dtype):
    x = torch.arange(1, 6)
    return [x.cumsum(0).tolist(), (x // 2).tolist(), (x % 3).tolist(), x.prod().item(),
            (x > 2).tolist(), torch.where(x > 2, x, -x).tolist(), str(x.dtype)]


def _random(torch, dtype):
    torch.manual_seed(0)
    a = torch.rand(4).tolist()
    b = torch.randn(4).tolist()
    c = torch.randint(0, 100, (4,)).tolist()
    d = torch.randperm(6).tolist()
    return [a, b, c, d]


def _nn(torch, dtype):
    torch.manual_seed(0)
    linear = torch.nn.Linear(4, 3)       # initialised in float32, then cast: the init is upstream's generator
    conv = torch.nn.Conv1d(1, 2, 3)
    linear, conv = linear.to(dtype), conv.to(dtype)
    x = torch.linspace(-1.0, 1.0, 8, dtype=dtype).reshape(2, 4)
    with torch.no_grad():
        h = linear(x)
        return [h.tolist(), torch.softmax(h, dim=-1).tolist(),
                torch.nn.functional.layer_norm(h, (3,)).tolist(),
                torch.nn.functional.gelu(h).tolist(),
                conv(x.reshape(2, 1, 4).repeat(1, 1, 2)).tolist()]


def _autograd(torch, dtype):
    torch.manual_seed(0)
    linear = torch.nn.Linear(3, 1).to(dtype)
    x = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]], dtype=dtype, requires_grad=True)
    loss = (linear(x).tanh() ** 2).mean()
    loss.backward()
    return [loss.item(), x.grad.tolist(), linear.weight.grad.tolist(), linear.bias.grad.tolist()]


def _dtypes(torch, dtype):
    x = torch.tensor([1.5, -2.25, 3.0e-3, 65504.0])
    return [x.half().float().tolist(), x.bfloat16().float().tolist(),
            (x.half() + x.half()).float().tolist(), str(x.half().dtype), str(x.bfloat16().dtype)]


CASES = [
    ("creation", "float", _creation),
    ("elementwise", "float", _elementwise),
    ("reductions", "float", _reductions),
    ("argmax", "exact", _argmax),
    ("matmul", "float", _matmul),
    ("indexing", "exact", _indexing),
    ("integers", "exact", _integers),
    ("random", "exact", _random),
    ("nn", "float", _nn),
    ("autograd", "float", _autograd),
    ("dtypes", "exact", _dtypes),
]


def compute(torch, dtype=None):
    """Run every case; a case that raises is recorded as its error, not skipped."""
    dtype = torch.float32 if dtype is None else dtype
    out = {}
    for name, _kind, fn in CASES:
        try:
            out[name] = {"value": fn(torch, dtype)}
        except Exception as exc:  # reported by name, never replaced
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _ulp32(x):
    x = abs(x)
    if x < 2.0 ** -126:
        return 2.0 ** -149
    return 2.0 ** (math.floor(math.log2(x)) - 23)


def tolerance(value32, value64):
    """Per element: max(|upstream f32 - upstream f64|, 8 float32 ulps of the f32 value)."""
    tol = []
    for a, b in zip(_flatten(value32), _flatten(value64)):
        if isinstance(a, (bool, str)) or not isinstance(a, (int, float)):
            tol.append(0.0)
        else:
            tol.append(max(abs(float(a) - float(b)), 8 * _ulp32(float(a))))
    return tol


def compare(name, kind, got, ref):
    """None when `got` agrees with `ref` (a reference.json entry), else the reason."""
    if "error" in got:
        return got["error"]
    a, b = list(_flatten(got["value"])), list(_flatten(ref["value"]))
    if len(a) != len(b):
        return f"{len(a)} elements, upstream has {len(b)}"
    if kind == "exact":
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y or type(x) is not type(y):  # 3 is not 3.0, and True is not 1
                return f"element {i}: {x!r}, upstream {y!r}"
        return None
    worst = None
    for i, (x, y, t) in enumerate(zip(a, b, ref["tol"])):
        if isinstance(y, str) or isinstance(y, bool):
            if x != y:
                return f"element {i}: {x!r}, upstream {y!r}"
            continue
        err = abs(float(x) - float(y))
        if not err <= t:  # NaN fails too
            if worst is None or err / max(t, 1e-300) > worst[0]:
                worst = (err / max(t, 1e-300), i, x, y, t)
    if worst:
        return f"element {worst[1]}: {worst[2]!r}, upstream {worst[3]!r}, tolerance {worst[4]:.3g}"
    return None


def main(reference_json):
    """Run the cases on this interpreter's torch and compare with `reference_json` (a str)."""
    import torch

    if not hasattr(torch._C, "_aten_implemented"):
        raise SystemExit("the torch imported here is not torchnative's (no torch._C._aten_implemented)")
    reference = json.loads(reference_json)
    print("PLATFORM " + json.dumps({
        "sys_platform": sys.platform, "python": sys.version.split()[0],
        "torch": torch.__version__, "torch_C": getattr(sys.modules["torch._C"], "__file__", None),
        "ipc_stand_ins": getattr(torch._C, "_shim_ipc_installed", None),
    }))
    results = compute(torch)
    failed = []
    for name, kind, _fn in CASES:
        reason = compare(name, kind, results[name], reference["cases"][name])
        line = {"case": name, "kind": kind, "agrees_with_upstream": reason is None}
        if reason is not None:
            line["reason"] = reason
            failed.append(name)
        print("BASICS " + json.dumps(line))
    print("SUMMARY " + json.dumps({"cases": len(CASES), "agree": len(CASES) - len(failed),
                                   "failed": failed, "upstream_torch": reference["torch"]}))
    return 1 if failed else 0
