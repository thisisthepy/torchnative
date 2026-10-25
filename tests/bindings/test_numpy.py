"""`Tensor.numpy()`, `np.asarray(tensor)` and `__array__` (issue #118).

Upstream 2.13.0 returns a numpy array that SHARES memory with a cpu strided
tensor, carries the tensor's strides, and refuses with fixed texts otherwise.
The shim used to refuse by name (`TensorBase.numpy`). Each case below runs in
a child process against upstream and again against the shim, and the two
answers must be identical, so every expectation is upstream's own output. The
shim child asserts it imported the shim.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip


_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")

_SCRIPT = r"""
import gc, json, sys
A = json.loads(sys.argv[1])
import torch
assert hasattr(torch._C, "_aten_implemented") == A["shim"], "wrong torch in this child"
import numpy as np

def refusal(f):
    try:
        f()
    except Exception as e:
        return [type(e).__name__, str(e)]
    return ["no error", ""]

def describe(a):
    return {"dtype": str(a.dtype), "shape": list(a.shape), "strides": list(a.strides),
            "values": a.tolist(), "type": type(a).__name__}

out = {}

DT = {"bool": ([True, False, True], torch.bool), "uint8": ([0, 7, 255], torch.uint8),
      "int8": ([-128, 0, 127], torch.int8), "int16": ([-32768, 5, 32767], torch.int16),
      "int32": ([-2**31, 9, 2**31 - 1], torch.int32), "int64": ([-2**62, 3, 2**62], torch.int64),
      "float16": ([0.5, -1.25, 2048.0], torch.float16),
      "float32": ([0.1, -2.5, 1e30], torch.float32),
      "float64": ([0.1, -2.5, 1e300], torch.float64)}
for name, (vals, dt) in DT.items():
    out["dtype_" + name] = describe(torch.tensor(vals, dtype=dt).numpy())

t = torch.ones(4)
a = t.numpy()
a[1] = 5.0
t[2] = 7.0
out["share"] = {"tensor": t.tolist(), "array": a.tolist(), "writeable": bool(a.flags.writeable)}
t.add_(1.0)
out["share_inplace_op"] = a.tolist()

base = torch.arange(12.).reshape(3, 4)
v = base.t()[1:]
av = v.numpy()
d = describe(av)
av[0, 0] = 99.0
base[1, 1] = -5.0
d["base_after"] = base.tolist()
d["view_after"] = av.tolist()
out["transpose_slice"] = d

b2 = torch.arange(12.).reshape(3, 4)
s = b2[1:, 2:]
as_ = s.numpy()
d = describe(as_)
as_[0, 0] = -1.0
d["base_after"] = b2.tolist()
out["offset_slice"] = d

out["scalar"] = describe(torch.tensor(3.5).numpy())
out["empty"] = describe(torch.zeros(0, 3).numpy())

big = torch.arange(100000.0)
arr = big.numpy()[::2]
del big
gc.collect()
junk = [torch.ones(100000) * i for i in range(20)]
gc.collect()
out["lifetime"] = [float(arr.sum()), float(arr[-1]), list(arr.shape), len(junk)]

r = torch.arange(6.0)
ar = r.numpy()
r.data = torch.zeros(3)
gc.collect()
junk = [torch.ones(6) * i for i in range(20)]
out["lifetime_after_rebind"] = ar.tolist()

out["bfloat16"] = refusal(lambda: torch.ones(2, dtype=torch.bfloat16).numpy())
out["bfloat16_force"] = refusal(lambda: torch.ones(2, dtype=torch.bfloat16).numpy(force=True))
out["requires_grad"] = refusal(lambda: torch.ones(2, requires_grad=True).numpy())
out["requires_grad_asarray"] = refusal(lambda: np.asarray(torch.ones(2, requires_grad=True)))
out["meta_device"] = refusal(lambda: torch.ones(2, device="meta").numpy())
out["force_type"] = refusal(lambda: torch.ones(2).numpy(force="x"))

try:
    m = torch.ones(2, device="mps")
    out["mps_device"] = refusal(lambda: m.numpy())
except Exception:
    out["mps_device"] = "unavailable"

g = torch.ones(2, requires_grad=True)
f = g.numpy(force=True)
f[0] = 7.0
out["force_grad"] = {"array": f.tolist(), "tensor": g.tolist(), "dtype": str(f.dtype)}
c = torch.ones(3)
fc = c.numpy(force=True)
fc[0] = 4.0
out["force_cpu_shares"] = c.tolist()

p = torch.arange(4.0)
ap = np.asarray(p)
ap[0] = 9.0
out["asarray_shares"] = {"tensor": p.tolist(), "dtype": str(ap.dtype)}
ac = np.array(p)
ac[1] = -3.0
out["array_copies"] = p.tolist()
out["asarray_dtype"] = str(np.asarray(torch.ones(2), dtype=np.float64).dtype)
out["dunder_array"] = describe(torch.arange(3).__array__())
out["numpy_scalar_mul"] = (torch.ones(2) * np.int64(3)).tolist()

st = torch.Generator().get_state().numpy()
out["generator_state"] = {"dtype": str(st.dtype), "shape": list(st.shape)}
print(json.dumps(out))
"""


def _run(upstream):
    env = dict(os.environ)
    env.pop("TORCH_C_ARTEFACT", None)
    if upstream:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    else:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, json.dumps({"shim": not upstream})],
                          capture_output=True, text=True, timeout=900, env=env,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    lines = proc.stdout.strip().splitlines()
    assert lines, "child produced no JSON (rc=%d):\n%s" % (proc.returncode, proc.stderr[-3000:])
    return json.loads(lines[-1])


_CACHE = {}


def _both(what):
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")):
        _skip.skip("   (skipped %s: no vendored shim at %s -- run scripts/vendor/install_shim.sh)"
                   % (what, _VENDOR_DIR))
        return None
    if not _CACHE:
        _CACHE["up"] = _run(True)
        _CACHE["shim"] = _run(False)
    return _CACHE["up"], _CACHE["shim"]


def _same(keys, what):
    both = _both(what)
    if both is None:
        return
    up, shim = both
    for k in keys:
        if up[k] == "unavailable" or shim[k] == "unavailable":
            _skip.skip("   (skipped %s: %s unavailable on this machine)" % (what, k))
            continue
        assert shim[k] == up[k], "%s differs from upstream:\n  shim:     %r\n  upstream: %r" % (
            k, shim[k], up[k])


def test_numpy_values_and_dtype_per_dtype():
    _same(["dtype_" + n for n in ("bool", "uint8", "int8", "int16", "int32", "int64",
                                   "float16", "float32", "float64")], "numpy dtypes")


def test_numpy_shares_memory_both_directions():
    _same(["share", "share_inplace_op"], "numpy memory sharing")
    both = _both("numpy memory sharing")
    if both:
        s = both[1]["share"]
        assert s["tensor"] == [1.0, 5.0, 7.0, 1.0] and s["array"] == s["tensor"], s


def test_numpy_non_contiguous_view_keeps_strides_and_offset():
    _same(["transpose_slice", "offset_slice", "scalar", "empty"], "numpy strided views")


def test_numpy_array_outlives_tensor():
    _same(["lifetime", "lifetime_after_rebind"], "numpy lifetime")


def test_numpy_refusals_carry_upstream_type_and_text():
    _same(["bfloat16", "bfloat16_force", "requires_grad", "requires_grad_asarray",
           "meta_device", "force_type", "mps_device"], "numpy refusals")


def test_numpy_force_true():
    _same(["force_grad", "force_cpu_shares"], "numpy force=True")


def test_np_asarray_and_dunder_array():
    _same(["asarray_shares", "array_copies", "asarray_dtype", "dunder_array",
           "numpy_scalar_mul"], "np.asarray")


def test_generator_state_numpy():
    _same(["generator_state"], "generator state numpy")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_numpy")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
