"""A block-quantised `QuantizedLinear` follows `module.to(device)` with its own bytes (issue #175).

`QuantizedLinear.qweight` is a plain attribute, not a parameter or buffer
(`torchnative/quant/__init__.py` says why), so `nn.Module._apply` never saw it.
`from_pretrained(..., gguf_file=..., keep_blocks=True).to("mps")` then left
every quantised weight on the CPU, and the first forward panicked inside
candle (`Cannot call metal matmul on non metal QTensor`) instead of refusing.

What this file asserts, each case in a fresh subprocess that imports the
vendored tree and asserts `hasattr(torch._C, "_aten_implemented")`:

* *the bytes move unchanged*: for q8_0 and q4_0, `_C._quantized_blob` of the
  weight after `.to("mps")`, and again after `.to("cpu")` back, is the blob the
  layer was built from, byte for byte. A move that dequantised and
  re-quantised would change q4_0's bytes (its scale is recomputed from the
  absmax) and fail here.
* *agrees*: the mps forward equals a dense `linear` on the CPU **bit for
  bit**. The operands are lossless on purpose: scale 1.0, integer weights,
  and activations whose every 32-block has absmax 127, so both candle's
  Q8_0-activation CPU path and its float-activation Metal path are exact.
* *placement*: the weight reports `device(type='mps')`, and
  `_C._metal_counters()` shows at least one compute encoder opened during
  the forward itself.
* *refusal*: a weight on one device and an activation on the other raise a
  `RuntimeError` that names `_quantized_linear` and both devices, never a
  `PanicException`.

What this file cannot find: whether the mps matmul ran candle's quantised
Metal kernel or a dequantised dense one; both open compute encoders and both
are exact on these operands.
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
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

_CHILD = r"""
import json, random, struct, sys
req = json.loads(sys.argv[1])
import torch
assert hasattr(torch._C, "_aten_implemented"), "the child imported upstream, not the shim"
from torchnative.quant import QuantizedLinear
out = {"no_mps": None}
try:
    torch._C._aten_dispatch("aten.ones.default", [1], device=torch._C.device("mps"))
except (NotImplementedError, RuntimeError) as e:
    out["no_mps"] = "no mps device -- %s" % str(e).splitlines()[0]
    print(json.dumps(out)); raise SystemExit(0)

fmt = req["format"]
out_f, in_f = 6, 64
rnd = random.Random(175)
blob = bytearray()
dense = []
for _ in range(out_f):
    row = []
    for _ in range(in_f // 32):
        if fmt == "q8_0":
            qs = [rnd.randint(-127, 127) for _ in range(32)]
            blob += struct.pack("<e", 1.0) + struct.pack("<32b", *qs)
            row += [float(v) for v in qs]
        else:
            nib = [rnd.randint(0, 15) for _ in range(32)]
            packed = bytes((nib[j] & 15) | ((nib[j + 16] & 15) << 4) for j in range(16))
            blob += struct.pack("<e", 1.0) + packed
            row += [float(v - 8) for v in nib]
    dense.append(row)
blob = bytes(blob)
q = torch._C._quantized_from_blob(blob, [out_f, in_f], fmt)
layer = QuantizedLinear(in_f, out_f, q, None, fmt)
model = torch.nn.Sequential(layer)

xs = []
for _ in range(3):
    r = [float(rnd.randint(-126, 126)) for _ in range(in_f)]
    r[0], r[32] = 127.0, -127.0
    xs.append(r)
x = torch.tensor(xs)
want = torch.nn.functional.linear(x, torch.tensor(dense))

def f32(t):
    return [struct.pack("<f", v).hex() for row in t.tolist() for v in row]

if req["case"] == "move":
    model.to("mps")
    out["device"] = str(layer.qweight.device)
    out["blob_mps"] = bytes(torch._C._quantized_blob(layer.qweight)) == blob
    xm = x.to("mps")
    c0 = torch._C._metal_counters()["compute_encoders"]
    got = model(xm)
    out["encoders"] = torch._C._metal_counters()["compute_encoders"] - c0
    out["out_device"] = str(got.device)
    out["exact"] = f32(got.cpu()) == f32(want)
    model.to("cpu")
    out["device_back"] = str(layer.qweight.device)
    out["blob_back"] = bytes(torch._C._quantized_blob(layer.qweight)) == blob
    out["exact_back"] = f32(model(x)) == f32(want)
elif req["case"] == "mismatch":
    errs = []
    for w_dev, x_dev in (("cpu", "mps"), ("mps", "cpu")):
        model.to(w_dev)
        try:
            model(x.to(x_dev))
            errs.append({"raised": None})
        except BaseException as e:
            errs.append({"raised": type(e).__name__, "msg": str(e)})
    out["errs"] = errs
print(json.dumps(out))
"""


def _run(case, fmt):
    if not os.path.isfile(_VENDOR_SHIM):
        raise _skip.Skip("vendored tree not built: no %s "
                         "(scripts/vendor/vendor_torch.sh + scripts/vendor/install_shim.sh)"
                         % _VENDOR_SHIM)
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    for var in ("CANDLE_DEQUANTIZE_ALL", "CANDLE_DEQUANTIZE_ALL_F16"):
        env.pop(var, None)
    req = {"case": case, "format": fmt}
    proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(req)],
                          capture_output=True, text=True, timeout=600, env=env)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert proc.returncode == 0 and lines, (
        "the mps child (%s, %s) exited %d:\n%s" % (case, fmt, proc.returncode, proc.stderr[-2000:]))
    out = json.loads(lines[-1])
    if out["no_mps"]:
        raise _skip.Skip(out["no_mps"])
    return out


def _check_move(fmt):
    out = _run("move", fmt)
    assert out["device"] == "mps:0", out
    assert out["blob_mps"], "%s: the blocks changed on the way to mps" % fmt
    assert out["out_device"] == "mps:0", out
    assert out["encoders"] >= 1, "%s: the mps forward opened no compute encoder: %s" % (fmt, out)
    assert out["exact"], "%s: the mps forward is not bit-identical to dense linear" % fmt
    assert out["device_back"] == "cpu", out
    assert out["blob_back"], "%s: the blocks changed on the way back to cpu" % fmt
    assert out["exact_back"], "%s: the cpu forward after the round trip is not exact" % fmt
    print("    mpsquant: %s QuantizedLinear follows .to('mps') and back with its bytes, "
          "%d compute encoder(s), exact" % (fmt, out["encoders"]))


def test_a_q8_0_linear_follows_module_to_mps_with_its_own_bytes():
    _check_move("q8_0")


def test_a_q4_0_linear_follows_module_to_mps_with_its_own_bytes():
    _check_move("q4_0")


def test_a_weight_and_activation_on_different_devices_refuse_by_name():
    out = _run("mismatch", "q8_0")
    for (w_dev, x_dev), err in zip((("cpu", "mps"), ("mps", "cpu")), out["errs"]):
        assert err["raised"] == "RuntimeError", (w_dev, x_dev, err)
        msg = err["msg"]
        assert "_quantized_linear" in msg and "cpu" in msg and "mps" in msg, msg
    print("    mpsquant: a weight/activation device mismatch refuses by name in both directions")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsquant")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
