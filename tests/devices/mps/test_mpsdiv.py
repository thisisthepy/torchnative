"""float32 division on mps is correctly rounded, like upstream (issue #113).

Metal's `/` on float is not correctly rounded even without fast math: `7/3`
and `10/3` came out one ulp high on the shim's mps while upstream's mps and
cpu (and the shim's cpu) round to nearest. IEEE division is exact-rounded, so
upstream cpu is a bit-exact oracle here: the shim on mps must match it bit for
bit, measured in a separate subprocess that asserts it imported upstream.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip


_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")

_SCRIPT = r"""
import json, sys, struct
A = json.loads(sys.argv[1])
import torch
assert hasattr(torch._C, "_aten_implemented") == A["shim"], "wrong torch in this child"
D = A["device"]
g = torch.Generator().manual_seed(113)
a = torch.randn(4096, generator=g) * 1e3
b = torch.randn(4096, generator=g) * 1e2
a[:4] = torch.tensor([7., 1., 2., 10.]); b[:4] = 3.
a, b = a.to(D), b.to(D)
out = {
    "tensor": (a / b).cpu(),
    "broadcast": (a.reshape(64, 64) / b[:64].reshape(1, 64)).cpu(),
    "inplace": a.clone().div_(b).cpu(),
}
def bits(t):
    return [struct.unpack("<I", struct.pack("<f", v))[0] for v in t.flatten().tolist()]
print(json.dumps({k: bits(v) for k, v in out.items()}))
"""


def _run(device, upstream):
    env = dict(os.environ)
    env.pop("TORCH_C_ARTEFACT", None)
    if upstream:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    else:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    payload = {"device": device, "shim": not upstream}
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, json.dumps(payload)],
                          capture_output=True, text=True, timeout=600, env=env,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    lines = proc.stdout.strip().splitlines()
    assert lines, "child produced no JSON (rc=%d):\n%s" % (proc.returncode, proc.stderr[-3000:])
    return json.loads(lines[-1])


def _mps_or_skip(what):
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")):
        _skip.skip("   (skipped %s: no vendored shim at %s -- run scripts/vendor/install_shim.sh)"
                   % (what, _VENDOR_DIR))
        return False
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return False
    return True


def test_float32_div_on_mps_is_bit_exact_with_upstream_cpu():
    if not _mps_or_skip("float32 div rounding on mps"):
        return
    want = _run("cpu", upstream=True)
    got = _run("mps", upstream=False)
    for form in ("tensor", "broadcast", "inplace"):
        bad = [i for i, (g, w) in enumerate(zip(got[form], want[form])) if g != w]
        assert not bad, "%s: %d of %d elements differ, first at %d: %08x vs %08x" % (
            form, len(bad), len(want[form]), bad[0], got[form][bad[0]], want[form][bad[0]])


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsdiv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
