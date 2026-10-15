"""mps without `MTLResidencySet`: issue #65.

`MTLResidencySet` is a macOS 15 API. candle built one unconditionally when it
created a Metal device, and `MTLResidencySetDescriptor::new()` panics on an OS
that lacks the class, so on macOS 11 to 14 (which the `macosx_11_0_arm64` wheel
tag admits) the first `mps` tensor panicked the process. The vendored
`candle-metal-kernels` now builds the set only when the OS has the class and
the device answers the selector (`src/metal/residency_set.rs`, recorded in
`torchnative/rust/vendor/int8-candle-metal-kernels-0.11.0.patch`); otherwise
the device runs without one, as Metal did before macOS 15.

`_C._metal_residency()` reports which path ran: `supported` (the class
exists), `created` and `skipped` (devices with and without a set).

What each test kills:

* `test_mps_computes_and_takes_the_path_this_os_has`
      the deliverable. In a fresh process: an mps tensor either computes the
      right numbers on the GPU or refuses by name, and never panics. On
      macOS >= 15 it asserts a set was created; below 15 it asserts none was
      and the device went without (only the macos-14 CI job runs that branch).
      Killed by: removing the class check (macOS 14 panics), and by gating
      the set off unconditionally (`created == 0` on this machine).
* `test_the_pre_15_path_computes_here`
      drives the no-set path on this machine with
      `CANDLE_METAL_NO_RESIDENCY_SET=1`, which can only disable the set. The
      kernel still runs on the GPU (`compute_encoders` moves) and the values
      are right. Killed by: ignoring the override (`created` > 0), and by a
      no-set path that breaks dispatch.
* `test_metal_residency_answers_its_names`
      the shape contract, and `built is False` off Apple.

The gate decision itself is a pure function with every input faked, tested in
`candle-metal-kernels` (`gate_tests` in `residency_set.rs`).
"""

import json
import os
import platform
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip

_SUPPORT = os.path.dirname(os.path.abspath(sys.modules["test_shim"].__file__))

_CHILD = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
from test_shim import _C
assert hasattr(_C, "_aten_implemented"), "the child imported upstream, not the shim"
out = {"computed": False, "refusal": None}
try:
    dev = _C.device("mps")
    before = _C._metal_counters()["compute_encoders"]
    a = _C._aten_dispatch("aten.ones.default", [2, 3], device=dev)
    r = _C._aten_dispatch("aten.add.Tensor", a, a)
    out["device"] = str(r.device)
    out["encoders"] = _C._metal_counters()["compute_encoders"] - before
    out["values"] = [float(v) for v in r.cpu().flatten().tolist()]
    out["computed"] = True
except (NotImplementedError, RuntimeError) as e:
    out["refusal"] = "%s: %s" % (type(e).__name__, e)
out["residency"] = _C._metal_residency()
print(json.dumps(out))
"""


def _mac_major():
    v = platform.mac_ver()[0]
    return int(v.split(".")[0]) if v else None


def _child(extra_env=None):
    env = dict(os.environ)
    env.pop("CANDLE_METAL_NO_RESIDENCY_SET", None)
    env.update(extra_env or {})
    proc = subprocess.run([sys.executable, "-c", _CHILD, _SUPPORT],
                          capture_output=True, text=True, timeout=600, env=env)
    assert "panicked" not in proc.stderr, (
        "creating an mps tensor panicked (issue #65):\n" + proc.stderr[-2000:])
    assert proc.returncode == 0, (
        "the mps child died with %d:\n%s" % (proc.returncode, proc.stderr[-2000:]))
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _apple_or_skip(what):
    if not _C._metal_residency()["built"]:
        _skip.skip("   (skipped %s: not an Apple build)" % what)
        return False
    return True


def test_metal_residency_answers_its_names():
    r = _C._metal_residency()
    assert set(r) == {"built", "supported", "created", "skipped"}, sorted(r)
    if r["built"]:
        assert isinstance(r["supported"], bool), r
        assert isinstance(r["created"], int) and isinstance(r["skipped"], int), r
    else:
        assert r["supported"] is None and r["created"] is None, r


def test_mps_computes_and_takes_the_path_this_os_has():
    if not _apple_or_skip("the residency path"):
        return
    out = _child()
    res = out["residency"]
    major = _mac_major()
    if major is not None:
        assert res["supported"] == (major >= 15), (
            "macOS %d but supported=%r" % (major, res["supported"]))
    if not out["computed"]:
        # A hosted runner may have no usable Metal device. That must be a
        # refusal with a name, not a panic (the child's exit code is 0).
        assert out["refusal"] and ("mps" in out["refusal"].lower()
                                   or "metal" in out["refusal"].lower()), out
        _skip.skip("   (skipped the value check: mps refused by name -- %s)"
                   % out["refusal"].splitlines()[0])
        return
    assert out["device"].startswith("mps"), out
    assert out["values"] == [2.0] * 6, out
    assert out["encoders"] >= 1, ("add on mps opened no compute encoder", out)
    if res["supported"]:
        assert res["created"] >= 1 and res["skipped"] == 0, (
            "this OS has MTLResidencySet, so the device must have one", res)
    else:
        assert res["created"] == 0 and res["skipped"] >= 1, (
            "this OS lacks MTLResidencySet, so the device must go without", res)


def test_the_pre_15_path_computes_here():
    if not _apple_or_skip("the no-residency-set path"):
        return
    out = _child({"CANDLE_METAL_NO_RESIDENCY_SET": "1"})
    res = out["residency"]
    if not out["computed"]:
        assert out["refusal"], out
        _skip.skip("   (skipped the no-set path: mps refused by name -- %s)"
                   % out["refusal"].splitlines()[0])
        return
    assert res["created"] == 0 and res["skipped"] >= 1, (
        "CANDLE_METAL_NO_RESIDENCY_SET=1 must take the path without a set", res)
    assert out["device"].startswith("mps"), out
    assert out["encoders"] >= 1, ("the no-set path ran no GPU kernel", out)
    assert out["values"] == [2.0] * 6, out


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsresidency")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
