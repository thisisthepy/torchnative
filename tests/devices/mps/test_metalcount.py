"""The Metal dispatch counter, and the substitution it can finally catch.

`docs/devices/matrix.md` §7.5 named the hole this file closes:

> **There is no Metal dispatch counter in this build.** `device.rs` has
> `_cuda_counters()` and `_vulkan_counters()`; Metal has neither, and candle's
> `MetalDevice` exposes no countable kernel launch.

Why that was the ceiling on every `mps` claim in this repository, stated as a
measurement rather than a worry: **three rounds replaced a device kernel with a
host-computed twin, and every value came back correct, every agreement test
stayed green, and the `.device` label never changed.** All three were Vulkan
(`7dff9f0`, `35e002f`, `22d9158`) -- not because Metal is safer but because
Vulkan was the only backend where the experiment could be *run* (AGENTS.md §13.1).
On Metal the failure mode could not be excluded, only hoped against.

`_C._metal_counters()` is the instrument. Its six numbers are incremented
inside `torchnative/rust/vendor/candle-core`'s Metal backend, at the doors candle itself opens:

    compute_encoders     MetalDevice::command_encoder()      kernel launches
    blit_encoders        MetalDevice::blit_command_encoder() device copies
    host_uploads(+bytes) MetalDevice::new_buffer_with_data() host -> device
    host_downloads(+b)   MetalStorage::to_cpu()              device -> host

They are in the fork rather than in this crate on purpose, and the reason is
the same one that makes them worth having: a counter on *this* side of the FFI
boundary would count what `aten.rs` intended, and a host-computed twin intends
exactly what the real kernel intends. Only candle can say whether a compute
encoder was opened.

**What `compute_encoders` is, narrowly.** A successful `command_encoder()`
call, which candle hands straight to a `candle_metal_kernels::call_*` that
encodes at least one `dispatch_thread*`. So it is a *lower bound* on GPU
dispatches and an exact count of candle's GPU op invocations. It is not a count
of `dispatch_threads`: those live in `candle-metal-kernels`, a crate this
vendoring does not cover (docs/devices/matrix.md §7.11). That distinction is
written down because the number would otherwise be quoted as something it is
not -- the failure `AGENTS.md` §13.1 records for the "four rounds"/"five rounds"
inflation.

**What each test here is for, and which mutant kills it.** A counter test is
worth nothing if the counter can be faked, so three of the four below are about
the instrument rather than about `abs`:

* `test_abs_on_mps_opens_a_metal_kernel_and_reads_nothing_back`
      the deliverable. Values against a subprocess oracle *and* a counter
      delta, so neither a wrong answer nor a right answer computed in the wrong
      place passes. Killed by: a host-computed twin planted in `abs` (verified
      -- see docs/devices/matrix.md §7.12), and by gutting either counter.
* `test_a_cpu_op_moves_no_metal_counter`
      kills the mutant that makes the instrument useless by making it say yes
      to everything: a `fetch_add` on a path every op takes, or a counter that
      increments on read.
* `test_a_readback_costs_exactly_the_tensors_bytes`
      calibration. `host_download_bytes` must equal the tensor's real size, so
      a download counter that only ever adds 1 or adds a constant is caught.
      Without this, "downloads == 0" above would be a claim about a number
      nobody has shown can move.
* `test_metal_counters_answers_with_all_six_names`
      the shape contract, and the honest `built is False` answer off Apple.
      This one is weak on purpose and is the only one here that a gutted
      implementation returning six zeros would survive; the three above are
      what stops that.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip

sys.path.insert(0, os.path.join(
    str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())), "tests", "golden"))
import dtypes as dt_utils  # noqa: E402

_NAMES = ("compute_encoders", "blit_encoders", "host_uploads",
          "host_upload_bytes", "host_downloads", "host_download_bytes")

# The four dtypes `abs` reaches on Metal in this build, from
# docs/devices/matrix.md §7.3. `float32` takes candle's `Tensor::abs`;
# `int64` takes `integral_abs_on_device`'s `maximum(x, 0 - x)`, which is a
# different code path and therefore a different thing to prove ran on the GPU.
_MPS_DTYPES = ("float32", "float16", "bfloat16", "int64")
_VALUES = [-3.0, -1.0, 0.0, 2.0, -7.0, 5.0]
_SHAPE = [2, 3]

_ORACLE_ABS = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for name in req["dtypes"]:
    t = torch.tensor(req["values"]).reshape(req["shape"]).to(getattr(torch, name))
    r = t.abs()
    out[name] = {
        "values": [float(v) for v in r.to(torch.float64).flatten().tolist()],
        "dtype": str(r.dtype).split(".")[-1],
    }
print(json.dumps(out))
"""


def _oracle(script, payload):
    """Upstream's answer, from an interpreter that has no `_C` on its path."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(payload)],
                          capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n" + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _mps_or_skip(what):
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


def _counters():
    c = _C._metal_counters()
    assert c["built"], "this build has no Metal backend"
    return {n: c[n] for n in _NAMES}


def _delta(before, after):
    return {n: after[n] - before[n] for n in _NAMES}


def _close(a, b, atol, rtol):
    if len(a) != len(b):
        return "length %d vs %d" % (len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x != x and y != y:
            continue
        if abs(x - y) > atol + rtol * abs(y):
            return "index %d: upstream=%r shim=%r" % (i, y, x)
    return None


def test_metal_counters_answers_with_all_six_names():
    """The shape contract: six names, `built`, and no exception anywhere.

    Off Apple every counter is `None` and not `0`, because a zero would read
    as "the GPU ran nothing" when the truth is "there is no Metal here".
    """
    c = _C._metal_counters()
    assert isinstance(c, dict), type(c)
    assert set(c) == set(_NAMES) | {"built"}, sorted(c)
    if c["built"]:
        for n in _NAMES:
            assert isinstance(c[n], int) and c[n] >= 0, (n, c[n])
    else:
        for n in _NAMES:
            assert c[n] is None, (n, c[n])
    # Reading must not perturb what it measures.
    again = _C._metal_counters()
    assert again == c, (c, again)


def test_a_cpu_op_moves_no_metal_counter():
    """A CPU op moves nothing -- the test that keeps the instrument specific.

    An instrument that says yes to everything proves nothing, and the cheapest
    way to break the one above is to put the `fetch_add` somewhere every
    dispatch passes. This fails immediately if that happens.
    """
    if not _C._metal_counters()["built"]:
        _skip.skip("not an Apple build: no Metal counters to move")
        return
    cpu = _C.device("cpu")
    t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), _C.float32, cpu)
    before = _counters()
    for _ in range(4):
        r = _C._aten_dispatch("aten.abs.default", t)
        _C._aten_dispatch("aten.mul.Tensor", r, r)
    after = _counters()
    d = _delta(before, after)
    assert all(v == 0 for v in d.values()), (
        "a cpu-only op moved a Metal counter: %r -- the counter is not "
        "measuring Metal work" % (d,))


def test_a_readback_costs_exactly_the_tensors_bytes():
    """`.cpu()` is one download of the tensor's real size.

    Calibration, and it is what gives the zero in the test below its meaning:
    a download counter that could never move would make "downloads == 0" a
    tautology. The byte count is asserted exactly -- a stub that adds a
    constant, or adds `1`, fails here.
    """
    mps = _mps_or_skip("the readback calibration")
    if mps is None:
        return
    n = _SHAPE[0] * _SHAPE[1]
    t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), _C.float32, mps)
    before = _counters()
    host = t.cpu()
    after = _counters()
    d = _delta(before, after)
    assert d["host_downloads"] == 1, (
        "one `.cpu()` should be exactly one download, got %r" % (d,))
    assert d["host_download_bytes"] == 4 * n, (
        "a float32 %r readback should cost %d bytes, got %r"
        % (_SHAPE, 4 * n, d))
    assert d["compute_encoders"] == 0, (
        "a readback is a blit, not a kernel: %r" % (d,))
    assert [float(v) for v in host.to(_C.float64).flatten().tolist()] == _VALUES


def test_abs_on_mps_opens_a_metal_kernel_and_reads_nothing_back():
    """The deliverable: `abs` on `mps` agrees with upstream **and** ran there.

    Two assertions that no single defect satisfies at once.

    *agrees* -- element-wise against an oracle computed in a separate
    subprocess with no `_C` on its path, at `tests/golden/dtypes.py`'s
    tolerance for the result dtype, not widened here.

    *ran on the GPU* -- `compute_encoders` rose across the dispatch and
    `host_downloads` did not move. A host-computed twin satisfies the first
    and fails the second, which is precisely the substitution that stayed
    invisible on this backend until this counter existed
    (docs/devices/matrix.md §7.12 records it being performed).

    The operand is built *before* the snapshot, so the upload that building it
    costs is not charged to `abs`, and the result is read back *after*, so the
    readback is not either.
    """
    mps = _mps_or_skip("the abs-on-mps dispatch count")
    if mps is None:
        return
    oracle = _oracle(_ORACLE_ABS,
                     {"dtypes": list(_MPS_DTYPES), "values": _VALUES,
                      "shape": _SHAPE})
    checked = 0
    for name in _MPS_DTYPES:
        t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE),
                                 dt_utils.c_dtype(_C, name), mps)
        before = _counters()
        r = _C._aten_dispatch("aten.abs.default", t)
        after = _counters()
        d = _delta(before, after)
        assert str(r.device).startswith("mps"), (name, str(r.device))
        assert d["compute_encoders"] >= 1, (
            "aten.abs %s on mps opened no Metal compute encoder: %r. The "
            "values may still be right -- that is the point of this "
            "assertion." % (name, d))
        assert d["host_downloads"] == 0, (
            "aten.abs %s on mps moved %d tensor(s) to the host mid-kernel "
            "(%d bytes): it is computing on the CPU under an mps label"
            % (name, d["host_downloads"], d["host_download_bytes"]))
        ref = oracle[name]
        tol = dt_utils.tolerance_for(ref["dtype"])
        got = [float(v) for v in r.cpu().to(_C.float64).flatten().tolist()]
        why = _close(got, ref["values"], tol.atol, tol.rtol)
        assert why is None, "aten.abs %s on mps: %s" % (name, why)
        checked += 1
    assert checked == len(_MPS_DTYPES), checked


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_metalcount")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
