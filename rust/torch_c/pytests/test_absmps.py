"""`abs` on Metal: the kernel stopped reading its tensor back to the host.

`docs/devices/matrix.md` publishes a (dtype x device) cell for every operator
this build implements, and `aten.abs.default` / `aten.abs_.default` read
`REFUSES` in **all eight** `mps` columns. That is a three-line hit:

    import torch
    x = torch.randn(1024, 1024, device="mps")
    x.abs()                       # NotImplementedError, before this round

The refusal was correct at the time and is the one `device.rs`'s
`mps_host_readback_gate` raises: a kernel that moves its operand's bytes to
the host computes on the CPU under an `mps` label, and this repository refuses
that rather than doing it silently (docs/devices/MPS.md §2). But it was *op*
granular, and `abs`'s body was only half a readback -- the floating path was
already one candle op (`Tensor::abs`), while the integral path went out
through `to_vec1::<i64>()` and a scalar loop. The whole operator was refused
on account of the integral half.

**The fix removes the readback rather than narrowing the gate**, because
narrowing the gate is the defeat `docs/devices/MPSATTN.md` §3.1 records: the
refusal list is *derived* by a source scan over each kernel's body
(`test_shim.py::test_the_mps_readback_list_is_what_the_kernels_actually_do`),
and anything that takes a name off the list while leaving the `to_vec1` in
place is a lie the scan is specifically there to catch. `integral_abs_on_device`
is `maximum(x, 0 - x)` -- two candle binary ops, both of which have real
integer arms (`op.rs`'s `bin_op!`, unlike `unary_op!`, whose integer arms are
`todo!()`, which is why the negation is a subtraction from zero and not
`Tensor::neg`).

**`abs_` was a different cell and stayed REFUSES on mps until 2026-09-20.**
Its kernel lost the readback too, so it left the list -- but `write_back`
refused an in-place write to a non-CPU tensor, which was a separate gate.
`docs/devices/matrix.md` §7.7 lifted that gate for a contiguous receiver
(`tensor.rs::write_on_device`, candle's `slice_set` onto a Metal blit), so
`abs_` now AGREES on all four `mps` dtypes Metal allows and the test below is
an agreement test rather than a refusal test. That change is recorded here
rather than silently made: this file landed green on 2026-09-19, and a test
that starts asserting the opposite of what it asserted needs the reason
written down beside it.

**What this file proves, and at which grade.**

* *agrees* -- element-wise against upstream, for every dtype on both devices,
  with the oracle computed in a **separate subprocess** that has no `_C` on
  its path at all (`_oracle_abs` below). The tolerance is
  `tools/golden/dtypes.py`'s for the result dtype and is not widened here.
* *placement* -- the honest statement is narrower than a CUDA or Vulkan cell's
  and this file says so rather than overclaiming. This build has **no Metal
  dispatch counter**: `device.rs` has `_cuda_counters()` and `_vulkan_counters()`
  and nothing of the kind for Metal, and candle's `MetalDevice` exposes no
  countable kernel launch. So the placement evidence here is (a) the kernel
  performs no host readback *by construction*, which the derivation scan in
  `test_shim.py` re-derives from source on every gate run, and (b)
  `test_abs_on_mps_does_not_come_back_through_the_readback_gate`, which checks
  the artefact's own table rather than the source constant. Both are the
  standard `softmax_on_device` was landed at (docs/devices/MPSATTN.md §3.1).
  A Metal counter is the piece of infrastructure that would raise every `mps`
  cell in `matrix.md` above this ceiling, and it is not built here.

Nullifications this file is meant to catch, each verified by making the break:

* `integral_abs_on_device` returning its input unchanged
      -> test_abs_agrees_with_upstream_on_both_devices (the negative values)
* `integral_abs_on_device` losing its unsigned identity arm
      -> the same test, on the uint8 row
* the wrapping arm changing (`abs(INT_MIN)` becoming `0` or saturating)
      -> test_abs_wraps_at_the_signed_minimum_exactly_as_upstream_does
* `aten.abs.default` going back on `MPS_HOST_READBACK_OPS`
      -> test_abs_on_mps_does_not_come_back_through_the_readback_gate
* `abs_` on `mps` regressing to a refusal, or coming back through the
  readback gate
      -> test_abs_inplace_agrees_with_upstream_on_cpu_and_on_mps
* a `to_vec1` reappearing in the kernel while the name stays off the list
      -> test_shim.py's derivation scan, which is why this file does not
         restate it
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools", "golden"))
import dtypes as dt_utils  # noqa: E402

# Real inputs, not a smoke shape: negatives on both sides of zero, a zero, and
# values that survive float16 exactly so the float16 row tests `abs` rather
# than testing rounding.
_VALUES = [-3.0, -1.0, 0.0, 2.0, -7.0, 5.0]
# `uint8` cannot hold a negative at all, and `_tensor_from_flat` and upstream
# disagree about what `-3.0 -> uint8` means (0 here, 253 there) -- a fact about
# the *constructor*, not about `abs`, and testing it here would make this file
# fail for a reason it is not about. So the unsigned row gets its own operands.
# They still catch the mutant that matters: dropping the unsigned identity arm
# makes `abs` compute `maximum(x, 0 - x)`, and `0u8 - 5` wraps to 251, so every
# nonzero value below comes back wrong.
_UNSIGNED_VALUES = [3.0, 1.0, 0.0, 2.0, 7.0, 5.0]
_SHAPE = [2, 3]


def _values_for(name):
    return _UNSIGNED_VALUES if name == "uint8" else _VALUES

# The dtypes `abs` is defined for, per device. `bool` is absent on purpose:
# upstream refuses it (`"abs_cpu" not implemented for 'Bool'`) and so does this
# shim, and that refusal is already pinned in test_shim.py.
_CPU_DTYPES = ("float32", "float16", "bfloat16", "float64",
               "int64", "int32", "int8", "uint8")
# `int32`/`int8` are absent from the mps row because this build refuses them on
# Metal for their own reasons, both already named and tested elsewhere:
# `_shim_mps_unsupported_int_dtypes` (test_intmps.py) and the candle fork's
# CPU-only `DType::I8` (docs/numerics/INT8.md §1.2). `float64` is absent
# because Metal has no double (docs/devices/matrix.md §3.1). Putting any of
# them here would make this file fail for a reason it is not about.
_MPS_DTYPES = ("float32", "float16", "bfloat16", "int64")


def _oracle(script, payload):
    """Upstream's answer, computed in a subprocess with no `_C` anywhere.

    The subprocess is the point. A same-process oracle shares an interpreter
    with the artefact under test, and this repository has already been bitten
    by a probe that silently imported the wrong `torch` (CLAUDE.md §3). Here
    the child's `PYTHONPATH` is emptied, so the only `torch` it can find is the
    real one in site-packages, and it asserts that what it got has **no**
    `_aten_implemented` -- i.e. that it is upstream and not the shim.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(payload)],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n" + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


_ORACLE_ABS = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for name, values in req["values"].items():
    t = torch.tensor(values).reshape(req["shape"]).to(getattr(torch, name))
    r = t.abs()
    out[name] = {
        "values": [float(v) for v in r.to(torch.float64).flatten().tolist()],
        "dtype": str(r.dtype).split(".")[-1],
    }
print(json.dumps(out))
"""

_ORACLE_WRAP = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for name, lo in req["mins"].items():
    t = torch.tensor([lo], dtype=getattr(torch, name))
    out[name] = int(t.abs().item())
print(json.dumps(out))
"""


def _mps_or_skip(what):
    """A live `mps` device, or None having said why not.

    Deliberately a local copy rather than an import from `test_dtmdev`: the
    skip line names the file that skipped, which is what
    docs/devices/VULKAN3.md §6.1 cost when it did not.
    """
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


def _shim_abs(name, device):
    t = _C._tensor_from_flat(list(_values_for(name)), list(_SHAPE),
                             dt_utils.c_dtype(_C, name), device)
    r = _C._aten_dispatch("aten.abs.default", t)
    host = r.cpu() if str(r.device) != "cpu" else r
    return [float(v) for v in host.to(_C.float64).flatten().tolist()], r


def _close(a, b, atol, rtol):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != x and y != y:
            continue
        if abs(x - y) > atol + rtol * abs(y):
            return "index %d: upstream=%r shim=%r (|diff|=%r > atol=%r rtol=%r)" % (
                i, y, x, abs(x - y), atol, rtol)
    if len(a) != len(b):
        return "length %d vs %d" % (len(a), len(b))
    return None


def test_abs_agrees_with_upstream_on_both_devices():
    """Element-wise agreement, cpu and mps, against a subprocess oracle.

    Grade *agrees*. The comparison is by value against upstream's own bytes,
    so a kernel that returned its input unchanged, returned zeros, or dropped
    the unsigned identity arm all fail here -- `_VALUES` has negatives, a zero
    and positives, and `uint8` holds the negatives as wrapped large values
    that `maximum(x, 0 - x)` would corrupt.
    """
    cpu = _C.device("cpu")
    mps = _mps_or_skip("abs agreement on mps")
    plan = [("cpu", cpu, _CPU_DTYPES)]
    if mps is not None:
        plan.append(("mps", mps, _MPS_DTYPES))
    wanted = sorted(set(_CPU_DTYPES) | set(_MPS_DTYPES))
    oracle = _oracle(_ORACLE_ABS,
                     {"values": {n: _values_for(n) for n in wanted},
                      "shape": _SHAPE})
    checked = 0
    for label, device, names in plan:
        for name in names:
            got, tensor = _shim_abs(name, device)
            ref = oracle[name]
            tol = dt_utils.tolerance_for(ref["dtype"])
            why = _close(got, ref["values"], tol.atol, tol.rtol)
            assert why is None, "aten.abs %s_%s: %s" % (name, label, why)
            assert str(tensor.device).startswith(label), (
                "aten.abs %s_%s answered on %s under a %s label"
                % (name, label, tensor.device, label))
            checked += 1
    assert checked >= len(_CPU_DTYPES), checked


def test_abs_wraps_at_the_signed_minimum_exactly_as_upstream_does():
    """`abs(INT_MIN)` is `INT_MIN`, and this is where `maximum(x, 0-x)` earns it.

    The scalar loop this replaced spelled the same thing as `wrapping_abs`. A
    rewrite that saturated, promoted, or returned zero would be invisible to
    every other test in this file, because `_VALUES` has no extreme in it.
    """
    mins = {"int8": -128, "int32": -2147483648, "int64": -9223372036854775808}
    oracle = _oracle(_ORACLE_WRAP, {"mins": mins})
    cpu = _C.device("cpu")
    for name, lo in mins.items():
        t = _C._tensor_from_flat([float(lo)], [1], dt_utils.c_dtype(_C, name), cpu)
        r = _C._aten_dispatch("aten.abs.default", t)
        got = int(r.to(_C.float64).flatten().tolist()[0])
        assert got == oracle[name], (
            "aten.abs %s at its minimum: shim=%r upstream=%r" % (name, got, oracle[name]))


def test_abs_on_mps_does_not_come_back_through_the_readback_gate():
    """The *artefact's* refusal table, not the constant in the source.

    A test that read `device.rs` on both sides would agree with itself after a
    change nobody rebuilt. `_shim_mps_host_readback_ops()` is the table the
    loaded `.so` is actually gating on.
    """
    refused = set(_C._shim_mps_host_readback_ops())
    for op in ("aten.abs.default", "aten.abs_.default"):
        assert op not in refused, (
            "%s is refused on mps again. If the kernel really did reacquire a "
            "host readback, that refusal is correct and this test should move "
            "with it -- but check first, because the whole point of "
            "integral_abs_on_device is that it has none." % op)
    allowed = set(_C._shim_mps_readback_but_allowed())
    assert "aten.abs.default" not in allowed, (
        "abs must not be *exempted* from the gate -- it must have nothing to "
        "exempt. An exemption here would be the MPSATTN.md §3.1 defeat.")


def test_abs_inplace_agrees_with_upstream_on_cpu_and_on_mps():
    """`abs_` is `abs`'s value through the receiver, now on both devices.

    **This test changed shape on 2026-09-19 and the previous version said to.**
    It used to assert that `abs_` *refuses* on `mps`, and it did: `abs_`'s
    kernel had lost its host readback along with `abs`'s, but the write itself
    stopped at `write_back`'s "writing through a view is implemented for the
    CPU backend only". Its docstring said that if the op ever started
    succeeding, this should become an agreement test provided the success was
    a device write and not a silent host round trip. It is -- see
    `test_mpsinplace.py`, which lifts that gate for a contiguous receiver by
    routing the write through candle's `slice_set` and pins the no-readback
    property structurally (docs/devices/matrix.md §7.7).

    So this is now grade *agrees* on both devices, with the oracle in a
    separate subprocess, and it keeps the two negative claims that were worth
    keeping: the receiver must come back on `mps`, and the readback sentence
    must not reappear.
    """
    wanted = sorted(set(_CPU_DTYPES) | set(_MPS_DTYPES))
    oracle = _oracle(_ORACLE_ABS,
                     {"values": {n: _values_for(n) for n in wanted},
                      "shape": _SHAPE})
    cpu = _C.device("cpu")
    for name in _CPU_DTYPES:
        t = _C._tensor_from_flat(list(_values_for(name)), list(_SHAPE),
                                 dt_utils.c_dtype(_C, name), cpu)
        r = _C._aten_dispatch("aten.abs_.default", t)
        ref = oracle[name]
        tol = dt_utils.tolerance_for(ref["dtype"])
        for who, which in (("returned", r), ("receiver", t)):
            got = [float(v) for v in which.to(_C.float64).flatten().tolist()]
            why = _close(got, ref["values"], tol.atol, tol.rtol)
            assert why is None, "aten.abs_ %s_cpu (%s): %s" % (name, who, why)

    mps = _mps_or_skip("abs_ agreement on mps")
    if mps is None:
        return
    for name in _MPS_DTYPES:
        t = _C._tensor_from_flat(list(_values_for(name)), list(_SHAPE),
                                 dt_utils.c_dtype(_C, name), mps)
        try:
            r = _C._aten_dispatch("aten.abs_.default", t)
        except NotImplementedError as e:
            msg = str(e).splitlines()[0]
            assert "reads the tensor back to host memory" not in msg, (
                "aten.abs_ %s_mps is refused by the host-readback gate again: "
                "%r" % (name, msg))
            raise AssertionError(
                "aten.abs_ %s_mps refuses again: %r. The device write door in "
                "tensor.rs::write_on_device is what makes it work; if that was "
                "removed, this is the regression." % (name, msg))
        ref = oracle[name]
        tol = dt_utils.tolerance_for(ref["dtype"])
        assert r is t, (
            "aten.abs_ %s_mps returned a different object from the receiver"
            % name)
        assert str(t.device).startswith("mps"), (
            "aten.abs_ %s_mps left the receiver on %s" % (name, t.device))
        got = [float(v) for v in t.cpu().to(_C.float64).flatten().tolist()]
        why = _close(got, ref["values"], tol.atol, tol.rtol)
        assert why is None, "aten.abs_ %s_mps (receiver): %s" % (name, why)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_absmps")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
