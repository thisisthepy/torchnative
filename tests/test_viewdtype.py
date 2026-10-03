"""`aten.view.dtype`: the one silent CPU fallback the matrix found.

`docs/devices/matrix.md` §4.1 recorded it and did not fix it. Measured again
on this branch, before this round's change, against the staged artefact:

    input.device=mps:0  ->  aten.view.dtype  ->  result.device=cpu
    float32->int32   in=mps:0  OUT=cpu
    int64->float64   in=mps:0  OUT=cpu
    int32->float32   in=mps:0  OUT=cpu

The values were right and the label was right on the way in. That is the
failure shape this namespace exists to prevent, and it is worse here than a
wrong number would be: because the *output* is a cpu tensor, everything
downstream of it leaves the device too, silently.

**Where the silence came from, exactly.** `view_dtype` in `aten.rs` is

    let bytes = crate::tensor::to_le_bytes(OP, input.tensor()?)?;
    let wrapped = crate::tensor::from_le_bytes(OP, &bytes, &dims, want)?;

`to_le_bytes` is a `to_vec1` per dtype -- a host readback -- and
`from_le_bytes` opens with `let device = candle_core::Device::Cpu;`, so the
result is constructed on the host whatever the input's device was.

`device.rs::MPS_HOST_READBACK_OPS` is the gate that refuses exactly this, and
it did not fire, **because the list is derived by a scan of `aten.rs` alone.**
The scan follows helper calls one level and by name, and all six names it
knows are functions defined in `aten.rs`. `to_le_bytes` lives in `tensor.rs`.
So the readback was one file away from every check that looks for it -- the
cross-file version of the defeat `docs/devices/MPSATTN.md` §3.1 records. The
classifier bug had to be fixed before the op could even be seen; this round
fixes it in `test_shim.py::_cross_file_readback_helpers` and the per-op
derivation now reaches across files by qualified path.

**Why this refuses rather than running on the device.** `view.dtype`
reinterprets bytes. candle 0.11.0 offers no bit-reinterpretation of a
`Tensor`, on any backend, and its Metal storage is not exposed at a level
where a buffer could be re-tagged; a device-resident implementation is a
Metal shader plus a candle-fork API, not a kernel edit. And one of the four
measured pairs cannot be done on Metal at *any* effort: `int64->float64`
asks for a double, which Metal does not have -- §3.1 of the matrix refuses
`float64` by name on all the other roads onto the device, and this op was the
road that stayed open. So the honest answer is a refusal that names the
reason, which is this repository's standing answer for a kernel that cannot
be written honestly (AGENTS.md §18).

**Evidence, per device, stated separately rather than blurred.**

* `mps` -- **structural, plus the artefact's own table.** This build has no
  Metal dispatch counter: `device.rs` exposes `_cuda_counters()` and
  `_vulkan_counters()` and nothing countable for Metal, because candle's
  `MetalDevice` exposes no kernel launch to count. So the claim here is (a)
  the op is refused before its kernel runs, checked against
  `_C._shim_mps_host_readback_ops()` -- the table *the loaded artefact* gates
  on, not the constant beside it -- and (b) the refusal is re-derived from
  `aten.rs` + `tensor.rs` on every gate run, so taking the name off the list
  without removing the `to_le_bytes` puts it straight back.
* `cuda` -- same list. `CUDA_HOST_READBACK_OPS` is an alias of
  `MPS_HOST_READBACK_OPS`, so the same one-line addition closes it, and
  `_cuda_counters()` reports the refusal. **Not measured: there is no CUDA
  device on this machine.** The claim is structural only and this file says
  so instead of implying a run.
* `vulkan` -- already refused, and not by this change. That backend is an
  allowlist (`vulkan::dispatch`), `aten.view.dtype` is not on it, and
  `PyTensorBase::tensor()` refuses to read a `VkBuffer` as CPU storage even
  if a kernel were reached by another route. Asserted below from the
  allowlist rather than from a run.
* `cpu` -- unaffected, and pinned at grade *agrees* against a subprocess
  oracle. This is the path that matters in practice: a safetensors checkpoint
  arrives as host bytes, is `view`ed into its dtype on the CPU, and is moved
  to the device afterwards.

Nullifications this file is meant to catch:

* `aten.view.dtype` coming off `MPS_HOST_READBACK_OPS`
      -> test_view_dtype_refuses_on_mps_rather_than_answering_from_the_host
         (runtime, needs Metal) and
         test_view_dtype_is_refused_by_the_artefacts_own_table (everywhere)
* the cross-file derivation being reverted to `aten.rs`-only
      -> test_shim.py's per-op derivation, which then reports the op as stale
* the byte round-trip being replaced by a numeric conversion
      -> test_view_dtype_agrees_with_upstream_on_cpu (`1.0 as int32` is
         1065353216, not 1)
* the three shape refusals being dropped
      -> test_view_dtype_keeps_upstreams_three_refusals
"""

import json
import os
import re
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import test_shim as _shim_tests
import _skip

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "torchnative", "rust", "torch_c", "src")

# Four (source dtype, target dtype) pairs. Three are the ones §4.1 measured as
# silently answering on the host; `bool->uint8` is the fourth it named and is
# kept because it is the only equal-width pair among them, so it exercises the
# path that does *not* go through the dimension arithmetic.
#
# The payloads are chosen so a numeric conversion cannot be mistaken for a
# reinterpretation: `1.0` as int32 is 1065353216, and a byte-for-byte view is
# the only route that says so.
_PAIRS = (
    ("float32", "int32", [1.0, -2.0, 0.5, 0.0, 3.25, -1.5]),
    ("int32", "float32", [1065353216, -1, 0, 1, 2139095040, 1078530011]),
    ("int64", "float64", [4607182418800017408, 0, -1, 1, 4611686018427387904, 7]),
    ("uint8", "uint8", [0, 1, 255, 128, 7, 64]),
)
_SHAPE = [2, 3]


def _oracle(script, payload):
    """Upstream's answer, computed in a subprocess with no `_C` on its path.

    A same-process oracle shares an interpreter with the artefact under test,
    and the child asserts it got upstream (`_aten_implemented` absent) rather
    than assuming it.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(payload)],
                          capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n" + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


_ORACLE_VIEW = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = []
for src, dst, values in req["pairs"]:
    t = torch.tensor(values, dtype=getattr(torch, src)).reshape(req["shape"])
    r = t.view(getattr(torch, dst))
    out.append({
        "src": src, "dst": dst,
        "shape": list(r.shape),
        "dtype": str(r.dtype).split(".")[-1],
        "values": [float(v) for v in r.to(torch.float64).flatten().tolist()],
    })
print(json.dumps(out))
"""


def _mps_or_skip(what):
    """A live `mps` device, or None having said why not, naming this file."""
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return None
    return _C.device("cpu"), _C.device("mps")


def _source(name):
    with open(os.path.join(_SRC, name), encoding="utf-8") as handle:
        return handle.read()


def test_view_dtype_agrees_with_upstream_on_cpu():
    """Grade *agrees*, element-wise, against a subprocess oracle.

    Exact equality, not a tolerance. `view.dtype` is a bit-level operation
    with a single right answer per element, so the derived tolerance from
    `tests/golden/dtypes.py` would be *looser* than the contract -- this
    narrows it rather than widening it, which is the only direction allowed.
    NaN is compared as NaN-for-NaN and its payload is not inspected; the
    inputs above include `0x7F800000` (inf) rather than relying on that.
    """
    want = _oracle(_ORACLE_VIEW, {"pairs": [list(p) for p in _PAIRS],
                                 "shape": _SHAPE})
    cpu = _C.device("cpu")
    assert len(want) == len(_PAIRS)
    for expected, (src, dst, values) in zip(want, _PAIRS):
        t = _C._tensor_from_flat([float(v) for v in values], list(_SHAPE),
                                 getattr(_C, src), cpu)
        r = _C._aten_dispatch("aten.view.dtype", t, getattr(_C, dst))
        assert str(r.device) == "cpu", (src, dst, str(r.device))
        assert str(r.dtype).split(".")[-1] == expected["dtype"], (
            src, dst, str(r.dtype), expected["dtype"])
        assert list(r.shape) == expected["shape"], (
            src, dst, list(r.shape), expected["shape"])
        got = [float(v) for v in r.to(_C.float64).flatten().tolist()]
        assert len(got) == len(expected["values"]), (src, dst, len(got))
        for i, (a, b) in enumerate(zip(got, expected["values"])):
            if a != a and b != b:
                continue
            assert a == b, (
                "aten.view.dtype %s->%s index %d: upstream=%r shim=%r -- a "
                "byte reinterpretation has one right answer and this is not "
                "it (a numeric conversion would give %r)"
                % (src, dst, i, b, a, float(values[i])))


def test_view_dtype_is_refused_by_the_artefacts_own_table():
    """The op is on the host-readback list the loaded artefact gates on.

    Reads `_C._shim_mps_host_readback_ops()` rather than the constant in
    `device.rs`: a test that read the source on both sides would agree with
    itself after a change nobody rebuilt. Runs everywhere, Metal or not.
    """
    refused = _C._shim_mps_host_readback_ops()
    assert "aten.view.dtype" in refused, (
        "aten.view.dtype is not refused on mps/cuda. Its kernel is "
        "`to_le_bytes` + `from_le_bytes`, and `from_le_bytes` constructs on "
        "`Device::Cpu` -- so with the name off this list it answers a device "
        "dispatch from the host and hands back a cpu tensor. If a "
        "device-resident byte reinterpretation landed, this test should "
        "become an agreement test on mps, not be deleted.")
    assert "aten.view.dtype" not in _C._shim_mps_readback_but_allowed(), (
        "aten.view.dtype was moved to the exemption list. The two exemptions "
        "are `.item()` and `uniform_`, and neither shape fits: this op "
        "returns a Tensor and its readback is not what the caller asked for.")


def test_view_dtype_refuses_on_mps_rather_than_answering_from_the_host():
    """The runtime half: every pair that used to answer `cpu` now refuses.

    Before this round all three placeable pairs returned a **cpu** tensor from
    an **mps** input. The assertion is deliberately two-sided -- it is not
    enough that an exception arrives, it has to be the readback refusal naming
    this op, and the op must not have quietly succeeded with a cpu result.
    """
    devices = _mps_or_skip("view.dtype's refusal on mps")
    if devices is None:
        return
    cpu, mps = devices
    tried = 0
    for src, dst, values in _PAIRS:
        host = _C._tensor_from_flat([float(v) for v in values], list(_SHAPE),
                                    getattr(_C, src), cpu)
        try:
            t = _C._aten_dispatch("aten._to_copy.default", host, device=mps)
        except (NotImplementedError, RuntimeError):
            # Metal cannot hold every dtype (no double, no I8/I16/I32 kernels).
            # A pair that cannot be placed is not this test's subject.
            continue
        tried += 1
        try:
            r = _C._aten_dispatch("aten.view.dtype", t, getattr(_C, dst))
        except NotImplementedError as e:
            msg = str(e).splitlines()[0]
            assert "aten.view.dtype" in msg, (src, dst, msg)
            assert "reads the tensor back to host memory" in msg, (
                "refused, but not for the readback reason: %r" % msg)
            assert "mps" in msg, (src, dst, msg)
        else:
            raise AssertionError(
                "aten.view.dtype %s->%s on an mps input returned a %s tensor "
                "instead of refusing -- this is docs/devices/matrix.md §4.1's "
                "silent CPU fallback, back again."
                % (src, dst, str(r.device)))
    assert tried >= 2, (
        "no dtype pair could be placed on mps, so this test proved nothing "
        "about the device; tried=%d" % tried)


def test_view_dtype_keeps_upstreams_three_refusals():
    """The shape refusals are upstream's own sentences and stay that way.

    They are the reason the kernel checks `stride(-1) == 1` even though it
    then makes a contiguous copy, and a round that "fixed" the fallback by
    loosening the kernel would show up here.
    """
    cpu = _C.device("cpu")
    zero = _C._tensor_from_flat([1.0], [], _C.float32, cpu)
    try:
        _C._aten_dispatch("aten.view.dtype", zero, _C.uint8)
    except RuntimeError as e:
        assert "self.dim() cannot be 0" in str(e), str(e)
    else:
        raise AssertionError("0-dim view to a different width must refuse")

    odd = _C._tensor_from_flat([1.0, 2.0], [2], _C.uint8, cpu)
    try:
        _C._aten_dispatch("aten.view.dtype", odd, _C.float32)
    except RuntimeError as e:
        assert "must be divisible by 4" in str(e), str(e)
    else:
        raise AssertionError("an indivisible last dim must refuse")


def test_no_aten_kernel_reaches_a_cross_file_readback_unrefused():
    """The general form of the hole `view.dtype` fell through.

    The per-op derivation in `test_shim.py` follows helpers **across files**
    now. This asserts the property that made the fix necessary and would make
    it necessary again: no kernel in the dispatch table reaches a host
    readback through a function defined in another module without being on the
    refusal list. It is source-level, so a machine with no Metal checks it too
    -- and those are the machines most likely to change a kernel under it.
    """
    parsed = _shim_tests._aten_rs_functions()
    if parsed is None:
        _skip.skip("   (skipped cross-file readback scan: torchnative/rust/torch_c/src is "
                   "not beside this file -- installed rather than in-tree)")
        return
    bodies, text = parsed
    ops = _shim_tests._aten_dispatch_targets(text)
    assert len(ops) > 200, len(ops)
    helpers = _shim_tests._cross_file_readback_helpers()
    assert ("tensor", "to_le_bytes") in helpers, (
        "`tensor::to_le_bytes` no longer reads bytes to the host, or the "
        "cross-file scan stopped finding it. If the first, `view.dtype` may "
        "have become device-resident and this whole file should be rewritten "
        "as an agreement test; if the second, the scan is blind again.")
    # And the linkage itself, not just the two ends of it. Without this the
    # whole test survives a `_reaches_cross_file_readback` that returns False
    # unconditionally -- measured: every assertion below still passed, because
    # the op is on the refusal list for other reasons and `unrefused` came out
    # empty. A scan that cannot connect a kernel to a helper is the blindness
    # this round fixed, so it is asserted directly.
    assert _shim_tests._reaches_cross_file_readback(
        bodies.get("view_dtype", ""), helpers), (
        "`view_dtype` no longer links to a cross-file readback helper. Either "
        "the kernel stopped calling `crate::tensor::to_le_bytes` -- in which "
        "case this file should become an mps agreement test -- or the scan "
        "went blind again, which is how §4.1 happened.")
    refused = set(_C._shim_mps_host_readback_ops())
    allowed = set(_C._shim_mps_readback_but_allowed())
    unrefused = {}
    for op, fn in ops.items():
        if op in refused or op in allowed:
            continue
        for mod, name in helpers:
            if re.search(r"(?:crate::)?" + mod + r"\s*::\s*" + name + r"\s*\(",
                         bodies.get(fn, "")):
                unrefused.setdefault(op, []).append("%s::%s" % (mod, name))
    assert not unrefused, (
        "these ops reach a host readback through a helper in another module "
        "and are not refused on mps/cuda -- the shape docs/devices/matrix.md "
        "§4.1 was: " + repr(sorted(unrefused.items())))


def test_vulkan_refuses_view_dtype_by_allowlist_not_by_this_rounds_change():
    """`vulkan` was never silent here, and the reason is worth pinning.

    That backend dispatches from an allowlist, so an op absent from it refuses
    naming itself. This test states the absence, so that adding `view.dtype`
    to the Vulkan table without a byte-reinterpreting shader would be a
    visible edit rather than a silent one.
    """
    src = _source("vulkan.rs")
    assert '"aten.view.default"' in src, (
        "vulkan.rs no longer names `aten.view.default`; this test is reading "
        "the wrong file or the allowlist moved")
    assert '"aten.view.dtype"' not in src, (
        "aten.view.dtype appeared in vulkan.rs. If a byte-reinterpreting "
        "shader landed, this should become an agreement test with the "
        "`_vulkan_counters()` assertion (shader_dispatches up, host_downloads "
        "unmoved); if not, it is the same silent fallback on a third device.")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_viewdtype")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
