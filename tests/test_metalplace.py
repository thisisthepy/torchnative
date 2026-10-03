"""The Metal counter pointed at the `mps` claims that landed without it.

`docs/devices/matrix.md` §7.11 built `_C._metal_counters()` and then said, in
its own words, what it had **not** done:

> no test in this round asserts a counter delta for `softmax_on_device`, for
> the §7.7 in-place family, or for any of the 43 operators §7.8 measured.
> Those remain where §7.5 put them until someone writes the assertion.

This file writes those assertions. It is an audit rather than a feature: no
kernel is added here and none is changed. What it produces is *counted*
evidence for claims that were graded on structural evidence -- "this kernel
performs no host readback, re-derived from source" -- and a **named record of
the eight operators where the counter says the structural evidence was
wrong** (§7.14).

**Why the counter and not the source scan.** Three Vulkan rounds and one Metal
round (§7.12) planted a host-computed twin of a device kernel and found every
value correct, every agreement test green, and the `.device` label unchanged.
The only instrument that saw it was a dispatch counter. So a cell whose
placement rests on reading `aten.rs` is resting on the thing that has failed
four times, and §7.14's twelve cells are that failure happening in production
rather than in a mutant.

**The bracket, stated because counters from different brackets read like
regressions side by side.** Every measurement below snapshots the counters
*immediately before* the single `_aten_dispatch` call under test and again
*immediately after*. Operands are built before the first snapshot, so the
upload that building them costs is not charged to the operator; results are
read back after the second, so the readback is not either. Nothing else runs
between the two reads.

**What `compute_encoders` is, and is not.** A successful
`MetalDevice::command_encoder()`, which candle hands to a
`candle_metal_kernels::call_*` that encodes at least one `dispatch_thread*`.
It is a **lower bound on GPU dispatches** and an exact count of candle's GPU
op invocations -- *not* a `dispatch_threads` count, which happens in
`candle-metal-kernels`, a crate this vendoring does not cover. `blit_encoders`
moves for a device-to-device copy *and* for a readback, because a readback
blits first; the two are not separated by this instrument. Consequently
`blit_encoders > 0` proves nothing on its own and is never asserted here as
evidence of device residency -- `host_downloads == 0` is.

**Why `compute_encoders >= 1` is not asserted for every cell.** Some
operators are correctly blit-only or metadata-only on the device: `zero_` is a
`const_set` plus a copy, `copy_` is a copy, `t_` is a layout change. Demanding
a compute encoder from them would be widening the claim to make it countable,
which §7.14 explicitly declines to do. For those the assertion is
`host_downloads == 0 and host_uploads == 0` -- a host-computed twin has to
move the bytes both ways and fails it.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import test_shim
from test_shim import _C
import _skip

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden"))
import dtypes as dt_utils  # noqa: E402

_NAMES = ("compute_encoders", "blit_encoders", "host_uploads",
          "host_upload_bytes", "host_downloads", "host_download_bytes")

_VALUES = [-3.0, -1.0, 0.5, 2.0, -7.0, 5.0]
_OTHER = [1.0, 2.0, 4.0, -1.0, 8.0, -2.0]
_SHAPE = [2, 3]

_FLOAT_DTYPES = ("float32", "float16", "bfloat16")
_ALL_DTYPES = ("float32", "float16", "bfloat16", "int64")


# ---------------------------------------------------------------------------
# §7.14's finding, as data -- and §7.16's fix, which is why the assertions
# below run in the opposite direction from the round that wrote them.
#
# Eight operators answered an `mps` dispatch by moving the operand to the
# host, computing there and uploading the answer. Twelve of their cells were
# published AGREES in §6's table. None of them was in
# `MPS_HOST_READBACK_OPS`, because the derivation followed exactly six helper
# names and each reached the host through a seventh (`order_along`,
# `argsort_core`, `floor_divide_impl`) or through another *operator's* kernel
# function (`scatter_inplace` calls `scatter_src`/`scatter_value`).
#
# **All eight are now refused** (§7.16), together with two more the deepened
# derivation found on its own (`linalg_vector_norm.default`,
# `norm.ScalarOpt_dim`, through `norm_pow_walk`). So what this file pins is no
# longer "they read back and nothing stops them" but "they are refused, and
# the refusal is why the counter no longer sees the download". The set is
# still data, still grown only by a visible edit, and still asserted in both
# directions -- a name silently leaving the refusal list while its readback
# stays is the docs/devices/MPSATTN.md §3.1 defeat and fails here.
#
# `test_mpsrefuse.py` owns the refusal itself and the note it carries. This
# file keeps the counter half: with the gate in place there is no dispatch to
# count, so the assertion is that the dispatch does not happen.
# ---------------------------------------------------------------------------
_HOST_COMPUTED_ON_MPS = {
    "aten.argsort.default": ("int64", "bool"),
    "aten.argsort.stable": ("int64", "bool"),
    "aten.floor_divide.Scalar": ("int64", "bool"),
    "aten.floor_divide.default": ("int64",),
    "aten.scatter_.src": ("int64",),
    "aten.scatter_.value": ("int64",),
    "aten.sort.default": ("int64", "bool"),
    "aten.topk.default": ("int64",),
}

# The two the deepened derivation added that §7.14 had not named. They are
# kept separate from the eight on purpose: the eight were found by a counter
# on a running device, these two by following calls in the source, and
# conflating the two kinds of evidence is how a number gets quoted into
# something it never measured (AGENTS.md §13.1).
_DERIVED_AS_WELL = ("aten.linalg_vector_norm.default", "aten.norm.ScalarOpt_dim")

_EXEMPT = ("aten._local_scalar_dense.default", "aten.uniform_.default")


def _counters():
    c = _C._metal_counters()
    assert c["built"], "this build has no Metal backend"
    return {n: c[n] for n in _NAMES}


def _delta(before, after):
    return {n: after[n] - before[n] for n in _NAMES}


def _mps_or_skip(what):
    if not _C._metal_counters()["built"]:
        _skip.skip("   (skipped %s: not an Apple build -- no Metal counters)" % what)
        return None
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


def _oracle(script, payload):
    """Upstream's answer, from an interpreter with no `_C` on its path."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(payload)],
                          capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n" + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _host(t):
    h = t.cpu() if str(t.device) != "cpu" else t
    return [float(v) for v in h.to(_C.float64).flatten().tolist()]


def _close(a, b, atol, rtol):
    if len(a) != len(b):
        return "length %d vs %d" % (len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x != x and y != y:
            continue
        if abs(x - y) > atol + rtol * abs(y):
            return "index %d: upstream=%r shim=%r (|diff|=%r > atol=%r rtol=%r)" % (
                i, y, x, abs(x - y), atol, rtol)
    return None


# ---------------------------------------------------------------------------
# 1. softmax
# ---------------------------------------------------------------------------

_ORACLE_SOFTMAX = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    dt = getattr(torch, case["dtype"])
    t = torch.tensor(case["values"]).reshape(case["shape"]).to(dt)
    if case["safe"]:
        r = torch.ops.aten._safe_softmax(t, case["dim"])
    else:
        r = torch.ops.aten._softmax(t, case["dim"], False)
    out[key] = {"values": [float(v) for v in r.to(torch.float64).flatten().tolist()],
                "dtype": str(r.dtype).split(".")[-1]}
print(json.dumps(out))
"""

_SOFTMAX_OPS = ("aten._softmax.default", "aten._safe_softmax.default")


def test_softmax_on_mps_opens_metal_kernels_and_reads_nothing_back():
    """`softmax_on_device`, counted -- the oldest `mps` claim in the document.

    MPSATTN.md §3.1 landed `softmax_on_device` at the structural standard: the
    kernel performs no host readback *by construction*, re-derived from
    `aten.rs` by the gate. §7.5 recorded that as weaker than a counter, and
    §7.11 built the counter without pointing it here.

    Pointed here it says: **five compute encoders per dispatch and zero host
    downloads**, on each of the three float dtypes Metal has, for both
    `_softmax` and `_safe_softmax`. Five, not one, because candle composes the
    reduction out of `max`, `broadcast_sub`, `exp`, `sum` and `broadcast_div`;
    the assertion is `>= 1` because that decomposition is candle's business
    and pinning it at five would make this test fail on a candle bump for a
    reason that is not a regression.

    Paired with element-wise agreement against a subprocess oracle, so neither
    a wrong answer nor a right answer computed in the wrong place passes.
    """
    mps = _mps_or_skip("the softmax dispatch count")
    if mps is None:
        return
    request = {}
    for op in _SOFTMAX_OPS:
        for name in _FLOAT_DTYPES:
            request["%s|%s" % (op, name)] = {
                "dtype": name, "values": _VALUES, "shape": _SHAPE, "dim": 1,
                "safe": op == "aten._safe_softmax.default"}
    oracle = _oracle(_ORACLE_SOFTMAX, request)

    checked = 0
    for op in _SOFTMAX_OPS:
        for name in _FLOAT_DTYPES:
            ctype = dt_utils.c_dtype(_C, name)
            t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), ctype, mps)
            before = _counters()
            if op == "aten._safe_softmax.default":
                r = _C._aten_dispatch(op, t, 1)
            else:
                r = _C._aten_dispatch(op, t, 1, False)
            after = _counters()
            d = _delta(before, after)
            assert str(r.device).startswith("mps"), (op, name, str(r.device))
            assert d["compute_encoders"] >= 1, (
                "%s %s on mps opened no Metal compute encoder: %r -- the "
                "values may still be right, which is exactly why this is "
                "asserted separately" % (op, name, d))
            assert d["host_downloads"] == 0, (
                "%s %s on mps moved %d tensor(s) to the host mid-kernel "
                "(%d bytes): it is computing on the CPU under an mps label"
                % (op, name, d["host_downloads"], d["host_download_bytes"]))
            ref = oracle["%s|%s" % (op, name)]
            tol = dt_utils.tolerance_for(ref["dtype"])
            why = _close(_host(r), ref["values"], tol.atol, tol.rtol)
            assert why is None, "%s %s on mps: %s" % (op, name, why)
            checked += 1
    assert checked == len(_SOFTMAX_OPS) * len(_FLOAT_DTYPES), checked


# ---------------------------------------------------------------------------
# 2. the §7.7 in-place family
# ---------------------------------------------------------------------------

# The same fourteen operators `test_mpsinplace.py` grades for *agreement*, so
# the two files are about the same cells and disagreeing between them is
# impossible without one of them going red.
_CASES = (
    ("aten.zero_.default", (), _ALL_DTYPES),
    ("aten.add_.Tensor", ("other",), _ALL_DTYPES),
    ("aten.add_.Scalar", (2,), _ALL_DTYPES),
    ("aten.sub_.Tensor", ("other",), _ALL_DTYPES),
    ("aten.mul_.Tensor", ("other",), _ALL_DTYPES),
    ("aten.mul_.Scalar", (3,), _ALL_DTYPES),
    ("aten.copy_.default", ("other",), _ALL_DTYPES),
    ("aten.neg_.default", (), _ALL_DTYPES),
    ("aten.abs_.default", (), _ALL_DTYPES),
    ("aten.div_.Tensor", ("other",), _FLOAT_DTYPES),
    ("aten.relu_.default", (), _FLOAT_DTYPES),
    ("aten.sigmoid_.default", (), _FLOAT_DTYPES),
    ("aten.tanh_.default", (), _FLOAT_DTYPES),
    ("aten.exp_.default", (), _FLOAT_DTYPES),
)

# Correctly blit-only: nothing is computed, so no compute encoder is due.
# `zero_` sets a constant on the device (`const_set`) and copies it in;
# `copy_` is the copy alone. Demanding a kernel from them would be the
# widening §7.14 refuses.
_BLIT_ONLY = {"aten.zero_.default", "aten.copy_.default"}

_ORACLE_INPLACE = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    dt = getattr(torch, case["dtype"])
    t = torch.tensor(case["values"]).reshape(case["shape"]).to(dt)
    other = torch.tensor(case["other"]).reshape(case["shape"]).to(dt)
    args = [other if a == "other" else a for a in case["args"]]
    getattr(t, case["op"].split(".")[1])(*args)
    out[key] = {"values": [float(v) for v in t.to(torch.float64).flatten().tolist()],
                "dtype": str(t.dtype).split(".")[-1]}
print(json.dumps(out))
"""


def test_the_in_place_family_on_mps_computes_on_the_device():
    """§7.7's 109 cells, counted -- 52 of them, one per (operator, dtype).

    §7.7 moved 109 cells REFUSES -> AGREES and said plainly that its placement
    evidence was structural: `write_on_device` calls no `flat_storage` and
    holds no readback marker, re-derived from `tensor.rs` every gate run. It
    also said why that is not enough -- three rounds had by then planted a
    host twin and watched every value test stay green.

    This is the counted version. For each cell: **zero host downloads and zero
    host uploads across the dispatch**, which is the pair a host-computed twin
    cannot satisfy (it must read the operand down and push the answer back);
    plus a compute encoder for the twelve operators that compute something,
    and `blit_encoders >= 1` with both byte counters at zero for the two that
    correctly only copy.

    Agreement is asserted alongside, against a subprocess oracle at
    `tests/golden/dtypes.py`'s derived tolerance, so that this file cannot
    pass by counting a kernel that writes the wrong answer.

    **`add_.Scalar`/`mul_.Scalar` upload and are supposed to.** The scalar is
    built on the host and becomes a 4- or 8-byte device buffer -- one upload
    of exactly one element. So the assertion for those is not "no upload" but
    "no upload larger than one element", which still refuses an operand-sized
    one. A twin reading six elements down and six back fails it.
    """
    mps = _mps_or_skip("the in-place family dispatch count")
    if mps is None:
        return
    request = {}
    for op, extra, dtypes in _CASES:
        for name in dtypes:
            request["%s|%s" % (op, name)] = {
                "op": op, "dtype": name, "args": list(extra),
                "values": _VALUES, "other": _OTHER, "shape": _SHAPE}
    oracle = _oracle(_ORACLE_INPLACE, request)

    checked = 0
    for op, extra, dtypes in _CASES:
        for name in dtypes:
            ctype = dt_utils.c_dtype(_C, name)
            t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), ctype, mps)
            other = _C._tensor_from_flat(list(_OTHER), list(_SHAPE), ctype, mps)
            args = [other if a == "other" else a for a in extra]
            before = _counters()
            r = _C._aten_dispatch(op, t, *args)
            after = _counters()
            d = _delta(before, after)
            assert r is t and str(t.device).startswith("mps"), (op, name)
            assert d["host_downloads"] == 0, (
                "%s %s on mps read %d tensor(s) back to the host (%d bytes) "
                "-- the write door is serving a device receiver from the CPU"
                % (op, name, d["host_downloads"], d["host_download_bytes"]))
            assert d["host_upload_bytes"] <= 8, (
                "%s %s on mps uploaded %d bytes -- more than the one scalar a "
                "Scalar overload is allowed, so an operand-sized buffer came "
                "from the host: %r" % (op, name, d["host_upload_bytes"], d))
            if op in _BLIT_ONLY:
                assert d["compute_encoders"] == 0 and d["blit_encoders"] >= 1, (
                    "%s %s is a copy on the device and should be blits only: %r"
                    % (op, name, d))
            else:
                assert d["compute_encoders"] >= 1, (
                    "%s %s on mps opened no Metal compute encoder: %r" % (op, name, d))
            ref = oracle["%s|%s" % (op, name)]
            tol = dt_utils.tolerance_for(ref["dtype"])
            why = _close(_host(t), ref["values"], tol.atol, tol.rtol)
            assert why is None, "%s %s on mps: %s" % (op, name, why)
            checked += 1
    assert checked == sum(len(d) for _, _, d in _CASES), checked


# ---------------------------------------------------------------------------
# 3. all 43 in-place operators
# ---------------------------------------------------------------------------

def _inplace_args(op, dtype, mps):
    """Operands for one in-place operator, or `None` if this file has none."""
    def t(vals, shape, dt):
        return _C._tensor_from_flat(list(vals), list(shape),
                                    dt_utils.c_dtype(_C, dt), mps)
    if op in ("aten.add_.Tensor", "aten.sub_.Tensor", "aten.mul_.Tensor",
              "aten.div_.Tensor", "aten.copy_.default"):
        return (t(_OTHER, _SHAPE, dtype),), {}
    if op in ("aten.add_.Scalar", "aten.sub_.Scalar", "aten.mul_.Scalar",
              "aten.div_.Scalar"):
        return (2,), {}
    if op == "aten.bernoulli_.float":
        return (0.5,), {}
    if op == "aten.clamp_.default":
        return (1.0, 4.0), {}
    if op == "aten.clamp_min_.default":
        return (1.0,), {}
    if op == "aten.fill_.Scalar":
        return (7.0,), {}
    if op == "aten.fill_.Tensor":
        return (t([7.0], [], dtype),), {}
    if op in ("aten.index_add_.default", "aten.index_copy_.default"):
        return (0, t([0.0, 1.0], [2], "int64"), t(_OTHER, _SHAPE, dtype)), {}
    if op == "aten.index_put_.default":
        return ([t([0.0, 1.0], [2], "int64")], t(_OTHER, _SHAPE, dtype)), {}
    if op == "aten.masked_fill_.Scalar":
        return (t([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], _SHAPE, "bool"), 7.0), {}
    if op in ("aten.normal_.default", "aten.uniform_.default"):
        return (0.0, 1.0), {}
    if op == "aten.round_.decimals":
        return (), {"decimals": 2}
    if op == "aten.scatter_.src":
        return (1, t([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], _SHAPE, "int64"),
                t(_OTHER, _SHAPE, dtype)), {}
    if op == "aten.scatter_.value":
        return (1, t([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], _SHAPE, "int64"), 7.0), {}
    return (), {}


def test_every_in_place_operator_that_reaches_on_mps_reads_nothing_back():
    """§7.8's 43 operators, counted -- placement only, and it says so.

    §7.8 dispatched every in-place operator `_aten_implemented()` reports on
    an `mps` tensor one at a time and partitioned them: 31 reach, 4 are held
    by the host-readback gate, 8 by `f64` on Metal. It measured *reachability*
    and inferred placement from the write door's source.

    This re-runs that sweep with the counters open and asserts placement.
    **Grade: reaches + placed, not agrees** -- values are graded by
    `test_mpsinplace.py` and by §6's table, and duplicating an oracle for 43
    operators here would make this file about agreement instead of about where
    the work happened. Stated rather than blurred, per AGENTS.md §16.

    Every operator that reaches must have `host_downloads == 0`, and there is
    **no longer an exception**. `scatter_.src` and `scatter_.value` were the
    two §7.14 named -- they read back, were not refused, and their
    `int64_mps` cells were published AGREES. §7.16 refuses them, so the
    assertion flipped: they must not reach at all, and they must be stopped
    by the host-readback gate specifically rather than by some earlier error
    that happens to raise first. A gate deleted and a gate that stopped
    firing produce different failures here, which is the point of checking
    the wording as well as the refusal.
    """
    mps = _mps_or_skip("the 43-operator placement sweep")
    if mps is None:
        return
    ops = sorted(o for o in _C._aten_implemented()
                 if o.split(".")[1].endswith("_"))
    assert len(ops) >= 43, (
        "the in-place family shrank to %d operators -- §7.8 measured 43" % len(ops))

    reached, refused, downloaded = [], [], []
    for op in ops:
        for dtype in ("float32", "int64"):
            args, kwargs = _inplace_args(op, dtype, mps)
            t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE),
                                     dt_utils.c_dtype(_C, dtype), mps)
            before = _counters()
            try:
                _C._aten_dispatch(op, t, *args, **kwargs)
            except (NotImplementedError, RuntimeError, TypeError) as e:
                refused.append("%s %s: %s" % (op, dtype, str(e).splitlines()[0][:240]))
                continue
            d = _delta(before, _counters())
            reached.append((op, dtype))
            if d["host_downloads"] > 0:
                downloaded.append((op, dtype, d))

    assert len(reached) >= 31, (
        "only %d (operator, dtype) pairs reached on mps; §7.8 measured 31 "
        "operators reaching. Refusals:\n  %s" % (len(reached), "\n  ".join(refused)))

    unexpected = [(op, dt, d) for op, dt, d in downloaded
                  if op not in _HOST_COMPUTED_ON_MPS and op not in _EXEMPT]
    assert not unexpected, (
        "these in-place operators reach on mps and compute on the host -- a "
        "new one, not among the eight docs/devices/matrix.md §7.14 names:\n  "
        + "\n  ".join("%s %s: %r" % x for x in unexpected))

    still_broken = {op for op, _, _ in downloaded if op in _HOST_COMPUTED_ON_MPS}
    assert not still_broken, (
        "%s reached on mps and downloaded. §7.16 refuses them by name, so "
        "reaching at all means the gate stopped firing -- the readback is "
        "still in the kernel." % sorted(still_broken))

    # The other half of the same claim, and the one that fails if the gate
    # were deleted rather than merely stopped firing: the two in-place
    # operators §7.14 named must be *refused*, by the readback gate and not by
    # some other error that happens to be raised first.
    held = [line for line in refused
            if line.startswith(("aten.scatter_.src ", "aten.scatter_.value "))
            and "reads the tensor back to host memory" in line]
    assert len(held) >= 2, (
        "scatter_.src/.value are meant to be held by the host-readback gate "
        "on mps (§7.16). What the sweep saw instead:\n  "
        + "\n  ".join(line for line in refused if "scatter_" in line))


# ---------------------------------------------------------------------------
# 4. the finding itself
# ---------------------------------------------------------------------------

_PROBES = {
    "aten.argsort.default": lambda T, d: ((0,), {}),
    "aten.argsort.stable": lambda T, d: ((True, 0), {}),
    "aten.sort.default": lambda T, d: ((0,), {}),
    "aten.topk.default": lambda T, d: ((2,), {}),
    "aten.floor_divide.default": lambda T, d: ((T(_OTHER, d),), {}),
    "aten.floor_divide.Scalar": lambda T, d: ((2,), {}),
}


def test_eight_operators_answer_an_mps_dispatch_from_the_host():
    """§7.14's finding, now pinned from the other side: the door is shut.

    Twelve cells published AGREES in §6's `mps` columns were computed on the
    CPU -- the operand downloaded, a scalar loop run on the host, the answer
    uploaded back. The values were right, which is why every agreement test
    stayed green, and `.device` still read `mps`. That is precisely the
    failure §7.12 had to plant deliberately in order to observe, and it was
    already here.

    **§7.16 refuses all eight**, so the assertion this test used to make --
    "the counter sees a download" -- is now unreachable: there is no dispatch
    to count. What replaces it is the thing that must stay true for the fix to
    mean anything, and it is asserted in three parts so that undoing any one
    of them reddens:

      1. every one of the eight is in `MPS_HOST_READBACK_OPS`;
      2. dispatching it on `mps` raises, with the *readback gate's* wording --
         not some other error raised earlier, which would make the gate
         removable without this failing;
      3. the counters do not move. A gate that refused after downloading
         would satisfy (2) and still have paid the round trip.

    (3) is the one that survives having the refusal list gutted and is stated
    plainly as the weak half: `host_downloads == 0` is satisfied by a counter
    that can never move, which is what M-B proved. It is meaningful only
    because `test_metalcount.py::
    test_a_readback_costs_exactly_the_tensors_bytes` runs in the same gate.
    """
    mps = _mps_or_skip("the host-computed mps operators")
    if mps is None:
        return
    refused_by_name = set(_C._shim_mps_host_readback_ops())

    def T(vals, dtype):
        return _C._tensor_from_flat(list(vals), list(_SHAPE),
                                    dt_utils.c_dtype(_C, dtype), mps)

    seen = {}
    for op, dtypes in sorted(_HOST_COMPUTED_ON_MPS.items()):
        assert op in refused_by_name, (
            "%s is not refused on mps by name any more. Its kernel still "
            "reaches `read_flat` through an un-named hop, so taking the name "
            "off the list without removing the readback is the "
            "docs/devices/MPSATTN.md §3.1 defeat -- the op would go straight "
            "back to answering an mps dispatch from the host." % op)
        dtype = dtypes[0]
        if op in _PROBES:
            args, kwargs = _PROBES[op](T, dtype)
        else:
            args, kwargs = _inplace_args(op, dtype, mps)
        t = T(_VALUES, dtype)
        before = _counters()
        try:
            _C._aten_dispatch(op, t, *args, **kwargs)
        except NotImplementedError as e:
            assert "reads the tensor back to host memory" in str(e), (
                "%s was refused on mps, but not by the host-readback gate: %s"
                % (op, str(e).splitlines()[0]))
        else:
            raise AssertionError(
                "%s %s answered on mps. §7.16 refuses it because it computes "
                "on the host; §6's table publishes REFUSES for every one of "
                "its mps cells." % (op, dtype))
        d = _delta(before, _counters())
        seen[op] = d
        assert d["host_downloads"] == 0 and d["host_download_bytes"] == 0, (
            "%s %s was refused on mps but the operand had already been "
            "downloaded (%r) -- the gate is firing after the round trip it "
            "exists to prevent." % (op, dtype, d))

    assert len(seen) == len(_HOST_COMPUTED_ON_MPS) == 8, sorted(seen)

    # And the two the *source* derivation added on its own are on the list
    # too. They are asserted here rather than in the loop because no counter
    # evidence was ever taken for them: the claim is "derived and refused",
    # and saying so is what keeps the two kinds of evidence apart.
    for op in _DERIVED_AS_WELL:
        assert op in refused_by_name, (
            "%s reaches read_flat through norm_pow_walk and is meant to be "
            "refused on mps (§7.16)." % op)


def test_the_readback_derivation_scan_reaches_these_kernels():
    """Half 3 of §7.14, inverted by §7.16 -- the instrument was the defect.

    A counter finding that a kernel reads back is only half a finding; the
    other half was that the gate built to refuse such kernels could not see
    it. §7.14 pinned the blind spot so that closing it would redden. This is
    that test after the fix, and it asserts the opposite: the derivation --
    `test_shim.py`'s own, imported rather than restated -- **does** reach all
    ten, and reaches each through a path of more than one hop, because a
    one-hop path would mean the kernel changed rather than the scan.

    Keeping the assertion here rather than deleting the test is deliberate.
    The blind spot has been mis-stated twice (§7.5 "one call deeper", §7.12
    "one file over") and both times the correction came from something
    failing. A test that says "the scan reaches through N hops" is the thing
    that fails if somebody re-flattens it.
    """
    witness = test_shim._ops_that_reach_the_host()
    if witness is None:
        _skip.skip("   (skipped the derivation depth check: torchnative/rust/torch_c/src "
                   "is not beside this file -- installed rather than in-tree)")
        return

    expected = sorted(set(_HOST_COMPUTED_ON_MPS) | set(_DERIVED_AS_WELL))
    missed = [op for op in expected if op not in witness]
    assert not missed, (
        "the derivation no longer reaches %r, which the counter measured "
        "reading back. That is the §7.14 blind spot reopening: these kernels "
        "reach `read_flat` through an in-file helper that is not one of the "
        "six names (`order_along`, `argsort_core`, `floor_divide_impl`, "
        "`norm_pow_walk`) or through another operator's kernel "
        "(`scatter_inplace` -> `scatter_src`)." % (missed,))

    shallow = {op: witness[op] for op in expected if len(witness[op]) < 3}
    assert not shallow, (
        "these are derived in fewer than two hops: %r. Either the kernel was "
        "rewritten -- in which case §7.16's account of *why* the scan missed "
        "them is now wrong and has to be corrected -- or the derivation is "
        "matching something other than a call." % shallow)

    # And the derivation is not simply everything: without this, an
    # `_ops_that_reach_the_host` that returned every dispatched op would
    # satisfy every assertion above.
    _, text = test_shim._aten_rs_functions()
    total = len(test_shim._aten_dispatch_targets(text))
    assert "aten.expm1.default" in witness and 40 < len(witness) < total * 0.6, (
        "the derivation reached %d of %d dispatched ops -- it has stopped "
        "discriminating, which makes the assertions above meaningless"
        % (len(witness), total))


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)
             and getattr(fn, "__module__", None) == __name__]
    failures = _skip.run_tests(items, suite="test_metalplace")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
