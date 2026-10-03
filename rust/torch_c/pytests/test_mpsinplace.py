"""The in-place family on Metal: one gate, thirty-three operators.

`docs/devices/matrix.md` §7.2 found that **112 cells across 33 in-place
operators** were published as BREAKS when they were merely refused, and that
every one of them stops at the *same* sentence:

    aten.zero_.default: writing through a view is implemented for the CPU
    backend only in torch._C shim; this tensor is on mps:0

That is one gate, not thirty-three gaps, and this file is about lifting it.

    import torch
    x = torch.randn(1024, 1024, device="mps")
    x.add_(1.0)                  # NotImplementedError, before this round
    x.zero_()                    # NotImplementedError, before this round

**What the gate actually was.** Not a missing kernel and not a missing
aliasing analysis -- both of those were already device independent.
`tensor.rs::write_into` is the single write door, and it was implemented as a
host-side strided scatter: `flat_storage` pulled the replacement out through
`to_vec1`, and `WriteThrough` implemented only candle's `InplaceOp1::cpu_fwd`,
walking the destination's layout over a `&mut CpuStorage` slice. There is no
CPU slice behind a Metal buffer, so the door could not open on a device. The
refusal was honest; what it named was the *door*, not the operator.

**What lifts it, and why it needs no change to `vendor/candle-core`.** For a
receiver whose layout is **contiguous** -- a whole tensor, or a `detach()`,
`view()`, `reshape()`, `unsqueeze()` or leading-axis `slice`/`select` of one --
the set of storage positions the view addresses is one contiguous run starting
at `layout.start_offset()`. candle already has a public door for exactly that
run: `Tensor::slice_set`, which reaches `BackendStorage::copy2d`, which on
Metal with matching strides is a **blit command on the GPU's own command
queue** (`metal_backend/mod.rs`, the `src_s == d2 && dst_s == d2` arm). So the
device path is candle's own code, through its published API, and the fork's
"two inputs and nothing else" contract is untouched.

**What is NOT lifted, stated first so the rest of this file is read
correctly.** A *non-contiguous* receiver -- `x[:, 0:2]`, an `expand`, a
transposed view -- still refuses on a device, because writing a strided run
needs a scatter kernel that candle does not expose. The refusal for that case
is narrower and names the layout, rather than naming the backend. **It must
not fall through to the host path**: doing so would compute the write on the
CPU under an `mps` label, which is what docs/devices/MPS.md §2 exists to
refuse. `test_the_device_write_door_performs_no_host_readback` is the
structural guard on that, and it is the weak half of this file's evidence --
see below.

**What this file proves, and at which grade.**

* *agrees* -- element-wise against upstream, on `mps`, for fourteen operators
  and four dtypes, with the oracle computed in a **separate subprocess** whose
  `PYTHONPATH` is emptied and which asserts it did not import the shim. The
  tolerance is `tools/golden/dtypes.py`'s for the result dtype and is not
  widened anywhere in this file.
* *aliasing* -- the property that made `write_into` a write-through rather
  than a rebinding in the first place (docs/kernels/VIEWS.md §6): a view taken
  **before** the call sees the write. On a device this is the test that
  separates a real write from a `replace_with`, and it is behavioural, not
  structural.
* *placement* -- **weaker than a counter, and this file says so rather than
  grading around it.** Three separate rounds planted a host-computed twin,
  found the values entirely correct and the agreement test green, and **only a
  dispatch counter caught the fallback** -- all three on Vulkan, which is the
  correction AGENTS.md §13.1 records against the inflated citation of that number.
  A Metal counter now exists (`_C._metal_counters()`,
  docs/devices/matrix.md §7.11) and **this file does not use it**: no test
  below asserts a counter delta for the in-place family, so these cells remain
  where §7.5 put them. So the placement evidence is (a) the device branch
  returns before `flat_storage` is ever reached, re-derived from `tensor.rs`
  by the scan below on every gate run, and (b) the receiver comes back on
  `mps` with a device-resident result. Neither can distinguish a write done by
  a blit from a write done by a hypothetical host path *inside candle*. It is
  the standard `softmax_on_device` and `abs` were landed at
  (docs/devices/MPSATTN.md §3.1) and no more.

Nullifications this file is meant to catch, each verified by making the break
and recorded in docs/devices/matrix.md §7.8:

* the device branch deleted (every op refuses again)
      -> test_in_place_ops_agree_with_upstream_on_mps
* `slice_set` given offset 0 against the *base* rather than the view
      -> test_writing_into_an_offset_slice_on_mps_touches_only_that_slice
* the device branch replaced by `receiver.replace_with(...)`
      -> test_a_view_taken_before_the_write_sees_it_on_mps
* the `.copy()` that breaks source/destination storage sharing removed
      -> test_a_self_overlapping_copy_on_mps_is_not_half_overwritten
* the contiguity check removed, so a strided receiver writes the wrong cells
      -> test_a_strided_receiver_on_mps_refuses_and_names_its_layout
* the contiguity check removed *and* the host path allowed to catch it
      -> test_the_device_write_door_performs_no_host_readback
"""

import json
import os
import re
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools", "golden"))
import dtypes as dt_utils  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_TENSOR_RS = os.path.join(_HERE, "..", "src", "tensor.rs")

# Real inputs: negatives on both sides of zero, a zero, and magnitudes that are
# exact in float16 so the float16 row tests the operator rather than testing
# rounding. `_OTHER` has no zero, so `div_` is defined on every element.
_VALUES = [-3.0, -1.0, 0.0, 2.0, -7.0, 5.0]
_OTHER = [1.0, 2.0, 4.0, -1.0, 8.0, -2.0]
_SHAPE = [2, 3]

# `float64` is absent because Metal has no double, and `int32`/`int8` because
# this build refuses them on Metal by name -- all three for reasons that are
# not this gate's and are pinned elsewhere (docs/devices/matrix.md §3.1,
# test_intmps.py, docs/numerics/INT8.md §1.2). Putting them here would make
# this file fail for a reason it is not about.
_FLOAT_DTYPES = ("float32", "float16", "bfloat16")
_ALL_DTYPES = ("float32", "float16", "bfloat16", "int64")

# (op, extra positional args, dtypes). The selection is by reachability, not by
# coverage: these are the in-place calls a person writing three lines actually
# makes. `fill_.Scalar` is deliberately absent and its own blocker is pinned by
# `test_fill_on_mps_agrees_with_upstream` below.
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

_ORACLE = r"""
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
    op = case["op"]
    name = op.split(".")[1]
    args = [other if a == "other" else a for a in case["args"]]
    getattr(t, name)(*args)
    out[key] = {
        "values": [float(v) for v in t.to(torch.float64).flatten().tolist()],
        "dtype": str(t.dtype).split(".")[-1],
    }
print(json.dumps(out))
"""

_ORACLE_SCRIPTS = {
    # `x[1:3]` written through, with the rest of `x` left alone. The oracle is
    # asked for the *whole* base, not the slice: a write that landed at the
    # wrong offset changes rows this comparison covers.
    "offset_slice": r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented")
req = json.loads(sys.argv[1])
x = torch.tensor(req["values"]).reshape(req["shape"]).to(torch.float32)
x[1:3].add_(100.0)
print(json.dumps([float(v) for v in x.to(torch.float64).flatten().tolist()]))
""",
    # A source that **shares the destination's storage without overlapping it
    # element-wise**. `x[0:2].copy_(x[1:3])` is not this case: upstream itself
    # raises for it ("some elements of the input tensor and the written-to
    # tensor refer to a single memory location"), measured. Disjoint rows of
    # one buffer is the case upstream writes -- and it is the one candle's
    # `slice_set` refuses, because its guard is `same_storage`, which is
    # coarser than element overlap.
    "self_overlap": r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented")
req = json.loads(sys.argv[1])
x = torch.tensor(req["values"]).reshape(req["shape"]).to(torch.float32)
x[0:2].copy_(x[2:4])
print(json.dumps([float(v) for v in x.to(torch.float64).flatten().tolist()]))
""",
}


def _oracle(script, payload):
    """Upstream's answer, computed in a subprocess with no `_C` anywhere."""
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


def test_in_place_ops_agree_with_upstream_on_mps():
    """Grade *agrees*: sixteen operators x four dtypes, on the device.

    The comparison is against upstream's own bytes from a separate
    subprocess, so an implementation that wrote zeros, wrote the source
    unchanged, or wrote nothing at all fails here -- `_VALUES` has negatives,
    a zero and positives, and `zero_`/`copy_`/`neg_` each move every element.

    It also asserts the receiver is **the same object**, still on `mps`, and
    that the returned handle is that object: an in-place op that rebound the
    wrapper would pass a value comparison and fail this.
    """
    mps = _mps_or_skip("in-place agreement on mps")
    if mps is None:
        return
    request = {}
    for op, extra, dtypes in _CASES:
        for name in dtypes:
            request["%s|%s" % (op, name)] = {
                "op": op, "dtype": name, "args": list(extra),
                "values": _VALUES, "other": _OTHER, "shape": _SHAPE,
            }
    oracle = _oracle(_ORACLE, request)

    checked = 0
    refused = []
    for op, extra, dtypes in _CASES:
        for name in dtypes:
            ctype = dt_utils.c_dtype(_C, name)
            t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), ctype, mps)
            other = _C._tensor_from_flat(list(_OTHER), list(_SHAPE), ctype, mps)
            args = [other if a == "other" else a for a in extra]
            try:
                r = _C._aten_dispatch(op, t, *args)
            except NotImplementedError as e:
                refused.append("%s %s: %s" % (op, name, str(e).splitlines()[0]))
                continue
            ref = oracle["%s|%s" % (op, name)]
            tol = dt_utils.tolerance_for(ref["dtype"])
            assert str(t.device).startswith("mps"), (
                "%s %s: the receiver came back on %s under an mps label"
                % (op, name, t.device))
            assert r is t, (
                "%s %s: returned a different object from the receiver -- an "
                "in-place op that rebinds is not writing through"
                % (op, name))
            why = _close(_host(t), ref["values"], tol.atol, tol.rtol)
            assert why is None, "%s %s_mps (receiver): %s" % (op, name, why)
            checked += 1
    assert not refused, (
        "these in-place ops still refuse on mps:\n  " + "\n  ".join(refused))
    assert checked == sum(len(d) for _, _, d in _CASES), checked


def test_a_view_taken_before_the_write_sees_it_on_mps():
    """The write-through property itself, on a device.

    `docs/kernels/VIEWS.md` §6 changed the in-place family from "replace the
    wrapper" to "write through the layout" so that an alias taken *before*
    the call sees the write, which is what upstream does. A device path built
    on `replace_with` would satisfy every value comparison in this file and
    fail only here.
    """
    mps = _mps_or_skip("alias visibility on mps")
    if mps is None:
        return
    x = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), _C.float32, mps)
    alias = _C._aten_dispatch("aten.detach.default", x)
    view = _C._aten_dispatch("aten.view.default", x, [6])
    _C._aten_dispatch("aten.zero_.default", x)
    assert _host(x) == [0.0] * 6, _host(x)
    assert _host(alias) == [0.0] * 6, (
        "detach() taken before zero_ did not see the write: %r -- the device "
        "path rebound the wrapper instead of writing through it" % (_host(alias),))
    assert _host(view) == [0.0] * 6, (
        "view() taken before zero_ did not see the write: %r" % (_host(view),))


def test_writing_into_an_offset_slice_on_mps_touches_only_that_slice():
    """`x[1:3].add_(100)` -- the test that `start_offset` is honoured.

    A device write that ignored the view's start offset would land the values
    at the top of the buffer and leave the slice untouched. Both halves are
    visible here because the oracle is asked for the whole base.
    """
    mps = _mps_or_skip("offset-slice write on mps")
    if mps is None:
        return
    values = [float(v) for v in range(12)]
    shape = [4, 3]
    want = _oracle(_ORACLE_SCRIPTS["offset_slice"],
                   {"values": values, "shape": shape})
    x = _C._tensor_from_flat(list(values), list(shape), _C.float32, mps)
    sl = _C._aten_dispatch("aten.slice.Tensor", x, 0, 1, 3, 1)
    _C._aten_dispatch("aten.add_.Scalar", sl, 100.0)
    tol = dt_utils.tolerance_for("float32")
    why = _close(_host(x), want, tol.atol, tol.rtol)
    assert why is None, "x[1:3].add_(100) on mps, whole base: %s" % why
    # Stated separately so the failure says which half went wrong.
    assert _host(x)[0:3] == values[0:3], (
        "rows outside the slice were written: %r" % (_host(x)[0:3],))


def test_a_self_overlapping_copy_on_mps_is_not_half_overwritten():
    """`x[0:2].copy_(x[2:4])` -- source and destination share one buffer.

    Disjoint rows, one storage. Upstream writes; `x[0:2].copy_(x[1:3])` is a
    different case and upstream *raises* for it (measured, and the reason this
    test does not use it).

    candle's `slice_set` guards on `same_storage`, which is coarser than
    element overlap: it refuses this pair outright. So the device door has to
    break the sharing itself, and if it does not, this raises candle's
    "cannot use slice_set when self and src share their storage" rather than
    producing upstream's answer.
    """
    mps = _mps_or_skip("self-overlapping copy_ on mps")
    if mps is None:
        return
    values = [float(v) for v in range(12)]
    shape = [4, 3]
    want = _oracle(_ORACLE_SCRIPTS["self_overlap"],
                   {"values": values, "shape": shape})
    x = _C._tensor_from_flat(list(values), list(shape), _C.float32, mps)
    dst = _C._aten_dispatch("aten.slice.Tensor", x, 0, 0, 2, 1)
    src = _C._aten_dispatch("aten.slice.Tensor", x, 0, 2, 4, 1)
    _C._aten_dispatch("aten.copy_.default", dst, src)
    tol = dt_utils.tolerance_for("float32")
    why = _close(_host(x), want, tol.atol, tol.rtol)
    assert why is None, "x[0:2].copy_(x[2:4]) on mps: %s" % why


def test_a_strided_receiver_on_mps_refuses_and_names_its_layout():
    """What is **not** lifted, pinned so it cannot be lifted by accident.

    A non-contiguous receiver (`x[:, 0:2]`) addresses a strided run, and
    candle exposes no device scatter for that. It must refuse -- and it must
    not refuse by naming the *backend*, which was the sentence
    docs/devices/matrix.md §7.2 found sending readers to the wrong fix.

    **The dangerous failure here is success, not refusal.** A device receiver
    that quietly fell through to the host path would compute the write on the
    CPU under an `mps` label, which is the thing docs/devices/MPS.md §2
    exists to refuse; that path is what `_host_written` below detects, by
    checking the values are *not* silently correct.
    """
    mps = _mps_or_skip("strided receiver refusal on mps")
    if mps is None:
        return
    x = _C._tensor_from_flat([float(v) for v in range(12)], [4, 3], _C.float32, mps)
    strided = _C._aten_dispatch("aten.slice.Tensor", x, 1, 0, 2, 1)
    try:
        _C._aten_dispatch("aten.add_.Scalar", strided, 100.0)
    except NotImplementedError as e:
        msg = str(e).splitlines()[0]
    else:
        raise AssertionError(
            "add_ on a NON-contiguous mps receiver succeeded. If a device "
            "scatter landed, that is good news and this test should become an "
            "agreement test -- but check first that the write did not go "
            "through the host path in tensor.rs, which would be a CPU "
            "computation under an mps label.")
    assert "non-contiguous" in msg, (
        "the strided refusal does not name the layout: %r" % msg)
    assert "writing through a view is implemented" not in msg, (
        "the strided refusal is still the old unqualified sentence -- the one "
        "that told 112 cells of matrix.md the *backend* was the limit when the "
        "limit is the layout: %r" % msg)
    assert "receiver does write on the device" in msg, (
        "the refusal does not tell the reader what does work, which is the "
        "half that sends them to the right fix: %r" % msg)


def test_the_device_write_door_performs_no_host_readback():
    """Re-derived from `tensor.rs` on every run, not restated from it.

    The placement evidence for every cell this round moves, and the weaker
    half of this file (there is no Metal dispatch counter -- see the module
    docstring). Two claims, both about source:

    1. the function that serves a device receiver contains none of the
       readback markers `test_shim.py` scans `aten.rs` for, and does not call
       `flat_storage`;
    2. `write_into` dispatches to it **before** it reaches `flat_storage`, so
       a device tensor cannot arrive at the host scatter at all.

    The second matters because `tensor.rs` is outside the scan in
    `test_shim.py`, which reads `aten.rs` only. `flat_storage`'s `to_vec1` was
    invisible to that scan the whole time; it did not matter while the door
    refused every device, and it starts mattering the moment the door opens.
    """
    if not os.path.exists(_TENSOR_RS):
        _skip.skip("   (skipped device write-door scan: rust/torch_c/src/tensor.rs "
                   "is not beside this file -- installed rather than in-tree)")
        return
    text = open(_TENSOR_RS, encoding="utf-8").read()
    markers = re.compile(r"\.to_vec[0-3]|\.to_scalar|\.to_cpu\(|to_cpu_storage")

    body = _rust_fn_body(text, "write_on_device")
    assert body is not None, (
        "tensor.rs has no `fn write_on_device` -- if the device write door was "
        "renamed, move this scan with it rather than deleting it")
    hit = markers.search(body)
    assert hit is None, (
        "the device write door reads bytes back to the host (%r). A device "
        "receiver served through the host is a CPU computation under an mps "
        "label -- docs/devices/MPS.md §2." % hit.group(0))
    assert "flat_storage" not in body, (
        "the device write door calls flat_storage, which is the host scatter's "
        "`to_vec1` one level down")

    door = _rust_fn_body(text, "write_into")
    assert door is not None
    at_device = door.find("write_on_device")
    at_host = door.find("flat_storage")
    assert at_device != -1, (
        "write_into no longer dispatches to write_on_device -- the device path "
        "is gone and every mps in-place op has silently refused again")
    assert at_host != -1, "write_into no longer calls flat_storage at all"
    assert at_device < at_host, (
        "write_into reaches flat_storage before it dispatches a device "
        "receiver; a device tensor can now be scattered on the host")


def test_fill_on_mps_agrees_with_upstream():
    """`fill_.Scalar` on `mps`, promoted from a recorded blocker to a grade.

    This test used to assert that `fill_` still **failed** on Metal, and said
    what to do when it stopped: "make this an agreement test and move the cell
    in docs/devices/matrix.md section 7.7". `host_full` (matrix.md section
    4.3b, cause D) made it stop, and the commit that did so did not come back
    here -- so the promotion is done now, and the cell in section 7.7 moves
    with it.

    The old blocker was `fill_inplace` building its replacement with
    `Tensor::full(value.as_f64(), ...)`, which asked Metal's `const_set` for an
    `f64` arm it does not have. The value is converted on the host now, once,
    before anything crosses.

    Graded **agrees**: element-wise against upstream in a separate subprocess,
    on the three float dtypes. `int64` is not here because `fill_` on `mps` is
    in this build's integer refusal, which matrix.md section 4.3c records by
    name.
    """
    mps = _mps_or_skip("fill_ on mps")
    if mps is None:
        return
    cases = {
        "fill_/%s" % dt: {"op": "aten.fill_.Scalar", "args": [5.0],
                          "dtype": dt, "values": list(_VALUES),
                          "shape": list(_SHAPE), "other": list(_VALUES)}
        for dt in _FLOAT_DTYPES
    }
    want = _oracle(_ORACLE, cases)
    bad = []
    for key, case in sorted(cases.items()):
        expect = want[key]
        assert "values" in expect, (key, expect)
        t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE),
                                 getattr(_C, case["dtype"]), mps)
        _C._aten_dispatch("aten.fill_.Scalar", t, 5.0)
        assert str(t.device).startswith("mps"), (
            "%s: the receiver left the mps device (%s), so this is not "
            "measuring a Metal fill" % (key, t.device))
        why = _close(_host(t), expect["values"], 0.0, 0.0)
        if why:
            bad.append("%s: %s" % (key, why))
    assert not bad, ("fill_ on mps does not agree with upstream:\n  "
                     + "\n  ".join(bad))


def test_fill_on_mps_is_refused_by_name_for_the_integer_dtypes():
    """The half of `fill_` that did **not** move, asserted rather than dropped.

    When a test that recorded a blocker becomes an agreement test, the easy
    mistake is to let the cells that still refuse disappear with it. `int32`
    on `mps` still refuses, and the requirement is that it refuses **by name**
    -- naming the operator, the dtype and the device -- rather than quoting a
    candle symbol.
    """
    mps = _mps_or_skip("fill_'s integer refusal on mps")
    if mps is None:
        return
    t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), _C.int32, mps)
    try:
        _C._aten_dispatch("aten.fill_.Scalar", t, 5)
    except (RuntimeError, NotImplementedError) as e:
        msg = str(e).splitlines()[0]
    else:
        raise AssertionError(
            "aten.fill_.Scalar now works on mps for int32. Good news: grade it "
            "in test_fill_on_mps_agrees_with_upstream and update "
            "docs/devices/matrix.md section 4.3c, which records it as refused.")
    for token in ("aten.fill_.Scalar", "int32", "mps"):
        assert token in msg, (
            "the int32/mps fill_ refusal does not name %r: %r" % (token, msg))


_ORACLE_SCRIPTS_NAN = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented")
req = json.loads(sys.argv[1])
nan = float("nan")
out = {}
for key, case in req.items():
    t = torch.tensor([nan, -1.0, 0.0, 2.0], dtype=getattr(torch, case["dtype"]))
    name = case["op"].split(".")[1]
    r = getattr(torch, name)(t, *case["args"]) if not name.endswith("_") else (
        getattr(t.clone(), name)(*case["args"]))
    out[key] = [float(v) for v in r.to(torch.float64).flatten().tolist()]
print(json.dumps(out))
"""

# `clamp_values` is shared by all four spellings, so all four are listed: a fix
# that only reached the in-place pair would leave the out-of-place pair wrong.
_NAN_CASES = (
    ("aten.clamp_min_.default", (0.0,)),
    ("aten.clamp_.default", (0.0, 1.0)),
    ("aten.clamp_min.default", (0.0,)),
    ("aten.clamp.default", (0.0, 1.0)),
)


def test_clamp_propagates_nan_on_mps_exactly_as_upstream_does():
    """A defect this round's re-measurement found, not one it introduced --
    and one it would otherwise have made *reachable*.

    `clamp_values` is `min(max(x, lo), hi)` through candle's scalar
    `maximum`/`minimum`. On the CPU those propagate NaN, which is upstream's
    rule: `torch.tensor([nan]).clamp_min_(0.)` is `nan`. **On Metal they do
    not** -- candle's kernels are MSL `max`/`min`, which return the non-NaN
    operand, so `nan` came back as `0.0`.

    Two things are worth separating here.

    * The **out-of-place** pair (`clamp`, `clamp_min`) has been wrong on
      Metal since `mps` landed. `docs/devices/matrix.md` grades both AGREES at
      `float32_mps` in every published table, and it is right to within what
      it can see: its cell is `tools/golden/cases.py`'s first case for the op,
      which has no NaN in it. §5's "one shape per op" is exactly this blind
      spot, and this is the first time it has been demonstrated rather than
      warned about.
    * The **in-place** pair (`clamp_`, `clamp_min_`) was *refused* on Metal
      until this round, by the write-back gate, so the divergence could not be
      reached at all. Lifting that gate without this fix would have converted
      a refusal into a silently wrong answer, which is the one direction
      AGENTS.md §16 does not permit. The sweep caught it because the in-place
      case builder does have a NaN in it and the out-of-place one does not.

    The fix is a device-resident `where(x != x, x, clamped)` in
    `clamp_values`, applied only off the CPU so the CPU hot path (mamba's
    discretisation clamps) keeps its two-kernel shape.
    """
    mps = _mps_or_skip("clamp NaN propagation on mps")
    if mps is None:
        return
    request = {}
    for op, extra in _NAN_CASES:
        for name in _FLOAT_DTYPES:
            request["%s|%s" % (op, name)] = {
                "op": op, "dtype": name, "args": list(extra)}
    oracle = _oracle(_ORACLE_SCRIPTS_NAN, request)

    nan = float("nan")
    wrong = []
    for op, extra in _NAN_CASES:
        for name in _FLOAT_DTYPES:
            ctype = dt_utils.c_dtype(_C, name)
            t = _C._tensor_from_flat([nan, -1.0, 0.0, 2.0], [4], ctype, mps)
            r = _C._aten_dispatch(op, t, *extra)
            got = _host(r)
            want = oracle["%s|%s" % (op, name)]
            # NaN-ness is the claim, so it is compared as NaN-ness rather than
            # by `_close`, which skips a pair where both are NaN and would also
            # skip the one where only upstream is.
            for i, (g, w) in enumerate(zip(got, want)):
                if (w != w) != (g != g):
                    wrong.append("%s %s_mps index %d: upstream=%r shim=%r"
                                 % (op, name, i, w, g))
            tol = dt_utils.tolerance_for(name)
            why = _close(got, want, tol.atol, tol.rtol)
            assert why is None, "%s %s_mps: %s" % (op, name, why)
    assert not wrong, (
        "clamp on Metal dropped NaN where upstream keeps it:\n  "
        + "\n  ".join(wrong))


def _rust_fn_body(text, name):
    """The brace-balanced body of `fn <name>` in a Rust source string."""
    match = re.search(r"\bfn\s+" + re.escape(name) + r"\b", text)
    if match is None:
        return None
    start = text.find("{", match.end())
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsinplace")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
