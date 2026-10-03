"""The constant gate: `Tensor::full(f64, .., device)` on a device with no `f64`.

`docs/devices/matrix.md` §4.3a split the `mps` failures by message. Cause B --
"a float64 cast gate placed wrong" -- is retracted: re-clustered by *the dtype
the caller actually asked for*, 116 of its 117 cells never mentioned
`float64` at all. Cause D -- the factories -- is the front of this file's
population, and §7.18 is its record.

**Two rounds fixed the same defect in parallel, and this file is the merge.**
§4.3b's `host_full` (the cause-D round, `test_constset.py`) landed first and
closed the float dtypes of nine factories. This round (`work/cmlsilence`)
had done the same with a helper of its own; at integration its helper was
dropped for `host_full` and what `host_full`'s round did not cover was kept:
the integer arms of the same call sites, five sites that called `host_full`
with **no** narrowing step and narrowed on the device afterwards, the RNG
writers, `scalar_arg`'s zero-dim widening, `fill_.Tensor`'s device path, and
`host_const`'s refusal of an `f64` on Metal.

**The shape.** A kernel that needs a scalar operand writes the obvious thing:

```rust
Tensor::full(value.as_f64(), shape, device).and_then(|t| t.fast_to(storage))
```

The narrowing is right there on the next line -- and it never runs, because
`Tensor::full` asks the **device** to materialise an `F64` fill first. Metal
has no `f64` at all, so the call dies in candle with `unsupported const-set
f64`, or, where the constant arrives as a vector instead, with `Metal
contiguous to_dtype F64 F32 not implemented`. Five of the six messages a
sweep collected name `f64`, a dtype **no probe ever asked for**, and none of
them names `mps` or says what to do.

**The population was never six.** A cell refuses for the first reason it
meets, so a sweep counts reasons that are *reachable*, not reasons that
*exist*. `grep -n "Tensor::full" aten.rs` returned 33 hits and roughly two
dozen were this shape. Measured before the fix, on this machine, ten
operators failed where the sweep had found six -- `aten.full.default`,
`aten.scalar_tensor.default`, `aten.where.Scalar` and
`aten.where.ScalarSelf` were behind the first four and invisible until they
were fixed.

**The route.** `aten.rs::host_const` already existed for exactly this, from
the SmolLM2 `mul.Scalar` stop in `docs/devices/MPSFWD.md` §2: build the one
element on the host, apply the narrowing steps **there**, then move. Its
shaped sibling `host_full` (§4.3b) and its vector sibling `host_vec`
(this round) remove the same intermediate. All three remove the intermediate rather than compensating for it.

**The trap this file exists to pin.** Narrowing on the device would have to
go `f64 -> f32 -> f16`, and that double-rounds:
`f16(f32(0.031265258789971995))` is `0.03125` where `f16` of the same value
in one step is `0.031280517578125`. The two differ by one ulp of `float16`
and **agree exactly on `float32`**, which is the dtype anyone would reach for
first. `test_one_step_narrowing_*` uses `float16` by name.

**That trap is aarch64's** (issue #28). c10 builds `Half` from `float16_t`
on aarch64 and from `float` elsewhere, so off aarch64 upstream *is*
f16(f32(x)) and the one-step tests skip by name. `bfloat16` is
bf16(f32(x)) on every platform; `test_f16_and_bf16_narrowing_follow_c10_*`
asks upstream for both, everywhere.

**What is not claimed.** Nothing here argues `float64` should compute on
Metal. The 260 `float64_mps` cells stay correctly refused, and
`test_float64_on_mps_is_still_refused_*` is the assertion that this round did
not open a hole while closing one: the host-side path could trivially have
built an `F64` constant on the host and shipped it to the GPU, so
`host_const` refuses that by name.

**The bracket.** Every counter measurement below snapshots
`_C._metal_counters()` immediately before the single `_aten_dispatch` under
test and again immediately after. Operands are built before the first
snapshot; results are read back after the second. Nothing else runs between
the two reads. Counters from a different bracket are not comparable with
these.
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
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools", "golden"))
import dtypes as dt_utils  # noqa: E402

_NAMES = ("compute_encoders", "blit_encoders", "host_uploads",
          "host_upload_bytes", "host_downloads", "host_download_bytes")

_SHAPE = [2, 3]
_VALUES = [-3.0, -1.0, 0.5, 2.0, -7.0, 5.0]
_MASK = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
_FLOAT_DTYPES = ("float32", "float16", "bfloat16")

# The scalar that separates one-step narrowing from two-step. `float16` only:
# every `float32` cell in this file would pass either way.
_ONE_ULP = 0.031265258789971995
_SINGLE_ROUNDED = 0.031280517578125      # f16(x)
_DOUBLE_ROUNDED = 0.03125                # f16(f32(x))

# Which of the two upstream itself gives is per-architecture (issue #28):
# c10 builds `Half` from `float16_t` on aarch64 (not CUDA) -- one rounding --
# and from `float` everywhere else -- f16(f32(x)). Off aarch64 the one-step
# witness has nothing to witness, so those tests skip by name. Every test that
# uses this also asks upstream, so a wrong guess is a FAIL.
import platform  # noqa: E402

_F16_SINGLE_ROUNDING = platform.machine().lower() in ("arm64", "aarch64")
_F16_UPSTREAM = _SINGLE_ROUNDED if _F16_SINGLE_ROUNDING else _DOUBLE_ROUNDED

# `float64 -> bfloat16` witnesses, with upstream's answer on every platform
# (c10 has no `BFloat16(double)`, so it is bf16(f32(x)) everywhere):
#   1 + 2^-8 + 2^-22: exact in f32, just above a bf16 tie. c10 gives
#     1.0078125; `half::bf16::from_f64` truncates the 2^-22 away, sees the
#     tie, and gives 1.0.
#   1 + 3*2^-8 - 2^-30: just below a bf16 tie, which f32 rounds onto. c10
#     gives 1.015625; a single rounding gives 1.0078125.
_BF16_WITNESSES = {1 + 2**-8 + 2**-22: 1.0078125,
                   1 + 3 * 2**-8 - 2**-30: 1.015625}


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
            return "index %d: upstream=%r shim=%r (|diff|=%r)" % (i, y, x, abs(x - y))
    return None


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


# ---------------------------------------------------------------------------
# The population, as data.
#
# Each entry is one `aten.rs` call site that used to hand an `f64` constant to
# the device. `sites` are the line numbers on develop *before* this round --
# recorded rather than derived, because a second round (cause D, `host_full`
# for the nine factories) is editing the same function and the reconciliation
# at merge needs to know which lines each touched.
#
# `expr` is evaluated upstream with `t`, `m`, `v` and `dt` bound; `call` runs
# the same thing through the shim. The two are written side by side on
# purpose: a case whose two halves drift apart is a case that proves nothing,
# and the element-wise comparison below is what notices.
# ---------------------------------------------------------------------------
_CASES = [
    # key, aten.rs sites, upstream expression, shim thunk (t, m, v, ctype, dev_kw)
    ("full", "6025,6028",
     "torch.full((2, 3), v, dtype=dt)",
     lambda t, m, v, ct, kw: _C._aten_dispatch(
         "aten.full.default", list(_SHAPE), v, dtype=ct, **kw)),
    ("full_like", "6058,6060",
     "torch.full_like(t, v)",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.full_like.default", t, v)),
    ("new_full", "6058,6060",
     "t.new_full((2, 3), v)",
     lambda t, m, v, ct, kw: _C._aten_dispatch(
         "aten.new_full.default", t, list(_SHAPE), v)),
    ("constant_pad_nd", "6058,6060",
     "torch.nn.functional.pad(t, (1, 1), value=v)",
     lambda t, m, v, ct, kw: _C._aten_dispatch(
         "aten.constant_pad_nd.default", t, [1, 1], v)),
    ("scalar_tensor", "9766,9768",
     "torch.scalar_tensor(v, dtype=dt)",
     lambda t, m, v, ct, kw: _C._aten_dispatch(
         "aten.scalar_tensor.default", v, dtype=ct, **kw)),
    ("masked_fill", "14234,14236",
     "t.masked_fill(m, v)",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.masked_fill.Scalar", t, m, v)),
    ("masked_fill_", "14234,14236",
     "t.clone().masked_fill_(m, v)",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.masked_fill_.Scalar", t, m, v)),
    ("mul_scalar", "arith_scalar's host_const, correct before this round",
     "t * v",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.mul.Scalar", t, v)),
    ("div_scalar", "arith_scalar's host_const, correct before this round",
     "t / v",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.div.Scalar", t, v)),
    ("where_scalar_other", "host_const at 14495, correct before this round",
     "torch.where(m, t, v)",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.where.ScalarOther", m, t, v)),
    ("fill_", "17639,17641",
     "t.clone().fill_(v)",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.fill_.Scalar", t, v)),
    ("fill_tensor", "scalar_arg widening at 25588 (widen_f64_host)",
     "t.clone().fill_(torch.tensor(v, dtype=dt))",
     lambda t, m, v, ct, kw: _C._aten_dispatch(
         "aten.fill_.Tensor", t,
         _C._aten_dispatch("aten.scalar_tensor.default", v, dtype=ct, **kw))),
    ("where_scalar", "27223,27225",
     "torch.where(m, v, 1.5)",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.where.Scalar", m, v, 1.5)),
    ("where_scalar_self", "27787,27789",
     "torch.where(m, v, t)",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.where.ScalarSelf", m, v, t)),
    ("round_decimals", "32357",
     "t.clone().round_(decimals=2)",
     lambda t, m, v, ct, kw: _C._aten_dispatch("aten.round_.decimals", t, 2)),
]

# Converted at the same time and **not exercised by a cell in this file**, so
# that the report can say which is which rather than implying every converted
# site is proven. Named, with the reason each one is hard to reach.
_CONVERTED_BUT_UNEXERCISED = {
    "13466": "extremum_default's all-NaN early return -- `max`/`min` with no "
             "`dim` are in _shim_mps_host_readback_ops(), so the site is "
             "unreachable on mps by design; covered on cpu by test_shim.",
    "13642": "nan_shaped_like -- reached from `mode`/`median` families, which "
             "are likewise refused on mps.",
    "15278,15280": "the reduced-float divisor branch of `floor_divide.Scalar`; "
                   "`floor_divide` is refused on mps by name "
                   "(test_metalplace._HOST_COMPUTED_ON_MPS).",
    # The integration round found these two labelled as exercised by the
    # `mul_scalar`/`div_scalar` cases above. They are not: those cases reach
    # `arith_scalar`, and 14791/14922 are `remainder_op`/`fmod_op`, whose
    # `.Scalar` forms are refused on mps by the host-readback gate (measured:
    # `aten.remainder.Scalar: not implemented for the mps device. This kernel
    # reads the tensor back to host memory`).
    "14791,14793": "remainder_op's scalar operand -- `remainder.Scalar` is "
                   "refused on mps by the host-readback gate.",
    "14922,14924": "fmod_op's scalar operand -- `fmod.Scalar` is refused on "
                   "mps by the host-readback gate.",
}

# Nothing in `_CASES` may read anything back -- including `fill_.Tensor`,
# whose value *is* a tensor.
#
# The first version of this file allowed it eight bytes, on the reasoning that
# `scalar_arg` has to learn what the zero-dim tensor's one value is in order
# to dispatch at all. `test_metalplace.py` disagreed, and it was right: on an
# accelerator that read is a host readback of a **dispatched** tensor, which
# docs/devices/MPS.md §1.1 refuses, and it only became visible because this
# round made the key reach on Metal in the first place. `fill_inplace` now
# builds the fill from the value tensor *where the value already is*, and the
# allowance is gone rather than written down. The place a zero-dim device
# tensor legitimately is read is `test_a_zero_dim_device_tensor_can_stand_in_for_a_scalar`
# below, which is about the `Scalar` overloads and says so.
_DOWNLOAD_ALLOWED = {}

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
    m = torch.tensor(case["mask"]).reshape(case["shape"]).to(torch.bool)
    v = case["v"]
    r = eval(case["expr"])
    out[key] = {"values": [float(x) for x in r.to(torch.float64).flatten().tolist()],
                "dtype": str(r.dtype).split(".")[-1]}
print(json.dumps(out))
"""


def _operands(dtype, dev_kw):
    ctype = dt_utils.c_dtype(_C, dtype)
    dev = dev_kw.get("device")
    t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), ctype, dev)
    m = _C._tensor_from_flat(list(_MASK), list(_SHAPE), dt_utils.c_dtype(_C, "bool"), dev)
    return t, m, ctype


def _agree_on(device_label, dev_kw, value):
    """Every case in `_CASES`, element-wise against upstream, on one device."""
    want = {}
    for dtype in _FLOAT_DTYPES:
        for key, _, expr, _ in _CASES:
            want["%s/%s" % (key, dtype)] = {
                "dtype": dtype, "values": _VALUES, "shape": _SHAPE,
                "mask": _MASK, "v": value, "expr": expr}
    ref = _oracle(_ORACLE, want)

    checked = 0
    for dtype in _FLOAT_DTYPES:
        for key, sites, _, call in _CASES:
            # Fresh operands per case. Five of the fifteen (`fill_`,
            # `masked_fill_`, `round_`, ...) write through their receiver, and
            # a shared `t` made every case after them measure a tensor an
            # earlier case had already overwritten -- which showed up as
            # `mul.Scalar` disagreeing by the *previous* case's fill value.
            t, m, ctype = _operands(dtype, dev_kw)
            name = "%s/%s" % (key, dtype)
            got = call(t, m, value, ctype, dev_kw)
            assert str(got.device).startswith(device_label), (
                "%s came back on %s, not %s -- the fix must not move the "
                "result off the device the caller asked for"
                % (name, got.device, device_label))
            tol = dt_utils.tolerance_for(ref[name]["dtype"])
            bad = _close(_host(got), ref[name]["values"], tol.atol, tol.rtol)
            assert bad is None, "%s (aten.rs %s) disagrees with upstream: %s" % (
                name, sites, bad)
            checked += 1
    assert checked == len(_CASES) * len(_FLOAT_DTYPES), checked
    return checked


def test_constant_gate_agrees_with_upstream_on_cpu():
    """The same fifteen cases on the CPU, where they always worked.

    This is the regression half. `host_const`/`host_full` changed what every
    one of these call sites executes on **every** device, not only on Metal,
    so a value that the device path fixed and the host path broke would look
    like a Metal success. Run first, and against the same oracle.
    """
    assert _agree_on("cpu", {}, 0.5) == len(_CASES) * len(_FLOAT_DTYPES)


def test_constant_gate_agrees_with_upstream_on_mps():
    """BREAKS -> AGREES, or BREAKS -> REFUSES, for every case that reaches.

    Before this round ten of these fifteen raised `candle: unsupported
    const-set f64` or a `Metal contiguous to_dtype` message on all three
    Metal float dtypes -- a message naming a dtype the caller never asked
    for. Element-wise agreement with a subprocess oracle is the grade; a
    right answer computed in the wrong place is caught by the counter test
    below, not by this one.
    """
    mps = _mps_or_skip("the mps half of the constant gate")
    if mps is None:
        return
    assert _agree_on("mps", {"device": mps}, 0.5) == len(_CASES) * len(_FLOAT_DTYPES)


# The five constant writers the witnesses go through, as upstream spells them.
# Evaluated by `_ORACLE` with `t`, `m`, `v` and `dt` in scope.
_NARROWING_EXPRS = {
    "full": "torch.full(list(t.shape), v, dtype=dt)",
    "scalar_tensor": "torch.scalar_tensor(v, dtype=dt)",
    "fill_": "t.clone().fill_(v)",
    "masked_fill": "t.masked_fill(m, v)",
    "full_like": "torch.full_like(t, v)",
}


def _narrowing_case(dev_kw, dtype="float16", value=_ONE_ULP):
    ctype = dt_utils.c_dtype(_C, dtype)
    dev = dev_kw.get("device")
    # A fresh `t` per writer. `fill_` writes through its receiver, and with
    # one shared `t` the `masked_fill` after it measured an already-filled
    # tensor -- invisible while this only looked for the witness value, and
    # the first thing a bit-exact comparison with upstream reported.
    fresh = lambda: _C._tensor_from_flat(list(_VALUES), list(_SHAPE), ctype, dev)
    m = _C._tensor_from_flat(list(_MASK), list(_SHAPE),
                             dt_utils.c_dtype(_C, "bool"), dev)
    out = {}
    out["full"] = _host(_C._aten_dispatch(
        "aten.full.default", list(_SHAPE), value, dtype=ctype, **dev_kw))
    out["scalar_tensor"] = _host(_C._aten_dispatch(
        "aten.scalar_tensor.default", value, dtype=ctype, **dev_kw))
    out["fill_"] = _host(_C._aten_dispatch("aten.fill_.Scalar", fresh(), value))
    out["masked_fill"] = _host(_C._aten_dispatch(
        "aten.masked_fill.Scalar", fresh(), m, value))
    out["full_like"] = _host(_C._aten_dispatch("aten.full_like.default", fresh(), value))
    return out


def _upstream_narrowing(dtype, value):
    """The same five writers, asked of upstream in a subprocess."""
    ref = _oracle(_ORACLE, {
        key: {"dtype": dtype, "values": _VALUES, "shape": _SHAPE,
              "mask": _MASK, "v": value, "expr": expr}
        for key, expr in _NARROWING_EXPRS.items()})
    return {key: ref[key]["values"] for key in _NARROWING_EXPRS}


def _assert_bit_exact(where, dtype, value, out, ref):
    assert sorted(out) == sorted(ref), (sorted(out), sorted(ref))
    for key in sorted(out):
        assert out[key] == ref[key], (
            "%s(%r) as %s on %s gave %r, upstream %r -- bit-exact is the "
            "only grade here, since the whole difference is one ulp"
            % (key, value, dtype, where, out[key], ref[key]))


def _assert_single_rounded(where, out):
    for key, values in sorted(out.items()):
        hits = [v for v in values if v == _SINGLE_ROUNDED]
        trap = [v for v in values if v == _DOUBLE_ROUNDED]
        assert not trap, (
            "%s/%s on %s produced %r -- that is f16(f32(x)), the DOUBLE-rounded "
            "value. The scalar is being narrowed in two steps again "
            "(f64 -> f32 -> f16). The one-step answer is %r, one ulp of "
            "float16 away, and the two agree exactly on float32, so no "
            "float32 cell in this file would have noticed."
            % (key, "float16", where, _DOUBLE_ROUNDED, _SINGLE_ROUNDED))
        assert hits, (
            "%s/%s on %s never produced %r at all: %r. Either the constant "
            "stopped arriving or the case stopped exercising it."
            % (key, "float16", where, _SINGLE_ROUNDED, values))


def _one_step(where, dev_kw):
    if not _F16_SINGLE_ROUNDING:
        _skip.skip("   (skipped the f16 one-step witness on %s: %s is not "
                   "aarch64, and upstream c10 narrows f64 -> f16 through float "
                   "there -- test_f16_narrowing_follows_c10_* carries it)"
                   % (where, platform.machine()))
        return
    ref = _upstream_narrowing("float16", _ONE_ULP)
    assert ref["full"][0] == _SINGLE_ROUNDED, (
        "upstream on %s gave %r for full(%r, float16); aarch64 c10 rounds "
        "once, which is %r" % (platform.machine(), ref["full"][0], _ONE_ULP,
                               _SINGLE_ROUNDED))
    out = _narrowing_case(dev_kw)
    _assert_single_rounded(where, out)
    _assert_bit_exact(where, "float16", _ONE_ULP, out, ref)


def test_one_step_narrowing_on_cpu():
    """`float16`, by name, because `float32` cannot see this. aarch64 only.

    `host_const` applies the narrowing steps the call site passes, in order,
    and every call site passes exactly one: the storage dtype. Insert `f32`
    in front of it -- which is what narrowing on the device would force, since
    Metal cannot hold the `f64` -- and `0.031265258789971995` comes back
    `0.03125` instead of `0.031280517578125`. That is a defect only where
    upstream rounds once, which is aarch64; elsewhere this skips by name.
    """
    _one_step("cpu", {})


def test_one_step_narrowing_on_mps():
    """The same ulp, on the device the two-step narrowing would have been for."""
    mps = _mps_or_skip("one-step narrowing on mps")
    if mps is None:
        return
    _one_step("mps", {"device": mps})


def _c10_narrowing(where, dev_kw):
    """f16 per platform and both bf16 witnesses, bit-exact against upstream."""
    ref = _upstream_narrowing("float16", _ONE_ULP)
    assert ref["full"][0] == _F16_UPSTREAM, (
        "upstream on %s gave %r for full(%r, float16), expected %r for this "
        "platform's c10 rule" % (platform.machine(), ref["full"][0], _ONE_ULP,
                                 _F16_UPSTREAM))
    _assert_bit_exact(where, "float16", _ONE_ULP,
                      _narrowing_case(dev_kw, "float16", _ONE_ULP), ref)
    for value, expect in sorted(_BF16_WITNESSES.items()):
        ref = _upstream_narrowing("bfloat16", value)
        assert ref["full"][0] == expect, (
            "upstream gave %r for full(%r, bfloat16), expected bf16(f32(x)) "
            "= %r" % (ref["full"][0], value, expect))
        _assert_bit_exact(where, "bfloat16", value,
                          _narrowing_case(dev_kw, "bfloat16", value), ref)


def test_f16_and_bf16_narrowing_follow_c10_on_cpu():
    """issue #28, on every platform: the narrowing is c10's, asked of upstream.

    `float16` one rounding on aarch64 and two elsewhere; `bfloat16` two
    everywhere, with one witness against `half`'s truncating `from_f64` and
    one against a single rounding.
    """
    _c10_narrowing("cpu", {})


def test_f16_and_bf16_narrowing_follow_c10_on_mps():
    """The same witnesses on the device, where the constant is built on the host."""
    mps = _mps_or_skip("c10 narrowing on mps")
    if mps is None:
        return
    _c10_narrowing("mps", {"device": mps})


def test_the_two_devices_narrow_identically():
    """cpu and mps give the same bits, which is the point of narrowing on the host.

    Not implied by the two tests above: each of those admits any value that is
    not the trap. This one fails if the devices ever part company at all.
    """
    mps = _mps_or_skip("cross-device narrowing")
    if mps is None:
        return
    for dtype, value in [("float16", _ONE_ULP)] + [
            ("bfloat16", v) for v in sorted(_BF16_WITNESSES)]:
        on_cpu = _narrowing_case({}, dtype, value)
        on_mps = _narrowing_case({"device": mps}, dtype, value)
        assert sorted(on_cpu) == sorted(on_mps)
        for key in sorted(on_cpu):
            assert on_cpu[key] == on_mps[key], (
                "%s(%r) as %s: cpu %r vs mps %r"
                % (key, value, dtype, on_cpu[key], on_mps[key]))


# ---------------------------------------------------------------------------
# Placement. The counter is the only instrument that sees a host fallback.
# ---------------------------------------------------------------------------

def test_the_constant_gate_operators_compute_on_the_device():
    """Zero host downloads, for every case, on every Metal float dtype.

    **Why this test and not the values.** A fix that moved the whole operator
    to the host would pass every agreement assertion above and leave the
    `.device` label unchanged -- four rounds have planted exactly that twin
    and found it invisible to values and labels alike. `host_downloads == 0`
    is what sees it.

    **What crosses the boundary, and the bound on it.** A constant is *one
    element*: built on the host, narrowed there, and moved. The assertion is
    therefore per transfer -- `host_upload_bytes <= 8 * host_uploads`, eight
    being one `f64`'s width, the widest a single constant can be on the wire
    -- and not `== 0`. `== 0` would be false for the right reason, and would
    push the next round into computing the constant on the GPU, which is the
    thing that does not work.

    Per transfer rather than per dispatch because a kernel may need several
    constants: `round_.decimals` uploads five four-byte scalars, one each for
    candle's `gt(0.5)`, `eq(0.5)`, `ne(0.0)`, `eq(0.0)` and the power of ten.
    Twenty bytes, and not one of them a buffer. A host-computed twin fails
    this on its first transfer, because the answer it ships back is the whole
    result.
    """
    mps = _mps_or_skip("constant-gate placement")
    if mps is None:
        return

    measured = {}
    for dtype in _FLOAT_DTYPES:
        for key, sites, _, call in _CASES:
            t, m, ctype = _operands(dtype, {"device": mps})
            name = "%s/%s" % (key, dtype)
            before = _counters()
            got = call(t, m, 0.5, ctype, {"device": mps})
            after = _counters()
            d = _delta(before, after)
            measured[name] = d
            assert str(got.device).startswith("mps"), got.device
            allowed = _DOWNLOAD_ALLOWED.get(key, 0)
            assert d["host_download_bytes"] <= allowed, (
                "%s (aten.rs %s) read %d byte(s) back to the host, allowance "
                "%d. Its only host-side value is a scalar the parser read out "
                "of a Python object, travelling host -> device; a download "
                "beyond the allowance means the kernel itself moved."
                % (name, sites, d["host_download_bytes"], allowed))
            assert d["host_upload_bytes"] <= 8 * d["host_uploads"], (
                "%s (aten.rs %s) uploaded %d bytes in %d transfer(s) -- wider "
                "than one element each. A constant is one element; a transfer "
                "wider than that is a whole buffer built on the host and "
                "shipped, which is the silent fallback this file exists to "
                "refuse."
                % (name, sites, d["host_upload_bytes"], d["host_uploads"]))
    assert len(measured) == len(_CASES) * len(_FLOAT_DTYPES), len(measured)

    # And the device was actually asked to do something for the shaped cases.
    # Without this the assertions above are satisfied by a kernel that does
    # nothing at all.
    for dtype in _FLOAT_DTYPES:
        for key in ("full", "fill_", "masked_fill", "mul_scalar", "where_scalar_self"):
            name = "%s/%s" % (key, dtype)
            assert measured[name]["compute_encoders"] >= 1, (
                "%s encoded no Metal compute kernel: %r" % (name, measured[name]))


def test_the_rng_writers_narrow_on_the_host_and_draw_the_same_stream():
    """`normal_` and `bernoulli_`, the two cases whose constant is a vector.

    These did not die in `const_set`; they died one line later, in
    `Tensor::from_vec(values_f64, shape, &metal).to_dtype(storage)` --
    `Metal contiguous to_dtype F64 F32 not implemented`. The draws are the
    host's either way (this crate's own RNG, reproducing torch's stream), so
    narrowing them on the host is not a fallback; it is the only place the
    conversion exists, and it halves what crosses the boundary.

    **The assertion is the stream, not the distribution.** Seeded identically,
    `mps` and `cpu` must produce bit-identical draws -- a statistical check
    would pass for an RNG that had quietly changed its consumption rate, which
    is the failure `docs/models/SAMPLING.md` cares about.
    """
    mps = _mps_or_skip("the rng writers on mps")
    if mps is None:
        return

    for dtype in _FLOAT_DTYPES:
        ctype = dt_utils.c_dtype(_C, dtype)
        for op, args in (("aten.normal_.default", (0.0, 1.0)),
                         ("aten.bernoulli_.float", (0.25,))):
            drawn = {}
            for label, dev in (("cpu", None), ("mps", mps)):
                _C._shim_manual_seed(1234)
                t = _C._tensor_from_flat(list(_VALUES), list(_SHAPE), ctype, dev)
                before = _counters() if dev is not None else None
                out = _C._aten_dispatch(op, t, *args)
                if dev is not None:
                    d = _delta(before, _counters())
                    assert d["host_downloads"] == 0, (
                        "%s/%s read %d bytes back" % (op, dtype, d["host_download_bytes"]))
                    assert str(out.device).startswith("mps"), out.device
                drawn[label] = _host(out)
            assert drawn["cpu"] == drawn["mps"], (
                "%s/%s drew a different stream on the two devices:\n  cpu %r\n"
                "  mps %r\nSeeded identically, the values are the host's on "
                "both -- a difference means the narrowing moved, or the "
                "generator was consumed at a different rate."
                % (op, dtype, drawn["cpu"], drawn["mps"]))
            if op == "aten.bernoulli_.float":
                assert set(drawn["mps"]) <= {0.0, 1.0}, drawn["mps"]


# ---------------------------------------------------------------------------
# The guard. Closing the constant gate must not open the float64 hole.
# ---------------------------------------------------------------------------

_FLOAT64_ENTRIES = (
    ("ones", lambda mps: _C._aten_dispatch(
        "aten.ones.default", [2], dtype=_C.float64, device=mps)),
    ("zeros", lambda mps: _C._aten_dispatch(
        "aten.zeros.default", [2], dtype=_C.float64, device=mps)),
    ("full", lambda mps: _C._aten_dispatch(
        "aten.full.default", [2], 1.0, dtype=_C.float64, device=mps)),
    ("empty", lambda mps: _C._aten_dispatch(
        "aten.empty.memory_format", [2], dtype=_C.float64, device=mps)),
    ("arange", lambda mps: _C._aten_dispatch(
        "aten.arange.start_step", 0, 2, 1, dtype=_C.float64, device=mps)),
    ("scalar_tensor", lambda mps: _C._aten_dispatch(
        "aten.scalar_tensor.default", 1.0, dtype=_C.float64, device=mps)),
)

# The two `_to_copy` forms need an operand, and building it inside the bracket
# would charge its construction to the refusal -- which is precisely the
# measurement error this round is correcting. Setup first, bracket after.
_FLOAT64_TRANSFERS = (
    ("to_copy_cpu_to_mps",
     lambda mps: _C._aten_dispatch("aten.ones.default", [2], dtype=_C.float64),
     lambda mps, t: _C._aten_dispatch("aten._to_copy.default", t, device=mps)),
    ("to_copy_widen_on_mps",
     lambda mps: _C._aten_dispatch(
         "aten.ones.default", [2], dtype=_C.float32, device=mps),
     lambda mps, t: _C._aten_dispatch(
         "aten._to_copy.default", t, dtype=_C.float64)),
)


def test_float64_on_mps_is_still_refused_in_upstreams_words():
    """The 260 `float64_mps` cells stay refused, and refused *before* any work.

    This round moved every one of those constants onto the host, where `F64`
    always works -- so the obvious regression is a `host_const` that builds an
    `f64` on the host and then ships it to a device that has no `f64`, which
    candle will happily allocate. `host_const` refuses that by name, and
    `storage_for`'s `metal_dtype_gate` refuses the caller-facing form.

    Both halves are asserted: the message is upstream's sentence (naming
    `float64` **and** MPS, which is what five of the six original messages did
    not do), and the counter delta across the refusal is zero on every
    channel -- a refusal that has already uploaded bytes or encoded a kernel
    is a refusal in the wrong place.
    """
    mps = _mps_or_skip("the float64 entry gate")
    if mps is None:
        return

    cases = [(name, None, entry) for name, entry in _FLOAT64_ENTRIES]
    # Operands built before the bracket: an `_to_copy` case has to construct
    # its input, and charging that construction to the refusal would read as
    # work-before-refusal that is not there.
    cases += [(name, setup(mps), call) for name, setup, call in _FLOAT64_TRANSFERS]

    for name, operand, entry in cases:
        before = _counters()
        try:
            entry(mps) if operand is None else entry(mps, operand)
            raise AssertionError(
                "%s: float64 on mps was accepted. Metal has no f64; a tensor "
                "made this way can be cloned and nothing else, and every "
                "later message names a candle symbol instead of the fact."
                % name)
        except TypeError as e:
            text = str(e)
        after = _counters()

        assert "float64" in text and "MPS" in text, (
            "%s refused with %r, which names neither the dtype nor the device. "
            "Upstream's sentence is the contract here." % (name, text))
        d = _delta(before, after)
        assert d == dict.fromkeys(_NAMES, 0), (
            "%s refused *after* doing work: %r. The gate belongs in front of "
            "the allocation, not behind it." % (name, d))


def test_the_sites_this_round_converted_are_accounted_for():
    """Every converted call site is either exercised here or named unexercised.

    A count of converted sites is not evidence. This test is the ledger that
    keeps the two lists from drifting: `_CASES` carries the `aten.rs` line
    numbers each case covers, `_CONVERTED_BUT_UNEXERCISED` carries the ones no
    cell reaches **and why**, and the two together have to equal the set the
    round actually edited. A site that silently leaves both lists is the
    failure this file is written against.
    """
    exercised = set()
    for _, sites, _, _ in _CASES:
        for part in sites.split(","):
            part = part.strip().split(" ")[0]
            if part.isdigit():
                exercised.add(part)
    unexercised = set()
    for sites in _CONVERTED_BUT_UNEXERCISED:
        unexercised.update(s.strip() for s in sites.split(","))

    converted = {
        "6025", "6028", "6058", "6060", "9766", "9768", "13466", "13642",
        "14234", "14236", "14791", "14793", "14922", "14924", "15278", "15280",
        "17639", "17641", "27223", "27225", "27787", "27789", "32357",
    }
    assert exercised <= converted, sorted(exercised - converted)
    assert unexercised <= converted, sorted(unexercised - converted)
    missing = converted - exercised - unexercised
    assert not missing, (
        "these call sites were converted and appear in neither list: %s. "
        "Either write a cell that reaches one, or name it unexercised with "
        "the reason." % sorted(missing))
    assert not (exercised & unexercised), sorted(exercised & unexercised)

    # `Tensor::full` survives only where the device *can* materialise the
    # fill: the `u8` bool paths and the two `BoolReduce` identities. Anything
    # that hands it an `f64` is the defect coming back.
    _, text = test_shim._aten_rs_functions()
    offenders = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if "Tensor::full(" not in line or line.lstrip().startswith("///"):
            continue
        if "u8::from(" in line:
            # The `bool` paths hand the device a `u8`, which Metal's
            # `const_set` has. They were never part of this failure and are
            # left alone; matching them here would make this check fire on
            # code that is correct.
            continue
        if "as_f64()" in line or "f64::NAN" in line:
            offenders.append((line_no, line.strip()))
    assert not offenders, (
        "an f64 constant is being handed to the device again at %r. On Metal "
        "that is `candle: unsupported const-set f64`, and the `.fast_to()` on "
        "the following line never runs." % (offenders,))


_SCALAR_BY_TENSOR = (
    ("mul", "t * v", lambda t, m, v: _C._aten_dispatch("aten.mul.Scalar", t, v)),
    ("add", "t + v", lambda t, m, v: _C._aten_dispatch("aten.add.Scalar", t, v)),
    ("masked_fill", "t.masked_fill(m, v)",
     lambda t, m, v: _C._aten_dispatch("aten.masked_fill.Scalar", t, m, v)),
)

_ORACLE_BY_TENSOR = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented")
req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    dt = getattr(torch, case["dtype"])
    t = torch.tensor(case["values"]).reshape(case["shape"]).to(dt)
    m = torch.tensor(case["mask"]).reshape(case["shape"]).to(torch.bool)
    v = torch.tensor(case["v"], dtype=dt)
    r = eval(case["expr"])
    out[key] = [float(x) for x in r.to(torch.float64).flatten().tolist()]
print(json.dumps(out))
"""


def test_a_zero_dim_device_tensor_can_stand_in_for_a_scalar():
    """The other half of `fill_.Tensor`'s message, and a wider population.

    torch accepts a zero-dim tensor anywhere a `Scalar` is taken, and
    `scalar_arg` reproduces that. On Metal it widened the value to `f64`
    **before** moving it, and Metal has no `F32 -> F64`, so *every* `.Scalar`
    overload called this way died -- `mul`, `add` and `masked_fill` among
    them, none of which appeared in the sweep because the sweep never passes a
    tensor where a scalar is allowed. `widen_f64_host` moves first and widens
    on the host: same bits, narrower transfer, and the kernel reaches.

    **This is the one download in this file that is allowed, and it is allowed
    because the caller asked for it.** The read is of a value the caller
    handed in as a scalar-shaped argument, not of the operand being computed;
    `test_shim.py` classifies `scalar_arg` as exactly that, with the reason
    written beside it. One download per dispatch is the bound -- a kernel that
    had moved to the host would need at least one more for the operand.
    """
    mps = _mps_or_skip("a zero-dim device tensor as a Scalar")
    if mps is None:
        return

    want = {}
    for dtype in _FLOAT_DTYPES:
        for key, expr, _ in _SCALAR_BY_TENSOR:
            want["%s/%s" % (key, dtype)] = {
                "dtype": dtype, "values": _VALUES, "shape": _SHAPE,
                "mask": _MASK, "v": 2.0, "expr": expr}
    ref = _oracle(_ORACLE_BY_TENSOR, want)

    checked = 0
    for dtype in _FLOAT_DTYPES:
        ctype = dt_utils.c_dtype(_C, dtype)
        for key, _, call in _SCALAR_BY_TENSOR:
            name = "%s/%s" % (key, dtype)
            t, m, _ = _operands(dtype, {"device": mps})
            v = _C._aten_dispatch("aten.scalar_tensor.default", 2.0,
                                  dtype=ctype, device=mps)
            before = _counters()
            got = call(t, m, v)
            d = _delta(before, _counters())

            assert str(got.device).startswith("mps"), got.device
            tol = dt_utils.tolerance_for(dtype)
            assert _close(_host(got), ref[name], tol.atol, tol.rtol) is None, (
                "%s disagrees with upstream: %r vs %r"
                % (name, _host(got), ref[name]))
            assert d["host_downloads"] == 1, (
                "%s made %d host downloads. Exactly one is right: the "
                "zero-dim value the caller passed. Zero would mean the value "
                "never arrived; more than one means the operand travelled too."
                % (name, d["host_downloads"]))
            checked += 1
    assert checked == len(_SCALAR_BY_TENSOR) * len(_FLOAT_DTYPES), checked


def test_the_integer_arm_narrows_on_the_host_too():
    """The integer half of the factories, which `host_full`'s round left alone.

    docs/devices/matrix.md §4.3c named `full` and `scalar_tensor` at
    `int32_mps` as still quoting `Metal contiguous to_dtype I64 I32 not
    implemented`: the integer constant was built at `i64` **on the device**
    and narrowed there. Narrowed on the host instead, `scalar_tensor` has no
    broadcast to do and agrees with upstream.

    `full` does **not** move, and that is pinned rather than hidden: the
    narrowed element reaches the device, but filling a shape from it is a
    strided copy, and candle's Metal backend has no `I32` one
    (`Metal copy_strided I32 not implemented`). That is a different wall from
    the one §4.3c named, and the moment it falls this test goes red and asks
    for the cell to be graded.
    """
    mps = _mps_or_skip("the integer arm on mps")
    if mps is None:
        return
    ref = _oracle(_ORACLE, {"scalar_tensor/int32": {
        "dtype": "int32", "values": _VALUES, "shape": _SHAPE, "mask": _MASK,
        "v": 3, "expr": "torch.scalar_tensor(v, dtype=dt)"}})
    ctype = dt_utils.c_dtype(_C, "int32")
    before = _counters()
    got = _C._aten_dispatch("aten.scalar_tensor.default", 3, dtype=ctype, device=mps)
    d = _delta(before, _counters())
    assert str(got.device).startswith("mps"), got.device
    assert ref["scalar_tensor/int32"]["dtype"] == "int32", ref
    assert _host(got) == ref["scalar_tensor/int32"]["values"], (
        "scalar_tensor/int32 on mps: %r, upstream %r"
        % (_host(got), ref["scalar_tensor/int32"]["values"]))
    assert d["host_downloads"] == 0, d

    try:
        _C._aten_dispatch("aten.full.default", list(_SHAPE), 3, dtype=ctype, device=mps)
    except (RuntimeError, NotImplementedError) as e:
        msg = str(e).splitlines()[0]
    else:
        raise AssertionError(
            "aten.full.default now answers for int32 on mps. Grade it against "
            "upstream here and move the cell in docs/devices/matrix.md §7.18.")
    assert "copy_strided I32" in msg, (
        "full/int32 on mps fails for a different reason than the one "
        "recorded: %r" % msg)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)
             and getattr(fn, "__module__", None) == __name__]
    failures = _skip.run_tests(items, suite="test_mpsconst")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
