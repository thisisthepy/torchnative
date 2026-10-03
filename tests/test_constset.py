"""Cause D of docs/devices/matrix.md section 4.3a: nine operators that answered
``candle: unsupported const-set f64`` on ``float32``/``float16``/``bfloat16``
tensors whose own dtype Metal supports perfectly well.

**Re-derived before it was implemented, and the clustering was misleading.**
Section 4.3a files these 25 cells under "candle cannot const-set the dtype",
which reads as nine candle gaps. It is one gap, it is not candle's, and the
door was already open in this file's own crate: ``aten.rs::host_const`` --
landed for ``mul.Scalar`` inside a rotary embedding (docs/devices/MPSFWD.md
section 2) -- builds the constant on the host and moves it, and its docstring
already states the whole argument. **Nine operators simply never adopted it.**
Every one of the 25 cells names ``f64`` while the cell's own dtype is
``float32``, ``float16`` or ``bfloat16``: the shim was asking the device to
materialise a *double* it had no reason to want, and Metal has no double at
all (docs/devices/matrix.md section 3.1).

So this is not a kernel. It is ``Tensor::full(v, shape, device)`` replaced by
``host_full``, which is ``host_const`` with a shape.

**The rounding trap, which is why this file exists rather than a one-line
diff.** The obvious cheap fix is to narrow the ``f64`` to ``f32`` on the host
and let the device const-set *that*, since Metal does have ``f32``. It is
wrong, and it is wrong in the direction nobody tests: for a ``float16``
destination it rounds twice.

    f16(f32(0.031265258789971995)) == 0.03125
    f16(    0.031265258789971995 ) == 0.031280517578125

``test_the_constant_is_rounded_once_and_not_twice`` is that witness, asked of
upstream as well as of this build, and it is the test that fails if anybody
"simplifies" ``host_full`` later.

Nullifications this file is meant to catch:

* ``host_full`` narrowing through ``f32`` instead of straight to the storage
  dtype -> test_the_constant_is_rounded_once_and_not_twice (aarch64 only:
  elsewhere upstream itself narrows ``f64 -> f16`` through ``f32``)
* ``candle_core::c10_bf16_from_f64`` reverted to ``half::bf16::from_f64``
  -> test_the_bf16_constant_is_narrowed_as_c10_does (issue #28)
* ``candle_core::c10_f16_from_f64`` rounding the wrong number of times for
  the platform -> test_the_f16_constant_follows_c10_on_this_platform
* ``host_full`` reverting to ``Tensor::full(.., device)``
  -> test_the_float_factories_answer_on_mps, and the structural derivation
* the fix applied to ``full`` only and not the family
  -> test_every_float_factory_in_cause_d_answers_on_mps
* the CPU path changing value -> test_the_cpu_answers_are_unchanged
* a value being invented instead of computed -> every AGREES test compares
  element-wise against upstream in a separate subprocess
"""

import json
import os
import re
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip

_HERE = os.path.dirname(os.path.abspath(__file__))
_ATEN_RS = os.path.join(_HERE, "..", "torchnative", "rust", "torch_c", "src", "aten.rs")

# The three float dtypes Metal has. `float64` is absent because Metal has no
# double on any of twenty-two roads (docs/devices/matrix.md section 3.1) and a
# cell for it here would fail for a reason this file is not about.
_DTYPES = ("float32", "float16", "bfloat16")

# A value whose f64 -> f16 rounding differs from its f64 -> f32 -> f16
# rounding. Derived, not chosen: it sits just above the tie between two
# `float16` neighbours, close enough that `float32` snaps it back onto the tie
# and then the tie rounds the other way.
_DOUBLE_ROUNDING_WITNESS = 0.031265258789971995


# Whether upstream narrows `f64 -> float16` in ONE rounding on this machine.
# c10 builds `Half` from `float16_t` on aarch64 (not CUDA) and from `float`
# everywhere else (torch/headeronly/util/Half.h:85-91), so on x86_64 upstream
# itself gives f16(f32(x)) and the one-step witness has nothing to witness.
# The tests below ask upstream as well, so a wrong guess here is a FAIL, not
# a silent pass.
import platform  # noqa: E402

_F16_SINGLE_ROUNDING = platform.machine().lower() in ("arm64", "aarch64")

# Two `float64 -> bfloat16` witnesses (issue #28). c10 has no
# `BFloat16(double)`, so upstream is bf16(f32(x)) on every platform.
#   _BF16_TRUNCATION_WITNESS: just above the tie between 1.0 and 1.0078125,
#     exact in f32. One and two roundings both give 1.0078125; `half`'s
#     `bf16::from_f64`, which drops the low 32 mantissa bits first, gives 1.0.
#   _BF16_TWO_STEP_WITNESS: just below the tie between 1.0078125 and
#     1.015625. f32 snaps it onto the tie, which goes to even: two roundings
#     give 1.015625 and one rounding gives 1.0078125.
_BF16_TRUNCATION_WITNESS = 1 + 2**-8 + 2**-22
_BF16_TWO_STEP_WITNESS = 1 + 3 * 2**-8 - 2**-30


def _bf16_round_once(x):
    """`float64 -> bfloat16` in one round-to-nearest-even step, exactly.

    Python's `round` on a float is exact ties-to-even, and dividing and
    multiplying by a power of two is exact, so this is the true single
    rounding (finite, normal-range inputs only, which is all it is used for).
    """
    import math
    e = math.frexp(abs(x))[1] - 1
    ulp = 2.0 ** (e - 7)
    return round(x / ulp) * ulp


def _bf16_rules(x):
    """The three candidate rules, computed here and not by either torch."""
    import struct
    f32 = struct.unpack("<f", struct.pack("<f", x))[0]
    hi = struct.unpack("<d", struct.pack(
        "<Q", struct.unpack("<Q", struct.pack("<d", x))[0] & ~0xFFFFFFFF))[0]
    return {"one": _bf16_round_once(x), "two": _bf16_round_once(f32),
            "truncate": _bf16_round_once(hi)}


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


_ORACLE = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    dt = getattr(torch, case["dtype"])
    name = case["case"]
    v = case["value"]
    if name == "full":
        t = torch.full(case["shape"], v, dtype=dt)
    elif name == "full_like":
        t = torch.full_like(torch.zeros(case["shape"], dtype=dt), v)
    elif name == "new_full":
        t = torch.zeros(case["shape"], dtype=dt).new_full(case["shape"], v)
    elif name == "scalar_tensor":
        t = torch.scalar_tensor(v, dtype=dt)
    elif name == "cast":
        t = torch.tensor([v], dtype=torch.float64).to(dt)
    elif name == "fill_":
        t = torch.zeros(case["shape"], dtype=dt)
        t.fill_(v)
    elif name == "constant_pad_nd":
        t = torch.nn.functional.pad(
            torch.zeros(case["shape"], dtype=dt), [1, 1], value=v)
    elif name == "round_decimals":
        t = torch.round(torch.tensor(case["source"], dtype=dt), decimals=2)
    elif name == "round__decimals":
        t = torch.tensor(case["source"], dtype=dt).clone()
        t.round_(decimals=2)
    elif name == "where_scalar_self":
        cond = torch.tensor(case["cond"], dtype=torch.bool)
        t = torch.where(cond, v, torch.tensor(case["source"], dtype=dt))
    else:
        raise AssertionError("unknown case " + name)
    out[key] = {
        "values": [float(x) for x in t.to(torch.float64).flatten().tolist()],
        "dtype": str(t.dtype).split(".")[-1],
    }
print(json.dumps(out))
"""


def _oracle(payload):
    """Upstream's answer, in a subprocess with no `_C` on the path."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run([sys.executable, "-c", _ORACLE, json.dumps(payload)],
                          capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n"
                             + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


_SHAPE = [2, 3]
_SOURCE = [1.25, -2.5, 0.5, 3.75, -0.125, 2.0]
_COND = [True, False, True, False, True, False]
_VALUE = 1.5


def _cases():
    """(key, case name, how to ask this build) for every operator in cause D."""
    def _zeros(dtype, device):
        return _C._aten_dispatch("aten.zeros.default", _SHAPE,
                                 dtype=getattr(_C, dtype), device=device)

    def _src(dtype, device):
        t = _C._tensor_from_flat(_SOURCE, [len(_SOURCE)], _C.float64) \
              .to(getattr(_C, dtype))
        return t.to("mps") if device is not None else t

    return (
        ("full", lambda dt, dev: _C._aten_dispatch(
            "aten.full.default", _SHAPE, _VALUE,
            dtype=getattr(_C, dt), device=dev)),
        ("full_like", lambda dt, dev: _C._aten_dispatch(
            "aten.full_like.default", _zeros(dt, dev), _VALUE)),
        ("new_full", lambda dt, dev: _C._aten_dispatch(
            "aten.new_full.default", _zeros(dt, dev), _SHAPE, _VALUE)),
        ("scalar_tensor", lambda dt, dev: _C._aten_dispatch(
            "aten.scalar_tensor.default", _VALUE,
            dtype=getattr(_C, dt), device=dev)),
        ("fill_", lambda dt, dev: _C._aten_dispatch(
            "aten.fill_.Scalar", _zeros(dt, dev), _VALUE)),
        ("constant_pad_nd", lambda dt, dev: _C._aten_dispatch(
            "aten.constant_pad_nd.default", _zeros(dt, dev), [1, 1], _VALUE)),
        ("round_decimals", lambda dt, dev: _C._aten_dispatch(
            "aten.round.decimals", _src(dt, dev), 2)),
        ("round__decimals", lambda dt, dev: _C._aten_dispatch(
            "aten.round_.decimals", _src(dt, dev).clone(), 2)),
        ("where_scalar_self", lambda dt, dev: _C._aten_dispatch(
            "aten.where.ScalarSelf",
            _C._tensor_from_flat([1. if c else 0. for c in _COND],
                                 [len(_COND)], _C.float64).to(_C.bool)
             .to("mps") if dev is not None else
            _C._tensor_from_flat([1. if c else 0. for c in _COND],
                                 [len(_COND)], _C.float64).to(_C.bool),
            _VALUE,
            _src(dt, dev))),
    )


def _payload(dtypes):
    req = {}
    for name, _ in _cases():
        for dt in dtypes:
            req["%s/%s" % (name, dt)] = {
                "case": name, "dtype": dt, "value": _VALUE,
                "shape": _SHAPE, "source": _SOURCE, "cond": _COND,
            }
    return req


def _compare(device, label):
    """Every cause-D operator, on `device`, against upstream. Returns failures."""
    want = _oracle(_payload(_DTYPES))
    bad = []
    for name, ask in _cases():
        for dt in _DTYPES:
            key = "%s/%s" % (name, dt)
            try:
                got = ask(dt, device)
            except Exception as e:  # noqa: BLE001 -- the refusal is the finding
                bad.append("%s on %s raised %s: %s"
                           % (key, label, type(e).__name__,
                              str(e).splitlines()[0]))
                continue
            values = _host(got)
            expect = want[key]["values"]
            if values != expect:
                bad.append("%s on %s gave %r, upstream gave %r"
                           % (key, label, values, expect))
            name_got = str(got.dtype).replace("torch.", "")
            if name_got != want[key]["dtype"]:
                bad.append("%s on %s returned dtype %s, upstream %s"
                           % (key, label, name_got, want[key]["dtype"]))
    return bad


def test_the_cpu_answers_are_unchanged():
    """Grade: **agrees**, on the CPU, for all nine operators.

    The regression half. `host_full` must not move a single CPU value: the
    conversion it performs is the same single `f64 -> storage` conversion the
    old `Tensor::full(f64, shape, cpu).fast_to(storage)` performed, and the
    only thing that changed is where the fill is materialised.

    Compared element-wise and exactly -- not within a tolerance. These are
    constants, and a constant that needs a tolerance has been computed.
    """
    bad = _compare(None, "cpu")
    assert not bad, "cause D regressed the cpu:\n  " + "\n  ".join(bad)


def test_every_float_factory_in_cause_d_answers_on_mps():
    """Grade: **agrees**, on `mps`, for all nine operators -- the 25 cells.

    This is the test the round exists to pass. Before the change every one of
    these raised `candle: unsupported const-set f64`; the assertion message
    prints whichever ones still do, so a partial fix reads as a list of names
    rather than as one failure.
    """
    device = _mps_or_skip("the cause D factories")
    if device is None:
        return
    bad = _compare(device, "mps")
    assert not bad, "cause D is not closed on mps:\n  " + "\n  ".join(bad)


def test_the_constant_is_rounded_once_and_not_twice():
    """The load-bearing measurement, and the trap in the cheap fix.

    Metal has `f32` but not `f64`, so the tempting repair is to narrow the
    scalar to `f32` on the host and let the device const-set it. For a
    `float16` destination that rounds twice and lands on a different number
    -- **on aarch64**, where upstream rounds `f64 -> f16` once. On every other
    platform upstream itself rounds through `float` (c10 `Half(float)`), the
    two paths agree, and this witness has nothing to tell apart: it is
    skipped there by name, and `test_the_f16_constant_follows_c10_on_this_platform`
    carries the agreement instead.

    Asked of upstream too, because the claim is about what upstream's answer
    *is*, not merely about internal consistency.
    """
    v = _DOUBLE_ROUNDING_WITNESS
    want = _oracle({"w/%s" % dt: {"case": "full", "dtype": dt, "value": v,
                                  "shape": [1], "source": _SOURCE,
                                  "cond": _COND}
                    for dt in _DTYPES})
    once = want["w/float16"]["values"][0]
    via32 = _host(_C._tensor_from_flat([v], [1], _C.float64)
                  .to(_C.float32).to(_C.float16))[0]

    if not _F16_SINGLE_ROUNDING:
        # State the platform's semantics before stepping aside: upstream here
        # must be the two-step value, or the platform guess above is wrong.
        assert once == via32 == 0.03125, (
            "on %s upstream gave %r for f16(%r) and f16(f32(x)) is %r; c10 "
            "narrows through float off aarch64, so these should both be "
            "0.03125" % (platform.machine(), once, v, via32))
        _skip.skip("   (skipped the f16 one-vs-two-step witness: %s is not "
                   "aarch64, and upstream c10 narrows f64 -> f16 through "
                   "float there, so both paths give %r)"
                   % (platform.machine(), once))
        return

    # The premise: the witness really does distinguish the two roundings.
    assert once != via32, (
        "the witness %r no longer distinguishes single from double rounding "
        "(f16 direct %r, f16 via f32 %r). This test proves nothing until a "
        "witness that does is put back -- do not delete it, replace it."
        % (v, once, via32))

    for dt in _DTYPES:
        for label, device in (("cpu", None), ("mps", _C.device("mps"))):
            if device is not None and _mps_or_skip("the rounding witness") is None:
                continue
            got = _C._aten_dispatch("aten.full.default", [1], v,
                                    dtype=getattr(_C, dt), device=device)
            assert _host(got) == want["w/%s" % dt]["values"], (
                "torch.full([1], %r, dtype=%s) on %s gave %r where upstream "
                "gives %r. Narrowing the constant through float32 before the "
                "device sees it rounds twice; the constant has to be "
                "converted straight to the storage dtype, once, on the host."
                % (v, dt, label, _host(got), want["w/%s" % dt]["values"]))


def _ask_narrowing(case, dt, v, device):
    if case == "full":
        return _C._aten_dispatch("aten.full.default", [1], v,
                                 dtype=getattr(_C, dt), device=device)
    t = _C._tensor_from_flat([v], [1], _C.float64).to(getattr(_C, dt))
    return t.to("mps") if device is not None else t


def _narrowing_agrees(dt, witnesses):
    """`full` and the f64 cast, on cpu and mps, against upstream. Failures."""
    cases = ("full", "cast")
    want = _oracle({"%s/%r" % (c, v): {"case": c, "dtype": dt, "value": v,
                                       "shape": [1], "source": _SOURCE,
                                       "cond": _COND}
                    for c in cases for v in witnesses})
    bad = []
    for label, device in (("cpu", None), ("mps", _C.device("mps"))):
        if device is not None and _mps_or_skip(
                "the %s narrowing witness on mps" % dt) is None:
            continue
        for c in cases:
            for v in witnesses:
                got = _host(_ask_narrowing(c, dt, v, device))
                if got != want["%s/%r" % (c, v)]["values"]:
                    bad.append("%s(%r) as %s on %s gave %r, upstream %r"
                               % (c, v, dt, label, got,
                                  want["%s/%r" % (c, v)]["values"]))
    return want, bad


def test_the_bf16_constant_is_narrowed_as_c10_does():
    """`float64 -> bfloat16` is bf16(f32(x)) upstream, on every platform.

    issue #28: `half`'s `bf16::from_f64` truncates the low 32 mantissa bits
    and then rounds, which is neither one rounding nor two. Two witnesses,
    each asked of upstream in a subprocess and of this build on cpu and on
    mps, through `full` and through `.to(torch.bfloat16)`:

    * the truncation witness, where the truncating rule disagrees with c10;
    * the two-step witness, where a single rounding disagrees with c10.
    """
    w = (_BF16_TRUNCATION_WITNESS, _BF16_TWO_STEP_WITNESS)
    want, bad = _narrowing_agrees("bfloat16", w)

    # The premises, against rules computed in this file rather than by either
    # torch: each witness separates c10's rule from the one it targets.
    for c in ("full", "cast"):
        up = [want["%s/%r" % (c, v)]["values"][0] for v in w]
        r0, r1 = _bf16_rules(w[0]), _bf16_rules(w[1])
        assert up[0] == r0["two"] == 1.0078125 != r0["truncate"], (
            "truncation witness: upstream %s gave %r; rules %r" % (c, up[0], r0))
        assert up[1] == r1["two"] == 1.015625 != r1["one"], (
            "two-step witness: upstream %s gave %r; rules %r" % (c, up[1], r1))

    assert not bad, "bfloat16 narrowing disagrees with upstream:\n  " + \
        "\n  ".join(bad)


def test_the_f16_constant_follows_c10_on_this_platform():
    """`float64 -> float16` against upstream, on every platform.

    The per-platform half of the witness above: one rounding on aarch64,
    f16(f32(x)) elsewhere. Upstream is asked; the value it gives is checked
    against the rule this platform should have, so the platform guess is
    itself under test.
    """
    v = _DOUBLE_ROUNDING_WITNESS
    want, bad = _narrowing_agrees("float16", (v,))
    expect = 0.031280517578125 if _F16_SINGLE_ROUNDING else 0.03125
    for c in ("full", "cast"):
        got = want["%s/%r" % (c, v)]["values"][0]
        assert got == expect, (
            "upstream %s(%r) as float16 on %s is %r, expected %r for this "
            "platform's c10 rule" % (c, v, platform.machine(), got, expect))
    assert not bad, "float16 narrowing disagrees with upstream:\n  " + \
        "\n  ".join(bad)


def test_no_float_constant_is_still_materialised_on_the_device():
    """Structural, derived from `aten.rs` rather than from a list kept by hand.

    The evidence here is **weaker than the agreement tests above and is meant
    to be**, but not for the reason first written here. That reason -- "there
    is no Metal dispatch counter in this build" -- was false when this file
    landed: docs/devices/matrix.md section 7.11 built one, and this file was
    scoped against a tree that predated it. Where the fill happens *is*
    observable, and `test_the_mps_fill_is_bracketed_by_the_metal_counters`
    below observes it.

    What this test adds on top of that is coverage of operators nobody has
    written yet: it refuses the *shape* of the defect -- an `f64` handed to
    `Tensor::full` together with a device that is not pinned to the host --
    so that a tenth operator written tomorrow in the old style is caught by
    arithmetic instead of by a sweep three weeks later.

    `&Device::Cpu` is excluded because that is `host_const`'s own body and the
    one place the pattern is correct by construction.
    """
    src = open(_ATEN_RS).read()
    offenders = []
    for m in re.finditer(r"Tensor::full\(([^;]*?)\)\s*\n?", src):
        call = m.group(1)
        if "as_f64()" not in call and "f64::" not in call and "power" not in call:
            continue
        if "&Device::Cpu" in call:
            continue
        line = src.count("\n", 0, m.start()) + 1
        offenders.append("aten.rs:%d  Tensor::full(%s)"
                         % (line, " ".join(call.split())[:90]))
    assert not offenders, (
        "an f64 constant is still being materialised on a device that may be "
        "Metal -- cause D by construction:\n  " + "\n  ".join(offenders))


def test_host_full_exists_and_converts_before_it_broadcasts():
    """The ordering inside `host_full`, which is what the witness above tests
    the *effect* of. Kept as a separate structural check so that a change to
    the helper is caught even on a machine with no `mps` device, where the
    interesting half of this file skips.
    """
    src = open(_ATEN_RS).read()
    m = re.search(r"fn host_full[\s\S]*?\n\}\n", src)
    assert m, "aten.rs no longer defines host_full"
    body = m.group(0)
    assert "host_const(" in body, (
        "host_full no longer routes through host_const, which is where the "
        "single-conversion argument lives")
    convert = body.index("host_const(")
    broadcast = body.index("broadcast_as")
    assert convert < broadcast, (
        "host_full broadcasts before it converts. Broadcasting an f64 and "
        "converting the block afterwards puts the f64 on the device, which is "
        "the defect this helper exists to remove")


# Bytes one element of each dtype occupies. The upload assertion below is
# built on these being the *storage* widths: an `f32` narrowing would upload 4
# bytes for a `float16` destination and an unconverted `f64` would upload 8.
_ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2}


def test_the_mps_fill_is_bracketed_by_the_metal_counters():
    """Where the fill happens, measured rather than argued.

    `_C._metal_counters()` (docs/devices/matrix.md section 7.11) is the only
    instrument in this project that can see a silent host fallback -- values
    and `.device` labels cannot, and AGENTS.md section 13.1 records three rounds
    where nothing but a counter caught it. This file originally said no such
    counter existed, which was true of the tree it was scoped against and is
    not true of this one.

    **Bracket**, stated because counters from different brackets read like
    regressions side by side: each reading is taken immediately before and
    immediately after **one** `_aten_dispatch` call, in this process, after an
    unmeasured dispatch has warmed the device.

    What the three deltas mean, and what each one would catch:

        host_downloads   == 0  -- nothing was read back. A fill computed on
                                  the host and returned under an `mps` label
                                  is the failure docs/graph/NPU2.md records.
        host_uploads     == 1  -- exactly one crossing. `host_const` converts
                                  a single scalar and moves it; a host fill of
                                  the whole block also uploads once, which is
                                  why the byte count below is the load-bearing
                                  assertion and not this one.
        host_upload_bytes == itemsize -- **one element of the storage dtype**,
                                  not `numel * itemsize` (a host-side fill of
                                  the whole tensor) and not 8 (an `f64` that
                                  was never converted). For `float16` this is
                                  2, so the `f32` narrowing that
                                  `test_the_constant_is_rounded_once_and_not_twice`
                                  catches by value is caught here by width.
        compute_encoders >= 1  -- the broadcast to `_SHAPE` ran on the device.

    `scalar_tensor` is deliberately not in this loop: its result has one
    element, so there is nothing to broadcast and `compute_encoders` is
    legitimately 0. Asserting `> 0` for it would be asserting a bug.
    """
    device = _mps_or_skip("the metal counter bracket")
    if device is None:
        return
    counters = getattr(_C, "_metal_counters", None)
    assert counters is not None, (
        "_C._metal_counters() is gone. It is the only instrument that can "
        "observe where a fill happened; without it this file is back to the "
        "structural argument it was written with (docs/devices/matrix.md 7.11)")
    assert counters().get("built"), (
        "the Metal counters report built=False, so no Metal device was ever "
        "constructed -- the deltas below would all be 0 and this test would "
        "pass while measuring nothing")

    numel = _SHAPE[0] * _SHAPE[1]
    _C._aten_dispatch("aten.full.default", _SHAPE, _VALUE,
                      dtype=_C.float32, device=device)  # warm; not measured
    bad = []
    for dt in _DTYPES:
        keys = ("host_downloads", "host_uploads", "host_upload_bytes",
                "compute_encoders")
        before = counters()
        got = _C._aten_dispatch("aten.full.default", _SHAPE, _VALUE,
                                dtype=getattr(_C, dt), device=device)
        after = counters()
        d = {k: after[k] - before[k] for k in keys}
        if not str(got.device).startswith("mps"):
            bad.append("full/%s answered on %s under an mps request"
                       % (dt, got.device))
        if d["host_downloads"] != 0:
            bad.append("full/%s moved host_downloads by %d -- something was "
                       "read back to the host, which is the silent fallback "
                       "this bracket exists to see"
                       % (dt, d["host_downloads"]))
        if d["host_uploads"] != 1:
            bad.append("full/%s crossed to the device %d times, expected "
                       "exactly 1 (host_const moves one converted scalar)"
                       % (dt, d["host_uploads"]))
        if d["host_upload_bytes"] != _ITEMSIZE[dt]:
            bad.append(
                "full/%s uploaded %d bytes, expected %d -- one element of the "
                "storage dtype. %d would mean the whole %d-element block was "
                "filled on the host, and 8 would mean an unconverted f64 "
                "crossed; for float16, 4 would mean the constant was narrowed "
                "through f32 first, which is the double-rounding defect"
                % (dt, d["host_upload_bytes"], _ITEMSIZE[dt],
                   numel * _ITEMSIZE[dt], numel))
        if d["compute_encoders"] < 1:
            bad.append("full/%s ran %d compute shaders, expected at least 1 "
                       "for the broadcast of the uploaded scalar to %r"
                       % (dt, d["compute_encoders"], _SHAPE))
    assert not bad, ("the mps fill is not where it is supposed to be:\n  "
                     + "\n  ".join(bad))


_NAN_ORACLE = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    t = torch.tensor(case["source"], dtype=getattr(torch, case["dtype"]))
    if case["case"] == "amax":
        r = torch.amax(t, dim=0)
    elif case["case"] == "amin":
        r = torch.amin(t, dim=0)
    elif case["case"] == "max_dim":
        r = torch.max(t, dim=0).values
    elif case["case"] == "min_dim":
        r = torch.min(t, dim=0).values
    else:
        raise AssertionError("unknown case " + case["case"])
    out[key] = {"values": [float(x) for x in
                           r.to(torch.float64).flatten().tolist()],
                "dtype": str(r.dtype).split(".")[-1]}
print(json.dumps(out))
"""

# A NaN in the input is what reaches the seed. Without one, `nan_along_dim`
# returns early and the `host_full(f64::NAN, ..)` line is never executed --
# which is exactly why the 2026-09-20 sweep never saw these call sites.
_NAN_SOURCE = [1.25, float("nan"), 0.5, 3.75]


_NAN_OPS = {"amax": "aten.amax.default", "max_dim": "aten.max.dim",
            "min_dim": "aten.min.dim"}


def _ask_nan(case, dt, on_mps):
    src = (_C._tensor_from_flat(_NAN_SOURCE, [len(_NAN_SOURCE)], _C.float64)
           .to(getattr(_C, dt)))
    if on_mps:
        src = src.to("mps")
    arg = [0] if case == "amax" else 0
    out = _C._aten_dispatch(_NAN_OPS[case], src, arg)
    return _host(out[0] if isinstance(out, tuple) else out)


def _nan_oracle(cases):
    req = {"%s/%s" % (c, dt): {"case": c, "dtype": dt, "source": _NAN_SOURCE}
           for c in cases for dt in _DTYPES}
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run(
        [sys.executable, "-c", _NAN_ORACLE, json.dumps(req)],
        capture_output=True, text=True, timeout=600, env=env)
    assert proc.returncode == 0, ("upstream NaN oracle failed:\n"
                                  + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _same(a, b):
    return (a != a and b != b) or a == b


def test_the_nan_seed_call_sites_answer_where_they_are_reachable():
    """The call sites section 4.3b found outside the sweep, actually run.

    `max.default`/`min.default`'s all-NaN early return and `nan_shaped_like`
    (used by `max.dim` and `min.dim`) build an `f64` NaN and move it, and the
    same change that adopted `host_full` for the nine cause-D operators
    adopted it here. Section 4.3b calls them "fixed too" and offers the
    structural derivation as the evidence -- which is a claim about the source
    text, not about what the machine does. **Nothing had ever executed them.**

    They are unreachable through the matrix sweep by construction: it builds
    one shape per operator and that shape has no NaN, so the seed is dead code
    for it. That is precisely the blindness AGENTS.md section 13.1 records for
    `clamp`, whose cell had no NaN either and graded AGREES for months while
    the operator was wrong.

    Graded **agrees** on the `cpu`, element-wise against upstream in a
    separate subprocess, NaN compared as NaN rather than by `==`, no
    tolerance: a seeded NaN either is one or is not.

    `amin` is not in this table because this build does not implement
    `aten.amin.default` at all, which is asserted rather than assumed -- an
    operator silently dropped from a table is how a gap stops being counted.
    """
    assert "aten.amin.default" not in _C._aten_implemented(), (
        "this build now implements aten.amin.default. It has the same NaN "
        "seed as the others and belongs in this table -- add it rather than "
        "leaving this assertion to fail")
    want = _nan_oracle(_NAN_OPS)
    saw_a_nan = False
    bad = []
    for case in _NAN_OPS:
        for dt in _DTYPES:
            key = "%s/%s" % (case, dt)
            expect = want[key]["values"]
            saw_a_nan = saw_a_nan or any(v != v for v in expect)
            try:
                values = _ask_nan(case, dt, on_mps=False)
            except Exception as e:  # noqa: BLE001 -- the refusal is the finding
                bad.append("%s on cpu raised %s: %s"
                           % (key, type(e).__name__, str(e).splitlines()[0]))
                continue
            if len(values) != len(expect) or not all(
                    _same(a, b) for a, b in zip(values, expect)):
                bad.append("%s on cpu gave %r, upstream gave %r"
                           % (key, values, expect))
    assert saw_a_nan, (
        "not one upstream answer in this table is NaN, so the seed these call "
        "sites exist for was never reached and this test proves nothing. Put "
        "a NaN back into _NAN_SOURCE rather than deleting this check")
    assert not bad, ("the NaN seed call sites do not agree:\n  "
                     + "\n  ".join(bad))


def test_the_nan_seed_on_mps_is_either_refused_or_a_recorded_defect():
    """What the `mps` half of those call sites actually does, measured.

    Two different answers, and only one of them is acceptable:

    * `max.dim` and `min.dim` **refuse by name** on `mps` -- they are in the
      host-readback family, so `nan_shaped_like` is not reachable there at
      all. The refusal is the right outcome and is asserted here so that a
      later round which makes them answer has to come back to this test.

    * `aten.amax.default` **answers, and answers wrongly.** With a NaN in the
      input it returns the largest non-NaN element where upstream returns NaN.
      This is not cause D and it is not this round's change: the cause is
      `aten.rs::amax_keepdim_anywhere`, which routes a Metal tensor to
      candle's `max_keepdim` -- the NaN-skipping fold that the docstring above
      `nan_along_dim` calls "the third repair of one predicate and ... meant
      to be the last". It is the fourth, it predates both this branch and the
      int8 round (byte-identical in `7457147`'s `aten.rs`), and the same
      helper is called from the softmax row-max, so the blast radius is wider
      than `amax`.

    **This test pins the defect rather than hiding it.** It asserts the wrong
    answer is still the wrong answer, so the moment somebody repairs
    `amax_keepdim_anywhere` this test goes RED and forces them here to replace
    it with the agreement assertion. A defect nobody has written down is
    rediscovered; a defect with a failing test is a decision. Deleting this
    test without fixing the operator puts it back in the first category.
    """
    device = _mps_or_skip("the NaN seed on mps")
    if device is None:
        return
    want = _nan_oracle(_NAN_OPS)
    bad = []

    for case in ("max_dim", "min_dim"):
        for dt in _DTYPES:
            try:
                values = _ask_nan(case, dt, on_mps=True)
            except (RuntimeError, NotImplementedError) as e:
                msg = str(e)
                for token in (_NAN_OPS[case], "mps"):
                    if token not in msg:
                        bad.append("%s/%s refuses on mps without naming %r: %s"
                                   % (case, dt, token, msg.splitlines()[0]))
                continue
            bad.append(
                "%s/%s now answers %r on mps. It is in the host-readback "
                "family, so either it has started returning a value the GPU "
                "did not compute, or the family changed and this test should "
                "be comparing against upstream %r instead"
                % (case, dt, values, want["%s/%s" % (case, dt)]["values"]))

    for dt in _DTYPES:
        expect = want["amax/%s" % dt]["values"]
        assert any(v != v for v in expect), (
            "upstream's amax/%s over %r is no longer NaN, so this pin is "
            "measuring nothing" % (dt, _NAN_SOURCE))
        values = _ask_nan("amax", dt, on_mps=True)
        if all(_same(a, b) for a, b in zip(values, expect)):
            bad.append(
                "amax/%s on mps now agrees with upstream (%r). The recorded "
                "defect in amax_keepdim_anywhere appears to be FIXED -- that "
                "is good news: delete this loop and move amax into "
                "test_the_nan_seed_call_sites_answer_where_they_are_reachable "
                "on mps as well." % (dt, values))
        elif values != [max(v for v in _NAN_SOURCE if v == v)]:
            bad.append(
                "amax/%s on mps gave %r, which is neither upstream's answer "
                "%r nor the known NaN-skipping one %r. The defect has changed "
                "shape and needs re-diagnosing rather than re-pinning"
                % (dt, values, expect,
                   [max(v for v in _NAN_SOURCE if v == v)]))

    assert not bad, ("the mps NaN seed behaviour is not what is recorded:\n  "
                     + "\n  ".join(bad))


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_constset")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
