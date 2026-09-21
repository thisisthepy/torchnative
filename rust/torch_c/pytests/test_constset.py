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
  dtype -> test_the_constant_is_rounded_once_and_not_twice
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
_ATEN_RS = os.path.join(_HERE, "..", "src", "aten.rs")

# The three float dtypes Metal has. `float64` is absent because Metal has no
# double on any of twenty-two roads (docs/devices/matrix.md section 3.1) and a
# cell for it here would fail for a reason this file is not about.
_DTYPES = ("float32", "float16", "bfloat16")

# A value whose f64 -> f16 rounding differs from its f64 -> f32 -> f16
# rounding. Derived, not chosen: it sits just above the tie between two
# `float16` neighbours, close enough that `float32` snaps it back onto the tie
# and then the tie rounds the other way.
_DOUBLE_ROUNDING_WITNESS = 0.031265258789971995


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
    `float16` destination that rounds twice and lands on a different number.

    Asked of upstream too, because the claim is about what upstream's answer
    *is*, not merely about internal consistency. If upstream double-rounded,
    this build would have to as well and the whole argument would invert.
    """
    v = _DOUBLE_ROUNDING_WITNESS
    want = _oracle({"w/%s" % dt: {"case": "full", "dtype": dt, "value": v,
                                  "shape": [1], "source": _SOURCE,
                                  "cond": _COND}
                    for dt in _DTYPES})

    # The premise: the witness really does distinguish the two roundings.
    once = want["w/float16"]["values"][0]
    via32 = _host(_C._tensor_from_flat([v], [1], _C.float64)
                  .to(_C.float32).to(_C.float16))[0]
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


def test_no_float_constant_is_still_materialised_on_the_device():
    """Structural, derived from `aten.rs` rather than from a list kept by hand.

    The evidence here is **weaker than the agreement tests above and is meant
    to be**: there is no Metal dispatch counter in this build
    (docs/devices/matrix.md section 7.5), so nothing can observe where the fill
    actually happened. What this can do is refuse the *shape* of the defect --
    an `f64` handed to `Tensor::full` together with a device that is not
    pinned to the host -- so that a tenth operator written tomorrow in the old
    style is caught by arithmetic instead of by a sweep three weeks later.

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


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_constset")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
