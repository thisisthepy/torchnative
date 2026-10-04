"""`int8` on Metal: the second vendored crate, and what it actually moved.

docs/devices/matrix.md §7.13 declined `DType::I8` on Metal and said why in one
sentence: a change confined to `candle-core` would let the *buffer* be built
and leave every kernel refusing for want of a shader symbol, because
`candle_metal_kernels::DType` has six variants and the MSL sources instantiate
for those six only. §7.15 then read the instantiation and found it
macro-driven. This file is the round that vendored the **second** crate and
measured the result rather than predicting it.

**The three grades, kept apart (AGENTS.md §16).**

    builds     the MSL compiles and the symbol resolves.
    reaches    the dispatch returns a tensor on `mps`.
    AGREES     every element matches upstream's answer, computed in a
               SEPARATE SUBPROCESS that has no `_C` on its path.

Only the third is claimed below, and it is claimed per operator.

**Why the inputs are the ones they are.** `int8` is eight bits and torch wraps
on overflow rather than saturating, so a shader that looks right on
`[1, 2, 3]` can be wrong at the only values that distinguish `int8` from a
wider type. `_VALUES` and `_OTHER` are chosen so that `add`, `sub` and `mul`
each wrap at least twice -- `-128 + -1` is `127`, `127 - -3` is `-126`,
`-128 * -1` is `-128` -- and so that both endpoints `-128` and `127` are
present in **both** operands (`-128`'s negation is not representable, which is
where a widened-then-narrowed kernel and a true 8-bit one part company). A previous round leaned on
`0u8 - 5 == 251` for `abs`, which is the same arithmetic read as a trick
rather than as a hazard; the answer here comes from upstream, never from
intuition.

**Why the counter and not a source argument.** AGENTS.md §13.1: a host-computed
twin leaves every value correct, every agreement test green and the `.device`
label unchanged, and the only instrument that has ever caught one is a
dispatch counter. So each agreement assertion below is paired with a
`_C._metal_counters()` bracket -- `host_downloads == 0` across the dispatch,
and `compute_encoders >= 1` wherever the operator is not legitimately
blit-only. Building operands happens before the first snapshot and reading
results back happens after the second, so neither is charged to the operator.

**What this file does not claim.** It does not claim the whole `int8_mps`
column. 99 of the 284 cells §4.3a counted do not have `int8_cpu` at AGREES
either, so Metal cannot be the thing that makes them agree; and `int8` has no
`sin`/`exp`/`sqrt` on Metal for the reason §7.15 gave and upstream gives too.
The sweep in `dtype_device_matrix.py` is what counts the column; this file
pins a named subset of it so that a regression has a name.

Nullifications this file is meant to catch:

* the `I8` buffer arms reverted in `candle-core`
      -> test_int8_builds_on_mps_and_makes_the_round_trip
* an `init_binary_k(..., i8, ...)` line dropped from `binary.metal`
      -> test_int8_binary_operators_agree_with_upstream_on_mps
* `init_cast_all(i8, int8_t)` dropped
      -> test_int8_casts_agree_with_upstream_on_mps
* any of it re-routed through the host to keep the values right
      -> every counter assertion in this file
* the wrapping cases quietly removed from the inputs
      -> test_the_inputs_actually_wrap
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip

_NAMES = ("compute_encoders", "blit_encoders", "host_uploads",
          "host_upload_bytes", "host_downloads", "host_download_bytes")

# Both operands carry -128 and 127. See the module docstring for why.
_VALUES = [-128, -128, 127, 127, -1, 0, 5, -64]
_OTHER = [-1, 1, 1, -3, 127, 3, -128, 64]
_SHAPE = [2, 4]

# `maximum` and `minimum` are absent, and not because `int8` is missing them:
# they are on `_shim_mps_host_readback_ops()` for **every** dtype, so the shim
# refuses them on `mps` before a dtype is looked at. Including them here would
# make this file fail for a reason that has nothing to do with `I8`, and
# asserting the refusal instead would put a second file's subject in this one.
# `test_mpsrefuse.py` owns that list.
_BINARY = ("aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor",
           "aten.eq.Tensor", "aten.ne.Tensor", "aten.lt.Tensor",
           "aten.le.Tensor", "aten.gt.Tensor", "aten.ge.Tensor")

# Casts candle instantiates in both directions once `i8` joins `init_cast_all`.
# `float64` is absent on purpose: §3.1 refuses it on Metal by name.
_CAST_TO = ("float32", "float16", "bfloat16", "int64", "uint8")


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


def _i8(values, shape, device=None):
    """An `int8` tensor. Built as float64 and narrowed, as the other suites do.

    The narrowing is exact for every value in range, and `test_the_inputs...`
    below checks the inputs are in range rather than trusting that.
    """
    t = _C._tensor_from_flat([float(v) for v in values], list(shape), _C.float64)
    t = t.to(_C.int8)
    return t.to(device) if device is not None else t


def _vals(t):
    h = t.cpu() if str(t.device) != "cpu" else t
    return [int(v) for v in h.to(_C.float64).flatten().tolist()]


def _oracle(script, payload):
    """Upstream's answer, from an interpreter with no `_C` on its path."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(payload)],
                          capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n"
                             + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# 0. The inputs themselves
# ---------------------------------------------------------------------------


def _wrap8(x):
    x &= 0xFF
    return x - 256 if x >= 128 else x


def test_the_inputs_actually_wrap():
    """The guard on every other test in this file.

    §5.5: a verification that cannot fail is not a verification. If someone
    softens `_VALUES` to a range where `int8` and `int64` cannot be told
    apart, every agreement test below keeps passing while testing nothing.
    So the wrapping is asserted as a property of the data, in Python, with no
    tensor involved.
    """
    assert len(_VALUES) == len(_OTHER) == _SHAPE[0] * _SHAPE[1]
    for name, vs in (("_VALUES", _VALUES), ("_OTHER", _OTHER)):
        assert all(-128 <= v <= 127 for v in vs), (name, vs)
        assert -128 in vs and 127 in vs, (
            "%s must carry both int8 endpoints, or the dtype is not exercised "
            "at the only places it differs from a wider one: %s" % (name, vs))
    wraps = {}
    for label, f in (("add", lambda a, b: a + b),
                     ("sub", lambda a, b: a - b),
                     ("mul", lambda a, b: a * b)):
        wraps[label] = sum(1 for a, b in zip(_VALUES, _OTHER)
                           if f(a, b) != _wrap8(f(a, b)))
    for label, n in wraps.items():
        assert n >= 2, (
            "%s wraps in only %d of %d element pairs. Fewer than two and a "
            "kernel that computes in a wider type and forgets to narrow can "
            "pass this file. wraps=%s" % (label, n, len(_VALUES), wraps))


# ---------------------------------------------------------------------------
# 1. builds -- the operand the whole 284 was blocked on
# ---------------------------------------------------------------------------


def test_int8_builds_on_mps_and_makes_the_round_trip():
    """§4.3a cause A, in one assertion.

    284 cells failed at stage `operands` because `storage_from_cpu_storage`
    refused `CpuStorage::I8`, so the tensor could not be *constructed* on the
    device and the kernel was never asked. This is the cell that has to move
    before any of the rest can be measured at all.

    It is not an agreement claim: it is `builds`, plus the readback that makes
    any later agreement claim checkable. The values are asserted exactly --
    `int8` is integral, so a round trip that loses anything is a defect and
    not a tolerance question.
    """
    mps = _mps_or_skip("the int8 operand build on mps")
    if mps is None:
        return
    t = _i8(_VALUES, _SHAPE, mps)
    assert str(t.device).startswith("mps"), t.device
    assert str(t.dtype).replace("torch.", "") == "int8", t.dtype
    back = t.cpu()
    assert str(back.device) == "cpu", back.device
    assert _vals(back) == _VALUES, (_vals(back), _VALUES)
    # A contiguous copy and a strided one, which are different Metal kernels
    # (`copy_i8` and `copy_i8_strided`) and fail separately.
    assert _vals(t.clone()) == _VALUES
    rows, cols = _SHAPE
    transposed = [_VALUES[r * cols + c] for c in range(cols) for r in range(rows)]
    assert _vals(t.t().contiguous()) == transposed, (
        _vals(t.t().contiguous()), transposed)


# ---------------------------------------------------------------------------
# 2. AGREES -- binary
# ---------------------------------------------------------------------------

_ORACLE_BINARY = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    a = torch.tensor(case["a"], dtype=torch.int8).reshape(case["shape"])
    b = torch.tensor(case["b"], dtype=torch.int8).reshape(case["shape"])
    r = getattr(torch.ops.aten, case["op"])(a, b)
    out[key] = {"values": [int(v) for v in r.to(torch.int64).flatten().tolist()],
                "dtype": str(r.dtype).split(".")[-1]}
print(json.dumps(out))
"""


def test_int8_binary_operators_agree_with_upstream_on_mps():
    """Eleven operators, element-wise against a subprocess oracle, counted.

    The comparison is exact. `int8` is integral and both sides are integral,
    so any tolerance here would be a place to hide a wrong answer, which
    /tmp/rules.txt forbids and `dtype_device_matrix.py` also refuses for
    integer cells.

    The dtype of the result is asserted too: the six comparisons return
    `bool`, and a kernel that returned `int8` would still match element-wise
    on these values.
    """
    mps = _mps_or_skip("int8 binary agreement on mps")
    if mps is None:
        return
    request = {}
    for op in _BINARY:
        request[op] = {"op": op.split("aten.")[1].rsplit(".", 1)[0],
                       "a": _VALUES, "b": _OTHER, "shape": _SHAPE}
    want = _oracle(_ORACLE_BINARY, request)
    for op in _BINARY:
        a = _i8(_VALUES, _SHAPE, mps)
        b = _i8(_OTHER, _SHAPE, mps)
        before = _counters()
        r = _C._aten_dispatch(op, a, b)
        d = _delta(before, _counters())
        assert str(r.device).startswith("mps"), (op, str(r.device))
        assert d["host_downloads"] == 0, (
            "%s on mps read %d tensor(s) back to the host (%d bytes) during "
            "the dispatch. A correct value the GPU did not compute is the "
            "failure AGENTS.md §16 exists to refuse."
            % (op, d["host_downloads"], d["host_download_bytes"]))
        assert d["compute_encoders"] >= 1, (
            "%s on mps opened no compute encoder, so no shader ran. %s"
            % (op, d))
        got = _vals(r)
        assert got == want[op]["values"], (op, got, want[op]["values"])
        assert str(r.dtype).replace("torch.", "") == want[op]["dtype"], (
            op, r.dtype, want[op]["dtype"])


# ---------------------------------------------------------------------------
# 3. AGREES -- casts
# ---------------------------------------------------------------------------

_ORACLE_CAST = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    t = torch.tensor(case["values"], dtype=torch.int8).reshape(case["shape"])
    r = t.to(getattr(torch, case["to"]))
    back = r.to(torch.int8)
    out[key] = {
        "to": [float(v) for v in r.to(torch.float64).flatten().tolist()],
        "back": [int(v) for v in back.to(torch.int64).flatten().tolist()],
    }
print(json.dumps(out))
"""


def test_int8_casts_agree_with_upstream_on_mps():
    """Both directions, because `init_cast_all` is two edits and one can be lost.

    `init_cast(tname, t, i8, int8_t)` inside the macro gives every dtype a
    cast *to* `i8`; the top-level `init_cast_all(i8, int8_t)` gives `i8` a
    cast *from* itself to every dtype. Dropping either leaves the other
    working, so both are asserted, and the narrowing direction carries `-128`
    and `127` where a saturating cast and a truncating one differ.
    """
    mps = _mps_or_skip("int8 cast agreement on mps")
    if mps is None:
        return
    request = {name: {"values": _VALUES, "shape": _SHAPE, "to": name}
               for name in _CAST_TO}
    want = _oracle(_ORACLE_CAST, request)
    for name in _CAST_TO:
        t = _i8(_VALUES, _SHAPE, mps)
        before = _counters()
        wide = t.to(getattr(_C, name))
        narrow = wide.to(_C.int8)
        d = _delta(before, _counters())
        assert str(wide.device).startswith("mps"), (name, str(wide.device))
        assert str(narrow.device).startswith("mps"), (name, str(narrow.device))
        assert d["host_downloads"] == 0, (
            "int8 <-> %s on mps read %d tensor(s) back to the host"
            % (name, d["host_downloads"]))
        assert d["compute_encoders"] >= 2, (
            "int8 -> %s -> int8 opened %d compute encoders, expected at least "
            "two (one cast each way). %s" % (name, d["compute_encoders"], d))
        got_to = [float(v) for v in _vals(wide)] if name != "float16" else None
        if got_to is None:
            # float16 loses nothing for int8's range, but read it as float64
            # rather than int so the assertion is about the same numbers.
            h = wide.cpu().to(_C.float64)
            got_to = [float(v) for v in h.flatten().tolist()]
        assert got_to == want[name]["to"], (name, got_to, want[name]["to"])
        assert _vals(narrow) == want[name]["back"], (
            name, _vals(narrow), want[name]["back"])


# ---------------------------------------------------------------------------
# 4. AGREES -- where, the ternary family
# ---------------------------------------------------------------------------

_ORACLE_WHERE = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
t = torch.tensor(req["a"], dtype=torch.int8).reshape(req["shape"])
o = torch.tensor(req["b"], dtype=torch.int8).reshape(req["shape"])
c = torch.tensor(req["c"], dtype=torch.bool).reshape(req["shape"])
r = torch.where(c, t, o)
print(json.dumps([int(v) for v in r.to(torch.int64).flatten().tolist()]))
"""


def test_int8_where_agrees_with_upstream_on_mps():
    """`WHERE_OP` with `int8` as the value type, on a live device."""
    mps = _mps_or_skip("int8 where agreement on mps")
    if mps is None:
        return
    cond = [1, 0, 1, 1, 0, 0, 1, 0]
    want = _oracle(_ORACLE_WHERE,
                   {"a": _VALUES, "b": _OTHER, "c": cond, "shape": _SHAPE})
    a = _i8(_VALUES, _SHAPE, mps)
    b = _i8(_OTHER, _SHAPE, mps)
    c = _C._tensor_from_flat([float(v) for v in cond], list(_SHAPE),
                             _C.float64).to(_C.bool).to(mps)
    before = _counters()
    r = _C._aten_dispatch("aten.where.self", c, a, b)
    d = _delta(before, _counters())
    assert str(r.device).startswith("mps"), str(r.device)
    assert d["host_downloads"] == 0, d
    assert d["compute_encoders"] >= 1, d
    assert _vals(r) == want, (_vals(r), want)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_i8mps")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
