"""Cause C of docs/devices/matrix.md section 4.3a: 27 cells filed under
"candle's matmul lacks the dtype", across ``mm``, ``bmm``, ``matmul``,
``addmm``, ``baddbmm`` and ``convolution``.

**Re-derived before implementation, and it is not one cause.** Section 4.3a
clusters by the text of candle's message, and three genuinely different
situations raise sentences that all contain the word ``matmul``:

===========================  =====  ==================================
what                         cells  answer
===========================  =====  ==================================
``bool`` on any device           6  **upstream refuses it too**
integer on the ``cpu``          15  a real kernel -- exact, and here
``int64`` on ``mps``             5  refusal, by name
``convolution`` ``bfloat16``     1  not attempted; see below
===========================  =====  ==================================

**The bool six are not a gap at all.** Measured against torch 2.13.0::

    torch.mm(bool, bool)       NotImplementedError: "addmm_impl_cpu_" not implemented for 'Bool'
    torch.bmm(bool, bool)      NotImplementedError: "bmm" not implemented for 'Bool'
    torch.baddbmm(...)         NotImplementedError: "baddbmm" not implemented for 'Bool'

``addmm`` in this build already answered exactly upstream's sentence;
``mm``/``matmul``/``bmm`` answered ``mlx matmul doesn't support U8``, which
names a candle-internal type token for an operator upstream declines by name.
Six cells move from a symbol refusal to upstream's own words. **No verdict
changes and no kernel was written** -- section 4.3 is explicit that this is
the work it names, so it is counted as a defect fixed and not as a feature.

**The upcast route is a fudge here, and this was checked rather than
assumed.** ``gemm_accumulate_in`` already widens ``float8_e4m3fn`` to ``f32``,
computes and narrows, and that is legitimate *because upstream's own answer
was measured to be bit-identical to it* over 700 cases. The same move on
integers is not, because upstream does not compute integer matmul in a wider
type -- it **wraps in the storage width**::

    torch.mm(int8[[100, 100]], int8[[100], [100]])   ==   32

not ``20000`` and not ``127``.  ``20000 mod 256 == 32``: two's complement, no
saturation, no widening.  An ``f64`` upcast answers ``20000`` and a saturating
one answers ``127``; both are wrong, and both are right on the single small
shape the sweep's cell happens to use.  This is exactly the ``clamp`` lesson
(docs/devices/matrix.md section 1): the cell is one shape, and that shape did
not overflow.  ``test_upstream_integer_matmul_wraps_in_the_storage_width`` is
the measurement, and the agreement test below is run on inputs chosen to
overflow every width.

So the kernel computes exactly, in ``i64`` with wrapping arithmetic, and
truncates to the storage width at the end -- which is the same number, because
reduction mod ``2**n`` is a ring homomorphism and ``2**8``, ``2**32`` all
divide ``2**64`` (the argument test_intmps.py established for ``add``/``mul``,
reused here for a sum of products, which is built from nothing else).

**Why the ``mps`` five stay refused.** The kernel above is a host computation.
Running it on an ``mps`` tensor would return a value the GPU did not compute
under an ``mps`` label, which is precisely what ``mps_host_readback_gate``
exists to refuse (docs/devices/MPS.md section 2). So ``int64`` matmul on
``mps`` refuses **by name**, and
``test_the_mps_integer_refusal_is_not_served_by_a_readback`` is the check that
nobody later "fixes" it by computing on the host anyway.

**What is not done: ``aten.convolution.default`` on ``bfloat16_cpu``, 1 cell.**
It is tempting to give it ``gemm_accumulate_in``'s widening, since that helper
already maps ``bf16 -> f32`` and the five gemms use it. It was not done
because the bit-identity argument does not carry over: widening a *cast* is
elementwise and cannot move a value, whereas a convolution widened before
im2col changes the **summation order** relative to upstream's kernel, and a
reassociated float sum is a different number. That claim needs its own
measurement against upstream, which this round did not make, so the cell is
left refusing rather than answered on an argument that was not checked.

Nullifications this file is meant to catch:

* the integer kernel replaced by an ``f64`` upcast
  -> test_integer_matmul_agrees_with_upstream_including_at_overflow
* the kernel saturating instead of wrapping -> the same test
* the wrap done at the wrong width -> the same test, all three dtypes
* ``bool`` answered instead of refused
  -> test_bool_matmul_refuses_in_upstreams_own_words
* the ``mps`` refusal replaced by a host readback
  -> test_the_mps_integer_refusal_is_not_served_by_a_readback
* the kernel applied to ``mm`` only -> every test loops the family
"""

import json
import os
import random
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip

_INT_DTYPES = ("int8", "int32", "int64")
_BITS = {"int8": 8, "int32": 32, "int64": 64}

# The five operators section 4.3a names, minus `convolution` (see the module
# note). `kind` is how the operands are shaped, not a coverage class.
_OPS = ("mm", "bmm", "matmul", "addmm", "baddbmm")


def _mps_or_skip(what):
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


def _wrap(x, bits):
    """Two's-complement reduction into `bits` bits -- arithmetic, not a table."""
    m = 1 << bits
    x &= m - 1
    return x - m if x >= (m >> 1) else x


# `_tensor_from_flat` is the only constructor this extension exposes and it
# takes `float64`, so every input has to survive a `float64` round trip
# exactly. That caps the draws at 2**52 for `int64` -- which costs this file
# nothing, because two operands of that size have a product of 2**104 and
# overflow `int64` by fifty bits.
_MAX_EXACT = 1 << 52


def _tensor(values, shape, dtype):
    flat = [float(v) for v in values]
    t = _C._tensor_from_flat(flat, [len(flat)], _C.float64) \
          .to(_C.int64).to(getattr(_C, dtype))
    got = [int(v) for v in t.reshape(-1).tolist()]
    assert got == list(values), (
        "constructing a %s tensor through float64 did not round trip: %r "
        "became %r. Every comparison in this file would then be against "
        "inputs upstream never saw." % (dtype, list(values)[:6], got[:6]))
    return t.reshape(*shape)


def _host(t):
    """Exact integers, and **not** through `float64`.

    The first version of this file read values back with
    `.to(_C.float64).tolist()`, copying the idiom from test_intmps.py where
    every dtype is narrower than `float64`'s 53-bit significand. At `int64` it
    silently rounds, and it made the kernel look wrong on exactly the rows
    where the kernel is most interesting -- every `int64` case disagreed while
    `int8` and `int32` passed. `.tolist()` on an integer tensor already gives
    Python ints of full width, which is what this compares.
    """
    h = t.cpu() if str(t.device) != "cpu" else t
    return [int(v) for v in h.reshape(-1).tolist()]


# ---------------------------------------------------------------------------
# The cases. Small shapes for the contract, and one shape per dtype whose
# products cannot fit the storage width -- which is where an upcast route and
# a saturating route both diverge from upstream, and where the sweep's single
# cell does not go.
# ---------------------------------------------------------------------------

def _random_case(rng, dtype, m, k, n, batch=None):
    hi = min((1 << (_BITS[dtype] - 1)) - 1, _MAX_EXACT)
    lo = -hi
    # Deliberately near the ends of the range: `k` of these multiplied and
    # summed overflows every width in the table, which is the point.
    def draw(count):
        return [rng.choice([hi, lo, hi - 1, lo + 1, 1, -1, 0,
                            rng.randint(lo, hi)]) for _ in range(count)]
    lead = [batch] if batch else []
    return {
        "a": draw((batch or 1) * m * k), "a_shape": lead + [m, k],
        "b": draw((batch or 1) * k * n), "b_shape": lead + [k, n],
        "c": draw((batch or 1) * m * n), "c_shape": lead + [m, n],
    }


def _cases():
    rng = random.Random(20260920)
    out = {}
    for dtype in _INT_DTYPES:
        for op in _OPS:
            batched = op in ("bmm", "baddbmm")
            for tag, (m, k, n) in (("small", (2, 3, 2)), ("wide", (3, 17, 4))):
                case = _random_case(rng, dtype, m, k, n,
                                    batch=2 if batched else None)
                case.update(op=op, dtype=dtype)
                out["%s/%s/%s" % (op, dtype, tag)] = case
    return out


_ORACLE = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")
req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    dt = getattr(torch, case["dtype"])
    a = torch.tensor(case["a"], dtype=torch.int64).reshape(case["a_shape"]).to(dt)
    b = torch.tensor(case["b"], dtype=torch.int64).reshape(case["b_shape"]).to(dt)
    c = torch.tensor(case["c"], dtype=torch.int64).reshape(case["c_shape"]).to(dt)
    op = case["op"]
    try:
        if op == "mm":
            t = torch.mm(a, b)
        elif op == "bmm":
            t = torch.bmm(a, b)
        elif op == "matmul":
            t = torch.matmul(a, b)
        elif op == "addmm":
            t = torch.addmm(c, a, b)
        else:
            t = torch.baddbmm(c, a, b)
    except Exception as e:
        out[key] = {"raised": type(e).__name__, "message": str(e)}
        continue
    out[key] = {
        "values": [int(v) for v in t.to(torch.int64).flatten().tolist()],
        "dtype": str(t.dtype).split(".")[-1],
        "shape": list(t.shape),
    }
print(json.dumps(out))
"""

_BOOL_ORACLE = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented")
a = torch.tensor([[True, False], [True, True]])
b = torch.tensor([[True, True], [False, True]])
out = {}
for op in ["mm", "bmm", "matmul", "addmm", "baddbmm"]:
    try:
        if op == "mm": torch.mm(a, b)
        elif op == "matmul": torch.matmul(a, b)
        elif op == "bmm": torch.bmm(a.unsqueeze(0), b.unsqueeze(0))
        elif op == "addmm": torch.addmm(a, a, b)
        else: torch.baddbmm(a.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0))
        out[op] = {"raised": None}
    except Exception as e:
        out[op] = {"raised": type(e).__name__, "message": str(e)}
print(json.dumps(out))
"""


def _oracle(script, payload=None):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    argv = [sys.executable, "-c", script]
    if payload is not None:
        argv.append(json.dumps(payload))
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=900,
                          env=env)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n"
                             + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _ask(case, device=None):
    dt = case["dtype"]
    a = _tensor(case["a"], case["a_shape"], dt)
    b = _tensor(case["b"], case["b_shape"], dt)
    c = _tensor(case["c"], case["c_shape"], dt)
    if device is not None:
        a, b, c = a.to("mps"), b.to("mps"), c.to("mps")
    op = case["op"]
    if op == "addmm":
        return _C._aten_dispatch("aten.addmm.default", c, a, b)
    if op == "baddbmm":
        return _C._aten_dispatch("aten.baddbmm.default", c, a, b)
    return _C._aten_dispatch("aten.%s.default" % op, a, b)


# ---------------------------------------------------------------------------
# The premise
# ---------------------------------------------------------------------------

def test_upstream_integer_matmul_wraps_in_the_storage_width():
    """The measurement the whole kernel rests on.

    If upstream widened, the right implementation would be an upcast and the
    exact kernel would be the wrong answer at every overflow. If it saturated,
    neither would be right. It does neither: it wraps, in the storage width.

    Asked in a subprocess of upstream itself, at values chosen so that the
    three possible behaviours give three different numbers -- ``32`` for
    wrapping, ``20000`` for widening, ``127`` for saturating.
    """
    want = _oracle(_ORACLE, {"probe": {
        "op": "mm", "dtype": "int8",
        "a": [100, 100], "a_shape": [1, 2],
        "b": [100, 100], "b_shape": [2, 1],
        "c": [0], "c_shape": [1, 1],
    }})["probe"]
    assert "values" in want, (
        "upstream refused int8 mm (%r) -- the premise of this file is that it "
        "does not" % want)
    assert want["values"] == [_wrap(20000, 8)] == [32], (
        "upstream int8 mm of 100*100 + 100*100 gave %r. This file's kernel is "
        "built on that answer being the wrapped one (32); 20000 would mean "
        "upstream widens and 127 would mean it saturates, and either would "
        "make the exact kernel below the wrong implementation."
        % want["values"])
    assert want["dtype"] == "int8", want


def test_integer_matmul_agrees_with_upstream_including_at_overflow():
    """Grade: **agrees**, element-wise and exactly, on the ``cpu``.

    Fifteen cells of cause C, plus a second shape per cell that the sweep does
    not contain. Integers, so the comparison is equality and there is no
    tolerance anywhere in this file to widen.
    """
    cases = _cases()
    want = _oracle(_ORACLE, cases)
    bad = []
    for key, case in sorted(cases.items()):
        expect = want[key]
        if "values" not in expect:
            bad.append("%s: upstream itself raised %s (%s) -- this case does "
                       "not belong in the table"
                       % (key, expect["raised"], expect["message"][:80]))
            continue
        try:
            got = _ask(case)
        except Exception as e:  # noqa: BLE001
            bad.append("%s raised %s: %s"
                       % (key, type(e).__name__, str(e).splitlines()[0]))
            continue
        values = _host(got)
        if values != expect["values"]:
            wrong = sum(1 for x, y in zip(values, expect["values"]) if x != y)
            bad.append("%s disagrees on %d/%d elements: got %r upstream %r"
                       % (key, wrong, len(expect["values"]),
                          values[:8], expect["values"][:8]))
        if list(got.shape) != expect["shape"]:
            bad.append("%s shape %r, upstream %r"
                       % (key, list(got.shape), expect["shape"]))
        if str(got.dtype).replace("torch.", "") != expect["dtype"]:
            bad.append("%s dtype %s, upstream %s"
                       % (key, got.dtype, expect["dtype"]))
    assert not bad, ("integer matmul does not agree with upstream:\n  "
                     + "\n  ".join(bad))


def test_the_overflow_cases_really_do_overflow():
    """Guards the test above against being quietly defanged.

    A previous round made two sentences the same length so the padding
    disappeared and the masked path was never taken (CLAUDE.md section 5.5).
    The equivalent here is inputs that stop overflowing -- every assertion
    would stay green while the interesting half went untested. So: at least
    one element of at least one case per dtype must have a true product-sum
    that does not fit the storage width.
    """
    cases = _cases()
    seen = {}
    for key, case in cases.items():
        dtype = case["dtype"]
        bits = _BITS[dtype]
        a_shape, b_shape = case["a_shape"], case["b_shape"]
        m, k = a_shape[-2], a_shape[-1]
        n = b_shape[-1]
        batch = a_shape[0] if len(a_shape) == 3 else 1
        for bi in range(batch):
            for i in range(m):
                for j in range(n):
                    total = 0
                    for x in range(k):
                        total += (case["a"][bi * m * k + i * k + x]
                                  * case["b"][bi * k * n + x * n + j])
                    if total != _wrap(total, bits):
                        seen[dtype] = (key, total)
    missing = [d for d in _INT_DTYPES if d not in seen]
    assert not missing, (
        "no case overflows for %s, so the wrapping half of this file is not "
        "actually exercised and an upcast implementation would pass it. Put "
        "larger inputs back rather than deleting this test." % missing)


def test_bool_matmul_refuses_in_upstreams_own_words():
    """Six cells, from a candle type token to upstream's own sentence.

    Upstream has no ``bool`` gemm and says so by name. A refusal is therefore
    the *correct* behaviour and the only defect was the wording -- this build
    answered ``mlx matmul doesn't support U8``, which names neither the dtype
    nor the operator nor anything the reader can act on.

    The exact sentence is taken from upstream at run time, not pasted here, so
    that a torch upgrade that rewords it is a visible failure rather than a
    silently stale string.
    """
    upstream = _oracle(_BOOL_ORACLE)
    a = _tensor([1, 0, 1, 1], [2, 2], "int64").to(_C.bool)
    b = _tensor([1, 1, 0, 1], [2, 2], "int64").to(_C.bool)
    a3, b3 = a.reshape(1, 2, 2), b.reshape(1, 2, 2)
    asks = {
        "mm": lambda: _C._aten_dispatch("aten.mm.default", a, b),
        "matmul": lambda: _C._aten_dispatch("aten.matmul.default", a, b),
        "bmm": lambda: _C._aten_dispatch("aten.bmm.default", a3, b3),
        "addmm": lambda: _C._aten_dispatch("aten.addmm.default", a, a, b),
        "baddbmm": lambda: _C._aten_dispatch("aten.baddbmm.default", a3, a3, b3),
    }
    bad = []
    for op, ask in sorted(asks.items()):
        assert upstream[op]["raised"], (
            "upstream no longer refuses bool %s -- this build refuses it, so "
            "that is now a divergence and a kernel is owed" % op)
        sentence = upstream[op]["message"]
        try:
            ask()
        except Exception as e:  # noqa: BLE001
            if sentence not in str(e):
                bad.append("%s said %r; upstream says %r"
                           % (op, str(e).splitlines()[0], sentence))
            continue
        bad.append("%s answered a value where upstream raises %s"
                   % (op, upstream[op]["raised"]))
    assert not bad, ("bool gemm does not refuse in upstream's words:\n  "
                     + "\n  ".join(bad))


def test_the_mps_integer_refusal_names_the_op_the_dtype_and_the_device():
    """Five cells, from ``mlx matmul doesn't support I64`` to a named refusal.

    The verdict does not change -- it was REFUSES and it stays REFUSES. What
    changes is that the sentence says which operator, which dtype, which
    device, why, and what to do instead, which is what section 4.3 counts as
    the work.
    """
    if _mps_or_skip("the mps integer gemm refusal") is None:
        return
    cases = {k: v for k, v in _cases().items()
             if v["dtype"] == "int64" and k.endswith("/small")}
    assert len(cases) == len(_OPS), cases.keys()
    bad = []
    for key, case in sorted(cases.items()):
        try:
            _ask(case, device=_C.device("mps"))
            bad.append("%s answered on mps; a host integer gemm under an mps "
                       "label is exactly the silent fallback this project "
                       "refuses" % key)
            continue
        except Exception as e:  # noqa: BLE001
            text = str(e)
        for token in (case["op"], "int64", "mps"):
            if token not in text:
                bad.append("%s refusal does not name %r: %r"
                           % (key, token, text.splitlines()[0]))
        if "mlx" in text or "candle:" in text:
            bad.append("%s still hands back a candle-internal symbol: %r"
                       % (key, text.splitlines()[0]))
    assert not bad, ("the mps integer gemm refusal is not named:\n  "
                     + "\n  ".join(bad))


def test_the_mps_integer_refusal_is_not_served_by_a_readback():
    """The refusal must stay a refusal.

    The exact kernel is a host computation, and the one thing that must never
    happen is it being reused for an ``mps`` operand -- that returns a correct
    number the GPU did not compute, under an ``mps`` label, which
    docs/graph/NPU2.md is the record of this project getting wrong before.

    **There is a Metal dispatch counter in this build, and this test now uses
    it.** The paragraph that stood here said there was not, citing
    docs/devices/matrix.md section 7.5 -- but section 7.11 built one, and this
    file was written against a tree that predated it: the round was scoped
    before the int8 / ``candle-metal-kernels`` work landed, and inherited that
    tree's ceiling along with its numbers. ``_C._metal_counters()`` is the
    only instrument that can see a silent host fallback (CLAUDE.md section 2),
    so the refusal is bracketed by it rather than argued for.

    **Bracket**: counters read immediately before and immediately after a
    single ``_aten_dispatch``, in this process, after an unmeasured dispatch
    has warmed the device. A refusal must not move ``host_downloads`` at all
    -- an answer served by a readback, or a readback begun and then discarded,
    appears there and nowhere else.
    """
    if _mps_or_skip("the readback check") is None:
        return
    counters = getattr(_C, "_metal_counters", None)
    assert counters is not None, (
        "_C._metal_counters() is gone. It is the only instrument that can "
        "tell this refusal apart from a host readback wearing an mps label; "
        "do not delete it, and do not weaken this test back into the "
        "structural argument it used to be (docs/devices/matrix.md 7.11)")
    a = _tensor([1, 2, 3, 4], [2, 2], "int64").to("mps")
    _C._aten_dispatch("aten.mul.Tensor", a, a)  # warm the device; not measured
    for op in ("mm", "matmul"):
        before = counters()["host_downloads"]
        try:
            out = _C._aten_dispatch("aten.%s.default" % op, a, a)
        except (RuntimeError, NotImplementedError):
            after = counters()["host_downloads"]
            assert after == before, (
                "aten.%s.default refused the int64 mps operand but moved "
                "host_downloads %d -> %d on the way. A refusal that reads the "
                "operand back to the host has already done the thing the "
                "refusal exists to prevent." % (op, before, after))
            continue
        raise AssertionError(
            "aten.%s.default answered %r for an int64 mps operand. There is no "
            "Metal integer gemm in this build, so the only way a value could "
            "have appeared is a host readback." % (op, _host(out)))


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_gemmint")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
