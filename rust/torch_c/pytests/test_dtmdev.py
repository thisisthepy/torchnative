"""Three guarantees about (dtype x device) that nothing was holding down.

`docs/devices/matrix.md` is the document; this file is what makes its three
load-bearing claims impossible to break quietly.

A previous round found and fixed these and added **no tests**, which is why
they could not be merged: each one guards real behaviour on a real device, and
nothing would have noticed if it came back. The fixes are re-derived here, one
of them rejected (see below), and each surviving one is paired with a test that
goes red when the implementation is gutted.

**The three, and what each is worth.**

1. `float64` on Metal must **refuse by name**. Metal has no `double` -- that is
   a property of MSL, not of candle and not of this build -- and upstream
   refuses at the boundary with a sentence naming the dtype and the framework.
   `metal_dtype_gate` (device.rs) already did this for the roads that go
   through `PyTensorBase::new`, and `test_dtypedev.py` already covers those
   three. What nothing covered was the **factories**: `ones`, `full`,
   `scalar_tensor` and `arange` reach candle *before* the gate and died with
   `candle: unsupported const-set f64` and `Metal contiguous to_dtype I64 F64
   not implemented` -- a candle symbol, for a dtype the API does not have.
   Measured here before the fix, all four.

   A refusal that names an internal symbol is the shape `test_intmps.py`
   already rejected for `int16`/`int32`, and CLAUDE.md §6 is the rule: a
   refusal says its own name. So this test asks every road onto Metal that
   takes a dtype, factories included, and requires upstream's sentence -- not
   merely *an* exception, which a typo would also produce.

2. `_tensor_from_flat` must land **upstream's values** on the device. It built
   on the device and cast there, so every dtype whose cast candle's Metal
   backend does not implement -- ten of the eleven storable ones -- died with
   `Metal contiguous to_dtype F64 <X> not implemented`. Building on the CPU,
   casting there, and moving the narrow result is the fix. The grade is
   **agrees**: every value is compared element-wise against upstream, per
   dtype, because "it stopped raising" would also be satisfied by a constructor
   that returned zeros.

3. `tolist` must read a device tensor's **values**, not raise and not garbage.
   `flat_objects` read the tensor where it lay, so `tolist()` on an `mps`
   tensor raised `Metal contiguous to_dtype F32 F64 not implemented` for five
   of the ten dtypes that can live there.

**The change this file deliberately does NOT make**, recorded because the next
round will otherwise re-derive it: a guard in `visit_for_device` (aten.rs)
refusing `float64` on a Metal device at *dispatch* time. It is unreachable.
`metal_dtype_gate` is called from `PyTensorBase::new`, the one constructor
every dense tensor passes through, so no `float64` Metal tensor exists to be
dispatched on -- thirteen roads onto the device were probed and not one
produces the object that guard inspects. A second guard behind the first is
the shape CLAUDE.md §5.5 records as "두 곳에 있어 서로를 가려주어": neither copy
can be tested alone, because nullifying either leaves the other holding the
line. The gate belongs at construction, and it is already there.

Nullifications this file is meant to catch, each verified by making the break
and watching it go red -- docs/devices/matrix.md §4 records what was seen:

* `metal_dtype_gate` returning `Ok(())`
      -> test_float64_refuses_by_name_on_every_road_onto_metal
* the factory gate removed, leaving the three `PyTensorBase::new` roads
      -> test_float64_refuses_by_name_on_every_road_onto_metal (factory roads)
* `_tensor_from_flat` building on the device again
      -> test_tensor_from_flat_lands_upstreams_values_on_the_device
* `_tensor_from_flat` returning zeros/garbage of the right shape
      -> the same test, which compares values rather than counting exceptions
* `flat_objects` reading the tensor where it lies
      -> test_tolist_on_a_device_tensor_is_upstreams_values
* a kernel routed through `flat_objects` to get a cheap host readback
      -> test_fixing_tolist_did_not_open_the_host_readback_hole
"""

import os

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C

try:
    import torch as _upstream_torch
except ImportError:  # pragma: no cover
    _upstream_torch = None


# Upstream's own sentence, which is what "by name" means for this cell. Kept
# character-for-character rather than matched loosely: the point of the refusal
# is that a reader of it learns the fact (Metal has no double) and the way out
# (use float32), and a paraphrase that drops either half is a different refusal.
_F64_REFUSAL = (
    "Cannot convert a MPS Tensor to float64 dtype as the MPS framework "
    "doesn't support float64. Please use float32 instead."
)

# Fragments that mean the refusal came from inside candle rather than from this
# shim's own vocabulary. `test_intmps.py` rejects exactly these for the integer
# dtypes; the float64 cell is held to the same standard, because a message
# about `const-set f64` sends the reader to candle's source for a fact about
# Metal's API.
_SYMBOL_LEAKS = ("const-set", "to_dtype", "candle:", "badd_", "bmul_")


# Values chosen so that every dtype in each group holds every one of them
# *exactly*. An input a dtype cannot represent would make a disagreement about
# the input look like a disagreement about the operation, and a tolerance would
# then be hiding it. Each set is also non-uniform, so a constructor that
# returned zeros -- or the same value four times -- fails rather than passes.
#
#   floats  -1.5, 0.25, 3.75, -8.0 are exact in float64/32/16, bfloat16 (8
#           mantissa bits) and float8_e4m3fn (3 mantissa bits): 1.5 = 1.1b,
#           3.75 = 1.111b x 2, and the other two are powers of two.
#   ints    0, 3, 100, 127 fit int8 (max 127) and every wider signed type, and
#           are non-negative so uint8/uint32 hold them too.
_FLOAT_VALUES = [-1.5, 0.25, 3.75, -8.0]
_INT_VALUES = [0.0, 3.0, 100.0, 127.0]
_BOOL_VALUES = [0.0, 1.0, 1.0, 0.0]
_SHAPE = [2, 2]

_FLOAT_DTYPES = ("float64", "float32", "float16", "bfloat16", "float8_e4m3fn")
_INT_DTYPES = ("int64", "int32", "int16", "int8", "uint8", "uint32")


def _values_for(name):
    if name == "bool":
        return _BOOL_VALUES
    return _FLOAT_VALUES if name in _FLOAT_DTYPES else _INT_VALUES


def _mps_or_skip(what):
    """A live `mps` device, or None having said why not.

    Same shape as `test_dtypedev._mps_or_skip` and deliberately not imported
    from it: the skip line is what this file promises to keep truthful
    (docs/devices/VULKAN3.md §6.1 is what a lying skip line cost), so it says
    which file skipped.
    """
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        print("   (skipped %s: no mps device on this machine -- %s)"
              % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


def _upstream_or_skip(what):
    if _upstream_torch is None:
        print("   (skipped %s: no upstream torch in this interpreter -- the "
              "oracle half of this test cannot run)" % what)
        return None
    return _upstream_torch


def _upstream_values(name, values):
    """Upstream's own bytes for this dtype, read back as float64.

    The oracle is asked on the **cpu** even for the mps cells, which is
    `test_dtypedev.py`'s choice and for its reason: the question here is
    whether the shim's number is upstream's number, not whether upstream would
    have produced it on that device.
    """
    t = _upstream_torch.tensor(values).reshape(_SHAPE)
    t = t.to(getattr(_upstream_torch, name))
    return [float(v) for v in t.to(_upstream_torch.float64).flatten().tolist()]


def _shim_values(t):
    """The shim's bytes, read back on the host as float64."""
    h = t.cpu() if str(t.device) != "cpu" else t
    return [float(v) for v in h.to(_C.float64).flatten().tolist()]


# ---------------------------------------------------------------------------
# 1. float64 on Metal refuses by name, on every road including the factories
# ---------------------------------------------------------------------------


def _f64_roads(mps):
    """Every spelling that asks for a float64 tensor on a Metal device.

    Three groups, and the middle one is the group this round added:

      * the conversion roads, which reach `PyTensorBase::new` and were already
        covered by `test_dtypedev.test_float64_refuses_by_name_on_every_road`;
      * the **factories**, which build inside candle first and so refused with
        a candle symbol until the gate was moved ahead of them;
      * the `*_like` factories, which take their device from a tensor argument
        and so are a third road again.
    """
    f32_cpu = _C._tensor_from_flat([1.0, 2.0], [2], _C.float32)
    f64_cpu = _C._tensor_from_flat([1.0, 2.0], [2], _C.float64)
    f32_mps = f32_cpu.to(mps)
    d = _C._aten_dispatch
    return {
        # conversion
        "_tensor_from_flat(device=mps)":
            lambda: _C._tensor_from_flat([1.0, 2.0], [2], _C.float64, mps),
        "cpu_float64.to(mps)": lambda: f64_cpu.to(mps),
        "_to_copy(cpu_float64, device=mps)":
            lambda: d("aten._to_copy.default", f64_cpu, device=mps),
        "mps_float32.to(float64)": lambda: f32_mps.to(_C.float64),
        # factories -- the four that reached candle before the gate
        "ones(device=mps, dtype=float64)":
            lambda: d("aten.ones.default", [2, 2], dtype=_C.float64, device=mps),
        "full(device=mps, dtype=float64)":
            lambda: d("aten.full.default", [2, 2], 1.0, dtype=_C.float64, device=mps),
        "scalar_tensor(device=mps, dtype=float64)":
            lambda: d("aten.scalar_tensor.default", 1.0, dtype=_C.float64, device=mps),
        "arange(device=mps, dtype=float64)":
            lambda: d("aten.arange.default", 4, dtype=_C.float64, device=mps),
        # factories that already refused by name, kept so a regression in the
        # other direction is caught too
        "zeros(device=mps, dtype=float64)":
            lambda: d("aten.zeros.default", [2, 2], dtype=_C.float64, device=mps),
        "empty(device=mps, dtype=float64)":
            lambda: d("aten.empty.memory_format", [2, 2], dtype=_C.float64, device=mps),
        "empty_strided(device=mps, dtype=float64)":
            lambda: d("aten.empty_strided.default", [2, 2], [2, 1],
                      dtype=_C.float64, device=mps),
        "eye(device=mps, dtype=float64)":
            lambda: d("aten.eye.default", 2, dtype=_C.float64, device=mps),
        "linspace(device=mps, dtype=float64)":
            lambda: d("aten.linspace.default", 0.0, 1.0, 4,
                      dtype=_C.float64, device=mps),
        # *_like: the device comes from the tensor, the dtype from the kwarg
        "ones_like(mps_float32, dtype=float64)":
            lambda: d("aten.ones_like.default", f32_mps, dtype=_C.float64),
        "full_like(mps_float32, dtype=float64)":
            lambda: d("aten.full_like.default", f32_mps, 1.0, dtype=_C.float64),
        "zeros_like(mps_float32, dtype=float64)":
            lambda: d("aten.zeros_like.default", f32_mps, dtype=_C.float64),
        "empty_like(mps_float32, dtype=float64)":
            lambda: d("aten.empty_like.default", f32_mps, dtype=_C.float64),
        "new_ones(mps_float32, dtype=float64)":
            lambda: d("aten.new_ones.default", f32_mps, [2, 2], dtype=_C.float64),
        "new_zeros(mps_float32, dtype=float64)":
            lambda: d("aten.new_zeros.default", f32_mps, [2, 2], dtype=_C.float64),
        "new_full(mps_float32, dtype=float64)":
            lambda: d("aten.new_full.default", f32_mps, [2, 2], 1.0, dtype=_C.float64),
        "new_empty(mps_float32, dtype=float64)":
            lambda: d("aten.new_empty.default", f32_mps, [2, 2], dtype=_C.float64),
        # A meta input with a real device: `new_ones` is a meta *factory* in
        # that case and is served by a different arm of the dispatcher
        # (the meta table in aten.rs), which is its own road onto Metal.
        "new_ones(meta_input, device=mps, dtype=float64)":
            lambda: d("aten.new_ones.default",
                      d("aten.empty.memory_format", [2, 2], device=_C.device("meta")),
                      [2, 2], dtype=_C.float64, device=mps),
    }


def test_float64_refuses_by_name_on_every_road_onto_metal():
    """Grade: **refuses by name**, on 21 roads.

    Three things are asserted per road and they are not the same thing:

      * it refused at all -- a road that *succeeds* has produced a tensor the
        device cannot compute with, which is the quiet form of the failure
        docs/graph/NPU2.md is about: a capability claim made by construction
        succeeding;
      * it refused with **upstream's sentence** -- not merely with *an*
        exception, which a typo in the kernel would also raise, and which is
        why this test would pass against a broken build if it stopped at
        `assertRaises`;
      * it did not refuse with a **candle symbol** -- `unsupported const-set
        f64` names an internal detail for a fact about Metal's API, and is what
        four of these roads said before this round.
    """
    mps = _mps_or_skip("the float64-on-Metal refusals")
    if mps is None:
        return
    bad = []
    for road, thunk in _f64_roads(mps).items():
        try:
            got = thunk()
        except BaseException as e:  # noqa: BLE001
            msg = str(e).splitlines()[0]
            if _F64_REFUSAL not in str(e):
                leaked = [s for s in _SYMBOL_LEAKS if s in msg]
                bad.append(
                    "%s refused with %r%s -- not upstream's sentence"
                    % (road, msg,
                       " (a candle symbol: %s)" % ", ".join(leaked) if leaked else ""))
            continue
        bad.append(
            "%s SUCCEEDED, producing dtype=%s on device=%s. Metal has no "
            "double; every arithmetic kernel refuses it, so this object can "
            "only be cloned." % (road, getattr(got, "dtype", "?"),
                                 getattr(got, "device", "?")))
    assert not bad, (
        "%d of %d roads onto Metal do not refuse float64 by name:\n  %s"
        % (len(bad), len(_f64_roads(mps)), "\n  ".join(bad)))


# ---------------------------------------------------------------------------
# 2. _tensor_from_flat lands upstream's values on the device
# ---------------------------------------------------------------------------
#
# Which dtypes reach the device at all is **recorded**, not derived, for the
# reason `test_dtypedev._MPS_REACHES` gives: both directions of change are
# news. A dtype that stops landing is a lost capability; one that starts
# landing is a capability nothing has compared against upstream yet -- and it
# will be, by this same test, as soon as it is added here.
#
# Measured on an arm64 Mac (Metal) against the artefact built from this tree,
# after the construction order was fixed: ten of the twelve storable dtypes
# land on Metal carrying upstream's values.
#
# `float8_e4m3fn` is in this list and that is worth a word, because it is the
# clearest evidence for *why* the order matters. Its cast is `F64 ->
# F8E4M3`, which candle's Metal backend does not implement at all -- so it
# could not be built on the device by any amount of trying. Cast on the host,
# it moves as eight-bit bytes and arrives exactly.
_LANDS_ON_MPS = (
    "float32", "float16", "bfloat16", "float8_e4m3fn",
    "int64", "int32", "int16", "uint8", "uint32",
    "bool",
)
# Refuses, and the fragment its message has to contain to count as naming what
# it refused. Both are real refusals; neither may become a wrong value.
#
# `float64` is a property of Metal's API (no `double`) and refuses with
# upstream's own sentence. `int8` is a property of *this build*: the
# candle-core fork that gave the CPU `DType::I8` (docs/numerics/INT8.md §1.2)
# is CPU-only, and Metal names the dtype it cannot take.
_REFUSED_ON_MPS = {
    "float64": _F64_REFUSAL,
    "int8": "I8",
}


def test_tensor_from_flat_lands_upstreams_values_on_the_device():
    """Grade: **agrees**, per dtype, on the device.

    `_tensor_from_flat` built on the target device and cast there, so it
    inherited every gap in candle's Metal `to_dtype` table -- ten of the eleven
    storable dtypes raised `Metal contiguous to_dtype F64 <X> not implemented`,
    measured before the fix. Building on the CPU, casting there, and moving the
    already-narrow result is the fix, and it is also the cheaper order: the
    bytes that cross to the device are the target dtype's, not float64's.

    This compares **values**, element-wise, against upstream. A test that only
    checked the call stopped raising would be green for a constructor that
    returned zeros of the right shape, and that is the failure worth catching
    here: a silently wrong tensor on a device is worse than a refusal.
    """
    up = _upstream_or_skip("the device-construction agreement")
    if up is None:
        return
    mps = _mps_or_skip("the device-construction agreement")
    if mps is None:
        return
    wrong, missing, unexpected = [], [], []
    for name in _LANDS_ON_MPS:
        values = _values_for(name)
        try:
            t = _C._tensor_from_flat(values, _SHAPE, getattr(_C, name), mps)
        except BaseException as e:  # noqa: BLE001
            missing.append("%s: %s: %s"
                           % (name, type(e).__name__, str(e).splitlines()[0]))
            continue
        assert str(t.device).startswith("mps"), (
            "%s: asked for mps and got device=%s" % (name, t.device))
        got, want = _shim_values(t), _upstream_values(name, values)
        if got != want:
            wrong.append("%s: shim=%s upstream=%s" % (name, got, want))
    for name, fragment in _REFUSED_ON_MPS.items():
        values = _values_for(name)
        try:
            t = _C._tensor_from_flat(values, _SHAPE, getattr(_C, name), mps)
        except BaseException as e:  # noqa: BLE001
            assert fragment in str(e), (
                "%s refused with %r, which does not name what was refused. A "
                "refusal that does not say what it refused sends the reader to "
                "the wrong place (CLAUDE.md §6)."
                % (name, str(e).splitlines()[0]))
            continue
        unexpected.append("%s landed on %s" % (name, t.device))
    assert not (wrong or missing or unexpected), (
        "_tensor_from_flat on mps:\n"
        "  wrong values (%d): %s\n"
        "  recorded as landing but raised (%d): %s\n"
        "  recorded as refusing but landed (%d): %s\n"
        "A dtype that starts landing is not automatically good -- it is a cell "
        "nothing has compared against upstream. Move it to _LANDS_ON_MPS and "
        "this test will grade it."
        % (len(wrong), wrong, len(missing), missing,
           len(unexpected), unexpected))


def test_tensor_from_flat_still_agrees_on_the_cpu():
    """The reordering's blast radius, asserted rather than assumed.

    Every operand the golden harness builds goes through this function
    (`tools/golden/build.py`), so a change to it that was wrong on the CPU
    would move 11420 golden cases at once. The cpu path now takes a
    `to_device` that is a no-op clone; this is what says the values survived
    it.
    """
    up = _upstream_or_skip("the cpu construction agreement")
    if up is None:
        return
    wrong = []
    for name in _FLOAT_DTYPES + _INT_DTYPES + ("bool",):
        values = _values_for(name)
        try:
            t = _C._tensor_from_flat(values, _SHAPE, getattr(_C, name))
        except BaseException as e:  # noqa: BLE001
            wrong.append("%s: %s: %s"
                         % (name, type(e).__name__, str(e).splitlines()[0]))
            continue
        got, want = _shim_values(t), _upstream_values(name, values)
        if got != want:
            wrong.append("%s: shim=%s upstream=%s" % (name, got, want))
    assert not wrong, (
        "_tensor_from_flat disagrees with upstream on the cpu for %d dtype(s):"
        "\n  %s" % (len(wrong), "\n  ".join(wrong)))


# ---------------------------------------------------------------------------
# 3. tolist reads a device tensor's values
# ---------------------------------------------------------------------------


def test_tolist_on_a_device_tensor_is_upstreams_values():
    """Grade: **agrees**, read straight off the device.

    `.tolist()` is called on the `mps` tensor itself -- no `.cpu()` first,
    which is what every other test in this tree does and is exactly why this
    gap survived. `flat_objects` read the tensor where it lay, so five of the
    ten dtypes that can live on Metal raised `Metal contiguous to_dtype F32
    F64 not implemented`; `int64`, `uint8`, `uint32` and `bool` happened to
    work because candle implements those casts on Metal.

    Values, not just absence of an exception: a `tolist` that returned the
    right *shape* full of uninitialised bytes would be the worse outcome, and
    the values here are chosen to be exact in every dtype so this compares with
    no tolerance at all.
    """
    up = _upstream_or_skip("the device tolist agreement")
    if up is None:
        return
    mps = _mps_or_skip("the device tolist agreement")
    if mps is None:
        return
    bad = []
    for name in _LANDS_ON_MPS:
        values = _values_for(name)
        t = _C._tensor_from_flat(values, _SHAPE, getattr(_C, name), mps)
        try:
            got = t.tolist()
        except BaseException as e:  # noqa: BLE001
            bad.append("%s: tolist() raised %s: %s"
                       % (name, type(e).__name__, str(e).splitlines()[0]))
            continue
        flat = [float(v) for row in got for v in row]
        want = _upstream_values(name, values)
        if flat != want:
            bad.append("%s: tolist()=%s -> %s, upstream %s"
                       % (name, got, flat, want))
        if name == "bool" and not all(isinstance(v, bool) for row in got for v in row):
            bad.append(
                "%s: tolist() gave %r, not Python bools. torch's tolist on a "
                "bool tensor yields bools, not 0/1 ints (BOOL.md §2.6), and "
                "the host hop must not have cost the tag its read." % (name, got))
    assert not bad, (
        "tolist() on an mps tensor is wrong or raises for %d dtype(s):\n  %s"
        % (len(bad), "\n  ".join(bad)))


def test_fixing_tolist_did_not_open_the_host_readback_hole():
    """The cost of moving `flat_objects` to the host, checked rather than
    argued.

    `flat_objects` now copies to the CPU before reading. That is legitimate for
    `tolist`, which *is* a host read by definition -- upstream's `tolist` copies
    too -- and it has exactly one caller. But the neighbouring function
    `read_flat` is what twenty-odd kernels use, and docs/devices/MPS.md §2
    measured what happens when a kernel reads a device tensor back and computes
    on the host: thirteen ops returned correct values that the GPU never
    computed, under an `mps` label. The only reason the float ones were *loud*
    is that `read_flat` widens through `f64` and Metal has no `F32 -> F64`.

    So the risk this fix creates is that a later round reaches for
    `flat_objects` to make a kernel's readback "just work". This test is
    behavioural rather than a source scan on purpose -- docs/devices/MPSATTN.md
    §3.1 records how to defeat a source scan -- and it fails if any refused
    readback op starts answering on mps.
    """
    mps = _mps_or_skip("the host-readback refusals")
    if mps is None:
        return
    refused = _C._shim_mps_host_readback_ops()
    assert len(refused) > 0, refused
    x = _C._tensor_from_flat([0.0, 1.0, 0.0, 3.0], [4], _C.int64, mps)
    answered = []
    for op in ("aten.nonzero.default", "aten.masked_select.default",
               "aten.gather.default"):
        if op not in refused:
            continue
        try:
            _C._aten_dispatch(op, x)
        except NotImplementedError as e:
            assert "mps" in str(e), (op, str(e).splitlines()[0])
            continue
        except (RuntimeError, TypeError):
            # Refused for another reason (argument shape); not a hole.
            continue
        answered.append(op)
    assert not answered, (
        "%s answered on mps. These ops read the tensor back to host memory and "
        "compute there, so an answer is a correct value the GPU did not "
        "compute, under an mps label -- docs/devices/MPS.md §2. If a kernel "
        "was routed through flat_objects' new host hop, that is the hole."
        % (answered,))


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
