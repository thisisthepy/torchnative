"""`Tensor.as_strided_` restrides its receiver in place, over its own storage
(issues #306 and #380; docs/kernels/STRIDED.md section 8, SPEC S1.13).

The wall it closes (CI log r382c, exir stage `xnnpack`): executorch's XNNPACK
pass `ConvertToUpsampleBilinear2d` runs a strict `torch.export` of an
`F.interpolate(mode="bilinear")` module, and Dynamo's builder copies each input
it wraps with `torch/_dynamo/utils.py::clone_input`, which calls
`result.as_strided_(x.size(), x.stride(), cache_line_offset)` on a fresh flat
buffer and then `result.copy_(x.clone())`. The shim raised
`NotImplementedError: not implemented in torch._C shim: TensorBase.as_strided_`.

Upstream's rule, measured on 2.13.0: the receiver object is returned (`r is
x`) with its storage, dtype and `requires_grad`; its shape, stride and storage
offset become the ones given; the bound is the **storage**, not what the
receiver addressed (`arange(12.)[2:4].as_strided_((10,), (1,), 2)` succeeds);
the default offset is the receiver's own; views made before the call keep
their layouts and share the buffer both ways; a root stays a root
(`_is_view()` False, `_base` None) and a view keeps its `_base`; the version
counter moves (not compared: the shim has no `_version`). Its refusals, in
order: `mismatch in length of strides and shape`, `as_strided: Negative
strides are not supported ...`, `numel: integer multiplication overflow`,
`Tensor: invalid storage offset -1`, `setStorage: ... out of bounds for
storage of size N`.

**Every assertion is upstream's answer, computed in a separate subprocess from
the same probe text**, never a table written beside the implementation. The
two exceptions are named where they are made: `clone_input`'s storage offset
depends on two allocation addresses on either side (`(x.data_ptr() -
result.data_ptr()) % 32 // itemsize`), so the probe records whether it lies in
`[0, 32 // itemsize)` instead of its value. (An empty layout past the end
of the storage, which upstream accepts, was a named refusal here until CI
run 43826381567's panic led to the fork's empty-layout rule; it is now
compared like every other row.)

Each scenario is one row, and an exception in it, a Rust panic included
(`pyo3_runtime.PanicException` derives from BaseException), is that row's
recorded error, so one defect is one differing row and never "the probe did
not run".

What each test turns red under (nullifications, none run locally; CI runs them):

* `restride_in_place` bounding by the receiver (`view_with_layout`) instead of
  the storage (`view_with_layout_in_storage`): the `grow*` rows of
  `test_the_receiver_takes_upstreams_layout_over_its_own_storage` (refused
  here, accepted upstream) and `w_grown`;
* the new candle tensor copying instead of sharing the storage `Arc`: every
  row of `test_writes_reach_views_made_before_the_call_and_back` and the
  `shares` field of the layout rows;
* the kernel returning a new tensor instead of the receiver: the `same`
  field of every row;
* the default offset taken as 0 instead of the receiver's own: the
  `view_default_offset` row;
* the view-ness pin removed (`restride_in_place` not recording a root as not
  a view): `is_view` and `base` of the offset rows (`_is_view()` True and
  `_base` refusing here) and `test_clone_input_answers_upstreams_layout_and_values`;
* the checks reordered or reworded: `test_refusals_are_upstreams_in_upstreams_order`;
* `tensor.rs::write_into`'s zero-element return, or the fork's empty-layout
  rule (`Layout::strided_blocks`, `Layout::contiguous_offsets`, `NdIter`),
  removed: `test_writes_into_an_empty_receiver_touch_nothing_at_any_offset`
  and the zero-element `clone_input` rows (a panic, recorded per row);
* the meta arm removed from `aten.rs::meta_table` or the op left out of
  `meta_stride_rule`'s `OwnLayout`: `test_meta_receivers_restride_as_upstreams_do`;
* the `methods.json` or `overloads.json` row removed: every test (method) or
  `test_torch_as_strided_and_keyword_arguments_reach_the_kernel` (function).

What this file cannot see: a CUDA or Metal receiver (the probe is CPU and
meta only; the device rule, restriding within what the receiver addresses and
refusing past it by name, has no CI run), and the strict `torch.export` that
called `clone_input` (`tests/export/test_exir_shim.py` reports that stage).

Nothing here needs numpy or a network.
"""

import json
import os
import subprocess
import sys

import _skip

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


def _available():
    return os.path.isfile(_VENDOR_SHIM)


# ---------------------------------------------------------------------------
# The probe. Identical text on both sides; only the answers can differ.
# ---------------------------------------------------------------------------

_PROBE = r'''
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented"),
       "layouts": {}, "refusals": {}, "writes": {}, "views": {}, "meta": {},
       "clone_input": {}, "calls": {}, "narrowing": {}}


def failed(exc):
    """One scenario's failure as its row. A Rust panic reaches Python as
    `pyo3_runtime.PanicException`, which derives from BaseException, so every
    row catches BaseException and comes here; an interrupt or an exit is
    raised again rather than recorded. One broken scenario is one differing
    row, never a probe that did not run."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc
    return {"raised": type(exc).__name__, "message": str(exc)}


def flat(t):
    acc = []

    def walk(x):
        if isinstance(x, list):
            for y in x:
                walk(y)
        else:
            acc.append(float(x))

    walk(t.tolist())
    return acc


def base_kind(t, root):
    try:
        b = t._base
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        return "raised: " + failed(exc)["raised"]
    if b is None:
        return "none"
    return "root" if b is root else "other"


def describe(root, t, values=True):
    d = {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "offset": t.storage_offset(),
        "contiguous": t.is_contiguous(),
        "is_view": t._is_view(),
        "base": base_kind(t, root),
        "dtype": str(t.dtype),
    }
    if t.device.type != "meta":
        d["shares"] = t.untyped_storage()._cdata == root.untyped_storage()._cdata
        d["nbytes"] = t.untyped_storage().nbytes()
    if values:
        d["values"] = flat(t)
    return d


def dense(n, dtype=torch.float32):
    if dtype == torch.bool:
        return torch.tensor([i % 2 == 1 for i in range(n)])
    return torch.arange(n).to(dtype)


def meta(n, dtype=torch.float32):
    return torch.empty(n, device="meta", dtype=dtype)


# `factory -> (root, receiver)`.
INPUTS = {
    "root": lambda f, dt: (lambda r: (r, r))(f(12, dt)),
    "slice24": lambda f, dt: (lambda r: (r, r[2:4]))(f(12, dt)),
    "slice3": lambda f, dt: (lambda r: (r, r[3:]))(f(12, dt)),
    "view34": lambda f, dt: (lambda r: (r, r.view(3, 4)))(f(12, dt)),
    "t34": lambda f, dt: (lambda r: (r, r.view(3, 4).t()))(f(12, dt)),
}

# (name, input, size, stride, storage_offset)
LAYOUTS = [
    ("transposed_offset", "root", (3, 2), (1, 3), 1),
    ("grow_past_receiver", "slice24", (10,), (1,), 2),
    ("grow_whole_storage", "slice24", (12,), (1,), 0),
    ("view_default_offset", "slice3", (2, 2), (2, 1), None),
    ("view_offset_zero", "slice3", (2, 2), (2, 1), 0),
    ("view_of_view", "view34", (2,), (1,), None),
    ("transposed_receiver", "t34", (2, 2), (1, 1), None),
    ("overlap", "root", (3, 3), (1, 1), None),
    ("zero_stride", "root", (4, 3), (0, 1), 2),
    ("last_two", "root", (2,), (1,), 10),
    ("scalar_last", "root", (), (), 11),
    ("empty_at_end", "root", (0, 3), (3, 1), 12),
    ("empty_inside", "slice3", (2, 0), (1, 1), 5),
    ("identity", "root", (12,), (1,), 0),
]
DTYPES = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64,
          "uint8": torch.uint8, "bool": torch.bool}

for dtname, dt in DTYPES.items():
    for name, key, size, stride, off in LAYOUTS:
        row = dtname + "|" + name
        try:
            root, x = INPUTS[key](dense, dt)
            r = x.as_strided_(size, stride, off) if off is not None else x.as_strided_(size, stride)
            d = describe(root, x)
            d["same"] = r is x
            out["layouts"][row] = d
        except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
            out["layouts"][row] = failed(exc)


# Upstream refuses each of these, in this order of checks.
REFUSALS = [
    ("mismatch", (2, 2), (1,), None),
    ("mismatch_before_negative", (2, 2), (-1,), None),
    ("negative_stride", (2, 2), (-1, 1), None),
    ("negative_stride_before_size", (-1, 2), (-1, 1), None),
    ("negative_size", (-1, 2), (1, 1), None),
    ("negative_offset", (2,), (1,), -1),
    ("negative_offset_before_bounds", (100,), (1,), -1),
    ("negative_offset_empty", (0,), (1,), -1),
    ("one_past", (2,), (1,), 11),
    ("too_long", (13,), (1,), 0),
    ("too_wide", (4, 4), (4, 1), 0),
    ("scalar_past", (), (), 12),
]
for name, size, stride, off in REFUSALS:
    for dtname in ("float32", "int64", "bool"):
        try:
            x = dense(12, DTYPES[dtname])
            x.as_strided_(size, stride, off) if off is not None else x.as_strided_(size, stride)
            out["refusals"][dtname + "|" + name] = {"ok": describe(x, x)}
        except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
            out["refusals"][dtname + "|" + name] = failed(exc)
# A view's bound is its storage: past the storage from a view still refuses,
# with the storage's size in the message.
try:
    dense(12)[2:4].as_strided_((11,), (1,), 2)
    out["refusals"]["view_past_storage"] = {"ok": True}
except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
    out["refusals"]["view_past_storage"] = failed(exc)


# Writes. Each returns what the side that did NOT take the write reads.
def w_restrided_to_old_view():
    root = dense(12)
    v = root[2:6]
    root.as_strided_((2, 2), (1, 2), 2)
    root.add_(100.0)
    return flat(v)


def w_old_view_to_restrided():
    root = dense(12)
    v = root[2:6]
    root.as_strided_((2, 2), (1, 2), 2)
    v[1] = -1.0
    v.mul_(2.0)
    return flat(root)


def w_grown():
    base = dense(12)
    w = base[2:4]
    w.as_strided_((10,), (1,), 2)
    w.mul_(-1.0)
    return flat(base)


def w_view_of_restrided():
    r = dense(12)
    pre = r.view(12)
    r.as_strided_((2, 3), (1, 2), 1)
    r[1].fill_(0.0)
    return flat(pre)


def w_setitem_restrided():
    r = dense(12)
    pre = r[:]
    r.as_strided_((3, 2), (1, 3), 1)
    r[2, 1] = 50.0
    return flat(pre)


def w_zero_stride_add():
    r = dense(12)
    r.as_strided_((4, 3), (0, 1), 2)
    r.add_(1.0)
    return flat(r)


def w_zero_stride_fill():
    r = dense(12)
    pre = r[:]
    r.as_strided_((4, 3), (0, 1), 2)
    r.fill_(5.0)
    return flat(pre)


def w_copy_into_restrided():
    r = torch.zeros(10)
    pre = r[:]
    r.as_strided_((2, 3), (1, 2), 3)
    r.copy_(dense(6).view(2, 3))
    return flat(pre)


# Writes into an empty receiver read and write nothing, wherever its offset
# lies (CI run 43826381567: the shim's `copy_` read an empty source over an
# empty storage through its upstream stride and panicked in candle).
def empty_writes(r):
    seen = {}
    r.copy_(torch.zeros(r.shape))
    r.fill_(5.0)
    r.add_(1.0)
    r.mul_(2.0)
    r.zero_()
    seen["after"] = [list(r.shape), list(r.stride()), r.storage_offset(), r.tolist()]
    seen["plus_one"] = (r + 1).tolist()
    seen["contiguous"] = [list(r.contiguous().shape), r.contiguous().tolist()]
    seen["repr"] = repr(r)
    return seen


def w_empty_restrided_at_offset_4():
    r = torch.zeros(10)
    pre = r[:]
    r.as_strided_((0, 3), (3, 1), 4)
    return {"empty": empty_writes(r), "storage": flat(pre)}


def w_empty_restrided_3d_at_offset_7():
    r = torch.zeros(10)
    pre = r[:]
    r.as_strided_((2, 0, 4), (4, 4, 1), 7)
    return {"empty": empty_writes(r), "storage": flat(pre)}


def w_empty_restrided_past_the_storage():
    r = torch.zeros(12)
    pre = r[:]
    r.as_strided_((0, 3), (3, 1), 100)
    return {"empty": empty_writes(r), "storage": flat(pre)}


def w_empty_clone_keeps_its_stride_over_no_storage():
    # Upstream's clone of `x[:, 0:0]` has stride (4, 1) and a 0-byte storage:
    # the layout CI's panicking `copy_` source had.
    e = torch.arange(12.0).view(3, 4)[:, 0:0].clone()
    head = [list(e.stride()), e.untyped_storage().nbytes()]
    again = e.clone()
    return {"clone": head, "clone_of_clone": [list(again.stride()), again.untyped_storage().nbytes()],
            "empty": empty_writes(e)}


def w_empty_clone_joined_on_an_inner_axis():
    # `cat` on dim 1 copies each entry block by block; the empty clone's
    # blocks sit at offsets 4 and 8 of an empty buffer (a panic in candle's
    # `cat_contiguous` before aten.rs `cat_default` replaced empty entries).
    e = torch.arange(12.0).view(3, 4)[:, 0:0].clone()
    ones = torch.ones(3, 2)
    left, right, alone = torch.cat([e, ones], 1), torch.cat([ones, e], 1), torch.cat([e, e], 1)
    return {"left": [list(left.shape), list(left.stride()), flat(left)],
            "right": [list(right.shape), list(right.stride()), flat(right)],
            "alone": [list(alone.shape), flat(alone)],
            "dim0": [list(torch.cat([e, e], 0).shape)]}


for name, fn in sorted(globals().copy().items()):
    if name.startswith("w_") and callable(fn):
        try:
            out["writes"][name] = {"ok": fn()}
        except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
            out["writes"][name] = failed(exc)


# Views made after the call: a root stays a root, a view keeps its base.
def v_root_offset():
    r = dense(12)
    r.as_strided_((2, 2), (2, 1), 5)
    return {"self": describe(r, r), "child": describe(r, r[0])}


def v_view_keeps_base():
    base = dense(12)
    v = base[3:]
    v.as_strided_((2,), (1,), 0)
    return {"self": describe(base, v), "child": describe(base, v[1:])}


def v_buffer_like_clone_input():
    r = torch.empty(5 + 32)
    r.as_strided_((3, 2), (1, 3), 3)
    return {"self": describe(r, r, values=False)}


def v_grad_leaf_no_grad():
    g = torch.zeros(6, requires_grad=True)
    with torch.no_grad():
        r = g.as_strided_((3,), (2,))
    return {"same": r is g, "requires_grad": g.requires_grad, "is_leaf": g.is_leaf,
            "grad_fn": g.grad_fn is None, "self": describe(g, g, values=False)}


for name, fn in sorted(globals().copy().items()):
    if name.startswith("v_") and callable(fn):
        try:
            out["views"][name] = fn()
        except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
            out["views"][name] = failed(exc)


# Meta receivers: the same layouts, without values, and the same bounds.
for name, key, size, stride, off in LAYOUTS:
    try:
        root, x = INPUTS[key](meta, torch.float32)
        r = x.as_strided_(size, stride, off) if off is not None else x.as_strided_(size, stride)
        d = describe(root, x, values=False)
        d["same"] = r is x
        out["meta"][name] = d
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["meta"][name] = failed(exc)
for name, size, stride, off in [("one_past", (2,), (1,), 11), ("too_wide", (4, 4), (4, 1), 0)]:
    try:
        meta(12).as_strided_(size, stride, off)
        out["meta"]["refused_" + name] = {"ok": True}
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["meta"]["refused_" + name] = failed(exc)


# `clone_input` itself, the function the strict export runs on every input.
try:
    from torch._dynamo.utils import clone_input
except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
    clone_input_error = failed(exc)

    def clone_input(x):
        raise RuntimeError(f"torch._dynamo.utils.clone_input did not import: {clone_input_error}")

CLONES = {
    "contiguous": lambda: torch.arange(6.0).view(2, 3),
    "transposed": lambda: torch.arange(12.0).view(3, 4).t(),
    "sliced_offset": lambda: torch.arange(20.0).view(4, 5)[1:, 1:4],
    "expanded": lambda: torch.arange(3.0).expand(4, 3),
    "int64_transposed": lambda: torch.arange(6).view(3, 2).t(),
    "leaf_requires_grad": lambda: torch.ones(2, 3, requires_grad=True),
    "zero_size": lambda: torch.arange(12.0).view(3, 4)[:, 0:0],
    # Zero-element inputs of three shapes: `needed_size` is -1, -1 and 3, so
    # the flat buffer holds 31, 31 and 35 floats and the restrided result
    # addresses none of them.
    "empty_0": lambda: torch.zeros(0),
    "empty_0x3": lambda: torch.zeros(0, 3),
    "empty_2x0x4": lambda: torch.zeros(2, 0, 4),
    "scalar": lambda: torch.tensor(3.0),
    "nchw": lambda: torch.arange(48.0).reshape(1, 3, 4, 4),
}
for name, make in CLONES.items():
    try:
        x = make()
        y = clone_input(x)
        d = describe(y, y)
        off = d.pop("offset")
        d["offset_in_cache_line"] = 0 <= off < 32 // y.element_size()
        d["shares_input"] = y.untyped_storage()._cdata == x.untyped_storage()._cdata
        d["requires_grad"] = y.requires_grad
        d["is_leaf"] = y.is_leaf
        d["equal"] = bool(torch.equal(y.detach(), x.detach()))
        out["clone_input"][name] = d
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["clone_input"][name] = failed(exc)


# The two other spellings: `torch.as_strided_` and keyword arguments.
def c_torch_function():
    x = dense(6)
    r = torch.as_strided_(x, (2,), (3,))
    return {"same": r is x, "self": describe(x, x)}


def c_keywords():
    x = dense(6)
    r = x.as_strided_(size=(2,), stride=(1,), storage_offset=4)
    return {"same": r is x, "self": describe(x, x)}


def c_ops_packet():
    x = dense(6)
    r = torch.ops.aten.as_strided_.default(x, [2], [2], storage_offset=1)
    return {"same": r is x, "self": describe(x, x)}


def c_torch_size_arguments():
    src = dense(12).view(3, 4).t()
    x = dense(12)
    r = x.as_strided_(src.size(), src.stride(), 0)
    return {"same": r is x, "self": describe(x, x)}


for name, fn in sorted(globals().copy().items()):
    if name.startswith("c_") and callable(fn):
        try:
            out["calls"][name] = fn()
        except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
            out["calls"][name] = failed(exc)


# What used to be the one narrowing: an empty layout past the end of the
# storage, on a dense and on a meta receiver.
for name, factory in (("empty_past_storage", dense), ("meta_empty_past_storage", meta)):
    try:
        x = factory(12)
        r = x.as_strided_((0, 3), (3, 1), 100)
        d = describe(x, x, values=factory is dense)
        d["same"] = r is x
        out["narrowing"][name] = {"ok": d}
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["narrowing"][name] = failed(exc)

print(json.dumps(out))
'''


def _run(shim, _cache={}):
    if shim in _cache:
        return _cache[shim]
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT, timeout=900,
    )
    assert proc.returncode == 0, (
        f"the {'shim' if shim else 'upstream'} probe did not run:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    # An empty vendored tree makes the shim side import upstream silently
    # (AGENTS.md section 15.2), so the side is asserted, not assumed.
    assert data["is_shim"] is shim, f"the probe meant for shim={shim} ran on the other side"
    _cache[shim] = data
    return data


def _both():
    if not _available():
        _skip.skip("no vendored shim installed (scripts/vendor/install_shim.sh)")
        return None
    return _run(True), _run(False)


def _diff(section, names, shim, up):
    bad = []
    for name in names:
        if shim[section].get(name) != up[section][name]:
            bad.append(f"  {name}\n    upstream {up[section][name]}\n    shim     {shim[section].get(name)}")
    return bad


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def test_the_receiver_takes_upstreams_layout_over_its_own_storage():
    """Every row of `LAYOUTS` in five dtypes: the receiver is returned; its
    shape, stride, offset, contiguity, view-ness, `_base`, dtype, storage
    (shared with the root, and its byte size) and values through the new
    layout. (Upstream's version counter moves from 0 to 1; the shim has no
    `_version`, so it is not compared.) The oracle is checked first on the
    rows the design turns on: upstream grows a view past its own extent, keeps
    a view's offset by default, and leaves a root at offset 1 a root."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["layouts"])
    assert len(names) == 70, len(names)
    refused = [n for n in names if "raised" in up["layouts"][n]]
    assert not refused, f"upstream refused rows this table expects it to answer: {refused}"
    grow = up["layouts"]["float32|grow_past_receiver"]
    assert (grow["shape"], grow["offset"], grow["shares"], grow["same"]) == ([10], 2, True, True), grow
    assert up["layouts"]["float32|view_default_offset"]["offset"] == 3
    root = up["layouts"]["float32|transposed_offset"]
    assert (root["is_view"], root["base"], root["stride"]) == (False, "none", [1, 3]), root
    bad = _diff("layouts", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} rows differ from upstream:\n" + "\n".join(bad)
    print(f"ASSTRIDED_INPLACE: {len(names)} of {len(names)} layout rows equal upstream's")


def test_refusals_are_upstreams_in_upstreams_order():
    """Twelve refusals in three dtypes, plus a view's: the exception type and
    the whole message, so the order of the checks (length, negative stride,
    negative size, negative offset, then the storage bound, whose byte counts
    carry the itemsize) is upstream's. Red when a check moves or rewords."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["refusals"])
    assert len(names) == 37, len(names)
    for name in names:
        assert up["refusals"][name].get("raised") == "RuntimeError", (name, up["refusals"][name])
    assert "storage of size 48" in up["refusals"]["view_past_storage"]["message"], up["refusals"]
    assert "itemsize 1" in up["refusals"]["bool|one_past"]["message"], up["refusals"]["bool|one_past"]
    bad = _diff("refusals", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} refusals differ:\n" + "\n".join(bad)


def test_writes_reach_views_made_before_the_call_and_back():
    """A view made before the call keeps its own layout over the same buffer:
    an in-place write through the restrided receiver is read by it, and a
    write through it (`__setitem__`, `mul_`) is read through the restrided
    receiver; a grown view's write reaches its base; `copy_` into a restrided
    buffer lands where its layout says (what `clone_input` does next); and a
    zero-stride receiver refuses `add_` and lets `fill_` through, as upstream.
    Red when the restride copies: the other side keeps its old values."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(n for n in up["writes"] if "_empty_" not in n)
    assert len(names) == 8, names
    assert up["writes"]["w_restrided_to_old_view"]["ok"] == [102.0, 103.0, 104.0, 105.0], up["writes"]
    assert up["writes"]["w_zero_stride_add"].get("raised") == "RuntimeError", up["writes"]["w_zero_stride_add"]
    bad = _diff("writes", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} write rows differ:\n" + "\n".join(bad)


def test_writes_into_an_empty_receiver_touch_nothing_at_any_offset():
    """`copy_`, `fill_`, `add_`, `mul_` and `zero_` into an empty receiver
    restrided at offset 4, at offset 7 in three dimensions and at offset 100
    of a 12-element storage, and into the clone of `x[:, 0:0]` (upstream's
    stride `(4, 1)` over a 0-byte storage, the source CI run 43826381567's
    `copy_` panicked on), and that clone joined by `cat` on an inner axis
    (red when aten.rs `cat_default` hands candle the empty entry as it is):
    each leaves the storage as it was, and the receiver
    then answers its layout, `tolist`, `+ 1`, `contiguous` and `repr` as
    upstream's. Red, as one differing row rather than a dead probe, when any
    of them reads or writes the storage through the empty layout (a Rust
    panic is recorded as `PanicException`): `tensor.rs::write_into`'s
    zero-element return or the fork's empty-layout rule in `Layout`
    and `NdIter` removed."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(n for n in up["writes"] if "_empty_" in n)
    assert len(names) == 5, names
    refused = [n for n in names if "raised" in up["writes"][n]]
    assert not refused, f"upstream refused: {refused}"
    past = up["writes"]["w_empty_restrided_past_the_storage"]["ok"]
    assert past["empty"]["after"][2] == 100 and past["storage"] == [0.0] * 12, past
    clone = up["writes"]["w_empty_clone_keeps_its_stride_over_no_storage"]["ok"]
    assert clone["clone"] == [[4, 1], 0], clone
    bad = _diff("writes", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} empty-receiver rows differ:\n" + "\n".join(bad)


def test_views_after_the_call_and_a_leaf_under_no_grad():
    """A root restrided to offset 5 is still a root (`_is_view()` False,
    `_base` None) and its own views' base; a restrided view keeps its `_base`
    and hands it on; the flat buffer `clone_input` makes is a root after the
    call; a leaf that requires grad, restrided under `no_grad`, is still a
    leaf that requires grad. Red when the view-ness pin is removed: the shim
    calls the offset root a view and `_base` refuses."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["views"])
    assert len(names) == 4, names
    assert up["views"]["v_root_offset"]["self"]["is_view"] is False, up["views"]["v_root_offset"]
    assert up["views"]["v_view_keeps_base"]["self"]["base"] == "root", up["views"]["v_view_keeps_base"]
    bad = _diff("views", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} view rows differ:\n" + "\n".join(bad)


def test_meta_receivers_restride_as_upstreams_do():
    """The same layouts on meta receivers (no values), and the storage bound
    with upstream's message. Red when the meta arm is missing (`as_strided_`
    refuses as an op with no meta kernel) or when it is not `OwnLayout` (the
    transposed receiver row refuses as an unverified stride rule)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["meta"])
    assert len(names) == len({n for n, *_ in _LAYOUT_NAMES}) + 2, names
    for name in ("refused_one_past", "refused_too_wide"):
        assert up["meta"][name].get("raised") == "RuntimeError", up["meta"][name]
    bad = _diff("meta", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} meta rows differ:\n" + "\n".join(bad)


def test_clone_input_answers_upstreams_layout_and_values():
    """`torch._dynamo.utils.clone_input` on a contiguous, a transposed, a sliced
    (offset 6), an expanded, an int64 transposed, a zero-size slice, three
    zero-element inputs (`(0,)`, `(0, 3)`, `(2, 0, 4)`), a 0-d, an NCHW and a
    leaf-that-requires-grad input: shape, stride, contiguity, view-ness,
    `_base`, dtype, storage size (`needed_size + 32` elements, or the plain
    clone's for the expanded input, where `copy_` refuses the zero stride and
    `clone_input` falls back to `torch.clone`), not sharing the input's
    storage, `requires_grad`, `is_leaf`, values, and an offset inside one
    cache line. The offset's value is not compared: it is computed from two
    allocation addresses. Red on the CI wall itself (`as_strided_` not
    implemented), when the restride or the `copy_` after it is wrong, and on
    CI run 43826381567's panic (the zero-size row's `copy_` read its empty
    source past the end of its empty storage)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["clone_input"])
    assert len(names) == 12, names
    refused = [n for n in names if "raised" in up["clone_input"][n]]
    assert not refused, f"upstream refused: {refused}"
    assert up["clone_input"]["transposed"]["stride"] == [1, 4], up["clone_input"]["transposed"]
    # `needed_size + 32` elements: (4 - 1) * 1 + (3 - 1) * 4 = 11, so 43 floats.
    assert up["clone_input"]["transposed"]["nbytes"] == (3 * 1 + 2 * 4 + 32) * 4
    assert up["clone_input"]["expanded"]["stride"] == [3, 1], up["clone_input"]["expanded"]
    # `needed_size` is -1, -1 and 3 for the three zero-element inputs.
    assert [up["clone_input"][n]["nbytes"] for n in ("empty_0", "empty_0x3", "empty_2x0x4")] \
        == [31 * 4, 31 * 4, 35 * 4], up["clone_input"]
    assert all(up["clone_input"][n]["offset_in_cache_line"] and up["clone_input"][n]["equal"]
               for n in names), up["clone_input"]
    bad = _diff("clone_input", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} clone_input rows differ:\n" + "\n".join(bad)


def test_torch_as_strided_and_keyword_arguments_reach_the_kernel():
    """`torch.as_strided_(x, ...)` (`overloads.json`), keyword arguments to the
    method, the `torch.ops` packet with `storage_offset=` by name, and
    `torch.Size` arguments (what `clone_input` passes): each returns the
    receiver restrided as upstream's."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["calls"])
    assert len(names) == 4, names
    assert all(up["calls"][n]["same"] for n in names), up["calls"]
    bad = _diff("calls", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} call rows differ:\n" + "\n".join(bad)


def test_an_empty_layout_past_the_storage_is_accepted_as_upstreams_is():
    """Upstream accepts `as_strided_((0, 3), (3, 1), 100)` on a 12-element
    storage (no element is read), on a dense and on a meta receiver. This was
    the one CPU narrowing until the fork's readers took an empty layout as one
    empty block at offset 0 (`Layout::strided_blocks`,
    `Layout::contiguous_offsets`, `NdIter`): the receiver now answers
    upstream's row (offset 100, storage still 48 bytes, no values). Red when
    the refusal comes back or the row differs; what the receiver then reads
    and writes is `test_writes_into_an_empty_receiver_touch_nothing_at_any_offset`."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["narrowing"])
    assert names == ["empty_past_storage", "meta_empty_past_storage"], names
    for name in names:
        assert up["narrowing"][name]["ok"]["offset"] == 100, up["narrowing"]
    bad = _diff("narrowing", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} rows differ:\n" + "\n".join(bad)


# The layout names the meta rows reuse, for the meta test's count.
_LAYOUT_NAMES = [
    ("transposed_offset",), ("grow_past_receiver",), ("grow_whole_storage",),
    ("view_default_offset",), ("view_offset_zero",), ("view_of_view",),
    ("transposed_receiver",), ("overlap",), ("zero_stride",), ("last_two",),
    ("scalar_last",), ("empty_at_end",), ("empty_inside",), ("identity",),
]


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_asstrided_inplace")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
