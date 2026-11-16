"""`aten._unsafe_index.Tensor`, the packets with no `default` overload, and the
fake propagation of `F.interpolate(mode="bilinear")` they unblock
(docs/design/FUNCTIONALIZATION.md section 5.4).

The wall (CI run 43826381567, exir stage `xnnpack`): Dynamo's strict export of
executorch's bilinear pattern module runs `F.interpolate(mode="bilinear")` on
fake tensors. FakeTensorMode decomposes `aten.upsample_bilinear2d.default`
through `torch/_decomp/decompositions.py::_upsample_linear`, which calls the
packet `aten._unsafe_index(input, [None, None, rows, cols])`; the shim's packet
dispatched `aten._unsafe_index.default`, an overload upstream does not have,
so no decomposition matched and the meta door refused:
`torch._C shim has no meta kernel for aten._unsafe_index.default`. Upstream's
packet matches the call against `_unsafe_index.Tensor`'s schema, and that
overload's decomposition calls the packet `aten.index(x, indices)`, which
upstream matches to `index.Tensor`.

**Every assertion is upstream's answer**, computed in a separate subprocess
from the same probe text. Each scenario is one row; an exception in it, a
Rust panic included (`pyo3_runtime.PanicException` derives from
BaseException), is that row's recorded error.

What each test turns red under (nullifications, none run locally; CI runs them):

* the `aten._unsafe_index.Tensor` dispatch arm removed: every dense row of
  `test_the_op_answers_index_tensors_values` (the op refuses by name);
* the op removed from the shared meta arm in `aten.rs::meta_table`: the meta
  rows and the fake rows of `test_fake_interpolate_answers_upstreams_shape`;
* `bootstrap.py::_packet_overload` answering None (the old `.default` key):
  `test_packets_without_a_default_reach_upstreams_overload` and the fake rows.

What this file cannot see: whether the strict export of the bilinear module
now completes and what graph it records (`tests/export/test_exir_shim.py`),
type-directed packet matching (`add.Tensor` against `add.Scalar`; not
reproduced, such a packet still dispatches `<name>.default` and refuses by
name), and a device receiver.

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


_PROBE = r'''
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented"),
       "dense": {}, "meta": {}, "packets": {}, "fake": {}}


def failed(exc):
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


def idx(values, shape):
    return torch.tensor(values, dtype=torch.int64).reshape(shape)


def describe(t, values=True):
    d = {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
         "contiguous": t.is_contiguous()}
    if values:
        d["values"] = flat(t)
    return d


IMAGE = lambda dt: torch.arange(48).to(dt).reshape(1, 3, 4, 4)
ROWS = [[0, 0, 0, 1, 1, 2, 2, 3], [1, 1, 1, 2, 2, 3, 3, 3]]
COLS = [[0, 0, 1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3, 3, 3]]

# (name, input, indices)
CASES = []
for dname, dt in (("float32", torch.float32), ("float64", torch.float64), ("int64", torch.int64)):
    for r in (0, 1):
        for c in (0, 1):
            CASES.append((f"{dname}|bilinear_r{r}_c{c}", lambda dt=dt: IMAGE(dt),
                          lambda r=r, c=c: [None, None, idx(ROWS[r], (8, 1)), idx(COLS[c], (8,))]))
CASES += [
    ("adjacent_middle", lambda: torch.arange(120.).reshape(2, 3, 4, 5),
     lambda: [None, idx([0, 1], (2, 1)), idx([0, 1, 2], (3,))]),
    ("separated", lambda: torch.arange(120.).reshape(2, 3, 4, 5),
     lambda: [idx([0, 1], (2, 1)), None, idx([0, 1, 2], (3,))]),
    ("zero_dim", lambda: torch.arange(120.).reshape(2, 3, 4, 5),
     lambda: [idx([1], ()), idx([2], ())]),
    ("empty_index", lambda: torch.arange(120.).reshape(2, 3, 4, 5),
     lambda: [idx([], (0,)), idx([], (0,))]),
    ("out_of_range", lambda: torch.arange(4.), lambda: [idx([7], (1,))]),
]

for name, make, indices in CASES:
    try:
        r = torch.ops.aten._unsafe_index.Tensor(make(), indices())
        out["dense"][name] = describe(r)
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["dense"][name] = failed(exc)
    try:
        m = make().to("meta")
        mi = [None if i is None else i.to("meta") for i in indices()]
        r = torch.ops.aten._unsafe_index.Tensor(m, mi)
        out["meta"][name] = describe(r, values=False)
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["meta"][name] = failed(exc)


# The packets. `_unsafe_index` and `index` have no `default` overload; the
# call is matched to `.Tensor`, as upstream's packet matches it.
PACKETS = {
    "_unsafe_index": lambda: torch.ops.aten._unsafe_index(
        IMAGE(torch.float32), [None, None, idx(ROWS[0], (8, 1)), idx(COLS[1], (8,))]),
    "index": lambda: torch.ops.aten.index(
        IMAGE(torch.float32), [None, None, idx(ROWS[1], (8, 1)), idx(COLS[0], (8,))]),
    "index_keyword": lambda: torch.ops.aten.index(
        IMAGE(torch.float32), indices=[None, idx([2, 0], (2,))]),
}
for name, call in PACKETS.items():
    try:
        out["packets"][name] = describe(call())
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["packets"][name] = failed(exc)


# The CI wall itself: fake propagation of a bilinear interpolate, as the
# strict export's Dynamo builder runs it.
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.nn.functional as F

FAKE = [
    ("scale2_align_false", (1, 3, 4, 4), dict(scale_factor=2, mode="bilinear", align_corners=False)),
    ("scale2_align_true", (1, 3, 4, 4), dict(scale_factor=2, mode="bilinear", align_corners=True)),
    ("size_6x10", (2, 2, 3, 5), dict(size=(6, 10), mode="bilinear", align_corners=False)),
]
for name, shape, kwargs in FAKE:
    try:
        with FakeTensorMode() as mode:
            x = mode.from_tensor(torch.zeros(shape))
            y = F.interpolate(x, **kwargs)
        out["fake"][name] = describe(y, values=False)
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["fake"][name] = failed(exc)

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


def test_the_op_answers_index_tensors_values():
    """`_unsafe_index.Tensor` on the four index patterns `_upsample_linear`
    builds, in float32, float64 and int64, and on four of `index.Tensor`'s
    placement rules (a block in the middle, a block moved to the front by a
    `None`, 0-d indices, empty indices): shape, stride, dtype, contiguity and
    values. An out-of-range index raises upstream's IndexError, whole
    message. Red when the dispatch arm is missing (a named refusal) or when it
    answers another op's layout."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["dense"])
    assert len(names) == 17, names
    assert up["dense"]["out_of_range"].get("raised") == "IndexError", up["dense"]["out_of_range"]
    assert up["dense"]["float32|bilinear_r0_c0"]["shape"] == [1, 3, 8, 8], up["dense"]
    bad = _diff("dense", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} dense rows differ:\n" + "\n".join(bad)


def test_meta_inputs_answer_upstreams_shape_and_stride():
    """The same calls on meta tensors: shape, stride, dtype and contiguity
    (no values; the out-of-range row is not refused on meta upstream, which
    reads no index values). Red when the op leaves the shared meta arm."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["meta"])
    assert len(names) == 17, names
    bad = _diff("meta", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} meta rows differ:\n" + "\n".join(bad)


def test_packets_without_a_default_reach_upstreams_overload():
    """`torch.ops.aten._unsafe_index(x, idx)`, `torch.ops.aten.index(x, idx)`
    and `torch.ops.aten.index(x, indices=...)`: the packet call answers what
    upstream's packet answers, which is `.Tensor`'s result. Red when the
    packet dispatches `<name>.default` (`aten op not implemented in torch._C
    shim: aten.index.default`)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["packets"])
    assert len(names) == 3, names
    assert not [n for n in names if "raised" in up["packets"][n]], up["packets"]
    bad = _diff("packets", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} packet rows differ:\n" + "\n".join(bad)


def test_fake_interpolate_answers_upstreams_shape():
    """`F.interpolate(mode="bilinear")` on a fake tensor, the call the strict
    export's Dynamo builder makes on executorch's pattern module (CI run
    43826381567), at scale 2 with and without `align_corners` and to an
    explicit size: the result's shape, stride, dtype and contiguity. Red on
    that CI wall (`no meta kernel for aten._unsafe_index.default`) and on
    any op of `_upsample_linear` without a meta kernel here."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["fake"])
    assert len(names) == 3, names
    assert up["fake"]["scale2_align_false"]["shape"] == [1, 3, 8, 8], up["fake"]
    bad = _diff("fake", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} fake rows differ:\n" + "\n".join(bad)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_unsafe_index")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
