"""`aten.index.Tensor` and `aten.gather.default` on mps (issue #32).

Both were refused by the host-readback gate: their kernels read the indices
(and, for `gather`, the operand) to the host, and candle's Metal
`index_select`/`gather` kernels do not bounds-check, so simply calling them
would read outside the buffer where upstream raises. Every transformers
forward under continuous batching reaches `index.Tensor`, so this was #32's
wall.

What computes now (`aten.rs::index_tensor_on_device`, `gather_on_device`):
integer index tensors (`int64` everywhere; `int32` on cuda), negative
indices wrapped on the device for `index.Tensor` (and refused for `gather`,
which upstream does not wrap), broadcasting among index tensors, slices and
separating axes, empty indices; values float32/float16/bfloat16/int64/bool.

**The readback budget, asserted exactly.** The bounds check reduces each
index tensor to its min and max on the device and reads that one short
vector back (`index_validation_readback`): **1 download per call** with a
non-empty result, **0** when the broadcast is empty, and **2** on an
out-of-range call (the second reads the indices only to name the first
offender, as upstream does, before raising). That readback is validation,
never an operand of the answer, and docs/devices/matrix.md §7.19 records it.

What still refuses, by name: boolean/uint8 mask indices on a device (the
result's length is data-dependent), and `int32` index tensors on mps (no
int32 Metal kernel, not even the cast).

Errors: upstream's **CPU** type and text (`IndexError: index 5 is out of
bounds for dimension 0 with size 3`, `RuntimeError` for `gather`). Upstream's
mps raises `torch.AcceleratorError` (a `RuntimeError`), with the same text
for `index` and `gather: index 5 is out of bounds for dimension with size 4`
for gather (measured, torch 2.13.0); the CPU wording is the oracle here.

Nullifications this file is meant to catch:

* the device bounds check removed (`index_bounds_on_device` returning Ok, or
  the `ends` test in `gather_on_device` deleted) -> the out-of-range rows
  (no raise, or candle reads garbage) and their download count
* the negative wrap removed -> the negative-index rows (wrong values or an
  out-of-range read)
* the router sending an mps tensor to `index_tensor_on_host`/`gather_on_host`
  -> every agreement row's download count (the host kernels read the whole
  index, and `gather_on_host` the operand too), and
  test_only_the_routers_call_the_host_halves
* a second validation download per index tensor -> the multi-index rows (2
  index tensors still 1 download)
* the mask refusal removed -> test_masks_and_mps_int32_indices_refuse_by_name
"""

import json
import math
import os
import re
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import test_shim
from test_shim import _C
import _skip


_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")


def _mps_or_skip(what):
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")):
        _skip.skip("   (skipped %s: no vendored shim at %s -- run scripts/vendor/install_shim.sh)"
                   % (what, _VENDOR_DIR))
        return False
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return False
    return True


def _run(payload, upstream):
    env = dict(os.environ)
    env.pop("TORCH_C_ARTEFACT", None)
    if upstream:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    else:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    payload = dict(payload, shim=not upstream)
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, json.dumps(payload)],
                          capture_output=True, text=True, timeout=1800, env=env)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise AssertionError("child produced no JSON (rc=%d):\n%s"
                             % (proc.returncode, proc.stderr[-3000:]))
    return json.loads(lines[-1])


# One expression per child (a candle Metal panic poisons the encoder's mutex
# for the rest of a process, test_mpsgen.py). The counters bracket only the
# block, and are recorded whether it answers or raises: the error rows assert
# their download count too.
_SCRIPT = r"""
import json, sys
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch in this child", IS_SHIM)
D = A["device"]
DT = getattr(torch, A["dtype"])
def T(rows, dtype=None):
    return torch.tensor(rows, dtype=dtype or DT).to(D)
L = torch.int64
out = {}
X = T(A["data"])
ns = dict(torch=torch, X=X, T=T, DT=DT, D=D, L=L)
before = torch._C._metal_counters() if IS_SHIM and D == "mps" else None
try:
    exec(A["code"], ns)
    r = ns["R"]
    out = {"shape": list(r.shape), "dtype": str(r.dtype), "device": str(r.device)}
except BaseException as exc:
    out = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:600])}
after = torch._C._metal_counters() if before is not None else None
if "error" not in out:
    out["values"] = ns["R"].cpu().double().flatten().tolist()
if before is not None:
    out["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
print(json.dumps(out))
"""

_GRID = [[1.0, -2.0, 3.0, 0.5], [4.0, 0.0, -1.0, 2.0], [-3.0, 5.0, 1.5, -0.5]]
_ALL = ["float32", "float16", "bfloat16", "int64", "bool"]
_FLOATS = ["float32", "float16", "bfloat16"]

# (name, code, dtypes, downloads). `X` is `_GRID` in the row's dtype on the
# device. `downloads` is the designed count: 1 for a non-empty result (the
# min/max vector of every index tensor, in one read), 0 for an empty one.
_INDEX_ROWS = [
    ("one index on dim 0", "R = X[T([0, 2, 1], L)]", _ALL, 1),
    ("negative indices wrap", "R = X[T([-1, 0, -3], L)]", _ALL, 1),
    ("index on dim 1 after a slice", "R = X[:, T([3, -4, 1], L)]", _ALL, 1),
    ("repeated indices", "R = X[T([1, 1, 1, 1], L)]", ["float32", "int64"], 1),
    ("two index tensors broadcast", "R = X[T([[0], [2]], L), T([1, -1, 0], L)]", _ALL, 1),
    ("multi-dim index", "R = X[T([[0, 1], [2, 0]], L)]", _ALL, 1),
    ("adjacent indices spliced at axis 1",
     "Y = X.reshape(3, 2, 2)\nR = Y[:, T([[1], [0]], L), T([0, 1, 1], L)]", _FLOATS + ["int64"], 1),
    ("separated indices go to the front",
     "Y = X.reshape(3, 2, 2)\nR = Y[T([0, 2], L), :, T([1, -2], L)]", _FLOATS + ["int64"], 1),
    ("index of a transposed view", "R = X.t()[T([3, 0], L)]", ["float32", "int64"], 1),
    ("0-d index through the aten op",
     "R = torch.ops.aten.index.Tensor(X, [T(1, L)])", ["float32"], 1),
    ("empty index", "R = X[T([], L)]", _ALL, 0),
    ("empty index on dim 1", "R = X[:, T([], L)]", ["float32", "int64"], 0),
]

_GATHER_ROWS = [
    ("gather dim 1", "R = X.gather(1, T([[0, 3], [1, 1], [2, 0]], L))", _ALL, 1),
    ("gather dim 0, index smaller off-axis", "R = X.gather(0, T([[2, 0], [1, 2]], L))", _ALL, 1),
    ("gather dim 0, index longer along dim",
     "R = X.gather(0, T([[0, 1, 2, 0], [2, 2, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0]], L))",
     ["float32", "int64"], 1),
    ("gather from a transposed view",
     "R = X.t().gather(1, T([[0, 2], [1, 0], [2, 2], [0, 1]], L))", _FLOATS + ["int64"], 1),
    ("gather with an empty index", "R = X.gather(1, T([[], [], []], L))", ["float32", "int64"], 0),
]

# (name, code). Shim mps against upstream cpu: same exception type, same
# text, and exactly 2 downloads (the min/max vector, then the indices to name
# the first offender).
_ERROR_ROWS = [
    ("index 5 past the end", "R = X[T([0, 5], L)]"),
    ("index -4 before the start", "R = X[T([0, -4], L)]"),
    ("first offender is named, not the extreme", "R = X[T([7, -9, 5], L)]"),
    ("dimension names the index position", "R = X[:, T([4], L)]"),
    ("second index tensor out of range", "R = X[T([0, 1], L), T([9, 1], L)]"),
    ("broadcast order picks the offender", "R = X[T([[0], [5]], L), T([1, 7], L)]"),
    ("gather past the end", "R = X.gather(1, T([[0], [5], [1]], L))"),
    ("gather does not wrap -1", "R = X.gather(1, T([[0], [-1], [1]], L))"),
    ("gather names the first offender", "R = X.gather(1, T([[7], [-1], [9]], L))"),
]


def _same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


def _agree(rows):
    problems = []
    for name, code, dtypes, downloads in rows:
        for dtype in dtypes:
            payload = dict(data=_GRID, code=code, dtype=dtype)
            want = _run(dict(payload, device="cpu"), upstream=True)
            got = _run(dict(payload, device="mps"), upstream=False)
            label = "%s [%s]" % (name, dtype)
            assert "error" not in want, (label, want)
            if "error" in got:
                problems.append("%s: shim raised %s" % (label, got["error"]))
                continue
            if not got["device"].startswith("mps"):
                problems.append("%s: result on %s" % (label, got["device"]))
            if (got["shape"], got["dtype"]) != (want["shape"], want["dtype"]):
                problems.append("%s: %s %s, upstream %s %s" % (
                    label, got["shape"], got["dtype"], want["shape"], want["dtype"]))
            elif not all(_same(a, b) for a, b in zip(got["values"], want["values"])):
                problems.append("%s: %s, upstream %s" % (label, got["values"], want["values"]))
            c = got["counters"]
            if c["host_downloads"] != downloads:
                problems.append("%s: %d host downloads, designed %d (%s)"
                                % (label, c["host_downloads"], downloads, c))
            if downloads and c["compute_encoders"] <= 0:
                problems.append("%s: no compute encoder ran (%s)" % (label, c))
    assert not problems, "\n".join(problems)


def test_index_tensor_computes_on_mps_and_agrees():
    """Integer `x[idx]` on mps equals upstream cpu exactly, in every listed
    dtype, stays on mps, and downloads exactly the designed count."""
    if not _mps_or_skip("index.Tensor on mps"):
        return
    _agree(_INDEX_ROWS)


def test_gather_computes_on_mps_and_agrees():
    """`gather` on mps equals upstream cpu exactly (index smaller off-axis,
    longer along dim, transposed self, empty), with the designed download
    count."""
    if not _mps_or_skip("gather on mps"):
        return
    _agree(_GATHER_ROWS)


def test_out_of_range_indices_raise_upstreams_error_on_mps():
    """Out-of-range positive and negative indices raise upstream cpu's
    exception type and exact text on mps, after exactly two downloads. This
    is the row candle's unchecked Metal kernels would otherwise turn into an
    out-of-buffer read."""
    if not _mps_or_skip("index bounds on mps"):
        return
    problems = []
    for name, code in _ERROR_ROWS:
        payload = dict(data=_GRID, code=code, dtype="float32")
        want = _run(dict(payload, device="cpu"), upstream=True)
        got = _run(dict(payload, device="mps"), upstream=False)
        assert "error" in want, (name, want)
        if "error" not in got:
            problems.append("%s: answered on mps (%s), upstream raised %s"
                            % (name, got.get("values"), want["error"]))
            continue
        if got["error"] != want["error"]:
            problems.append("%s: shim %r, upstream %r" % (name, got["error"], want["error"]))
        c = got["counters"]
        if c["host_downloads"] != 2:
            problems.append("%s: %d host downloads on the error path, designed 2 (%s)"
                            % (name, c["host_downloads"], c))
    assert not problems, "\n".join(problems)


_REFUSAL_ROWS = [
    ("bool mask", "R = X[X > 0]", "aten.index.Tensor", "mask index is not implemented for the mps device"),
    ("uint8 mask", "R = X[(X > 0).to(torch.uint8)]", "aten.index.Tensor",
     "mask index is not implemented for the mps device"),
    ("int32 index", "R = X[T([0, 2], torch.int32)]", "aten.index.Tensor",
     "int32 index tensor is not implemented on the mps device"),
    ("int32 gather index", "R = X.gather(1, T([[0], [1], [2]], torch.int32))", "aten.gather.default",
     "int32 index tensor is not implemented on the mps device"),
]


def test_masks_and_mps_int32_indices_refuse_by_name():
    """The paths not implemented on the device refuse with the op's name and
    the reason, never candle's words, and read nothing back first."""
    if not _mps_or_skip("index refusals on mps"):
        return
    problems = []
    for name, code, op, says in _REFUSAL_ROWS:
        got = _run(dict(data=_GRID, code=code, dtype="float32", device="mps"), upstream=False)
        if "error" not in got:
            problems.append("%s: answered on mps; if a device kernel landed, move this row "
                            "to an agreement test with counters" % name)
            continue
        err = got["error"]
        if not err.startswith("NotImplementedError: %s:" % op):
            problems.append("%s: not a refusal naming %s: %s" % (name, op, err[:300]))
        elif says not in err or "candle:" in err:
            problems.append("%s: refusal does not say %r: %s" % (name, says, err[:300]))
        if got["counters"]["host_downloads"] != 0:
            problems.append("%s: read back before refusing (%s)" % (name, got["counters"]))
    assert not problems, "\n".join(problems)


def test_the_two_ops_left_the_readback_list_and_the_rest_did_not():
    """The artefact's table: `index.Tensor` and `gather` are off it, nothing
    moved into the excused column instead, and their neighbours that still
    read back are still on it."""
    refused = set(_C._shim_mps_host_readback_ops())
    for op in ("aten.index.Tensor", "aten.gather.default"):
        assert op not in refused, op
    for op in ("aten.scatter.src", "aten.sort.default", "aten.multinomial.default",
               "aten.view.dtype", "aten.masked_select.default"):
        assert op in refused, op
    assert set(_C._shim_mps_readback_but_allowed()) == {
        "aten._local_scalar_dense.default", "aten.uniform_.default"}


def test_only_the_routers_call_the_host_halves():
    """`index_tensor_on_host`, `gather_on_host` and `index_validation_readback`
    are barriers in test_shim.py's readback derivation. A barrier is only
    honest while its callers are the ones its reason names, so the call graph
    is read here: each host half is called only by its router, which tests
    for `Device::Cpu`; the validation readback only by the device paths."""
    mods = test_shim._all_rs_modules()
    if mods is None:
        _skip.skip("   (skipped router claim: torchnative/rust/torch_c/src is not beside "
                   "this file -- installed rather than in-tree)")
        return
    callers = {}
    for module, (bodies, _) in mods.items():
        for fn, body in bodies.items():
            for name in test_shim._CALL_IN_MODULE.findall(body):
                if module == "aten":
                    callers.setdefault(name, set()).add(fn)
            for other, name in test_shim._CALL_QUALIFIED.findall(body):
                if other == "aten":
                    callers.setdefault(name, set()).add("%s::%s" % (module, fn))
    bodies = mods["aten"][0]
    for host, router in (("index_tensor_on_host", "index_tensor"),
                         ("gather_on_host", "gather_default")):
        assert host in bodies, host
        assert callers.get(host, set()) - {host} == {router}, (host, callers.get(host))
        assert re.search(r"Device::Cpu", bodies[router]), router
    assert callers.get("index_validation_readback", set()) <= {
        "index_extremes", "index_bounds_on_device", "gather_on_device"}, \
        callers.get("index_validation_readback")
    for name in ("index_tensor_on_host", "gather_on_host", "index_validation_readback"):
        assert name in test_shim._MPS_READBACK_EXEMPT, name


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_mpsindex")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
