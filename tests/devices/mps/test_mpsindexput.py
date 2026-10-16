"""`aten.index_put_.default` on mps for integer index tensors.

`index_put_` was refused on mps through `device.rs::MPS_HOST_READBACK_OPS`:
its kernel (`aten.rs::index_put_on_host`) reads `self`, the indices and the
values to the host and walks them there, and candle's Metal scatter and
index_add kernels do not bounds-check, so calling them directly would write
outside the buffer where upstream raises.

What computes now (`aten.rs::index_put_on_device`): int64 index tensors,
negative indices wrapped on the device, broadcasting among index tensors,
adjacent indices spliced in place and separated ones fronted (the
`index.Tensor` placement rules), `values` broadcast to the indexing result,
a zero-dim CPU `values` (the scalar `x[idx] = 5` lifts to), `accumulate=False`
and `accumulate=True`. Dtypes float32/float16/bfloat16/int64/bool.

How it computes. The indexed axes are moved to the front and `self` is viewed
as `(block, span)`, exactly as `index_tensor_on_device` does; the flat index
is one `i64` per broadcast element. `accumulate=True` is one candle
`index_add` on that view. `accumulate=False` finds, per destination row, the
last source row that names it (a candle `scatter` of the source positions),
then `index_select`s those rows and `where`s them over `self`.

**The readback budget, asserted exactly.** The bounds check is #89's
(`index_bounds_on_device`): each index tensor's min and max reduced on the
device and read back as one short vector. **1 download per call** with a
non-empty write, **0** when the broadcast is empty, **2** on an out-of-range
call (the second names the first offender before raising). Nothing is
written before that check passes: the error rows assert `self` unchanged.

Oracles (AGENTS.md §16): every agreement row is compared with upstream torch
on **mps** in a separate subprocess; upstream cpu is recorded and must agree
with it (a row where they disagree is reported, not graded). The error rows
use upstream **cpu**'s exception type and text, because upstream mps raises
asynchronously: measured on torch 2.13.0, `x[:, tensor([9])] = 1.0` on mps
returns without raising and the `AcceleratorError` surfaces later (at exit),
so it is not an oracle for when or whether the call raises.

**Duplicate indices with `accumulate=False` have no fixed winner upstream.**
Measured on 2.13.0: `zeros(5)[[1, 1, 1, 3]] = [1, 2, 3, 4]` gives `3` at
position 1 on cpu (last wins) and `1` on mps (first wins); with 1000
duplicates per position mps gave `992` and `1984` (neither first nor last).
So the duplicate row asserts only that each written element is one of the
values aimed at it. The sum (`accumulate=True`) is defined, and is graded.

What still refuses, by name, with zero downloads: a boolean or uint8 mask
index that is not the `masked_fill_` lowering `__setitem__` already makes
(its write count is data-dependent), and an `int32` index tensor on mps (this
base has no int32 Metal kernel; PR #100 adds the cast, and this row flips
when that lands).

Nullifications this file is meant to catch:

* the bounds check removed (`index_bounds_on_device` call deleted from
  `index_put_on_device`) -> every row of
  test_out_of_range_indices_raise_and_leave_self_unchanged (no raise, a
  write outside the buffer, or 0 downloads instead of 2)
* the negative wrap removed -> the negative-index rows
* the router sending an mps receiver to `index_put_on_host` -> every
  agreement row's download count (the host kernel reads everything)
* `index_put_` put back on `MPS_HOST_READBACK_OPS` -> every agreement row
  (refused) and test_index_put_left_the_readback_list
* the separated/adjacent placement swapped -> the two 3-d rows
"""

import json
import math
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import test_shim  # noqa: F401  (puts the shim's _C first, as the other mps suites do)
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


# One block per child (a candle Metal panic poisons the encoder's mutex for
# the rest of a process, test_mpsgen.py). `X` is `_GRID` in the row's dtype on
# the device, built before the bracket; the block leaves its answer in `R`.
# The counters bracket only the block and are recorded whether it answers or
# raises; on a raise, `X` after the block is recorded too.
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
else:
    out["self_after"] = X.cpu().double().flatten().tolist()
if before is not None:
    out["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
print(json.dumps(out))
"""

_GRID = [[1.0, -2.0, 3.0, 0.5], [4.0, 0.0, -1.0, 2.0], [-3.0, 5.0, 1.5, -0.5]]
_FLAT = [v for row in _GRID for v in row]
_FLOATS = ["float32", "float16", "bfloat16"]
_ALL = _FLOATS + ["int64", "bool"]

# (name, code, dtypes, downloads). `downloads` is the designed count: 1 for a
# non-empty write (the min/max vector of every index tensor, one read), 0 for
# an empty one.
_ROWS = [
    ("x[idx] = scalar", "X[T([0, 2], L)] = 4\nR = X", _ALL, 1),
    ("negative indices wrap, values broadcast over rows",
     "X[T([-1, 0], L)] = T([1, 2, 3, 4])\nR = X", _ALL, 1),
    ("x[:, idx] = v", "X[:, T([3, -4], L)] = T([[1, 2], [3, 4], [5, 6]])\nR = X", _ALL, 1),
    ("x[idx0, idx1] = v", "X[T([0, 2, -1], L), T([1, -1, 0], L)] = T([9, 8, 7])\nR = X",
     _ALL, 1),
    ("index tensors broadcast", "X[T([[0], [2]], L), T([1, 3], L)] = T([[1, 2], [3, 4]])\nR = X",
     _FLOATS + ["int64"], 1),
    ("multi-dim index", "X[T([[0, 2]], L)] = 6\nR = X", ["float32", "int64"], 1),
    ("adjacent indices spliced at axis 1",
     "Y = X.reshape(3, 2, 2)\nY[:, T([1, 0], L), T([0, 1], L)] = T([5, 6])\nR = Y",
     _FLOATS + ["int64"], 1),
    ("separated indices go to the front",
     "Y = X.reshape(3, 2, 2)\nY[T([0, 2], L), :, T([1, -2], L)] = T([[1, 2], [3, 4]])\nR = Y",
     _FLOATS + ["int64"], 1),
    ("0-dim cpu value through the aten op",
     "X.index_put_((T([1], L),), torch.tensor(5, dtype=DT))\nR = X", _ALL, 1),
    ("accumulate=True sums duplicates",
     "X.index_put_((T([0, 0, 2, 0], L),), T([1, 2, 3, 4]).reshape(4, 1), accumulate=True)\nR = X",
     _ALL, 1),
    ("accumulate=True with two index tensors",
     "X.index_put_((T([1, 1, -1], L), T([2, 2, 0], L)), T([5, 6, 7]), accumulate=True)\nR = X",
     _FLOATS + ["int64"], 1),
    ("accumulate=True runs at the receiver's precision",
     "R = torch.zeros(1, dtype=DT).to(D)\n"
     "R.index_put_((torch.zeros(64, dtype=L).to(D),), torch.full((64,), 0.01, dtype=DT).to(D), "
     "accumulate=True)", _FLOATS, 1),
    ("write through a view taken before the call",
     "V = X[1]\nV[T([0, 3], L)] = 6\nR = X", ["float32", "int64"], 1),
    ("int64 rows not named stay put",
     "X[T([1], L)] = T([[10, 11, 12, 13]])\nR = X", ["int64", "float32"], 1),
    ("empty index writes nothing", "X[T([], L)] = 1\nR = X", ["float32", "int64"], 0),
]

# (name, code). Shim mps against upstream cpu: same exception type and text,
# exactly 2 downloads, and `X` unchanged.
_ERROR_ROWS = [
    ("index 3 past the end", "X[T([0, 3], L)] = 1"),
    ("index -4 before the start", "X[T([0, -4], L)] = 1"),
    ("dimension names the index position", "X[:, T([4], L)] = 1"),
    ("second index tensor out of range", "X[T([0, 1], L), T([9, 1], L)] = 1"),
    ("accumulate=True is checked too",
     "X.index_put_((T([0, 5], L),), T(1.0), accumulate=True)"),
]


def _same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


def _counted(label, got, downloads, problems):
    c = got["counters"]
    if c["host_downloads"] != downloads:
        problems.append("%s: %d host downloads, designed %d (%s)"
                        % (label, c["host_downloads"], downloads, c))
    if downloads and c["compute_encoders"] <= 0:
        problems.append("%s: no compute encoder ran (%s)" % (label, c))


def test_index_put_computes_on_mps_and_agrees():
    """Every row: the shim's mps result equals upstream mps exactly (which
    equals upstream cpu), stays on mps, and downloads exactly the designed
    count with at least one compute encoder."""
    if not _mps_or_skip("index_put_ on mps"):
        return
    problems = []
    for name, code, dtypes, downloads in _ROWS:
        for dtype in dtypes:
            payload = dict(data=_GRID, code=code, dtype=dtype)
            oracle = _run(dict(payload, device="mps"), upstream=True)
            host = _run(dict(payload, device="cpu"), upstream=True)
            label = "%s [%s]" % (name, dtype)
            assert "error" not in oracle, (label, "upstream mps", oracle)
            assert "error" not in host, (label, "upstream cpu", host)
            if not all(_same(a, b) for a, b in zip(oracle["values"], host["values"])):
                problems.append("%s: upstream mps %s and cpu %s disagree; not graded"
                                % (label, oracle["values"], host["values"]))
                continue
            got = _run(dict(payload, device="mps"), upstream=False)
            if "error" in got:
                problems.append("%s: shim raised %s" % (label, got["error"]))
                continue
            if not got["device"].startswith("mps"):
                problems.append("%s: result on %s" % (label, got["device"]))
            if (got["shape"], got["dtype"]) != (oracle["shape"], oracle["dtype"]):
                problems.append("%s: %s %s, upstream %s %s" % (
                    label, got["shape"], got["dtype"], oracle["shape"], oracle["dtype"]))
            elif not all(_same(a, b) for a, b in zip(got["values"], oracle["values"])):
                problems.append("%s: %s, upstream mps %s" % (label, got["values"], oracle["values"]))
            _counted(label, got, downloads, problems)
    assert not problems, "\n".join(problems)


def test_duplicate_indices_without_accumulate_write_one_of_their_values():
    """`x[[1, 1, 1]] = rows` without accumulate: upstream fixes no winner
    (cpu takes the last, mps the first, measured), so each element of row 1
    must be one of the three values aimed at it and every other row must be
    untouched. One download, as for any non-empty write."""
    if not _mps_or_skip("duplicate index_put_ on mps"):
        return
    code = ("X[T([1, 1, 1], L)] = T([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])\n"
            "R = X")
    problems = []
    for dtype in ("float32", "int64"):
        label = "duplicates [%s]" % dtype
        got = _run(dict(data=_GRID, code=code, dtype=dtype, device="mps"), upstream=False)
        if "error" in got:
            problems.append("%s: shim raised %s" % (label, got["error"]))
            continue
        values = got["values"]
        for k in range(4):
            if values[4 + k] not in (10 + k, 20 + k, 30 + k):
                problems.append("%s: column %d holds %r, none of the values aimed at it"
                                % (label, k, values[4 + k]))
        # `_GRID` in this dtype: int64 truncates toward zero (0.5 -> 0, 1.5 -> 1,
        # -0.5 -> 0), so the untouched rows are compared with that, not `_FLAT`.
        start = _FLAT if dtype == "float32" else [float(int(v)) for v in _FLAT]
        if values[:4] != start[:4] or values[8:] != start[8:]:
            problems.append("%s: rows 0 and 2 changed: %s" % (label, values))
        _counted(label, got, 1, problems)
    assert not problems, "\n".join(problems)


def test_out_of_range_indices_raise_and_leave_self_unchanged():
    """Out-of-range positive and negative indices raise upstream cpu's type
    and exact text on mps, after exactly two downloads, and nothing was
    written: candle's unchecked Metal scatter would otherwise write outside
    the buffer."""
    if not _mps_or_skip("index_put_ bounds on mps"):
        return
    problems = []
    for name, code in _ERROR_ROWS:
        payload = dict(data=_GRID, code=code + "\nR = X", dtype="float32")
        want = _run(dict(payload, device="cpu"), upstream=True)
        got = _run(dict(payload, device="mps"), upstream=False)
        assert "error" in want, (name, want)
        if "error" not in got:
            problems.append("%s: answered on mps (%s), upstream cpu raised %s"
                            % (name, got.get("values"), want["error"]))
            continue
        if got["error"] != want["error"]:
            problems.append("%s: shim %r, upstream %r" % (name, got["error"], want["error"]))
        if got["self_after"] != _FLAT:
            problems.append("%s: self changed before the raise: %s" % (name, got["self_after"]))
        c = got["counters"]
        if c["host_downloads"] != 2:
            problems.append("%s: %d host downloads on the error path, designed 2 (%s)"
                            % (name, c["host_downloads"], c))
    assert not problems, "\n".join(problems)


_REFUSAL_ROWS = [
    ("bool mask not the receiver's shape", "X[T([True, False, True], torch.bool)] = 1\nR = X",
     "mask index is not implemented for the mps device"),
    ("int32 index", "X[T([0, 2], torch.int32)] = 1\nR = X",
     "int32 index tensor is not implemented on the mps device"),
]


def test_masks_and_int32_indices_refuse_by_name():
    """The paths not implemented on the device refuse naming
    `aten.index_put_.default` and the reason, never candle's words, and read
    nothing back first."""
    if not _mps_or_skip("index_put_ refusals on mps"):
        return
    problems = []
    for name, code, says in _REFUSAL_ROWS:
        got = _run(dict(data=_GRID, code=code, dtype="float32", device="mps"), upstream=False)
        if "error" not in got:
            problems.append("%s: answered on mps; if a device kernel landed, move this row "
                            "to an agreement row with counters" % name)
            continue
        err = got["error"]
        if not err.startswith("NotImplementedError: aten.index_put_.default:"):
            problems.append("%s: not a refusal naming index_put_: %s" % (name, err[:300]))
        elif says not in err or "candle:" in err:
            problems.append("%s: refusal does not say %r: %s" % (name, says, err[:300]))
        if got["counters"]["host_downloads"] != 0:
            problems.append("%s: read back before refusing (%s)" % (name, got["counters"]))
    assert not problems, "\n".join(problems)


def test_index_put_left_the_readback_list():
    """The artefact's table: `index_put_` is off it, nothing moved into the
    excused column instead, and its neighbours that still read back are on
    it."""
    refused = set(_C._shim_mps_host_readback_ops())
    assert "aten.index_put_.default" not in refused
    for op in ("aten.index_add_.default", "aten.index_add.default", "aten.scatter.src",
               "aten.masked_scatter.default"):
        assert op in refused, op
    assert set(_C._shim_mps_readback_but_allowed()) == {
        "aten._local_scalar_dense.default", "aten.uniform_.default"}


def test_only_the_router_calls_the_host_half():
    """`index_put_on_host` is a barrier in test_shim.py's readback
    derivation; that is honest only while its one caller is the router that
    tests for `Device::Cpu`."""
    mods = test_shim._all_rs_modules()
    if mods is None:
        _skip.skip("   (skipped router claim: torchnative/rust/torch_c/src is not beside "
                   "this file -- installed rather than in-tree)")
        return
    callers = set()
    for module, (bodies, _) in mods.items():
        for fn, body in bodies.items():
            if module == "aten" and "index_put_on_host" in test_shim._CALL_IN_MODULE.findall(body):
                callers.add(fn)
            for other, name in test_shim._CALL_QUALIFIED.findall(body):
                if other == "aten" and name == "index_put_on_host":
                    callers.add("%s::%s" % (module, fn))
    bodies = mods["aten"][0]
    assert "index_put_on_host" in bodies and "index_put_on_device" in bodies
    assert callers - {"index_put_on_host"} == {"index_put_inplace"}, callers
    assert "Device::Cpu" in bodies["index_put_inplace"]
    assert "index_bounds_on_device" in bodies["index_put_on_device"]
    assert "index_put_on_host" in test_shim._MPS_READBACK_EXEMPT


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_mpsindexput")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
