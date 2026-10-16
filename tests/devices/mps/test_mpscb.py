"""Continuous batching's walls on mps (issue #32).

Found by #13 (transformers continuous batching on mps). Four groups, and this
file pins each one to either a computed answer or a refusal by name
(AGENTS.md §18):

1. **candle-symbol errors.** `cumsum` on a float mps tensor, `bool.copy_`,
   `any` of a float tensor, and the float comparisons died with candle's own
   `Metal contiguous to_dtype F32 F64 not implemented` (or `U8 F64`): the
   kernels widened to `f64` first, and Metal has no `f64`. The comparisons
   were fixed by #49; `cumsum` now accumulates in `opmath_in` off the CPU, and
   the two `!= 0` tests run in the operand's own dtype. All compute on Metal.
2. **int32 on mps** computes on Metal since the rest of issue #32: the
   vendored candle-metal-kernels fork now instantiates the `int32_t` casts,
   strided copy and fill, and the claimed operators widen to int64 on the
   device and narrow back (`aten.rs::mps_int32_widened`). The five rows that
   used to pin the refusal now agree with upstream here; test_mpsint32.py
   grades the whole set, overflow included, and pins what still refuses.
3. **Strided in-place writes** compute on Metal. `tensor.rs::write_on_device`
   splits a non-contiguous receiver into its contiguous runs and blits each
   with candle's `slice_set`; `fill_.Tensor` accepts a zero-dim CPU value
   beside an mps receiver, as upstream does, so `x[..., :k] = 0` works.
4. **Host-readback ops.** `argmax` already computes on Metal (#29) and is
   pinned here by counters. `index.Tensor` and `gather` compute on Metal for
   integer indices, bounds-checked on the device (test_mpsindex.py pins
   them, with their one validation download). `scatter.src`, `sort`,
   `multinomial` and `view.dtype` have no device kernel in this build that
   matches upstream's contract (candle's Metal `scatter` does no bounds
   check, so an out-of-range index writes outside the buffer where upstream
   raises), so they stay refused by the readback gate, by name.

Grades (AGENTS.md §16): every *computed* row *agrees* with upstream torch on
the CPU, run in a separate subprocess that asserts it imported upstream, and
is *on the GPU* by `_metal_counters()` bracketing only the expression: at
least one compute or blit encoder and zero host downloads. Every *refused*
row raises `NotImplementedError` whose first word is the aten op and which
carries no `candle:` text.

Nullifications this file is meant to catch:

* `cumsum_default`'s off-CPU accumulator back to `F64` -> the cumsum rows
* `any_from`'s device branch removed -> the `any` rows
* `copy_inplace`'s bool device branch removed -> the bool `copy_` row
* `write_on_device`'s run split removed (back to the refusal) -> every
  strided row; split wrong (one run only) -> their values
* the `fill_.Tensor` CPU-scalar exception in `visit_for_device` removed ->
  the `[..., :1] = 0` row (cross-device RuntimeError)
* the int32 casts / fill / widening removed -> test_int32_rows_compute_on_mps
* a readback op leaving `MPS_HOST_READBACK_OPS` without a device kernel ->
  test_readback_ops_refuse_by_name_on_mps (and test_metalplace.py)
"""

import json
import math
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

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


# One statement block per child (a candle Metal panic poisons the encoder's
# mutex for the rest of a process, test_mpsgen.py). `X` is the input on the
# device; the block leaves its answer in `R`. Only the block is bracketed.
_SCRIPT = r"""
import json, sys
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch in this child", IS_SHIM)
D = A["device"]
inf, nan = float("inf"), float("nan")
DT = getattr(torch, A["dtype"])
def T(rows, dtype=None):
    return torch.tensor(rows, dtype=dtype or DT).to(D)
out = {}
try:
    X = T(A["data"])
    ns = dict(torch=torch, X=X, T=T, DT=DT, D=D, inf=inf, nan=nan)
    before = torch._C._metal_counters() if IS_SHIM and D == "mps" else None
    exec(A["code"], ns)
    after = torch._C._metal_counters() if before is not None else None
    r = ns["R"]
    parts = list(r) if isinstance(r, tuple) else [r]
    out = {"parts": [{"shape": list(p.shape), "dtype": str(p.dtype),
                      "device": str(p.device),
                      "values": p.cpu().double().flatten().tolist()} for p in parts]}
    if before is not None:
        out["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
except BaseException as exc:
    out = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:600])}
print(json.dumps(out))
"""

_INF, _NAN = float("inf"), float("nan")
_GRID = [[1.0, -2.0, 3.0, 0.5], [4.0, 0.0, -1.0, 2.0], [-3.0, 5.0, 1.5, -0.5]]
_FLOATS = ["float32", "float16", "bfloat16"]

# (name, data, code, dtypes). Values are exact in every dtype listed, and the
# cumsum partial sums are too, so the comparison is exact (NaN == NaN).
_CANDLE_SYMBOL_ROWS = [
    ("cumsum dim=1", _GRID, "R = torch.cumsum(X, dim=1)", _FLOATS),
    ("cumsum dim=0", _GRID, "R = X.cumsum(0)", _FLOATS),
    ("cumsum to float32", _GRID, "R = X.cumsum(1, dtype=torch.float32)", ["float16", "bfloat16"]),
    ("lt.Tensor", _GRID, "R = X < X.flip(1)", _FLOATS),
    ("le.Tensor", _GRID, "R = X <= X.flip(0)", _FLOATS),
    ("lt.Scalar", [[1.0, -_INF, 2.5, _NAN]], "R = X < 2.0", _FLOATS),
    ("gt.Scalar", [[1.0, _INF, 2.5, _NAN]], "R = X > 1.0", _FLOATS),
    ("bool copy_", _GRID,
     "B = X > 0\nC = torch.zeros_like(B)\nC.copy_(B)\nR = C", _FLOATS),
    ("bool copy_ into a view", _GRID,
     "B = X > 0\nC = torch.zeros(3, 6, dtype=torch.bool, device=D)\nC[:, 1:5].copy_(B)\nR = C",
     ["float32"]),
    ("any of a float tensor, dim=1", [[0.0, -0.0, 0.0], [0.0, _NAN, 0.0], [0.0, 0.0, 2.0]],
     "R = X.any(dim=1)", _FLOATS),
]

_STRIDED_ROWS = [
    ("zero_ on [:, 0:2]", _GRID, "Y = X.clone()\nY[:, 0:2].zero_()\nR = Y", _FLOATS + ["int64"]),
    ("bulk input [:2, :3].zero_()", _GRID, "Y = X.clone()\nY[:2, :3].zero_()\nR = Y", ["int64"]),
    ("fill_ on [:, 1:3]", _GRID,
     "Y = X.clone()\nY[:, 1:3].fill_(torch.finfo(DT).min)\nR = Y", _FLOATS),
    ("x[..., :1] = 0", _GRID, "Y = X.clone()\nY[..., :1] = 0\nR = Y", _FLOATS + ["int64"]),
    ("copy_ into [:, 1:]", _GRID, "Y = X.clone()\nY[:, 1:].copy_(X[:, :-1])\nR = Y", _FLOATS),
    ("4D mask [:, :, :2, :3].fill_(min)", [[0.0]],
     "M = torch.zeros(2, 2, 3, 4, dtype=DT, device=D)\n"
     "M[:, :, :2, :3].fill_(torch.finfo(DT).min)\nR = M", _FLOATS),
    ("add_ on a column", _GRID, "Y = X.clone()\nY[:, 2].add_(10)\nR = Y", _FLOATS + ["int64"]),
    ("fill_ on a transposed view", _GRID, "Y = X.clone()\nY.t()[1:3].fill_(7)\nR = Y", ["float32"]),
]

_ARGMAX_ROWS = [
    ("argmax dim=1 with ties and NaN", [[1.0, 3.0, 3.0], [_NAN, 1.0, _NAN], [-_INF, -_INF, -_INF]],
     "R = X.argmax(dim=1)", _FLOATS),
    ("argmax whole tensor", _GRID, "R = X.argmax()", _FLOATS),
]


def _same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


def _agree(rows):
    problems = []
    for name, data, code, dtypes in rows:
        for dtype in dtypes:
            payload = dict(data=data, code=code, dtype=dtype)
            want = _run(dict(payload, device="cpu"), upstream=True)
            got = _run(dict(payload, device="mps"), upstream=False)
            label = "%s [%s]" % (name, dtype)
            assert "error" not in want, (label, want)
            if "error" in got:
                problems.append("%s: shim raised %s" % (label, got["error"]))
                continue
            for i, (g, w) in enumerate(zip(got["parts"], want["parts"])):
                if not g["device"].startswith("mps"):
                    problems.append("%s: result %d on %s" % (label, i, g["device"]))
                if (g["shape"], g["dtype"]) != (w["shape"], w["dtype"]):
                    problems.append("%s: result %d is %s %s, upstream %s %s"
                                    % (label, i, g["shape"], g["dtype"], w["shape"], w["dtype"]))
                elif not all(_same(a, b) for a, b in zip(g["values"], w["values"])):
                    problems.append("%s: result %d is %s, upstream %s"
                                    % (label, i, g["values"], w["values"]))
            c = got["counters"]
            if c["compute_encoders"] + c["blit_encoders"] <= 0:
                problems.append("%s: no compute or blit encoder ran (%s)" % (label, c))
            if c["host_downloads"] != 0:
                problems.append("%s: something was read back to the host (%s)" % (label, c))
    assert not problems, "\n".join(problems)


def test_candle_symbol_errors_now_compute_on_mps():
    """cumsum, lt/le.Tensor, lt/gt.Scalar, bool copy_ and any on float mps
    tensors: they used to die with candle's `Metal contiguous to_dtype ... F64`.
    Each now equals upstream exactly, stays on mps, and downloads nothing."""
    if not _mps_or_skip("candle-symbol rows on mps"):
        return
    _agree(_CANDLE_SYMBOL_ROWS)


def test_strided_in_place_writes_compute_on_mps():
    """zero_, fill_, copy_ and add_ into non-contiguous views (column slices,
    a 4D mask slice, a transposed view, `x[..., :1] = 0`) on mps: the whole
    base equals upstream's, and the bracket shows device encoders and no
    download."""
    if not _mps_or_skip("strided writes on mps"):
        return
    _agree(_STRIDED_ROWS)


def test_argmax_computes_on_mps():
    """argmax stays a device computation (first maximal index, first NaN)."""
    if not _mps_or_skip("argmax on mps"):
        return
    _agree(_ARGMAX_ROWS)


def _refused(rows, must_say):
    problems = []
    for name, data, code, dtype in rows:
        got = _run(dict(data=data, code=code, dtype=dtype, device="mps"), upstream=False)
        if "error" not in got:
            problems.append("%s: answered on mps (%s); if a device kernel landed, "
                            "move this row to an agreement test with counters"
                            % (name, got["parts"][0]["device"]))
            continue
        err = got["error"]
        if not err.startswith("NotImplementedError: aten."):
            problems.append("%s: not a refusal naming its op: %s" % (name, err[:300]))
        elif "candle:" in err:
            problems.append("%s: refusal still carries candle's words: %s" % (name, err[:300]))
        elif not any(s in err for s in must_say):
            problems.append("%s: refusal gives no reason (%s): %s" % (name, must_say, err[:300]))
    assert not problems, "\n".join(problems)


_INT32_ROWS = [
    ("embedding with int32 ids", [[0, 2], [1, 0]],
     "W = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(D)\n"
     "R = torch.nn.functional.embedding(X, W)", ["int32"]),
    ("fill_ int32", [[1, 2], [3, 4]], "X.fill_(3)\nR = X", ["int32"]),
    ("copy_ int32 <- int64", [[1, 2], [3, 4]], "X.copy_(X.long())\nR = X", ["int32"]),
    ("_to_copy to int32", [[1, 2], [3, 4]], "R = X.long().to(torch.int32)", ["int32"]),
    ("sub.Tensor int32", [[1, 2], [3, 4]], "R = X - X", ["int32"]),
]


def test_int32_rows_compute_on_mps():
    """The five int32 rows continuous batching stopped at (they used to
    refuse by name): each equals upstream, stays on mps, downloads nothing."""
    if not _mps_or_skip("int32 rows on mps"):
        return
    _agree(_INT32_ROWS)


_READBACK_ROWS = [
    ("scatter.src", _GRID,
     "R = X.scatter(1, T([[0], [1], [2]], torch.int64), X[:, :1].contiguous())", "float32"),
    ("sort", _GRID, "R = X.sort(dim=1)", "float32"),
    ("multinomial", [[0.2, 0.8]], "R = torch.multinomial(X, 1)", "float32"),
    ("view.dtype", _GRID, "R = X.view(torch.int32)", "float32"),
]


def test_readback_ops_refuse_by_name_on_mps():
    """The host-readback ops with no device kernel matching upstream's
    contract stay refused by the readback gate, by name."""
    if not _mps_or_skip("readback refusals on mps"):
        return
    _refused(_READBACK_ROWS, ("not implemented for the mps device",))


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpscb")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
