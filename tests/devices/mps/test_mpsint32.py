"""int32 on mps (issue #32): stored as int32, computed on Metal.

Continuous batching keeps its token ids, positions, sequence lengths and
block table in int32 tensors on the model's device, and every one of its
eight measured cells stopped at the first op on mps with
`aten.embedding.default: not implemented for int32 tensors on the mps device`.

What this build does now, in two halves:

1. **Buffer operations compute natively.** The vendored candle-metal-kernels
   fork (`torchnative/rust/vendor/int8-candle-metal-kernels-0.11.0.patch`)
   instantiates the `int32_t` casts to and from `i64 f32 f16 bf16 u8 u32 i8`,
   the strided copy and the fill; the candle-core fork
   (`int8-candle-0.11.0-cpu.patch`) names them in `to_dtype`,
   `copy_strided_src` and `const_set`. So `_to_copy` to and from int32,
   `fill_`/`zero_`/`copy_` into int32 (strided views and broadcasts too),
   `.contiguous()` of an int32 view, and int32 *index* tensors beside a
   non-int32 `self` all run on Metal kernels.
2. **Arithmetic computes by widening on the device.** For the operators in
   `_C._shim_mps_int32_widened_ops()`, every int32 operand is cast to int64
   with the Metal cast, candle's int64 kernels compute, and the result is
   narrowed back with the truncating `i64 -> i32` Metal cast where upstream's
   result is int32. Wrap-identical for `add sub mul neg` (ring homomorphism
   Z/2**64 -> Z/2**32), exact for comparisons, `abs`, `maximum`, `where`,
   `max` and indexing, and never narrowed for `sum`/`cumsum`/`argmax` (int64
   upstream) or for true division, `exp` and `sqrt` (float upstream).

Every other operator on an int32 mps tensor refuses by name.

Grades (AGENTS.md §16): every computed row *agrees* with upstream torch on the
CPU, in a separate subprocess that asserts it imported upstream, including the
overflow rows (`int32 max + 1`, `int32 min - 1`, `max * max`, `-min`,
`abs(min)`); and is *on the GPU* by `_metal_counters()` bracketing only the
expression: at least one compute or blit encoder and **zero** host downloads
(an indexing row reads back exactly what its int64 twin reads back: the
bounds-check extremes, test_mpsindex.py).

Nullifications this file is meant to catch:

* a cast kernel dropped from cast.metal or its name from candle-core's
  `to_dtype` table -> the `_to_copy` rows and every widened row (candle names
  the missing cast, the translator rewords it, the row reports a refusal)
* `init_const_set(i32, ...)` dropped, or its `const_set` arm -> the fill rows
* `init_unary(copy, uid, i32, ...)` dropped, or the `copy_strided_src` arm ->
  the broadcast-copy and `t().contiguous()` rows
* `mps_int32_widened` not called (back to the plain dispatch) -> every
  arithmetic row refuses by name
* the narrowing skipped -> the arithmetic rows come back int64 (dtype check)
* the narrowing applied to `Keep` ops -> `sum`/`cumsum` of `max + max` wrap
  and `argmax` comes back int32
* `Promote`'s zero-dim rule wrong -> the `+ int64 0-dim` / `+ int64 [2,1]` rows
* the in-place write-back skipped -> the `add_`/`mul_` rows leave the
  receiver unchanged
* an operator not claimed starting to answer, or its refusal losing its name
  -> test_unclaimed_int32_ops_refuse_by_name
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
# device, in `DT`; the block leaves its answer in `R`. Only the block is
# bracketed by the counters. Values are read as Python ints for integer
# results, so int32 and int64 extremes compare exactly.
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
out = {}
try:
    X = T(A["data"])
    ns = dict(torch=torch, X=X, T=T, DT=DT, D=D)
    before = torch._C._metal_counters() if IS_SHIM and D == "mps" else None
    exec(A["code"], ns)
    after = torch._C._metal_counters() if before is not None else None
    r = ns["R"]
    parts = list(r) if isinstance(r, tuple) else [r]
    def vals(p):
        h = p.cpu()
        if h.dtype.is_floating_point:
            return h.double().flatten().tolist()
        return [int(v) for v in h.to(torch.int64).flatten().tolist()]
    out = {"parts": [{"shape": list(p.shape), "dtype": str(p.dtype),
                      "device": str(p.device), "values": vals(p)} for p in parts]}
    if before is not None:
        out["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
except BaseException as exc:
    out = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:600])}
print(json.dumps(out))
"""

_MAX, _MIN = 2147483647, -2147483648
_EDGE = [[_MAX, _MIN, 1], [-1, 0, _MAX - 1]]
_SMALL = [[3, -1, 4], [1, 5, -2]]
_IDS = [[0, 2], [1, 0]]

# (name, data, code). Each row runs at int32 against upstream's int32.
_BUFFER_ROWS = [
    ("_to_copy int32 -> int64", _EDGE, "R = X.long()"),
    ("_to_copy int32 -> float32", _SMALL, "R = X.float()"),
    ("_to_copy int32 -> float16", _SMALL, "R = X.half()"),
    ("_to_copy int32 -> bfloat16", _SMALL, "R = X.to(torch.bfloat16)"),
    ("_to_copy int32 -> uint8", [[3, 1, 4], [1, 5, 2]], "R = X.to(torch.uint8)"),
    ("_to_copy int64 -> int32 (wraps)", _EDGE, "R = (X.long() + 1).to(torch.int32)"),
    ("_to_copy float32 -> int32", _SMALL, "R = (X.float() * 2.5).to(torch.int32)"),
    ("_to_copy bfloat16 -> int32", _SMALL, "R = X.to(torch.bfloat16).to(torch.int32)"),
    ("clone", _EDGE, "R = X.clone()"),
    ("t().contiguous()", _EDGE, "R = X.t().contiguous()"),
    ("fill_ whole", _SMALL, "X.fill_(7)\nR = X"),
    ("fill_ int32 min", _SMALL, "X.fill_(-2147483648)\nR = X"),
    ("fill_ on a column view", _SMALL, "Y = X.clone()\nY[:, 1:].fill_(-1)\nR = Y"),
    ("zero_ on [:1, :2]", _SMALL, "Y = X.clone()\nY[:1, :2].zero_()\nR = Y"),
    ("copy_ int32 <- int64 (wraps)", _EDGE, "Y = X.clone()\nY.copy_(X.long() * 2)\nR = Y"),
    ("copy_ into a view", _SMALL, "Y = X.clone()\nY[:, 1:].copy_(X[:, :2])\nR = Y"),
    ("broadcast setitem (CB defaults)", _SMALL,
     "Y = torch.zeros(2, 5, dtype=torch.int32, device=D)\nY[:] = X[:, :1]\nR = Y"),
    ("setitem from a fresh int32 tensor (CB inputs)", _SMALL,
     "Y = torch.zeros(2, 6, dtype=torch.int32, device=D)\nY[0, :3] = X[1]\nR = Y"),
    ("empty + zero_ (CB output_ids)", _SMALL,
     "Y = torch.empty(2, 4, dtype=torch.int32, device=D)\nY.zero_()\nR = Y"),
    ("copy_ int64 argmax into int32 view (CB sampling)", _SMALL,
     "Y = torch.zeros(1, 4, dtype=torch.int32, device=D)\n"
     "Y[0, :2].copy_(X.float().argmax(1))\nR = Y"),
]

_ARITH_ROWS = [
    ("add.Tensor overflow", _EDGE, "R = X + X"),
    ("add.Scalar max + 1", _EDGE, "R = X + 1"),
    ("sub.Scalar min - 1", _EDGE, "R = X - 1"),
    ("sub.Tensor", _EDGE, "R = X - X.flip(1)"),
    ("mul.Tensor max * max", _EDGE, "R = X * X"),
    ("mul.Scalar", _EDGE, "R = X * 3"),
    ("neg of min", _EDGE, "R = -X"),
    ("abs of min", _EDGE, "R = X.abs()"),
    ("abs_ of min on the receiver", _EDGE, "Y = X.clone()\nY.abs_()\nR = Y"),
    ("maximum", _EDGE, "R = torch.maximum(X, X.flip(1))"),
    ("where.self", _EDGE, "R = torch.where(X > 0, X, X * 2)"),
    ("add with a zero-dim int64 stays int32", _EDGE,
     "R = X + torch.tensor(5, dtype=torch.int64).to(D)"),
    ("add with a dimensioned int64 is int64", _EDGE,
     "R = X + T([[1], [2]], torch.int64)"),
    ("add with a python float is float32", _SMALL, "R = X + 0.5"),
    ("add with uint8 stays int32", _EDGE, "R = X + T([[1, 2, 3], [4, 5, 6]], torch.uint8)"),
    ("add_ on the receiver (wraps)", _EDGE, "Y = X.clone()\nY.add_(1)\nR = Y"),
    ("add_ on a column view", _EDGE, "Y = X.clone()\nY[:, 0].add_(X[:, 1])\nR = Y"),
    ("mul_ on the receiver", _EDGE, "Y = X.clone()\nY.mul_(2)\nR = Y"),
    ("float32 receiver add_ int32", _SMALL,
     "Y = X.float()\nY.add_(X)\nR = Y"),
    ("eq.Tensor", _EDGE, "R = X == X.flip(0)"),
    ("ne.Scalar", _EDGE, "R = X != 1"),
    ("lt.Scalar", _EDGE, "R = X < 0"),
    ("ge.Tensor", _EDGE, "R = X >= X.flip(1)"),
    ("gt.Scalar", _EDGE, "R = X > -1"),
    ("le.Tensor", _EDGE, "R = X <= X.flip(1)"),
    ("div.Tensor is float", _SMALL, "R = X / X.flip(1)"),
    ("div.Scalar is float", _SMALL, "R = X / 2"),
    ("exp is float", _SMALL, "R = X.exp()"),
    ("sqrt is float", [[4, 9, 0], [1, 16, 25]], "R = X.sqrt()"),
    ("sum is int64 and does not wrap", _EDGE, "R = X.sum()"),
    ("sum dim is int64", _EDGE, "R = X.sum(1)"),
    ("cumsum is int64 and does not wrap", _EDGE, "R = X.cumsum(1)"),
    ("argmax is int64", _EDGE, "R = X.argmax(1)"),
    ("max is int32", _EDGE, "R = X.max()"),
]

# (name, data, code). Indexing reads back the bounds-check extremes, so the
# download count is checked against the same code at int64.
_INDEX_ROWS = [
    ("embedding with int32 ids", _IDS,
     "W = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(D)\n"
     "R = torch.nn.functional.embedding(X, W)"),
    ("embedding of an int32 weight", _IDS,
     "W = torch.tensor([[2147483647, 1], [2, -2147483648], [3, 4]], dtype=torch.int32).to(D)\n"
     "R = torch.nn.functional.embedding(X.long(), W)"),
    ("index_select float by int32", _IDS,
     "W = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(D)\n"
     "R = W.index_select(0, X[0])"),
    ("index_select int32 by int64", _EDGE,
     "R = X.index_select(1, T([2, 0], torch.int64))"),
    ("index.Tensor float by int32 (CB logits)", _IDS,
     "W = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(D)\n"
     "R = W[X[0]]"),
    ("index.Tensor int32 by int32 (CB input_ids)", _EDGE,
     "R = X[0, T([2, 0], torch.int32)]"),
    ("gather int32 by int64", _EDGE, "R = X.gather(1, T([[1], [2]], torch.int64))"),
    ("gather float by int32", _IDS,
     "W = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(D)\n"
     "R = W.gather(1, X)"),
]


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
    return a == b


def _agree(rows, twin=False):
    problems = []
    for name, data, code in rows:
        payload = dict(data=data, code=code, dtype="int32")
        want = _run(dict(payload, device="cpu"), upstream=True)
        got = _run(dict(payload, device="mps"), upstream=False)
        assert "error" not in want, (name, want)
        if "error" in got:
            problems.append("%s: shim raised %s" % (name, got["error"]))
            continue
        if len(got["parts"]) != len(want["parts"]):
            problems.append("%s: %d results, upstream %d"
                            % (name, len(got["parts"]), len(want["parts"])))
            continue
        for i, (g, w) in enumerate(zip(got["parts"], want["parts"])):
            if not g["device"].startswith("mps"):
                problems.append("%s: result %d on %s" % (name, i, g["device"]))
            if (g["shape"], g["dtype"]) != (w["shape"], w["dtype"]):
                problems.append("%s: result %d is %s %s, upstream %s %s"
                                % (name, i, g["shape"], g["dtype"], w["shape"], w["dtype"]))
            elif not all(_same(a, b) for a, b in zip(g["values"], w["values"])):
                problems.append("%s: result %d is %s, upstream %s"
                                % (name, i, g["values"], w["values"]))
        c = got["counters"]
        if c["compute_encoders"] + c["blit_encoders"] <= 0:
            problems.append("%s: no compute or blit encoder ran (%s)" % (name, c))
        allowed = 0
        if twin:
            wide = _run(dict(payload, dtype="int64", device="mps"), upstream=False)
            if "error" in wide:
                problems.append("%s: the int64 twin raised %s" % (name, wide["error"]))
                continue
            allowed = wide["counters"]["host_downloads"]
        if c["host_downloads"] != allowed:
            problems.append("%s: %d host downloads, the int64 path makes %d (%s)"
                            % (name, c["host_downloads"], allowed, c))
    assert not problems, "\n".join(problems)


def test_int32_buffer_ops_compute_on_mps():
    """Casts to and from int32, fill_/zero_/copy_ into int32 (views and
    broadcasts), clone and `.t().contiguous()`: upstream's values, on mps, no
    download. These are the new Metal kernels with no widening involved."""
    if not _mps_or_skip("int32 buffer ops on mps"):
        return
    _agree(_BUFFER_ROWS)


def test_int32_arithmetic_widens_on_the_device_and_agrees():
    """The claimed operators at the int32 boundary: wrapped where upstream
    wraps, int64 where upstream returns int64, float where upstream returns
    float, the receiver updated in place; on mps, no download."""
    if not _mps_or_skip("int32 arithmetic on mps"):
        return
    _agree(_ARITH_ROWS)


def test_int32_indexing_computes_on_mps():
    """embedding / index_select / index.Tensor / gather with int32 ids or an
    int32 `self`: upstream's values, on mps, and no more downloads than the
    same code at int64 (the bounds-check extremes)."""
    if not _mps_or_skip("int32 indexing on mps"):
        return
    _agree(_INDEX_ROWS, twin=True)


_REFUSED_ROWS = [
    ("matmul", _SMALL, "R = X[:, :2] @ X[:, :2].t()"),
    ("pow.Tensor_Scalar", _SMALL, "R = X.pow(2)"),
]


def test_unclaimed_int32_ops_refuse_by_name():
    """An operator `_shim_mps_int32_widened_ops` does not claim still refuses
    on an int32 mps tensor, naming the op, never in candle's words."""
    if not _mps_or_skip("int32 refusals on mps"):
        return
    problems = []
    for name, data, code in _REFUSED_ROWS:
        got = _run(dict(data=data, code=code, dtype="int32", device="mps"), upstream=False)
        if "error" not in got:
            problems.append("%s: answered on mps; if it is now claimed, grade it in "
                            "an agreement table above" % name)
            continue
        err = got["error"]
        if not err.startswith("NotImplementedError: aten."):
            problems.append("%s: not a refusal naming its op: %s" % (name, err[:300]))
        elif "candle:" in err:
            problems.append("%s: refusal still carries candle's words: %s" % (name, err[:300]))
        elif not ("not implemented for int32 tensors on the mps device" in err
                  or "not implemented for the mps device" in err):
            problems.append("%s: refusal gives no reason: %s" % (name, err[:300]))
    assert not problems, "\n".join(problems)


_CLAIMED = {
    "promote": {"aten.add.Tensor", "aten.add.Scalar", "aten.sub.Tensor", "aten.sub.Scalar",
                "aten.mul.Tensor", "aten.mul.Scalar", "aten.neg.default", "aten.abs.default",
                "aten.maximum.default", "aten.where.self", "aten.abs_.default",
                "aten.add_.Tensor",
                "aten.add_.Scalar", "aten.sub_.Tensor", "aten.mul_.Tensor", "aten.mul_.Scalar"},
    "self": {"aten.max.default", "aten.embedding.default", "aten.index_select.default",
             "aten.index.Tensor", "aten.gather.default"},
    "keep": {"aten.eq.Tensor", "aten.eq.Scalar", "aten.ne.Tensor", "aten.ne.Scalar",
             "aten.lt.Tensor", "aten.lt.Scalar", "aten.le.Tensor", "aten.le.Scalar",
             "aten.gt.Tensor", "aten.gt.Scalar", "aten.ge.Tensor", "aten.ge.Scalar",
             "aten.div.Tensor", "aten.div.Scalar", "aten.exp.default", "aten.sqrt.default",
             "aten.sum.default", "aten.sum.dim_IntList", "aten.cumsum.default",
             "aten.argmax.default"},
}


def test_the_artefact_claims_exactly_the_reviewed_operators():
    """The table in the artefact is the reviewed one: a new claim needs its
    widen-compute-narrow argument (and a row above), and `sum`/`cumsum`/
    `argmax` must never be in a narrowing mode. int16 is still sealed."""
    got = {}
    for op, mode in _C._shim_mps_int32_widened_ops():
        got.setdefault(mode, set()).add(op)
    assert got == _CLAIMED, (got, _CLAIMED)
    assert list(_C._shim_mps_unsupported_int_dtypes()) == ["int16"], \
        _C._shim_mps_unsupported_int_dtypes()


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsint32")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
