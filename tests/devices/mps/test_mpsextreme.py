"""Extrema, comparisons and random draws on mps (issue #49).

Found by #29's design round, on the build #29 landed:

* `aten.amax.default` on mps answered `-FLT_MAX` (f16 `-65504`, bf16
  `-3.39e38`) for an all-`-inf` input and dropped NaN (`amax([nan, 1])` was
  `1.0`). candle's Metal reduce is seeded with the finite lowest value and
  skips NaN. `max.default`, `min.default`, `max.dim` and `min.dim` had already
  moved to `first_extremum_index` + a gather in #29; `amax` had not, because
  it goes through `amax_keepdim_anywhere`, which on Metal was candle's
  `max_keepdim` alone.
* Every float comparison on mps (`eq`, `ne`, `lt`, ... in both the `Tensor`
  and `Scalar` overloads) and `to(torch.bool)` of a float mps tensor died with
  `Metal contiguous to_dtype F32 F64 not implemented`: `compare_common` and
  `to_copy_default` widened to `f64` first, and Metal has no `f64`. The issue
  named the empty tensor; the matrix (docs/devices/matrix.md §6) shows the
  non-empty float cells REFUSES too.
* `torch.randn(..., device="mps")` was reported failing in `aten.normal_` with
  `Metal contiguous to_dtype F64 F32`. Reading this tree, `normal_inplace`
  narrows its draws on the host (`host_vec`) before the upload, so the report
  may predate it; this file measures it either way.

Grades, in AGENTS.md §16's terms:

* *agrees* -- every value against upstream torch, in a separate subprocess
  that asserts it imported upstream. **The oracle is upstream on the CPU, not
  upstream on mps**, and the difference is not hypothetical: measured on torch
  2.13.0 on this Mac, upstream mps answers index `-1` for `max(dim=)` of an
  all-`-inf` row and the *last* NaN's index where upstream CPU answers `0` and
  the first NaN's (12 of the 72 rows below differ, all of them `max.dim` /
  `min.dim` indices; every value and every comparison agrees). The CPU answer
  is the documented one (the first maximal index) and is what `generate`
  parity in #29 was measured against. Compared exactly, NaN equal to NaN and
  the sign of a zero ignored (`amax`'s zero sign is a vectorisation artefact
  upstream).
* *ran on the GPU* -- `_metal_counters()` bracketing only the dispatch:
  compute encoders > 0 and zero host downloads. A host-computed twin can keep
  every value right; it cannot launch a compute encoder, and it has to read
  its operand back.
* the random draws are made on the host by this crate's RNG and uploaded
  (that is the design: the stream is upstream's CPU stream, docs/numerics/RNG.md),
  so their bracket asserts an upload and **no download**, and the values are
  checked against this shim's own CPU draws from the same seed, bit for bit,
  plus a distribution check. They are not upstream mps's draws (a different
  generator), and this file does not claim they are.

Nullifications this file is meant to catch:

* `amax_keepdim_anywhere` back to `t.max_keepdim(dim)` on Metal
      -> the all--inf and NaN rows of test_amax_agrees_on_mps
* the NaN half of it (`any_nan.where_cond`) dropped -> the NaN rows only
* `compare_common`'s device branch removed -> test_comparisons_agree_on_mps
  (every float row raises again)
* `compare_scalar` not narrowing the scalar -> the f16 `== 0.1` row
  (`host_const` refuses an f64 constant on Metal)
* `to_copy_default`'s device branch removed -> the `to(bool)` rows
* `normal_inplace` uploading f64 -> test_randn_and_normal_on_mps
"""

import json
import math
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip


_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")


def _shim_tree_or_skip(what):
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")):
        _skip.skip("   (skipped %s: no vendored shim at %s -- run scripts/vendor/install_shim.sh)"
                   % (what, _VENDOR_DIR))
        return False
    return True


def _mps_or_skip(what):
    if not _shim_tree_or_skip(what):
        return False
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return False
    return True


def _run(script, payload, upstream):
    """Run `script` in a child. upstream=True empties PYTHONPATH so the child
    can only import site-packages torch; otherwise the child imports the
    vendored shim tree. Either way the script asserts which torch it got."""
    env = dict(os.environ)
    env.pop("TORCH_C_ARTEFACT", None)
    if upstream:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    else:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(payload)],
                          capture_output=True, text=True, timeout=1800, env=env)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise AssertionError("child produced no JSON (rc=%d):\n%s"
                             % (proc.returncode, proc.stderr[-3000:]))
    return json.loads(lines[-1])


# One expression per child: a panic inside candle's Metal backend poisons the
# encoder's mutex for the rest of that process (test_mpsgen.py measured it), so
# one failure must not take the other rows with it.
_EXPR_SCRIPT = r"""
import json, sys
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch in this child", IS_SHIM)
D = A["device"]
inf, nan = float("inf"), float("nan")
DT = getattr(torch, A["dtype"])
def T(rows):
    return torch.tensor(rows, dtype=DT).to(D)
out = {}
try:
    X = T(A["data"])
    before = torch._C._metal_counters() if IS_SHIM and D == "mps" else None
    r = eval(A["expr"])
    after = torch._C._metal_counters() if before is not None else None
    parts = list(r) if isinstance(r, tuple) else [r]
    out = {"parts": [{"shape": list(p.shape), "dtype": str(p.dtype),
                      "device": str(p.device),
                      "values": p.cpu().double().flatten().tolist()} for p in parts]}
    if before is not None:
        out["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
except BaseException as exc:
    out = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
print(json.dumps(out))
"""


_INF, _NAN = float("inf"), float("nan")

# (name, data, expr). `X` is the input on the device.
_EXTREMA_ROWS = [
    ("amax all -inf, dim=1",
     [[-_INF, -_INF, -_INF], [1.0, -_INF, 2.0]], "torch.amax(X, dim=1)"),
    ("amax all -inf, whole tensor",
     [[-_INF, -_INF], [-_INF, -_INF]], "torch.amax(X)"),
    ("amax all +inf, dim=0",
     [[_INF, 1.0], [_INF, _INF]], "torch.amax(X, dim=0)"),
    ("amax NaN not first",
     [[1.0, _NAN, 3.0], [_NAN, 1.0, 2.0], [4.0, 5.0, _NAN]], "torch.amax(X, dim=1)"),
    ("amax NaN, keepdim, dim=0",
     [[1.0, 2.0], [_NAN, 0.5]], "torch.amax(X, dim=0, keepdim=True)"),
    ("amax over two dims with -inf and NaN",
     [[[-_INF, -_INF], [-_INF, -_INF]], [[1.0, _NAN], [2.0, 3.0]]], "torch.amax(X, dim=(1, 2))"),
    ("max.default all -inf", [[-_INF, -_INF, -_INF]], "X.max()"),
    ("min.default all +inf", [[_INF, _INF]], "X.min()"),
    ("max.default NaN last", [[1.0, 2.0, _NAN]], "X.max()"),
    ("min.default NaN middle", [[1.0, _NAN, -2.0]], "X.min()"),
    ("max.dim all -inf and ties",
     [[-_INF, -_INF, -_INF], [2.0, 5.0, 5.0], [-1.0, -1.0, -3.0]], "X.max(dim=1)"),
    ("min.dim all +inf and ties",
     [[_INF, _INF, _INF], [5.0, 2.0, 2.0], [-3.0, 0.0, -3.0]], "X.min(dim=1)"),
    ("max.dim NaN first index", [[1.0, _NAN, _NAN, 4.0]], "X.max(dim=1)"),
    ("min.dim NaN, keepdim, dim=0",
     [[1.0, 2.0], [_NAN, 0.0], [_NAN, -1.0]], "X.min(dim=0, keepdim=True)"),
]

_COMPARE_ROWS = [
    ("eq.Tensor", [[1.0, _NAN, -0.0, 3.0]], "X == T([[1.0, nan, 0.0, 4.0]])"),
    ("ne.Tensor", [[1.0, _NAN, 2.0, 3.0]], "X != X"),
    ("lt.Scalar", [[1.0, -_INF, 2.5, _NAN]], "X < 2.0"),
    ("ge.Tensor broadcast", [[1.0, 4.0], [3.0, 2.0]], "X >= X[0:1]"),
    ("eq.Scalar 0.1 is narrowed to the dtype", [[0.1, 0.2]], "X == 0.1"),
    ("eq.Scalar int scalar", [[1.0, 2.0]], "X == 1"),
    ("to(bool)", [[0.0, -0.0, _NAN, 2.0, -_INF]], "X.to(torch.bool)"),
    ("eq on an empty tensor", [[1.0]], "X[:0] == X[:0]"),
    ("eq.Scalar on an empty tensor", [[1.0]], "X[:0] == 1.0"),
    ("to(bool) on an empty tensor", [[1.0]], "X[:0].to(torch.bool)"),
]

_FLOAT_DTYPES = ["float32", "float16", "bfloat16"]


def _same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


def _agree(rows):
    problems = []
    for dtype in _FLOAT_DTYPES:
        for name, data, expr in rows:
            payload = dict(data=data, expr=expr, dtype=dtype)
            want = _run(_EXPR_SCRIPT, dict(payload, shim=False, device="cpu"), upstream=True)
            got = _run(_EXPR_SCRIPT, dict(payload, shim=True, device="mps"), upstream=False)
            label = "%s [%s] %s" % (name, dtype, expr)
            assert "error" not in want, (label, want)
            if "error" in got:
                problems.append("%s: shim raised %s" % (label, got["error"]))
                continue
            if len(got["parts"]) != len(want["parts"]):
                problems.append("%s: %d results, upstream %d"
                                % (label, len(got["parts"]), len(want["parts"])))
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
            empty = all(len(p["values"]) == 0 for p in got["parts"])
            if not empty and c["compute_encoders"] <= 0:
                problems.append("%s: no compute encoder ran (%s)" % (label, c))
            if c["host_downloads"] != 0:
                problems.append("%s: the operand was read back to the host (%s)" % (label, c))
    assert not problems, "\n".join(problems)


def test_amax_agrees_on_mps():
    """amax / max / min (with and without dim) on rows of all -inf, all +inf,
    NaN in every position and ties, in float32/float16/bfloat16: values and
    indices equal upstream's exactly (indices: the first extremum, or the
    first NaN), and each dispatch launched a compute encoder and downloaded
    nothing."""
    if not _mps_or_skip("extrema on mps"):
        return
    _agree(_EXTREMA_ROWS)


def test_comparisons_agree_on_mps():
    """Float eq/ne/lt/ge (Tensor and Scalar overloads) and to(torch.bool), on
    non-empty and empty mps tensors: the booleans equal upstream's, the result
    stays on mps, and nothing is read back."""
    if not _mps_or_skip("comparisons on mps"):
        return
    _agree(_COMPARE_ROWS)


_RANDN_SCRIPT = r"""
import json, sys
A = json.loads(sys.argv[1])
import torch
assert hasattr(torch._C, "_aten_implemented"), "this child must import the shim"
out = {}
try:
    DT = getattr(torch, A["dtype"])
    draws = {}
    for D in ("cpu", "mps"):
        torch.manual_seed(A["seed"])
        before = torch._C._metal_counters() if D == "mps" else None
        if A["how"] == "randn":
            r = torch.randn(A["n"], dtype=DT, device=D)
        else:
            r = torch.empty(A["n"], dtype=DT, device=D).normal_(2.0, 0.5)
        after = torch._C._metal_counters() if D == "mps" else None
        draws[D] = {"device": str(r.device), "values": r.cpu().double().tolist()}
        if before is not None:
            draws[D]["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
    out = draws
except BaseException as exc:
    out = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
print(json.dumps(out))
"""


def test_randn_and_normal_on_mps():
    """torch.randn and Tensor.normal_ on mps, at sizes on both sides of
    normal_'s 16-element path split: they succeed, the result is on mps, the
    draws equal this shim's CPU draws from the same seed bit for bit (one
    stream, narrowed on the host), the bracket uploaded and downloaded
    nothing, and the 4096-draw sample has the asked-for mean and spread
    (|mean - mu| < 5 sigma/sqrt(n), std within 10%)."""
    if not _mps_or_skip("randn on mps"):
        return
    problems = []
    for dtype in _FLOAT_DTYPES:
        for how in ("randn", "normal_"):
            for n in (15, 17, 4096):
                label = "%s(%d) %s" % (how, n, dtype)
                got = _run(_RANDN_SCRIPT, dict(dtype=dtype, how=how, n=n, seed=7),
                           upstream=False)
                if "error" in got:
                    problems.append("%s: %s" % (label, got["error"]))
                    continue
                mps, cpu = got["mps"], got["cpu"]
                if not mps["device"].startswith("mps"):
                    problems.append("%s: result on %s" % (label, mps["device"]))
                if mps["values"] != cpu["values"]:
                    problems.append("%s: mps draws differ from the CPU stream" % label)
                c = mps["counters"]
                if c["host_uploads"] <= 0 or c["host_downloads"] != 0:
                    problems.append("%s: expected an upload and no download, got %s" % (label, c))
                if n == 4096:
                    mu, sigma = (0.0, 1.0) if how == "randn" else (2.0, 0.5)
                    v = mps["values"]
                    mean = sum(v) / n
                    std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))
                    if abs(mean - mu) > 5 * sigma / math.sqrt(n) or abs(std / sigma - 1) > 0.1:
                        problems.append("%s: mean %.4f std %.4f, asked for %s, %s"
                                        % (label, mean, std, mu, sigma))
    assert not problems, "\n".join(problems)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsextreme")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
