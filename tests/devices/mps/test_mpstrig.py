"""Transcendentals on mps (issue #101): sin cos exp log tanh sigmoid erf.

candle compiles its Metal kernels with fast math (`CANDLE_METAL_ENABLE_FAST_MATH`,
default on). Which oracle does "agrees" mean on mps? Upstream torch has an mps
backend that also runs Metal kernels, so the shim is graded against upstream
mps *and* upstream cpu, both against upstream's own float64 result as truth.

Tolerance (AGENTS.md section 16, derived, not chosen): per dtype, the p90 over
all cells of upstream-cpu's error against its float64 result (in ulps of the
output dtype), floored at 8 ulp. An element is graded in ulps of the float64
truth rounded into the dtype; non-finite truths must match by class.

A cell is "agrees" when the shim's worst ulp error against the float64 truth is
within that tolerance. The cells that diverge are not hidden: DIVERGES names
each one, and test_known_divergences_still_diverge asserts the measured
divergence is still there (and beyond the tolerance), so a kernel fix turns the
entry visible and forces the table, and docs/devices/MPSTRIG.md, to be updated.

Nullifications this file is meant to catch:

* a transcendental kernel replaced by a different function (e.g. `sin` -> `cos`)
  -> every agreeing cell of that op
* an op falling back to a host twin changes nothing here (values only);
  that is test_metalcount.py's instrument
* fast math made less accurate -> the agreeing cells near their ulp bound
* a divergent cell starting to agree -> test_known_divergences_still_diverge
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

OPS = ["sin", "cos", "exp", "log", "tanh", "sigmoid", "erf"]
DTYPES = ["float32", "float16", "bfloat16"]
MANT = {"float32": 24, "float16": 11, "bfloat16": 8}      # significand bits
EMIN = {"float32": -126, "float16": -14, "bfloat16": -126}
MAXV = {"float32": 3.4028234663852886e38, "float16": 65504.0, "bfloat16": 3.3895313892515355e38}

# (range name, lo, hi, shape). Ops skip ranges outside their meaning.
RANGES = [
    ("unit", -1.0, 1.0, (3,)),
    ("unit", -1.0, 1.0, (64, 64)),
    ("mid", -10.0, 10.0, (37, 129)),
    ("wide", -80.0, 80.0, (4096,)),
    ("large", -1e4, 1e4, (64, 64)),
    ("pos", 1e-3, 1e3, (64, 64)),
    ("tiny", 1e-6, 1e-3, (64, 64)),
    ("neg", -50.0, 0.0, (4096,)),
]
_LOG_RANGES = {"unit", "pos", "tiny", "neg"}      # neg: log of <= 0 -> nan / -inf
_NO_LARGE = {"exp", "log", "tanh", "sigmoid", "erf"}   # "large" is for sin/cos only
_NO_POS_NEG_FOR = {"sin", "cos", "tanh", "sigmoid", "erf", "exp"}


def _cells():
    out = []
    for op in OPS:
        for rname, lo, hi, shape in RANGES:
            if rname == "large" and op in _NO_LARGE:
                continue
            if rname in ("pos", "tiny", "neg") and op != "log":
                continue
            if op == "log" and rname in ("mid", "wide", "large"):
                continue
            if op == "exp" and rname == "unit" and shape == (3,):
                pass
            out.append((op, rname, lo, hi, shape))
    return out


# One child per (torch, device, fast math). Inputs are generated from a seeded
# generator in float64 and rounded into the dtype, identically in every child.
_SCRIPT = r"""
import json, random, sys
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch in this child", IS_SHIM)
res = {}
for dt in A["dtypes"]:
    DT = getattr(torch, dt)
    for (op, rname, lo, hi, shape) in A["cells"]:
        n = 1
        for s in shape: n *= s
        rng = random.Random("%s/%s/%s" % (op, rname, shape))
        vals = [rng.uniform(lo, hi) for _ in range(n)]
        if rname == "neg":
            vals[0] = 0.0
        x64 = torch.tensor(vals, dtype=torch.float64).reshape(shape)
        xd = x64.to(DT)                       # the dtype-rounded input
        key = "%s|%s|%s|%s" % (dt, op, rname, "x".join(map(str, shape)))
        entry = {"x": xd.double().flatten().tolist()}
        if A["truth"]:
            entry["truth"] = getattr(torch, op)(xd.double()).flatten().tolist()
        try:
            y = getattr(torch, op)(xd.to(A["device"])).cpu()
            assert y.dtype == DT, (y.dtype, DT)
            assert list(y.shape) == list(shape)
            entry["y"] = y.double().flatten().tolist()
        except Exception as e:
            entry["error"] = str(e).splitlines()[0][:200]
        res[key] = entry
print(json.dumps(res))
"""


def _run(device, upstream, fast=True, truth=False, cells=None, dtypes=None):
    env = dict(os.environ)
    env.pop("TORCH_C_ARTEFACT", None)
    env.pop("CANDLE_METAL_ENABLE_FAST_MATH", None)
    if upstream:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    else:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
        if not fast:
            env["CANDLE_METAL_ENABLE_FAST_MATH"] = "0"
    payload = {"device": device, "shim": not upstream, "truth": truth,
               "cells": cells or _cells(), "dtypes": dtypes or DTYPES}
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, json.dumps(payload)],
                          capture_output=True, text=True, timeout=1800, env=env,
                          cwd=os.path.join(_REPO_ROOT, ".scratch") if os.path.isdir(
                              os.path.join(_REPO_ROOT, ".scratch")) else None)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise AssertionError("child produced no JSON (rc=%d):\n%s"
                             % (proc.returncode, proc.stderr[-3000:]))
    return json.loads(lines[-1])


def _ulp(t, dt):
    """Spacing of dtype at the float64 value t (finite)."""
    a = abs(t)
    e = EMIN[dt] if a == 0 or math.frexp(a)[1] - 1 < EMIN[dt] else math.frexp(a)[1] - 1
    return 2.0 ** (e - (MANT[dt] - 1))


def _err_ulp(y, t, dt):
    """Error of y against truth t in ulps of dt; inf when classes differ."""
    if math.isnan(t):
        return 0.0 if math.isnan(y) else math.inf
    if math.isnan(y):
        return math.inf
    if abs(t) >= MAXV[dt] * (1 + 2.0 ** -MANT[dt]):      # overflows the dtype
        return 0.0 if (math.isinf(y) and (y > 0) == (t > 0)) else (
            0.0 if abs(y) >= MAXV[dt] and (y > 0) == (t > 0) else math.inf)
    if math.isinf(t):
        return 0.0 if y == t else math.inf
    if math.isinf(y):
        return math.inf
    return abs(y - t) / _ulp(t, dt)


def _cell_stats(entry, key):
    dt = key.split("|")[0]
    errs = [_err_ulp(y, t, dt) for y, t in zip(entry["y"], entry["truth"])]
    fin = [(abs(y - t), e) for y, t, e in zip(entry["y"], entry["truth"], errs)
           if math.isfinite(y) and math.isfinite(t)]
    return {"max_ulp": max(errs), "max_abs": max((a for a, _ in fin), default=0.0),
            "errs": errs}


def _p90(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(math.ceil(0.9 * len(v))) - 1)]


def measure(fast=True):
    """Grade every cell. Returns (tolerance per dtype, rows)."""
    mps_ok = True
    up_cpu = _run("cpu", True, truth=True)
    up_mps = _run("mps", True)
    shim = _run("mps", False, fast=fast)
    tol, p90 = {}, {}
    for dt in DTYPES:
        # the oracle error: upstream cpu in dt against upstream's own float64
        pool = []
        for key, e in up_cpu.items():
            if key.startswith(dt + "|"):
                pool += [x for x in _cell_stats(e, key)["errs"] if math.isfinite(x)]
        p90[dt] = _p90(pool)
        tol[dt] = max(8.0, p90[dt])
    rows = []
    for key, c in up_cpu.items():
        assert up_mps[key]["x"] == c["x"] == shim[key]["x"], key   # same inputs
        row = {"key": key, "tol": tol[key.split("|")[0]]}
        for name, src in (("cpu", up_cpu), ("upmps", up_mps), ("shim", shim)):
            e = src[key]
            if "error" in e:
                row[name] = {"error": e["error"]}
                continue
            s = _cell_stats(dict(e, truth=c["truth"]), key)
            row[name] = {"max_ulp": s["max_ulp"], "max_abs": s["max_abs"]}
        rows.append(row)
    return tol, p90, rows


def _grade(row):
    s = row["shim"]
    if "error" in s:
        return "refuses"
    return "agrees" if s["max_ulp"] <= row["tol"] else "diverges"


# Cells measured diverging with fast math ON on 2026-10-25 (the default build):
# name -> a floor under the measured max ulp (about half of it). The numbers are
# in docs/devices/MPSTRIG.md.
DIVERGES = {
    "float32|sin|unit|3": 10.0,
    "float32|sin|unit|64x64": 740.0,
    "float32|sin|mid|37x129": 4600.0,
    "float32|sin|wide|4096": 16000.0,
    "float32|sin|large|64x64": 800000.0,
    "float32|cos|mid|37x129": 19000.0,
    "float32|cos|wide|4096": 5800.0,
    "float32|cos|large|64x64": 6400000.0,
    "float32|exp|wide|4096": 29.0,
    "float32|log|unit|64x64": 4100.0,
    "float32|sigmoid|wide|4096": 29.0,
    "float32|erf|unit|64x64": 2000.0,
    "float32|erf|mid|37x129": 300.0,
    "float32|erf|wide|4096": 76.0,
    "float16|sin|unit|3": 15.0,
    "float16|sin|unit|64x64": 1000.0,
    "float16|sin|mid|37x129": 1000.0,
    "float16|sin|wide|4096": 1000.0,
    "float16|sin|large|64x64": 2400.0,
    "float16|cos|mid|37x129": 780.0,
    "float16|cos|wide|4096": 1000.0,
    "float16|cos|large|64x64": 2300.0,
}

# With CANDLE_METAL_ENABLE_FAST_MATH=0 (Relaxed, precise functions) the float32
# sin/cos/exp/log/sigmoid cells agree; only these still diverge: float32 erf
# (candle's own erf polynomial, not a fast-math effect) and float16 sin/cos
# (their error does not change with the switch).
OFF_DIVERGES = {
    "float32|erf|unit|64x64": 1000.0,
    "float32|erf|mid|37x129": 370.0,
    "float32|erf|wide|4096": 57.0,
    **{k: v for k, v in DIVERGES.items() if k.startswith("float16|")},
}


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


_CACHE = {}


def _measured():
    if "m" not in _CACHE:
        _CACHE["m"] = measure(fast=True)
    return _CACHE["m"]


def test_every_cell_is_graded_and_tolerance_is_derived():
    if not _mps_or_skip("mps transcendental grading"):
        return
    tol, p90, rows = _measured()
    assert len(rows) == 93, len(rows)
    for dt in DTYPES:
        assert tol[dt] >= 8.0 and tol[dt] == max(8.0, p90[dt]), (dt, tol, p90)
    for r in rows:
        assert "error" not in r["cpu"], r         # the oracle itself must answer


def test_agreeing_cells_agree_with_the_float64_truth():
    if not _mps_or_skip("mps transcendental agreement"):
        return
    _, _, rows = _measured()
    bad = [(r["key"], r["shim"]) for r in rows
           if _grade(r) != "agrees" and r["key"] not in DIVERGES]
    assert not bad, bad


def test_known_divergences_still_diverge():
    if not _mps_or_skip("mps transcendental divergences"):
        return
    _, _, rows = _measured()
    by = {r["key"]: r for r in rows}
    for key, floor in DIVERGES.items():
        r = by[key]
        assert _grade(r) == "diverges", ("now agrees or refuses; update MPSTRIG.md", key, r)
        assert r["shim"]["max_ulp"] >= floor, (key, r["shim"], floor)


def test_fast_math_off_fixes_exactly_the_float32_fast_math_cells():
    """The switch fixes the cells it can fix and no others: the divergence set
    with CANDLE_METAL_ENABLE_FAST_MATH=0 is exactly OFF_DIVERGES."""
    if not _mps_or_skip("mps transcendental fast-math-off"):
        return
    _, _, rows = measure(fast=False)
    diverging = {r["key"] for r in rows if _grade(r) != "agrees"}
    assert diverging == set(OFF_DIVERGES), (sorted(diverging ^ set(OFF_DIVERGES)))
    by = {r["key"]: r for r in rows}
    for key, floor in OFF_DIVERGES.items():
        assert by[key]["shim"]["max_ulp"] >= floor, (key, by[key]["shim"], floor)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpstrig")
    return 1 if failures else 0


def _report(fast):
    tol, p90, rows = measure(fast=fast)
    print(json.dumps({"tol": tol, "p90": p90, "fast": fast}))
    for r in rows:
        def f(d):
            return d["error"] if "error" in d else "%.3g/%.3g" % (d["max_ulp"], d["max_abs"])
        print("ROW|%s|cpu=%s|upmps=%s|shim=%s|%s" % (
            r["key"], f(r["cpu"]), f(r["upmps"]), f(r["shim"]), _grade(r)))


if __name__ == "__main__":
    if os.environ.get("TRIG_REPORT"):
        _report(os.environ["TRIG_REPORT"] == "fast")
        raise SystemExit(0)
    raise SystemExit(_main())
