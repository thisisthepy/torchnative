"""`triu_` / `tril_` and `nansum` / `nanmean`, against upstream, on cpu and mps.

Gemstone (LogitAI/Gemstone #177, run 39023653125 on 0.1.0b6) found two of
these refused by name on every build:

* `TensorBase.triu_`: transformers' continuous batching builds its attention
  mask with it, so every generation on that path stopped (16 tests).
* `TensorBase.nansum`: `typical_p` sampling.

`tril_` and `nanmean` are the same gap's siblings and land with them.

Every comparison runs the shim and upstream (the interpreter's own torch) in
separate subprocesses. The inputs are small integers and NaN/inf, exact in every
dtype here, and every reduction is over at most a few dozen of them, so the
float sums are exact on both sides and the comparison is equality: no tolerance
is involved. What is compared:

* *agrees*: values, dtype and shape for float32, float64, float16, bfloat16,
  int32, int64, uint8 and bool where upstream accepts the dtype, over several
  `dim` / `keepdim` / `dtype=` / `diagonal` choices; refusals carry upstream's
  type and wording.
* in-place: `triu_` returns its receiver, and on a view (`x.t()`, `x[:, 1:]`)
  writes through to the base, as upstream does.
* mps: the same cases on `mps` against upstream's own mps answers, and
  `_metal_counters()` shows no host download during the op (the sub-ops run
  on the GPU, not a host twin under an mps label, AGENTS.md §13.1).

What this cannot find: summation order on long float reductions. With exact
inputs both orders give the same sum; a reordering that rounds differently on
large random inputs would pass here.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())), "tests", "_support"))
import _skip  # noqa: E402

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.environ.get(
    "TORCHNATIVE_VENDOR_DIR", os.path.join(_REPO_ROOT, "torchnative", "python")
)


def _run_side(script: str, shim: bool, timeout: int = 600) -> dict:
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{'shim' if shim else 'upstream'} side failed "
                           f"(rc={proc.returncode}):\n{proc.stderr[-3000:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out.pop("is_shim") is shim, "the side imported the wrong torch"
    return out


def _upstream_available() -> bool:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", "import torch"],
                          capture_output=True, text=True, env=env, timeout=300)
    return proc.returncode == 0


def _both(script: str) -> tuple:
    if not _upstream_available():
        raise _skip.Skip("upstream torch is not importable by this interpreter: no oracle")
    return _run_side(script, shim=True), _run_side(script, shim=False)


_HEAD = r"""
import json, math, torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
out = {"is_shim": IS_SHIM}
DEV = "%s"
if DEV == "mps" and not torch.backends.mps.is_available():
    out["no_mps"] = True
    print(json.dumps(out)); raise SystemExit(0)

def enc(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else ("inf" if v == math.inf else ("-inf" if v == -math.inf else v))
    if isinstance(v, list):
        return [enc(e) for e in v]
    return v

def show(t):
    return [str(t.dtype), list(t.shape), str(t.device.type), enc(t.cpu().tolist())]

LAST_DOWNLOADS = [0]

def run(f):
    # The download bracket closes before `show`, whose `.cpu()` is a download
    # by definition; inside it are the input's upload and the op.
    before = counters()
    try:
        t = f()
    except Exception as e:
        LAST_DOWNLOADS[0] = counters() - before
        return [type(e).__name__, str(e).splitlines()[0]]
    LAST_DOWNLOADS[0] = counters() - before
    return show(t)

def counters():
    if IS_SHIM and DEV == "mps":
        return torch._C._metal_counters()["host_downloads"]
    return 0

def src(values, dt):
    return torch.tensor(values, dtype=torch.float64).to(dt).to(DEV)
"""
_TAIL = "\nprint(json.dumps(out))\n"

_TRI = r"""
FLOATS = ["float32", "float16", "bfloat16"] + ([] if DEV == "mps" else ["float64"])
DTYPES = FLOATS + ["int32", "int64", "uint8", "bool"]
GRID = [[float(3 * i + j + 1) for j in range(4)] for i in range(3)]
# Non-negative: a negative float cast to uint8 is undefined behaviour in C++,
# and upstream and the shim disagree about it for a reason that is not this op.
BATCH = [[[float(12 * b + 4 * i + j + 1) for j in range(4)] for i in range(3)] for b in range(2)]
res, downloads = {}, {}
for name in DTYPES:
    dt = getattr(torch, name)
    for diag in (-2, -1, 0, 1, 3):
        for op in ("triu_", "tril_"):
            def go(op=op, diag=diag, dt=dt):
                x = src(GRID, dt)
                y = getattr(x, op)(diag)
                assert y is x, "the in-place op did not return its receiver"
                return x
            res[f"{op} {name} {diag}"] = run(go)
            downloads[f"{op} {name} {diag}"] = LAST_DOWNLOADS[0]
    for op in ("triu_", "tril_"):
        res[f"{op} {name} batch"] = run(lambda op=op, dt=dt: getattr(src(BATCH, dt), op)(1))
        res[f"{op} {name} diagonal kw"] = run(lambda op=op, dt=dt: getattr(src(GRID, dt), op)(diagonal=-1))
        # Through a view: the base changes, as upstream's does.
        def through_t(op=op, dt=dt):
            base = src(GRID, dt).t().contiguous()
            getattr(base.t(), op)(0)
            return base
        def through_slice(op=op, dt=dt):
            base = src(GRID, dt)
            getattr(base[:, 1:], op)(0)
            return base
        res[f"{op} {name} through t()"] = run(through_t)
        res[f"{op} {name} through slice"] = run(through_slice)
res["triu_ 1-d"] = run(lambda: src([1.0, 2.0], torch.float32).triu_())
res["tril_ 0-d"] = run(lambda: src(1.0, torch.float32).tril_())
res["triu_ empty"] = run(lambda: src([[]], torch.float32).triu_())
out["res"] = res
out["downloads"] = downloads
"""

_NAN = r"""
FLOATS = ["float32", "float16", "bfloat16"] + ([] if DEV == "mps" else ["float64"])
INTS = ["int32", "int64", "uint8", "bool"]
nan, inf = float("nan"), float("inf")
# Every row, column and the whole has a power-of-two count of non-NaN values
# (rows 2, 0, 4, 2; columns 2 each; all 8), so `nanmean`'s division is exact
# and this compares the composition, not `div`'s rounding. Metal's
# `div.Tensor` itself is 1 ulp off upstream on 7/3 (issue #113); that is
# measured on its own and does not belong to these ops.
DATA = [[1.0, nan, 2.0, nan], [nan, nan, nan, nan], [3.0, -1.0, 5.0, 0.0], [nan, 6.0, nan, -2.0]]
WITH_INF = [[1.0, inf, nan], [-inf, nan, 2.0]]
INT_DATA = [[1.0, 0.0, 2.0], [3.0, 1.0, 0.0]]
res, downloads = {}, {}
CHOICES = [
    ("all", (), {}),
    ("dim0", (0,), {}),
    ("dim1", (1,), {}),
    ("dim -1 keep", (-1,), {"keepdim": True}),
    ("dims01", ((0, 1),), {}),
    ("dim list keep", ([1],), {"keepdim": True}),
    ("kw dim", (), {"dim": 1}),
]
for name in FLOATS:
    dt = getattr(torch, name)
    for label, args, kw in CHOICES:
        for op in ("nansum", "nanmean"):
            key = f"{op} {name} {label}"
            res[key] = run(lambda op=op, args=args, kw=kw, dt=dt: getattr(src(DATA, dt), op)(*args, **kw))
            downloads[key] = LAST_DOWNLOADS[0]
            res[key + " fn"] = run(lambda op=op, args=args, kw=kw, dt=dt: getattr(torch, op)(src(DATA, dt), *args, **kw))
    res[f"nansum {name} inf"] = run(lambda dt=dt: src(WITH_INF, dt).nansum(1))
    res[f"nanmean {name} inf"] = run(lambda dt=dt: src(WITH_INF, dt).nanmean(0))
    res[f"nansum {name} dtype f32"] = run(lambda dt=dt: src(DATA, dt).nansum(dtype=torch.float32))
    res[f"nanmean {name} dtype f32"] = run(lambda dt=dt: src(DATA, dt).nanmean(1, dtype=torch.float32))
    res[f"nansum {name} empty"] = run(lambda dt=dt: torch.empty(0, 3, dtype=dt, device=DEV).nansum(0))
    res[f"nanmean {name} all nan"] = run(lambda dt=dt: src([nan, nan], dt).nanmean())
for name in INTS:
    dt = getattr(torch, name)
    for label, args, kw in CHOICES:
        key = f"nansum {name} {label}"
        res[key] = run(lambda args=args, kw=kw, dt=dt: src(INT_DATA, dt).nansum(*args, **kw))
        downloads[key] = LAST_DOWNLOADS[0]
    res[f"nanmean {name}"] = run(lambda dt=dt: src(INT_DATA, dt).nanmean())
    res[f"nansum {name} dtype f32"] = run(lambda dt=dt: src(INT_DATA, dt).nansum(dtype=torch.float32))
out["res"] = res
out["downloads"] = downloads
"""


def _compare(script_body: str, dev: str):
    script = (_HEAD % dev) + script_body + _TAIL
    shim, upstream = _both(script)
    if shim.get("no_mps") or upstream.get("no_mps"):
        raise _skip.Skip(f"no {dev} device on this machine")
    assert set(shim["res"]) == set(upstream["res"]), set(shim["res"]) ^ set(upstream["res"])
    wrong = {k: (v, upstream["res"][k]) for k, v in shim["res"].items() if v != upstream["res"][k]}
    assert not wrong, f"{len(wrong)} of {len(shim['res'])} cases differ from upstream on {dev}:\n" + "\n".join(
        f"  {k}:\n    shim     {s}\n    upstream {u}" for k, (s, u) in sorted(wrong.items())[:20])
    downloaded = {k: n for k, n in shim["downloads"].items() if n}
    assert not downloaded, f"{dev}: these cases downloaded to the host during the op: {downloaded}"
    return len(shim["res"])


def test_triu_and_tril_in_place_agree_with_upstream_on_cpu():
    assert _compare(_TRI, "cpu") > 0


def test_triu_and_tril_in_place_agree_with_upstream_on_mps():
    assert _compare(_TRI, "mps") > 0


def test_nansum_and_nanmean_agree_with_upstream_on_cpu():
    assert _compare(_NAN, "cpu") > 0


def test_nansum_and_nanmean_agree_with_upstream_on_mps():
    assert _compare(_NAN, "mps") > 0


def test_nansum_and_triu_on_meta_give_upstreams_shape_and_dtype():
    script = (_HEAD % "meta") + r"""
res = {}
x = torch.empty(2, 3, 4, device="meta")
res["nansum"] = [str(x.nansum(1).dtype), list(x.nansum(1).shape)]
res["nansum keep"] = list(x.nansum((0, 2), keepdim=True).shape)
res["nansum int"] = str(torch.empty(3, dtype=torch.int32, device="meta").nansum().dtype)
res["nanmean"] = [str(x.nanmean(-1).dtype), list(x.nanmean(-1).shape)]
y = torch.empty(3, 3, device="meta")
res["triu_"] = [y.triu_(1) is y, list(y.shape), str(y.device)]
out["res"] = res
""" + _TAIL
    shim, upstream = _both(script)
    assert shim["res"] == upstream["res"], (shim["res"], upstream["res"])


def _main():
    failures = _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")],
        suite="test_nanreduce_tri",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
