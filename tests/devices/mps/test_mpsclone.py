"""`clone()` on mps is a copy, not a second name for the same buffer (issue #97).

candle's `MetalStorage::try_clone` returned `self.clone()`, which shares the
`Arc<Buffer>`. candle never writes a buffer in place, so that was safe for
candle. torch._C does (`add_`, `fill_`, setitem), so on mps

    x = torch.ones(3, device="mps")
    y = x.clone(); y.add_(1)      # x changed too, before the fix

The fix lives in the vendored fork (`int8-candle-0.11.0-cpu.patch`): a fresh
buffer of the same element count, filled by one blit.

What this file asserts, each case in a fresh subprocess that imports the
vendored tree and asserts `hasattr(torch._C, "_aten_implemented")`:

* *agrees*: for `clone()`, `clone(memory_format=torch.contiguous_format)`,
  `detach().clone()`, `x.t().clone()` and `x[:, 1:].clone()`, crossed with
  `add_(1)`, `fill_(7)`, `mul_(0)` and `y[0] = <mps tensor>`, in float32,
  float16, bfloat16 and int64: the source still holds its original values and
  the clone holds upstream's answer. Upstream's answer is computed here in
  plain Python; the inputs are small integers, exact in every one of the four
  dtypes, so no tolerance is involved.
* *placement*: `_C._metal_counters()` shows at least one blit or compute
  encoder opened during the `clone` call itself. A shared-buffer clone opens
  none.
* *negative control*: `x.detach().add_(1)` and `torch.ops.aten.alias(x).add_(1)`
  DO change `x`, because upstream defines both as views. A fix that made every
  aliasing op copy would pass the cases above and fail these.

What this file cannot find: a clone that copies through the host (download
then upload) would still open a blit for the readback and pass the counter
check; the counter bracket only proves the GPU was asked to do something.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

_DTYPES = ("float32", "float16", "bfloat16", "int64")
_SHAPE = (2, 3)
_VALUES = [[-3, -1, 0], [2, -7, 5]]
_ROW = 9  # the value setitem writes into row 0

_VARIANTS = ("clone", "clone_contiguous", "detach_clone", "t_clone", "slice_clone")
_OPS = ("add_", "fill_", "mul_", "setitem")

_CHILD = r"""
import json, os, sys
req = json.loads(sys.argv[1])
import torch
assert hasattr(torch._C, "_aten_implemented"), "the child imported upstream, not the shim"
out = {"no_mps": None, "results": {}}
try:
    torch._C._aten_dispatch("aten.ones.default", [1], device=torch._C.device("mps"))
except (NotImplementedError, RuntimeError) as e:
    out["no_mps"] = "no mps device -- %s" % str(e).splitlines()[0]
    print(json.dumps(out)); raise SystemExit(0)

def counters():
    c = torch._C._metal_counters()
    return c["blit_encoders"] + c["compute_encoders"]

def take(x, variant):
    if variant == "clone":
        return x.clone()
    if variant == "clone_contiguous":
        return x.clone(memory_format=torch.contiguous_format)
    if variant == "detach_clone":
        return x.detach().clone()
    if variant == "t_clone":
        return x.t().clone()
    if variant == "slice_clone":
        return x[:, 1:].clone()
    if variant == "detach_view":
        return x.detach()
    if variant == "alias_view":
        return torch.ops.aten.alias(x)
    raise ValueError(variant)

def mutate(y, op, dt):
    if op == "add_":
        y.add_(1)
    elif op == "fill_":
        y.fill_(7)
    elif op == "mul_":
        y.mul_(0)
    elif op == "setitem":
        y[0] = torch.full((y.shape[1],), req["row"], dtype=dt, device="mps")
    else:
        raise ValueError(op)

for name in req["dtypes"]:
    dt = getattr(torch, name)
    rec = {}
    try:
        x = torch.tensor(req["values"], dtype=dt).to("mps")
        before = counters()
        y = take(x, req["variant"])
        rec["encoders"] = counters() - before
        mutate(y, req["op"], dt)
        rec["device"] = str(y.device)
        rec["x"] = [[float(v) for v in r] for r in x.cpu().tolist()]
        rec["y"] = [[float(v) for v in r] for r in y.cpu().tolist()]
    except Exception as e:
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    out["results"][name] = rec
print(json.dumps(out))
"""


def _run(variant, op):
    if not os.path.isfile(_VENDOR_SHIM):
        raise _skip.Skip("vendored tree not built: no %s "
                         "(scripts/vendor/vendor_torch.sh + scripts/vendor/install_shim.sh)"
                         % _VENDOR_SHIM)
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    req = {"variant": variant, "op": op, "dtypes": list(_DTYPES),
           "values": _VALUES, "row": _ROW}
    proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(req)],
                          capture_output=True, text=True, timeout=600, env=env)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert proc.returncode == 0 and lines, (
        "the mps child (%s, %s) exited %d:\n%s" % (variant, op, proc.returncode, proc.stderr[-2000:]))
    out = json.loads(lines[-1])
    if out["no_mps"]:
        raise _skip.Skip(out["no_mps"])
    return out["results"]


def _view(variant):
    """Upstream's value of the cloned view, in plain Python."""
    v = [list(r) for r in _VALUES]
    if variant == "t_clone":
        return [list(c) for c in zip(*v)]
    if variant == "slice_clone":
        return [r[1:] for r in v]
    return v


def _apply(m, op):
    if op == "add_":
        return [[e + 1 for e in r] for r in m]
    if op == "fill_":
        return [[7 for _ in r] for r in m]
    if op == "mul_":
        return [[0 for _ in r] for r in m]
    if op == "setitem":
        return [[_ROW for _ in m[0]]] + [list(r) for r in m[1:]]
    raise ValueError(op)


def _f(m):
    return [[float(e) for e in r] for r in m]


def _check_clone(variant, op):
    results = _run(variant, op)
    expected_y = _f(_apply(_view(variant), op))
    expected_x = _f(_VALUES)
    for name in _DTYPES:
        rec = results[name]
        tag = "%s().%s on mps, %s" % (variant, op, name)
        # A strided clone (`x.t().clone()`, `x[:, 1:].clone()`) keeps the view's
        # strides, and an in-place write through a strided receiver on mps is
        # issue #32's, landing with PR #89. Until then that refusal is named
        # and skipped, so this suite still runs and passes the moment the
        # writes land; any other refusal stays a failure.
        if ("error" in rec and variant in ("t_clone", "slice_clone")
                and "writing through a non-contiguous view" in rec["error"]):
            raise _skip.Skip("%s: in-place writes through a strided mps receiver are "
                             "refused on this tree (issue #32, PR #89)" % tag)
        assert "error" not in rec, "%s refused: %s" % (tag, rec["error"])
        assert rec["device"].startswith("mps"), (tag, rec["device"])
        assert rec["x"] == expected_x, (
            "%s changed the source (the clone shares its buffer): x=%r, expected %r"
            % (tag, rec["x"], expected_x))
        assert rec["y"] == expected_y, (
            "%s: clone holds %r, upstream says %r" % (tag, rec["y"], expected_y))
        assert rec["encoders"] >= 1, (
            "%s: clone opened %d blit/compute encoders; a copy opens at least one"
            % (tag, rec["encoders"]))
    return len(_DTYPES)


def _make(variant, op):
    def test():
        assert _check_clone(variant, op) == len(_DTYPES)
    test.__name__ = "test_%s_then_%s_leaves_the_source_alone_on_mps" % (variant, op.rstrip("_"))
    return test


for _v in _VARIANTS:
    for _o in _OPS:
        _t = _make(_v, _o)
        globals()[_t.__name__] = _t
del _v, _o, _t


def _check_view_writes_through(variant):
    results = _run(variant, "add_")
    expected = _f(_apply(_VALUES, "add_"))
    for name in _DTYPES:
        rec = results[name]
        tag = "%s.add_(1) on mps, %s" % (variant, name)
        assert "error" not in rec, "%s refused: %s" % (tag, rec["error"])
        assert rec["x"] == expected, (
            "%s did not change the source, but upstream defines it as a view: "
            "x=%r, expected %r" % (tag, rec["x"], expected))
        assert rec["y"] == expected, (tag, rec["y"], expected)


def test_detach_is_still_a_view_on_mps():
    _check_view_writes_through("detach_view")


def test_aten_alias_is_still_a_view_on_mps():
    _check_view_writes_through("alias_view")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsclone")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
