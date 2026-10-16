"""`w[...] = python_scalar` on mps (issue #99).

The Python scalar on the right of a setitem is materialised as a zero-dim CPU
tensor (`bootstrap.py::_lift`, upstream's `lift_fresh`). Two paths died on it:

1. **`copy_`** (`w[0] = 9` on a 1-D tensor: `select.int` to a 0-d view, then
   `copy_`, upstream's `copy_to` "sizes equal" arm) reached candle's
   `slice_set` with a Metal destination and a CPU source and raised candle's
   raw `device mismatch in slice-set`. `copy_` is the op whose definition is
   a transfer, and upstream's `mps.copy_(cpu)` copies for any shape, so
   `copy_inplace` now moves the source to the receiver's device first (an
   upload, never a host computation).
2. **`index_put_`** (`w[w > 1] = 0`) failed the dispatcher's device check.
   The zero-dim CPU value is now accepted there, as `fill_.Tensor`'s is
   (#32), so a non-zero-dim CPU value still gets upstream's own
   `Expected all tensors to be on the same device` message. `index_put_`
   itself reads back to the host and is refused on mps by the readback gate,
   so a **bool mask the shape of the receiver with a scalar or zero-dim
   value** is lowered by `__setitem__` to `masked_fill_.Scalar` (Python or
   CPU scalar) or `where.self` (zero-dim value on the receiver's device),
   which compute on Metal. An **integer index tensor** with a scalar still
   reaches `index_put_` and refuses by name: an on-device lowering would have
   to bounds-check the index on the device, which this build does not do.

Grades (AGENTS.md §16): every computed row *agrees* with upstream torch on
mps, run in a separate subprocess that asserts it imported upstream; upstream
on the CPU is recorded too and must agree with upstream mps (a row where they
disagree is reported, not graded). Every computed row is *on the GPU* by
`_metal_counters()` bracketing only the assignment: at least one compute or
blit encoder and zero host downloads. The scalar's one-element upload is
allowed and is not a computation.

Nullifications this file is meant to catch:

* `copy_inplace`'s `to_device` of the source removed -> every `w[0] = ...`
  and `copy_ from cpu` row (candle's slice-set error)
* the `index_put_` zero-dim CPU exception in `visit_for_device` removed ->
  the int-index refusal row (raises the device-mismatch RuntimeError instead
  of the named NotImplementedError)
* `__setitem__`'s mask lowering removed -> every `w[w > 1] = ...` row
  (refused by the readback gate)
* the narrowing in the mask lowering removed (`9.5` passed through to an
  int64 receiver) -> the int64 `= 9.5` mask row's value, if `masked_fill_`
  does not truncate itself
* the where.self dtype refusal removed (cast instead) -> the zero-dim mps
  float32 rows of test_non_scalar_cpu_value_refuses_like_upstream
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
# One assignment block per child (a candle Metal panic poisons the
# encoder's mutex for the rest of a process, test_mpsgen.py). `W` is the
# receiver on the device, built before the bracket; `M` a mask and `I` an
# int64 index on the same device, also built before it. Only the block is
# bracketed, and the answer is `W` after it.
_SCRIPT = r"""
import json, sys
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch in this child", IS_SHIM)
D = A["device"]
DT = getattr(torch, A["dtype"])
out = {}
try:
    W = torch.tensor(A["data"], dtype=DT).to(D)
    M = W > 1
    I = torch.tensor([1, 3], dtype=torch.int64).to(D)
    Z = torch.tensor(7.0).to(D)
    CPU = torch.ones(W.shape)
    ns = dict(torch=torch, W=W, M=M, I=I, Z=Z, CPU=CPU, D=D, DT=DT)
    before = torch._C._metal_counters() if IS_SHIM and D == "mps" else None
    exec(A["code"], ns)
    after = torch._C._metal_counters() if before is not None else None
    out = {"shape": list(W.shape), "dtype": str(W.dtype), "device": str(W.device),
           "values": W.cpu().double().flatten().tolist()}
    if before is not None:
        out["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
except BaseException as exc:
    out = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:600])}
print(json.dumps(out))
"""

_ALL = ["float32", "float16", "bfloat16", "int64"]
_VEC = [0.0, 1.0, 2.0, 3.0]
_MAT = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

_COMPUTED_ROWS = [
    ("w[0] = 9", _VEC, "W[0] = 9"),
    ("w[0] = 9.5 (int64 truncates)", _VEC, "W[0] = 9.5"),
    ("w[0] = -9.5 (int64 truncates toward zero)", _VEC, "W[0] = -9.5"),
    ("w[0] = True", _VEC, "W[0] = True"),
    ("w[0] = 0-dim mps tensor", _VEC, "W[0] = Z"),
    ("w[1, 2] = 9", _MAT, "W[1, 2] = 9"),
    ("w[1] = 9 (fill_ arm)", _MAT, "W[1] = 9"),
    ("w[1:3] = 5", _VEC, "W[1:3] = 5"),
    ("copy_ from a cpu tensor", _VEC, "W.copy_(CPU)"),
    ("w[w > 1] = 0", _VEC, "W[M] = 0"),
    ("w[w > 1] = 9.5", _VEC, "W[M] = 9.5"),
    ("w[w > 1] = True", _VEC, "W[M] = True"),
    ("w[w > 1] = 0 on a matrix", _MAT, "W[M] = 0"),
    ("w[w > 1] = 0-dim cpu tensor", _VEC, "W[M] = torch.tensor(6.0)"),
    ("w[w > 1] = 0-dim mps tensor of its dtype", _VEC, "W[M] = Z.to(DT)"),
]


def _same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


def _agree(rows, dtypes):
    problems = []
    for name, data, code in rows:
        for dtype in dtypes:
            payload = dict(data=data, code=code, dtype=dtype)
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
                problems.append("%s: result is %s %s, upstream %s %s"
                                % (label, got["shape"], got["dtype"],
                                   oracle["shape"], oracle["dtype"]))
            elif not all(_same(a, b) for a, b in zip(got["values"], oracle["values"])):
                problems.append("%s: result is %s, upstream mps %s"
                                % (label, got["values"], oracle["values"]))
            c = got["counters"]
            if c["compute_encoders"] + c["blit_encoders"] <= 0:
                problems.append("%s: no compute or blit encoder ran (%s)" % (label, c))
            if c["host_downloads"] != 0:
                problems.append("%s: something was read back to the host (%s)" % (label, c))
    assert not problems, "\n".join(problems)


def test_scalar_setitem_computes_on_mps():
    """`w[i] = scalar`, `w[mask] = scalar` and `copy_` from a CPU tensor on
    mps, for f32/f16/bf16/int64: the receiver equals upstream mps exactly
    (which equals upstream cpu), stays on mps, and the assignment ran at
    least one Metal encoder with zero host downloads."""
    if not _mps_or_skip("scalar setitem on mps"):
        return
    _agree(_COMPUTED_ROWS, _ALL)


def test_int_index_with_scalar_refuses_by_name():
    """`w[int64_index_on_mps] = scalar`: upstream computes; this build has no
    bounds-checked device scatter, so `index_put_` refuses by name through
    the readback gate rather than raising the device-mismatch message (which
    is what the zero-dim CPU scalar used to trigger) or candle's words."""
    if not _mps_or_skip("int-index refusal on mps"):
        return
    problems = []
    for dtype in _ALL:
        label = "w[I] = 4 [%s]" % dtype
        payload = dict(data=_VEC, code="W[I] = 4", dtype=dtype)
        oracle = _run(dict(payload, device="mps"), upstream=True)
        assert "error" not in oracle, (label, oracle)
        got = _run(dict(payload, device="mps"), upstream=False)
        err = got.get("error", "")
        if not err.startswith("NotImplementedError: aten.index_put_.default"):
            problems.append("%s: not index_put_'s named refusal: %r" % (label, got))
        elif "candle:" in err or "not implemented for the mps device" not in err:
            problems.append("%s: refusal gives no reason: %s" % (label, err[:300]))
    assert not problems, "\n".join(problems)


def test_non_scalar_cpu_value_refuses_like_upstream():
    """A CPU value that is not zero-dim beside an mps receiver: upstream
    raises `Expected all tensors to be on the same device, but found at least
    two devices, mps:0 and cpu!`, and so does the shim (the zero-dim
    exception must not widen to every CPU tensor). A zero-dim mps value of
    another dtype gets upstream's `Index put requires ... dtypes match`."""
    if not _mps_or_skip("non-scalar cpu value on mps"):
        return
    want = ("Expected all tensors to be on the same device, but found at least "
            "two devices, mps:0 and cpu!")
    problems = []
    rows = [("w[w > 1] = cpu (2,)", "W[M] = torch.tensor([5.0, 6.0])", "float32", want),
            ("w[I] = cpu (2,)", "W[I] = torch.tensor([5.0, 6.0])", "float32", want)]
    # A zero-dim value already on mps is not a scalar to upstream: another
    # dtype refuses rather than casting.
    for dtype, name in (("float16", "Half"), ("bfloat16", "BFloat16"), ("int64", "Long")):
        rows.append(("w[w > 1] = 0-dim mps float32", "W[M] = Z", dtype,
                     "Index put requires the source and destination dtypes match, got "
                     "%s for the destination and Float for the source." % name))
    for name, code, dtype, want in rows:
        payload = dict(data=_VEC, code=code, dtype=dtype, device="mps")
        oracle = _run(payload, upstream=True)
        got = _run(payload, upstream=False)
        if want not in oracle.get("error", ""):
            problems.append("%s [%s]: upstream no longer says %r: %r" % (name, dtype, want, oracle))
        if not got.get("error", "").startswith("RuntimeError: " + want):
            problems.append("%s [%s]: shim did not refuse like upstream: %r" % (name, dtype, got))
    assert not problems, "\n".join(problems)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpssetitem")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
