"""Issue #50, path 1: where a python-float scalar is narrowed to ``float16``
when the tensor lives on ``mps``.

The issue's premise was that the shim passes Metal scalars ``as f32`` and the
shader narrows them again (two roundings where c10 on aarch64 rounds once).
**Measured, the premise is the other way round.** Upstream's own MPS backend is
the one that rounds twice, and only in some kernels. With the #28 float16
witness ``w = 0.031265258789971995`` (``f16(w) = 0.031280517578125``,
``f16(f32(w)) = 0.03125``), torch 2.13.0 on this Mac's ``mps``:

    add.Scalar(0, w), sub.Scalar(0, -w), rsub.Scalar(0, w)  -> 0.03125  (two)
    add.Scalar(0, 1, alpha=w)                               -> 0.031280517578125 (one)
    fill_, full, clamp, clamp_min, hardtanh, masked_fill,
    where.ScalarSelf, add.Tensor(alpha=w)                   -> 0.031280517578125 (one)
    mul.Scalar(1, w)                                        -> 0.03125  (opmath f32, same as cpu)

and on ``cpu`` every one of the add/sub/rsub cells is ``0.031280517578125``.
Over 4096 random ``float16`` inputs ``add.Scalar`` on ``mps`` equals
``f16(x + f16(f32(w)))`` in all 4096 and ``f16(x + f16(w))`` in 592: the
*operand* is narrowed through ``float``, the addition itself is exact-then-once.

The shim narrowed the operand through ``float_narrower`` (c10's one-step rule
on aarch64) on every device, so its ``mps`` answer was its ``cpu`` answer and
not upstream ``mps``'s. ``bfloat16`` is not affected: c10's rule is already
``bf16(f32(x))`` everywhere, and upstream ``mps`` agrees (asked below).

Nullifications this file catches:

* ``arith_scalar``/``rsub_scalar`` using ``float_narrower`` for the operand on
  Metal again -> test_add_sub_rsub_scalar_narrow_through_float_on_mps
* the Metal rule leaking onto the ``alpha`` operand
  -> test_alpha_is_still_rounded_once_on_mps
* the Metal rule leaking onto ``cpu`` -> test_the_cpu_answers_are_unchanged
* the ops that round once on upstream mps being "fixed" too
  -> test_the_once_rounded_ops_stay_once_rounded_on_mps
"""

import json
import os
import platform
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip

W16 = 0.031265258789971995
ONCE = 0.031280517578125     # f16(W16)
TWICE = 0.03125              # f16(f32(W16))
BF16_W = 1 + 3 * 2**-8 - 2**-30   # bf16(f32(x)) = 1.015625, bf16(x) = 1.0078125

_ON_AARCH64 = platform.machine().lower() in ("arm64", "aarch64")


def _mps_or_skip(what):
    if not _C._metal_counters()["built"]:
        _skip.skip("   (skipped %s: not an Apple build -- no Metal counters)" % what)
        return None
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


def _oracle(device):
    """Upstream's answers on `device`, from an interpreter with no `_C` on its path."""
    script = r"""
import json, sys, torch
A = torch.ops.aten
dev, w, bw = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
if dev == "mps" and not torch.backends.mps.is_available():
    print(json.dumps(None)); raise SystemExit
h = lambda: torch.zeros(1, dtype=torch.float16, device=dev)
one = lambda: torch.ones(1, dtype=torch.float16, device=dev)
m = lambda: torch.tensor([True], device=dev)
r = {
  "add.Scalar": A.add.Scalar(h(), w),
  "sub.Scalar": A.sub.Scalar(h(), -w),
  "rsub.Scalar": A.rsub.Scalar(h(), w),
  "add.Scalar/alpha": A.add.Scalar(h(), 1.0, alpha=w),
  "rsub.Scalar/alpha": A.rsub.Scalar(one(), 0.0, alpha=w),
  "clamp": A.clamp.default(h(), w),
  "clamp_min": A.clamp_min.default(h(), w),
  "hardtanh": A.hardtanh.default(h(), w, 1.0),
  "masked_fill": A.masked_fill.Scalar(h(), m(), w),
  "fill_": A.fill_.Scalar(h(), w),
  "full": A.full.default([1], w, dtype=torch.float16, device=dev),
  "where.ScalarSelf": A.where.ScalarSelf(m(), w, h()),
  "add.Tensor/alpha": A.add.Tensor(h(), one(), alpha=w),
  "mul.Scalar": A.mul.Scalar(one(), w),
  "bf16/add.Scalar": A.add.Scalar(torch.zeros(1, dtype=torch.bfloat16, device=dev), bw),
}
print(json.dumps({k: v.item() for k, v in r.items()}))
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run([sys.executable, "-c", script, device, repr(W16), repr(BF16_W)],
                          capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n" + proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _shim(device):
    def h(dt=_C.float16):
        return _C._aten_dispatch("aten.zeros.default", [1], dtype=dt, device=device)

    def one():
        return _C._aten_dispatch("aten.ones.default", [1], dtype=_C.float16, device=device)

    def m():
        t = _C._tensor_from_flat([1.0], [1], _C.float64).to(_C.bool)
        return t.to("mps") if device is not None else t

    d = _C._aten_dispatch
    cells = {
        "add.Scalar": lambda: d("aten.add.Scalar", h(), W16),
        "sub.Scalar": lambda: d("aten.sub.Scalar", h(), -W16),
        "rsub.Scalar": lambda: d("aten.rsub.Scalar", h(), W16),
        "add.Scalar/alpha": lambda: d("aten.add.Scalar", h(), 1.0, alpha=W16),
        "rsub.Scalar/alpha": lambda: d("aten.rsub.Scalar", one(), 0.0, alpha=W16),
        "clamp": lambda: d("aten.clamp.default", h(), W16),
        "clamp_min": lambda: d("aten.clamp_min.default", h(), W16),
        "hardtanh": lambda: d("aten.hardtanh.default", h(), W16, 1.0),
        "masked_fill": lambda: d("aten.masked_fill.Scalar", h(), m(), W16),
        "fill_": lambda: d("aten.fill_.Scalar", h(), W16),
        "full": lambda: d("aten.full.default", [1], W16, dtype=_C.float16, device=device),
        "where.ScalarSelf": lambda: d("aten.where.ScalarSelf", m(), W16, h()),
        "add.Tensor/alpha": lambda: d("aten.add.Tensor", h(), one(), alpha=W16),
        "mul.Scalar": lambda: d("aten.mul.Scalar", one(), W16),
        "bf16/add.Scalar": lambda: d("aten.add.Scalar", h(_C.bfloat16), BF16_W),
    }
    out = {}
    for k, f in cells.items():
        t = f()
        if device is not None:
            assert str(t.device).startswith("mps"), (k, t.device)
            t = t.cpu()
        out[k] = t.to(_C.float64).flatten().tolist()[0]
    return out


def _compare(keys, shim, ref, where):
    bad = ["%s: shim %r, upstream %s %r" % (k, shim[k], where, ref[k])
           for k in keys if shim[k] != ref[k]]
    assert not bad, "float16 scalar narrowing disagrees with upstream:\n  " + "\n  ".join(bad)


def test_add_sub_rsub_scalar_narrow_through_float_on_mps():
    mps = _mps_or_skip("add/sub/rsub.Scalar witness on mps")
    if mps is None:
        return
    ref = _oracle("mps")
    if ref is None:
        _skip.skip("   (skipped add/sub/rsub.Scalar witness: upstream has no mps here)")
        return
    keys = ("add.Scalar", "sub.Scalar", "rsub.Scalar")
    # The premise, asked of upstream: these three really are two roundings.
    for k in keys:
        assert ref[k] == TWICE, "upstream mps %s moved: %r (was %r)" % (k, ref[k], TWICE)
    _compare(keys, _shim(mps), ref, "mps")


def test_alpha_is_still_rounded_once_on_mps():
    mps = _mps_or_skip("alpha witness on mps")
    if mps is None:
        return
    ref = _oracle("mps")
    if ref is None:
        _skip.skip("   (skipped alpha witness: upstream has no mps here)")
        return
    _compare(("add.Scalar/alpha", "rsub.Scalar/alpha", "add.Tensor/alpha"),
             _shim(mps), ref, "mps")


def test_the_once_rounded_ops_stay_once_rounded_on_mps():
    mps = _mps_or_skip("once-rounded ops on mps")
    if mps is None:
        return
    ref = _oracle("mps")
    if ref is None:
        _skip.skip("   (skipped once-rounded ops: upstream has no mps here)")
        return
    _compare(("clamp", "clamp_min", "hardtanh", "masked_fill", "fill_", "full",
              "where.ScalarSelf", "mul.Scalar", "bf16/add.Scalar"),
             _shim(mps), ref, "mps")


def test_the_cpu_answers_are_unchanged():
    if not _ON_AARCH64:
        # Off aarch64 c10 itself narrows f64 -> f16 through float, so the
        # witness cannot tell the Metal rule from the cpu rule there.
        _skip.skip("   (skipped cpu witness: not aarch64, f16 one/two roundings coincide in c10)")
        return
    ref = _oracle("cpu")
    keys = ("add.Scalar", "sub.Scalar", "rsub.Scalar", "add.Scalar/alpha",
            "rsub.Scalar/alpha", "bf16/add.Scalar")
    for k in ("add.Scalar", "sub.Scalar", "rsub.Scalar"):
        assert ref[k] == ONCE, "upstream cpu %s moved: %r (was %r)" % (k, ref[k], ONCE)
    _compare(keys, _shim(None), ref, "cpu")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)
             and getattr(fn, "__module__", None) == __name__]
    failures = _skip.run_tests(items, suite="test_mpsnarrow")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
