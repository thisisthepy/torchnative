"""The *reference* Q4 implementation for issue #16, and the exact arithmetic it is judged by.

Two things live here, and they are kept apart on purpose:

1. **llama.cpp's own Python codec** (`gguf-py`, the `gguf` package the
   llama.cpp repository ships). `load_gguf_quants()` imports it; nothing in
   this file re-derives what it does. It is the independent reference that
   `rust/torch_c/pytests/ggml_ref.py` says it is *not*: ggml_ref was
   transcribed from the format by the same hands that read candle, so a
   misreading shared by both would pass. gguf-py was written by llama.cpp's
   authors against their C (`Q8_0` says so in a comment: "bit-exact same
   results as reference implementation in ggml-quants.c").

   gguf-py has **dequantisers for Q4_0 and Q4_K, but a quantiser for Q4_0
   only** -- k-quant writers are not in it. That gap decides what can be
   compared byte for byte (Q4_0) and what can only be compared on the reading
   side (Q4_K). `measure.py`'s docstring has the whole design, and AGENTS.md
   §16 says why the grade of each comparison is stated separately.

2. **Transcriptions of llama.cpp C**, each marked with the function and the
   llama.cpp commit it was read at. They exist because gguf-py lacks them:

   * `quantize_q4_k_llamacpp` -- `quantize_row_q4_K_ref` with
     `make_qkx2_quants`. **This is not what candle does** (candle's Q4K writer
     is `make_qkx1_quants(15, 5, x)`, the older unweighted search that
     llama.cpp left behind as a comment). It is the quality baseline: what
     llama.cpp's encoder costs on the same weights. A transcription can be
     wrong; `measure.py --gguf` closes that by reading a real llama.cpp-written
     file instead.
   * `quantize_q8_k_ref` -- `quantize_row_q8_K_ref`, the activation-side
     format candle quantises the *input* to inside a Q4_K matmul. Needed to
     build the exact reference for `_quantized_linear`.

   **Float semantics.** Every transcription runs in `float32` with the
   operation order the C has, and sums that the C accumulates sequentially are
   accumulated sequentially here (`np.cumsum(...)[..., -1]`, never `np.sum`,
   which is pairwise). The C itself is not bit-stable across builds: a
   compiler allowed to contract `a*b + c` into an FMA produces different last
   bits, and llama.cpp's own SIMD quantisers round ties to even where the
   `_ref` ones round half away. So these match the `_ref` C *without* FP
   contraction, and say so rather than pretending to match every build.

Reading llama.cpp: `ggml/src/ggml-quants.c` at commit
bed0a856606ee4a24a164066f73d2379447033f5 (fetched 2026-10-03).
"""

import hashlib
import importlib
import math
import os
import sys
import types

import numpy as np

LLAMACPP_COMMIT = "bed0a856606ee4a24a164066f73d2379447033f5"

QK4_0 = 32
QK8_0 = 32
QK_K = 256
K_SCALE_SIZE = 12
TYPE_SIZE = {"q4_0": 18, "q8_0": 34, "q4_k": 144}
BLOCK_SIZE = {"q4_0": QK4_0, "q8_0": QK8_0, "q4_k": QK_K}

# `float32` unit roundoff. Every bound in this file is written in it.
U32 = 2.0 ** -24
U64 = 2.0 ** -53


# --- gguf-py ---------------------------------------------------------------


def load_gguf_quants(spec=None):
    """Import `gguf.quants` and say exactly which one was imported.

    `spec` (or `$TORCHNATIVE_GGUF_PY`) is a directory that contains a `gguf/`
    package -- a llama.cpp checkout's `gguf-py/`, say -- or a `gguf-*.whl`,
    which is a zip and imports as one because the package is pure Python.
    With neither, the installed `gguf` is used.

    **The package `__init__` is bypassed.** `gguf/__init__.py` imports the
    reader, writer, vocab and metadata modules, and those pull `yaml`,
    `sentencepiece` and friends; `quants.py` itself needs `constants.py`,
    `lazy.py` and numpy. A missing optional dependency of a module this
    measurement does not use is not a reason to fail, so a bare package
    object is registered and only `gguf.quants` is imported through it.

    Returns `(quants_module, provenance)`. The provenance carries the sha256
    of `quants.py`'s bytes, so a result can say which reference produced it.
    """
    spec = spec or os.environ.get("TORCHNATIVE_GGUF_PY")
    if "gguf.quants" in sys.modules:
        mod = sys.modules["gguf.quants"]
    elif spec:
        root = os.path.abspath(spec)
        pkg = types.ModuleType("gguf")
        pkg.__path__ = [os.path.join(root, "gguf")]
        sys.modules["gguf"] = pkg
        if root.endswith(".whl") and root not in sys.path:
            # zipimport resolves "<archive>/gguf" through sys.path_hooks once
            # the archive itself is importable.
            sys.path.insert(0, root)
        mod = importlib.import_module("gguf.quants")
    else:
        mod = importlib.import_module("gguf.quants")
    loader = getattr(mod, "__loader__", None)
    try:
        src = loader.get_data(mod.__file__) if loader is not None else open(mod.__file__, "rb").read()
    except Exception as exc:  # noqa: BLE001 -- provenance is reported, not assumed
        raise RuntimeError(f"cannot read gguf.quants source for provenance: {exc}") from exc
    if not hasattr(mod, "Q4_0") or not hasattr(mod, "Q4_K"):
        raise RuntimeError(f"{mod.__file__} has no Q4_0/Q4_K classes; not llama.cpp's gguf-py")
    return mod, {
        "file": mod.__file__,
        "sha256": hashlib.sha256(src).hexdigest(),
    }


def gguf_quantize(gq, fmt, x):
    """gguf-py's writer. `x` is float32 with a last dim divisible by the block."""
    cls = {"q4_0": gq.Q4_0, "q8_0": gq.Q8_0}[fmt]
    return cls.quantize(np.ascontiguousarray(x, dtype=np.float32)).tobytes()


def gguf_dequantize(gq, fmt, blob, n):
    cls = {"q4_0": gq.Q4_0, "q8_0": gq.Q8_0, "q4_k": gq.Q4_K}[fmt]
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(-1, TYPE_SIZE[fmt])
    out = cls.dequantize(raw).reshape(-1)
    assert out.size == n, (fmt, out.size, n)
    return out.astype(np.float32, copy=False)


# --- llama.cpp transcriptions ----------------------------------------------


def _nearest_int(v):
    """`nearest_int` in ggml-quants.c: `fval + 12582912.f` read back as bits.

    Adding 1.5 * 2**23 forces the fraction out of the significand under the
    current rounding mode, which is round-to-nearest-**even**. `np.rint` is the
    same rule on `float32`. (candle's `nearest_int` is `f32::round`, half
    *away* from zero -- the two differ only on exact ties.)
    """
    return np.rint(np.asarray(v, dtype=np.float32)).astype(np.int32)


def _seqsum(a, axis=-1):
    """A float32 sum in the order a C `for` loop accumulates it."""
    return np.cumsum(np.asarray(a, dtype=np.float32), axis=axis, dtype=np.float32).take(-1, axis=axis)


def _make_qkx2_quants(x, weights, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20):
    """`make_qkx2_quants(32, 15, x, weights, L, &min, Laux, -1.f, 0.1f, 20, false)`.

    Vectorised over sub-blocks: `x` and `weights` are `(B, 32)` float32.
    Returns `(scale, the_min)` as `(B,)` float32, `the_min = -min` as in C.
    """
    f = np.float32
    x = x.astype(f)
    w = weights.astype(f)
    mn = np.minimum(x.min(axis=1), f(0))
    mx = x.max(axis=1)
    sum_w = _seqsum(w)
    sum_x = _seqsum(w * x)
    flat = mx == mn

    span = np.where(flat, f(1), mx - mn).astype(f)
    iscale = (f(nmax) / span).astype(f)
    scale = (f(1) / iscale).astype(f)
    L = np.clip(_nearest_int(iscale[:, None] * (x - mn[:, None])), 0, nmax)
    diff = (scale[:, None] * L.astype(f) + mn[:, None] - x).astype(f)
    best = _seqsum(w * (diff * diff))
    best_scale, best_min = scale.copy(), mn.copy()

    for step in range(nstep + 1):
        iscale = ((f(rmin) + f(rdelta) * f(step) + f(nmax)) / span).astype(f)
        La = np.clip(_nearest_int(iscale[:, None] * (x - mn[:, None])), 0, nmax).astype(f)
        sum_l = _seqsum(w * La)
        sum_l2 = _seqsum(w * La * La)
        sum_xl = _seqsum(w * La * x)
        D = (sum_w * sum_l2 - sum_l * sum_l).astype(f)
        ok = D > 0
        Ds = np.where(ok, D, f(1))
        this_scale = ((sum_w * sum_xl - sum_x * sum_l) / Ds).astype(f)
        this_min = ((sum_l2 * sum_x - sum_l * sum_xl) / Ds).astype(f)
        pos = this_min > 0
        this_scale = np.where(pos, (sum_xl / np.where(sum_l2 == 0, f(1), sum_l2)).astype(f), this_scale)
        this_min = np.where(pos, f(0), this_min)
        d2 = (this_scale[:, None] * La + this_min[:, None] - x).astype(f)
        cur = _seqsum(w * (d2 * d2))
        better = ok & (cur < best)
        best = np.where(better, cur, best)
        best_scale = np.where(better, this_scale, best_scale)
        best_min = np.where(better, this_min, best_min)

    best_scale = np.where(flat, f(0), best_scale)
    best_min = np.where(flat, mn, best_min)
    return best_scale.astype(f), (-best_min).astype(f)


def _f16_bits(v):
    return np.asarray(v, dtype=np.float32).astype(np.float16).view(np.uint16)


def quantize_q4_k_llamacpp(x):
    """`quantize_row_q4_K_ref` (ggml-quants.c), no importance matrix.

    `x`: float32, size a multiple of 256. Returns GGUF Q4_K bytes, 144 per
    super-block, in the layout candle's `BlockQ4K` and gguf-py's `Q4_K` read.
    """
    f = np.float32
    xs = np.asarray(x, dtype=f).reshape(-1, QK_K)
    nb = xs.shape[0]
    sub = xs.reshape(nb * 8, 32)
    sum_x2 = _seqsum(sub * sub)
    av_x = np.sqrt((sum_x2 / f(32)).astype(f)).astype(f)
    weights = (av_x[:, None] + np.abs(sub)).astype(f)
    scales, mins = _make_qkx2_quants(sub, weights)
    scales = scales.reshape(nb, 8)
    mins = mins.reshape(nb, 8)
    max_scale = np.maximum(scales.max(axis=1), f(0))
    max_min = np.maximum(mins.max(axis=1), f(0))
    inv_scale = np.where(max_scale > 0, f(63) / np.where(max_scale > 0, max_scale, f(1)), f(0)).astype(f)
    inv_min = np.where(max_min > 0, f(63) / np.where(max_min > 0, max_min, f(1)), f(0)).astype(f)
    ls = np.minimum(_nearest_int(inv_scale[:, None] * scales), 63).astype(np.uint8)
    lm = np.minimum(_nearest_int(inv_min[:, None] * mins), 63).astype(np.uint8)

    sc = np.zeros((nb, K_SCALE_SIZE), dtype=np.uint8)
    sc[:, 0:4] = ls[:, 0:4]
    sc[:, 4:8] = lm[:, 0:4]
    sc[:, 8:12] = (ls[:, 4:8] & 0xF) | ((lm[:, 4:8] & 0xF) << 4)
    sc[:, 0:4] |= (ls[:, 4:8] >> 4) << 6
    sc[:, 4:8] |= (lm[:, 4:8] >> 4) << 6

    d_bits = _f16_bits((max_scale / f(63)).astype(f))
    dmin_bits = _f16_bits((max_min / f(63)).astype(f))
    d = d_bits.view(np.float16).astype(f)
    dmin = dmin_bits.view(np.float16).astype(f)

    sc6, m6 = _unpack_scale_min_k4(sc)
    dd = (d[:, None] * sc6.astype(f)).astype(f)  # (nb, 8)
    dm = (dmin[:, None] * m6.astype(f)).astype(f)
    xsub = xs.reshape(nb, 8, 32)
    safe = np.where(dd == 0, f(1), dd)
    L = np.clip(_nearest_int((xsub + dm[:, :, None]) / safe[:, :, None]), 0, 15)
    L = np.where(dd[:, :, None] == 0, 0, L).astype(np.uint8)  # C: `if (!d) continue;` leaves L at 0

    # Nibbles: each 64-element group -> 32 bytes, low nibble = first 32.
    Lg = L.reshape(nb, 4, 2, 32)
    qs = (Lg[:, :, 0, :] | (Lg[:, :, 1, :] << 4)).reshape(nb, 128)

    out = np.concatenate(
        [d_bits.reshape(nb, 1).view(np.uint8), dmin_bits.reshape(nb, 1).view(np.uint8), sc, qs], axis=1
    )
    assert out.shape == (nb, TYPE_SIZE["q4_k"])
    return out.tobytes()


def _unpack_scale_min_k4(scales):
    """`get_scale_min_k4` for all eight sub-blocks of `(nb, 12)` scale bytes."""
    s = scales.astype(np.uint8)
    sc = np.empty((s.shape[0], 8), dtype=np.uint8)
    m = np.empty((s.shape[0], 8), dtype=np.uint8)
    sc[:, 0:4] = s[:, 0:4] & 63
    m[:, 0:4] = s[:, 4:8] & 63
    sc[:, 4:8] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, m


def quantize_q8_k_ref(x):
    """`quantize_row_q8_K_ref`. Returns `(d f32 (nb,), qs int8 (nb,256))`.

    `iscale = -127/max` keeps the sign of the largest-magnitude element (the
    first one, under `>`), and `d = 1/iscale` is stored as `float32`, not f16.
    """
    f = np.float32
    xs = np.asarray(x, dtype=f).reshape(-1, QK_K)
    ax = np.abs(xs)
    idx = ax.argmax(axis=1)  # first index of the maximum, as `ax > amax` picks it
    mx = xs[np.arange(xs.shape[0]), idx]
    zero = ax.max(axis=1) == 0
    iscale = np.where(zero, f(0), (f(-127) / np.where(zero, f(1), mx)).astype(f)).astype(f)
    q = np.minimum(_nearest_int(iscale[:, None] * xs), 127)
    q = np.where(zero[:, None], 0, q).astype(np.int8)
    d = np.where(zero, f(0), (f(1) / np.where(zero, f(1), iscale)).astype(f)).astype(f)
    return d, q


def q8_k_ties(x):
    """Positions where `iscale*x` lands exactly on k + 0.5.

    There half-even (llama.cpp's `nearest_int`) and half-away (candle's
    `f32::round`) disagree, so a fixture with any is ambiguous and is refused
    rather than graded.
    """
    f = np.float32
    xs = np.asarray(x, dtype=f).reshape(-1, QK_K)
    ax = np.abs(xs)
    mx = xs[np.arange(xs.shape[0]), ax.argmax(axis=1)]
    iscale = (f(-127) / np.where(mx == 0, f(1), mx)).astype(f)
    v = (iscale[:, None] * xs).astype(f)
    return int(np.sum(np.abs(v - np.floor(v)) == f(0.5)))


def q8_0_ties(x):
    """The same question for Q8_0, whose rounding is `roundf` in the `_ref`
    C and `vcvtnq` (half-even) in llama.cpp's NEON quantiser."""
    f = np.float32
    xs = np.asarray(x, dtype=f).reshape(-1, QK8_0)
    d = (np.abs(xs).max(axis=1) / f(127)).astype(f)
    inv = np.where(d == 0, f(0), f(1) / np.where(d == 0, f(1), d)).astype(f)
    v = (xs * inv[:, None]).astype(f)
    return int(np.sum(np.abs(v - np.floor(v)) == f(0.5)))


def signed_max_ties(x, block):
    """Blocks whose largest |x| is reached by two elements of opposite sign.

    The scalar `_ref` writers take the first; a lane-parallel max (candle's
    NEON `quantize_row_q8k`) may take either. Refused like a rounding tie.
    """
    xs = np.asarray(x, dtype=np.float32).reshape(-1, block)
    ax = np.abs(xs)
    amax = ax.max(axis=1, keepdims=True)
    at = (ax == amax) & (amax > 0)
    has_pos = np.any(at & (xs > 0), axis=1)
    has_neg = np.any(at & (xs < 0), axis=1)
    return int(np.sum(has_pos & has_neg))


# --- the exact value of a quantised matmul ---------------------------------
#
# A Q4 matmul on CPU (candle and llama.cpp alike) does not compute
# `x @ dequant(W).T`. It first quantises each row of `x` to the activation
# format (Q8_0 for a Q4_0 weight, Q8_K for a Q4_K one), takes **integer** dot
# products block by block, and only then scales by the block scales in
# float32. So the exact answer is a sum of dyadic rationals that can be
# written down from the two blobs, and the only thing left to a bound is the
# float32 accumulation of those block terms.
#
# The bound is Higham's: a sum of terms t_i, each computed with at most K
# roundings on its path to the output (its own multiplications plus the
# additions it passes through, in any order or tree), satisfies
#
#     |fl(sum) - sum|  <=  gamma_K * sum |t_i|,   gamma_K = K u / (1 - K u)
#
# with u = 2**-24. `sum |t_i|` is taken at the **element** level
# (|q_w| * |q_x| * |scales| per element), which is >= the same sum over any
# grouping a kernel uses, so the bound holds whichever grouping it is.
#
# K is counted from candle's aarch64 kernels, not chosen:
#
#   q4_0 x q8_0 (neon.rs `vec_dot_q4_0_q8_0`): the block scale product
#     `d_w * d_x` is exact (two f16 significands, 22 bits < 24); the integer
#     partial sum converts exactly (|sum| <= 32*8*127 < 2**24); then one
#     multiply, nb lane accumulations, and a 4-lane horizontal add of depth 2.
#     K = nb + 3.
#
#   q4_k x q8_k: two kernels, and the larger count is used.
#     `vec_dot_q4k_q8k`: per super-block one rounding for `y.d * x.d`, one
#     for the int->f32 conversion (the scaled sum can exceed 2**24), one
#     multiply, and two sequential accumulations per super-block: K = 2 nb + 3.
#     `vec_dot_8_q4k_q8k` (the repacked path, built only with `+dotprod`, taken
#     only when out_features % 8 == 0): one rounding for the scale product,
#     then eight fused multiply-adds and one multiply-subtract per super-block
#     into the same accumulator: K = 9 nb + 2.
#     So K = 9 nb + 3.
#
# **What this cannot see:** an x86 build. Its kernels (avx.rs, and the
# scalar fallback) were not counted, and K must be recounted before this
# bound is applied there.


def gamma(k):
    return k * U32 / (1.0 - k * U32)


def accumulation_k(fmt, in_features):
    if fmt == "q4_0":
        return in_features // QK4_0 + 3
    if fmt == "q4_k":
        return 9 * (in_features // QK_K) + 3
    raise KeyError(f"no counted accumulation depth for {fmt}")


def _q4_0_parts(blob, out_features, in_features):
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(out_features, in_features // QK4_0, TYPE_SIZE["q4_0"])
    d = raw[:, :, 0:2].copy().view(np.float16).astype(np.float64)[..., 0]  # (out, nb)
    qs = raw[:, :, 2:]
    lo = (qs & 0x0F).astype(np.int64) - 8
    hi = (qs >> 4).astype(np.int64) - 8
    q = np.concatenate([lo, hi], axis=2)  # (out, nb, 32): element j and j+16
    return d, q


def _q8_0_parts(blob, rows, in_features):
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(rows, in_features // QK8_0, TYPE_SIZE["q8_0"])
    d = raw[:, :, 0:2].copy().view(np.float16).astype(np.float64)[..., 0]
    q = raw[:, :, 2:].copy().view(np.int8).astype(np.int64)
    return d, q


def _sum_terms(terms):
    """Sum the last axis of float64 `terms`, and say how far the sum can be off.

    Small problems (the gate's) are summed with `math.fsum`, which is
    correctly rounded: off by at most half an ulp of float64, bounded here by
    `U64 * sum|t|`. A real layer (512 x 2048 outputs) is too many for a Python
    loop, so it is summed by numpy and the float64 summation bound
    `gamma64(n) * sum|t|` is returned instead. Either way the reference's own
    error is part of the derived ceiling rather than assumed to be zero.
    """
    lead = terms.shape[:-1]
    n = terms.shape[-1]
    if int(np.prod(lead)) <= 4096:
        flat = terms.reshape(-1, n)
        value = np.array([math.fsum(row) for row in flat]).reshape(lead)
        rel = U64
    else:
        value = terms.sum(axis=-1)
        rel = n * U64 / (1.0 - n * U64)
    return value, rel


def exact_linear_q4_0(x_q8_blob, w_blob, rows, out_features, in_features):
    """`(value, abs_sum, ref_err)` per output, each `(rows, out)` float64.

    Every block term is exact in float64 (11 + 11 + 15 bits), so the only
    rounding in the reference is the summation, and `ref_err` says how much
    that can be. `abs_sum` is the element-level `sum |t|` the float32 bound is
    written in.
    """
    dw, qw = _q4_0_parts(w_blob, out_features, in_features)
    dx, qx = _q8_0_parts(x_q8_blob, rows, in_features)
    ints = np.einsum("rbk,obk->rob", qx, qw)  # exact int64 (rows, out, nb)
    absints = np.einsum("rbk,obk->rob", np.abs(qx), np.abs(qw))
    scale = dx[:, None, :] * dw[None, :, :]  # exact
    value, rel = _sum_terms(ints.astype(np.float64) * scale)
    abs_sum = (absints.astype(np.float64) * np.abs(scale)).sum(axis=2)
    return value, abs_sum, rel * abs_sum


def _q4_k_parts(blob, out_features, in_features):
    nb = in_features // QK_K
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(out_features, nb, TYPE_SIZE["q4_k"])
    d = raw[:, :, 0:2].copy().view(np.float16).astype(np.float64)[..., 0]
    dmin = raw[:, :, 2:4].copy().view(np.float16).astype(np.float64)[..., 0]
    sc, m = _unpack_scale_min_k4(raw[:, :, 4:16].reshape(-1, K_SCALE_SIZE))
    sc = sc.reshape(out_features, nb, 8).astype(np.int64)
    m = m.reshape(out_features, nb, 8).astype(np.int64)
    qs = raw[:, :, 16:].reshape(out_features, nb, 4, 32)
    q = np.stack([qs & 0x0F, qs >> 4], axis=3).reshape(out_features, nb, 8, 32).astype(np.int64)
    return d, dmin, sc, m, q


def exact_linear_q4_k(x_d, x_q, w_blob, rows, out_features, in_features):
    """The Q4_K x Q8_K value, from the two quantised operands.

    Per super-block b and sub-block j (32 elements):

        y.d * x.d * sc_j * sum(q4 * q8)  -  y.d * x.dmin * m_j * sum(q8)

    `x_d` is `(rows, nb)` float32, `x_q` `(rows, nb, 256)` int8, from
    `quantize_q8_k_ref`. Each term is rounded to float64 once (24 + 11 + 6 +
    16 bits can exceed 53); that and the summation's own error come back as
    `ref_err`.
    """
    nb = in_features // QK_K
    d, dmin, sc, m, q4 = _q4_k_parts(w_blob, out_features, in_features)
    xq = x_q.astype(np.int64).reshape(rows, nb, 8, 32)
    xd = x_d.astype(np.float64).reshape(rows, nb)
    dots = np.einsum("rbjk,objk->robj", xq, q4)  # exact int64
    absdots = np.einsum("rbjk,objk->robj", np.abs(xq), q4)
    qsum = xq.sum(axis=3)  # (rows, nb, 8)
    absqsum = np.abs(xq).sum(axis=3)
    sd = xd[:, None, :, None] * d[None, :, :, None]  # exact: 24 + 11 bits
    sm = xd[:, None, :, None] * dmin[None, :, :, None]
    a = sd * sc[None] * dots  # one float64 rounding at most
    b = sm * m[None] * qsum[:, None, :, :]  # exact: 24 + 11 + 6 + 12 bits
    terms = np.concatenate([a.reshape(rows, out_features, -1), -b.reshape(rows, out_features, -1)], axis=2)
    value, rel = _sum_terms(terms)
    abs_sum = (np.abs(sd) * sc[None] * absdots + np.abs(sm) * m[None] * absqsum[:, None, :, :]).sum(axis=(2, 3))
    # `rel` for the sum, plus one float64 rounding per `a` term.
    return value, abs_sum, (rel + U64) * abs_sum


def linear_bound(fmt, in_features, abs_sum, ref_err):
    """The derived ceiling on `|ours - reference|`, per output element:
    candle's float32 accumulation (`gamma_K * sum|t|`) plus the reference's
    own float64 error. Nothing in it is a free parameter."""
    return gamma(accumulation_k(fmt, in_features)) * abs_sum + ref_err
