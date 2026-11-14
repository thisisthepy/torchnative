"""The Q4_K writer against llama.cpp's own algorithm, byte for byte (issue #27).

`test_q4ref.py` pins the Q4_K *reader* to llama.cpp and says why it could not
pin the writer: gguf-py has no k-quant writer, and candle 0.11.0's writer was
not llama.cpp's (`make_qkx1_quants(15, 5)`, unweighted, ties away from zero,
against `make_qkx2_quants` weighted by `av_x + |x|` with a 21-step sweep and a
ties-to-even `nearest_int`). The fork now carries llama.cpp's algorithm
(`torchnative/rust/vendor/int8-candle-0.11.0-cpu.patch`, `BlockQ4K::from_float`),
and this suite requires `_C._quantize(.., "q4_k")` to produce exactly the bytes
llama.cpp's `quantize_row_q4_K_ref` produces. There is no tolerance anywhere.

**Where the expected bytes come from.** Not from the new code: from the scalar
functions below, which are a statement-for-statement **transcription** of
llama.cpp's C, read in `ggml/src/ggml-quants.c` at commit
3de52ac38818978d6da8f386eda43a856e53714a:

  * `nearest_int`            lines 630-635 (the `+ 12582912.f` bit trick);
  * `make_qkx2_quants`       lines 808-887;
  * `get_scale_min_k4`       lines 889-896;
  * `quantize_row_q4_K_ref`  lines 1467-1535 (what `quantize_q4_K` runs when
    there is no importance matrix, line 1639).

Every `float` operation is one `np.float32` operation in the C's order, and
every sum is accumulated in the C's loop order. The gguf package has no Q4_K
quantiser (gguf-py 0.17.1 `Q4_K` has `dequantize_blocks` only), so a
transcription is the only reference available in the gate.

**How the transcription was checked** (once, by hand, outside the gate): it was
run against `quantize_row_q4_K_ref` exported by a compiled llama.cpp
(`libggml-base.dylib`, macOS arm64, clang -O3) through ctypes. Every fixture
in this file agrees byte for byte, and 694 of 700 random super-blocks do
(Gaussian, Student-t, integer-valued, half-integer, one-signed). The 6 that
do not are FP contraction: that build fuses `a*b + c` into FMA (its
`quantize_row_q4_K_ref` disassembly has 107 `fmadd`), so its last bits depend
on the compiler. A copy of the transcription with `fma()` at the contractible
expressions reproduced both blocks it was tried on, but still missed 3 of 210
others, so clang's exact contraction pattern was not reproduced.

**What this cannot see** (AGENTS.md §17.4):

  * A llama.cpp build that contracts. llama.cpp's Q4_K bytes are not a single
    answer: an FMA-contracting build (clang's default `-ffp-contract=on`,
    GCC's `fast`) differs from the ISO reading on about 1 % of super-blocks
    (6 of 700 above). The fork follows the uncontracted reading, which is what
    Rust computes (rustc never contracts). A real llama.cpp GGUF is the
    remaining confirmation (#27's second box), and its result depends on how
    that llama.cpp was built.
  * The importance-matrix writer (`from_float_imatrix`, `quantize_row_q4_K_impl`).
  * Any other k-quant writer. Q2_K and Q5_K still use `make_qkx1_quants` and
    candle's `nearest_int`; they were left alone.

Nullification, done on the transcription (a mutated copy of the functions
below run on the fixtures, standing in for the same change to the Rust; the
gate itself was not run). Which fixture goes red:

  * weights `1.0` instead of `av_x + |x|`        -> tests 1 and 2 (8 fixtures)
  * ties away from zero everywhere                -> test 1 (`ties`), test 2
  * ties away only in the final pass              -> test 1 (`final_pass_tie`)
  * the sweep keeps the first `min` (no update)   -> tests 1 and 2
  * `if (!d)` writes 0 instead of keeping L        -> test 1 (`zero_scale_sub`)
  * a 20-step sweep instead of 21                  -> test 2 (`last_sweep_step`)
  * candle's old `make_qkx1_quants(15, 5)` writer  -> tests 1 and 2 (every
    fixture but `zeros`)
  * NOT caught: `cur_error <= best_error` for `<`, on any fixture or on 800
    searched blocks (no two candidates tie on error); and `MIN(63, ..)` before
    the `uint8_t` cast, which is unobservable because scales and mins are
    never negative.
"""

import math
import os
import struct
import sys

import numpy as np

import _C
import _skip

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "q4quality"))
import q4_gguf_ref  # noqa: E402

F = np.float32
QK_K = 256
BLOCK_BYTES = 144


# --- the transcription ------------------------------------------------------


def nearest_int(fval):
    """ggml-quants.c 630-635: `val = fval + 12582912.f`, then the low 23 bits.

    The addition rounds under the current mode, round to nearest, ties to even.
    """
    val = F(F(fval) + F(12582912.0))
    i = struct.unpack("<i", struct.pack("<f", float(val)))[0]
    return (i & 0x007FFFFF) - 0x00400000


def nearest_int_away(fval):
    """Ties away from zero (candle's `nearest_int`, `f32::round`). Used only to
    prove a fixture contains a tie that decides a byte."""
    v = float(F(fval))
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def make_qkx2_quants(n, nmax, x, weights, L, rmin, rdelta, nstep, rnd=nearest_int):
    """ggml-quants.c 808-887 with `use_mad = false`. Returns `(scale, the_min)`
    and writes the levels into `L`, as the C does."""
    mn = x[0]
    mx = x[0]
    sum_w = weights[0]
    sum_x = F(sum_w * x[0])
    for i in range(1, n):
        if x[i] < mn:
            mn = x[i]
        if x[i] > mx:
            mx = x[i]
        w = weights[i]
        sum_w = F(sum_w + w)
        sum_x = F(sum_x + F(w * x[i]))
    if mn > 0:
        mn = F(0)
    if mx == mn:
        for i in range(n):
            L[i] = 0
        return F(0), F(-mn)
    iscale = F(F(nmax) / F(mx - mn))
    scale = F(F(1) / iscale)
    best_error = F(0)
    for i in range(n):
        l = rnd(F(iscale * F(x[i] - mn)))
        L[i] = max(0, min(nmax, l))
        diff = F(F(F(scale * F(L[i])) + mn) - x[i])
        diff = F(diff * diff)
        best_error = F(best_error + F(weights[i] * diff))
    if nstep < 1:
        return scale, F(-mn)
    Laux = [0] * n
    for is_ in range(nstep + 1):
        # `min` here is the *current best* min: the C overwrites it below.
        iscale = F(F(F(F(rmin) + F(F(rdelta) * F(is_))) + F(nmax)) / F(mx - mn))
        sum_l = F(0)
        sum_l2 = F(0)
        sum_xl = F(0)
        for i in range(n):
            l = rnd(F(iscale * F(x[i] - mn)))
            l = max(0, min(nmax, l))
            Laux[i] = l
            w = weights[i]
            sum_l = F(sum_l + F(w * F(l)))
            sum_l2 = F(sum_l2 + F(F(w * F(l)) * F(l)))
            sum_xl = F(sum_xl + F(F(w * F(l)) * x[i]))
        D = F(F(sum_w * sum_l2) - F(sum_l * sum_l))
        if D > 0:
            this_scale = F(F(F(sum_w * sum_xl) - F(sum_x * sum_l)) / D)
            this_min = F(F(F(sum_l2 * sum_x) - F(sum_l * sum_xl)) / D)
            if this_min > 0:
                this_min = F(0)
                this_scale = F(sum_xl / sum_l2)
            cur_error = F(0)
            for i in range(n):
                diff = F(F(F(this_scale * F(Laux[i])) + this_min) - x[i])
                diff = F(diff * diff)
                cur_error = F(cur_error + F(weights[i] * diff))
            if cur_error < best_error:
                for i in range(n):
                    L[i] = Laux[i]
                best_error = cur_error
                scale = this_scale
                mn = this_min
    return scale, F(-mn)


def get_scale_min_k4(j, q):
    """ggml-quants.c 889-896."""
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)


def _f16_bits(v):
    return int(np.array([F(v)], dtype=F).astype(np.float16).view(np.uint16)[0])


def _f16_to_f32(bits):
    return F(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def quantize_row_q4_K_ref(xs, rnd=nearest_int, trace=None, final_rnd=None, nstep=20):
    """ggml-quants.c 1467-1535. Returns the GGUF Q4_K bytes, 144 per 256.

    `final_rnd` and `nstep` exist only so the fixture checks can show that a
    fixture depends on them; the C has `nearest_int` and `20`.

    `trace`, when a list, receives one dict per super-block with the 6-bit
    scales and the per-sub-block `d`, so the fixture checks can see inside.
    """
    xs = [F(v) for v in np.asarray(xs, dtype=F).reshape(-1)]
    assert len(xs) % QK_K == 0
    out = bytearray()
    for i in range(len(xs) // QK_K):
        x = xs[i * QK_K : (i + 1) * QK_K]
        L = [0] * QK_K
        scales = [F(0)] * 8
        mins = [F(0)] * 8
        max_scale = F(0)
        max_min = F(0)
        for j in range(8):
            sum_x2 = F(0)
            for l in range(32):
                sum_x2 = F(sum_x2 + F(x[32 * j + l] * x[32 * j + l]))
            av_x = F(np.sqrt(F(sum_x2 / F(32))))
            weights = [F(av_x + F(abs(x[32 * j + l]))) for l in range(32)]
            Lj = [0] * 32
            scales[j], mins[j] = make_qkx2_quants(32, 15, x[32 * j : 32 * j + 32], weights, Lj, -1.0, 0.1, nstep, rnd)
            L[32 * j : 32 * j + 32] = Lj
            if scales[j] > max_scale:
                max_scale = scales[j]
            if mins[j] > max_min:
                max_min = mins[j]
        inv_scale = F(F(63) / max_scale) if max_scale > 0 else F(0)
        inv_min = F(F(63) / max_min) if max_min > 0 else F(0)
        sc = [0] * 12
        for j in range(8):
            ls = rnd(F(inv_scale * scales[j])) & 0xFF  # uint8_t ls = nearest_int(..)
            lm = rnd(F(inv_min * mins[j])) & 0xFF
            ls = min(63, ls)
            lm = min(63, lm)
            if j < 4:
                sc[j] = ls
                sc[j + 4] = lm
            else:
                sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
                sc[j - 4] |= (ls >> 4) << 6
                sc[j - 0] |= (lm >> 4) << 6
        d_bits = _f16_bits(F(max_scale / F(63)))
        dmin_bits = _f16_bits(F(max_min / F(63)))
        ds = []
        for j in range(8):
            s, m = get_scale_min_k4(j, sc)
            d = F(_f16_to_f32(d_bits) * F(s))
            ds.append(d)
            if not d:
                continue  # L keeps make_qkx2_quants' levels
            dm = F(_f16_to_f32(dmin_bits) * F(m))
            for ii in range(32):
                l = (final_rnd or rnd)(F(F(x[32 * j + ii] + dm) / d))
                L[32 * j + ii] = max(0, min(15, l))
        qs = bytearray()
        for j in range(0, QK_K, 64):
            for l in range(32):
                qs.append(L[j + l] | (L[j + l + 32] << 4))
        if trace is not None:
            trace.append({"sc": list(sc), "d": ds, "L": list(L)})
        out += struct.pack("<HH", d_bits, dmin_bits) + bytes(sc) + bytes(qs)
    return bytes(out)


# --- fixtures ---------------------------------------------------------------


def _fixtures():
    i = np.arange(QK_K)
    fx = {}
    fx["zeros"] = np.zeros((1, QK_K), F)
    fx["equal_negative"] = np.full((1, QK_K), -0.5, F)
    fx["equal_positive"] = np.full((1, QK_K), 0.75, F)
    fx["negative_only"] = (-(np.abs(np.sin(i * 0.37)) + 0.05)).astype(F).reshape(1, QK_K)
    fx["ties"] = ((i * 11 + i // 32) % 33).astype(F).reshape(1, QK_K)
    zs = np.sin(i * 1.3).astype(F)
    zs[:32] = (np.cos(i[:32] * 0.9) * 1e-3).astype(F)
    fx["zero_scale_sub"] = zs.reshape(1, QK_K)
    mix = np.sin(i * 0.61).astype(F)
    mix[32:64] = -np.abs(mix[32:64]) - 0.3
    mix[64:96] = np.abs(mix[64:96]) + 0.3
    fx["mixed_sign_subs"] = mix.reshape(1, QK_K)
    # One element placed so that `(x + dm) / d` in the final pass is exactly
    # 6.5 (found by iterating x = 6.5*d - dm to a fixed point): ties to even
    # gives 6, ties away gives 7. The sweep's ties are `ties`' job.
    ft = np.sin(i * 0.61).astype(F)
    ft[5] = np.array([0xBE0FF480], dtype=np.uint32).view(F)[0]
    fx["final_pass_tie"] = ft.reshape(1, QK_K)
    rng = np.random.default_rng(27)
    # Real-layer-like: a linear weight's scale, two rows of 1024 (8 super-blocks),
    # and a heavy-tailed row, where the outliers set each sub-block's range.
    fx["gaussian_2x1024"] = (rng.standard_normal((2, 1024)) * 0.02).astype(F)
    fx["student_t_1x512"] = (rng.standard_t(3, (1, 512)) * 0.02).astype(F)
    # A Gaussian super-block whose best candidate is the sweep's last step
    # (is = 20), so a 20-step sweep would miss it.
    fx["last_sweep_step"] = (np.random.default_rng(110).standard_normal((1, QK_K)) * 0.02).astype(F)
    return fx


_CACHE = {}


def _expected(name, x):
    if name not in _CACHE:
        trace = []
        _CACHE[name] = (quantize_row_q4_K_ref(x, trace=trace), trace)
    return _CACHE[name]


def _ours(x):
    t = _C._tensor_from_flat(x.reshape(-1).tolist(), list(x.shape), dtype=_C.float32)
    return _C._quantized_blob(_C._quantize(t, "q4_k"))


def _first_difference(got, want):
    for b in range(len(want) // BLOCK_BYTES):
        g = got[b * BLOCK_BYTES : (b + 1) * BLOCK_BYTES]
        w = want[b * BLOCK_BYTES : (b + 1) * BLOCK_BYTES]
        if g != w:
            off = [k for k in range(BLOCK_BYTES) if g[k] != w[k]]
            return f"super-block {b}: {len(off)} bytes differ, first at {off[:6]} (0-3 d/dmin, 4-15 scales, 16+ nibbles)"
    return "lengths differ"


def _check(names):
    fx = _fixtures()
    for name in names:
        x = fx[name]
        want, _ = _expected(name, x)
        got = _ours(x)
        assert len(got) == len(want) == x.size // QK_K * BLOCK_BYTES, (name, len(got), len(want))
        assert got == want, f"{name}: not llama.cpp's Q4_K bytes; {_first_difference(got, want)}"


# --- 1-2: the writer ---------------------------------------------------------


EDGE = (
    "zeros", "equal_negative", "equal_positive", "negative_only", "ties", "final_pass_tie",
    "zero_scale_sub", "mixed_sign_subs",
)


def test_q4_k_writer_produces_llama_cpp_bytes_on_edge_blocks():
    """Zeros, all-equal blocks of either sign, a negative-only block, a block
    whose bytes depend on how an exact .5 is rounded, a sub-block whose 6-bit
    scale rounds to zero, and sub-blocks of opposite sign in one super-block."""
    _check(EDGE)


def test_q4_k_writer_produces_llama_cpp_bytes_on_layer_like_data():
    """Gaussian rows at a linear layer's scale, a heavy-tailed row, and a
    Gaussian block whose optimum is the sweep's last step."""
    _check(("gaussian_2x1024", "student_t_1x512", "last_sweep_step"))


# --- 3-4: the fixtures and the reference ------------------------------------


def test_the_transcriptions_nearest_int_is_ties_to_even():
    """The bit trick, checked on the ties it exists to decide."""
    assert [nearest_int(v) for v in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.49, 7.0)] == [0, 2, 2, 0, -2, -2, 3, 7]
    assert [nearest_int_away(v) for v in (0.5, 1.5, 2.5, -0.5, -2.5)] == [1, 2, 3, -1, -3]


def test_every_fixture_still_exercises_what_it_was_built_for():
    """The check on the fixtures. A fixture regenerated into something easier
    must fail here, not quietly stop testing its edge."""
    fx = _fixtures()
    exp = {name: _expected(name, x) for name, x in fx.items()}

    def head(name, b=0):
        blob = exp[name][0]
        return struct.unpack("<ee", blob[b * BLOCK_BYTES : b * BLOCK_BYTES + 4])

    assert exp["zeros"][0] == bytes(BLOCK_BYTES)
    # llama.cpp keeps `the_min = -min` for a flat negative sub-block; candle's
    # old writer returned (0, 0) and so wrote dmin = 0.
    d, dmin = head("equal_negative")
    assert d == 0.0 and dmin > 0.0, (d, dmin)
    # A flat *positive* block is not flat to the search: min is clamped to 0.
    d, dmin = head("equal_positive")
    assert d > 0.0, d
    assert float(fx["negative_only"].max()) < 0.0
    # The tie must decide at least one byte: rounding ties away from zero
    # instead of to even must change the blob.
    assert quantize_row_q4_K_ref(fx["ties"], rnd=nearest_int_away) != exp["ties"][0]
    assert quantize_row_q4_K_ref(fx["final_pass_tie"], final_rnd=nearest_int_away) != exp["final_pass_tie"][0]
    assert quantize_row_q4_K_ref(fx["last_sweep_step"], nstep=19) != exp["last_sweep_step"][0]
    # Sub-block 0 has a 6-bit scale of 0 (so d == 0) yet nonzero levels, which
    # only survive if `if (!d) continue;` keeps make_qkx2_quants' L.
    tr = exp["zero_scale_sub"][1][0]
    assert tr["d"][0] == 0.0 and any(tr["L"][:32]), (tr["d"][0], tr["L"][:32])
    assert all(tr["d"][j] != 0.0 for j in range(1, 8))
    mix = fx["mixed_sign_subs"][0]
    assert mix[32:64].max() < 0 < mix[64:96].min()
    assert fx["gaussian_2x1024"].size // QK_K == 8


def test_the_measurement_reference_agrees_with_the_transcription():
    """`scripts/q4quality/q4_gguf_ref.quantize_q4_k_llamacpp` is the vectorised
    llama.cpp writer `measure.py` grades the #16 reconstruction error with. It
    must give the scalar transcription's bytes on every fixture, or #16's
    "llama.cpp" column is not llama.cpp's (its first version missed the C's
    overwritten `min` and its `if (!d)` levels, and disagreed with a compiled
    llama.cpp on 366 of 400 Gaussian super-blocks)."""
    for name, x in _fixtures().items():
        want, _ = _expected(name, x)
        got = q4_gguf_ref.quantize_q4_k_llamacpp(x)
        assert got == want, f"{name}: {_first_difference(got, want)}"


def _main():
    return 1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")],
        suite="test_q4kwriter",
    ) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
