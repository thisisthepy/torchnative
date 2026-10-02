"""Q4 against llama.cpp's own answers, and the quantised matmul against its exact value.

Issue #16. `docs/graph/QUANT2.md` §2 built the verification axis for block
quantisation on `ggml_ref.py`, and that file states the axis' blind spot in
its own docstring: it was transcribed from the format by the same reading that
checked candle, so **a misreading shared by both would pass**, and "a GGUF
file written by llama.cpp compared against `_quantized_blob` would close it".
This suite closes the half of that which can be closed without a model:

  1. **llama.cpp's bytes and bits, frozen.** `tools/q4quality/golden_gguf_py.json`
     holds what llama.cpp's own Python codec (`gguf-py`) writes and reads for
     ten blocks, each built to expose one misreading (the generator's
     docstring lists them). Nothing in this repository computed those
     numbers. Candle must reproduce them **byte for byte and bit for bit**;
     there is no tolerance anywhere in tests 1-4.

  2. **The matmul, against its exact value.** A Q4 matmul on CPU is not
     `x @ dequant(W).T`: candle (like llama.cpp) first quantises each row of
     `x` to Q8_0 or Q8_K, takes integer block dots, and scales in `float32`.
     So the exact answer is a finite sum of dyadic rationals computable from
     the two blobs, and only the `float32` accumulation is left to a bound --
     Higham's `gamma_K * sum|t|`, with `K` *counted* from candle's aarch64
     kernels (`tools/q4quality/gguf_ref.py`, the block above `gamma`). The
     bound is not a tolerance anybody picked, and each test also proves it is
     **tight enough to refuse** the nearest wrong implementation: a matmul
     that skipped activation quantisation must land outside it.

**What this cannot see** (AGENTS.md §17.4):

  * Model-level quality. That is `tools/q4quality/measure.py`, which needs a
    quiet machine and a real model; the gate never loads one.
  * The Q4_K *writer*. gguf-py has no k-quant writer, so there is no
    llama.cpp answer to freeze, and candle's writer is **not** llama.cpp's
    (`make_qkx1_quants` against llama.cpp's `make_qkx2_quants`). Both produce
    valid blobs; they produce different ones. Only the reader is pinned here.
  * An x86 build. `K` was counted on aarch64 NEON kernels.

Nullification plan -- what to break, and which test must go red:

  * candle Q4_0 reader pairs nibbles `j, j+1`        -> test 2 (and 3 via the dot)
  * candle Q4_0 writer scale `amax/8` (sign lost)      -> test 1
  * candle Q4_K `get_scale_min_k4` without the split    -> test 3
  * one byte of `golden_gguf_py.json` flipped           -> tests 1-3 (and 5 if a
    coverage property is destroyed)
  * `CANDLE_DEQUANTIZE_ALL=1` (no activation quantisation, dense matmul over
    dequantised weights) -> tests 6 and 7 -- runnable without a rebuild
  * `accumulation_k` returning `100 * K`                 -> tests 6 and 7, through
    their own "the bound must refuse the wrong matmul" assertion
"""

import json
import os
import struct
import sys

import numpy as np

import _C
import _skip

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "q4quality"))
sys.path.insert(0, HERE)
import ggml_ref  # noqa: E402
import gguf_ref  # noqa: E402

GOLDEN = os.path.join(REPO, "tools", "q4quality", "golden_gguf_py.json")


def _golden():
    with open(GOLDEN) as fh:
        doc = json.load(fh)
    return {(c["format"], c["name"]): c for c in doc["cases"]}, doc


def _f32_list(hexstr):
    raw = bytes.fromhex(hexstr)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _bits(values):
    return struct.pack(f"<{len(values)}f", *values)


def _flat(t):
    out, stack = [], [t.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _dense(values, shape):
    return _C._tensor_from_flat(list(values), list(shape), dtype=_C.float32)


# --- 1-4: the format, against llama.cpp -------------------------------------


def test_q4_0_and_q8_0_writers_produce_llama_cpp_bytes():
    """Candle's writer, on llama.cpp's inputs, gives llama.cpp's bytes.

    `ggml_ref` already checks this against a transcription; this checks it
    against gguf-py's output, which was produced outside this repository.
    """
    cases, _ = _golden()
    checked = 0
    for (fmt, name), c in sorted(cases.items()):
        if "input_f32" not in c:
            continue
        q = _C._quantize(_dense(_f32_list(c["input_f32"]), c["shape"]), fmt)
        got = _C._quantized_blob(q)
        want = bytes.fromhex(c["blob"])
        assert got == want, (fmt, name, [i for i, (a, b) in enumerate(zip(got, want)) if a != b][:8])
        checked += 1
    assert checked == 9, f"expected 9 writer cases, found {checked} -- the golden file was trimmed"


def test_q4_0_and_q8_0_readers_produce_llama_cpp_bits():
    """Candle's reader, handed llama.cpp's bytes, gives llama.cpp's floats.

    The blob goes in through `_quantized_from_blob`, so the reader is checked
    on bytes candle did not write -- the half of the axis that matters for the
    GGUF reader (#17), which will hand candle exactly such bytes.
    """
    cases, _ = _golden()
    for (fmt, name), c in sorted(cases.items()):
        if fmt not in ("q4_0", "q8_0"):
            continue
        q = _C._quantized_from_blob(bytes.fromhex(c["blob"]), c["shape"], fmt)
        got = _bits(_flat(_C._dequantize(q)))
        want = bytes.fromhex(c["dequant_f32"])
        assert got == want, (fmt, name)


def test_q4_k_reader_on_a_blob_no_writer_produced_gives_llama_cpp_bits():
    """The Q4_K reader on random scale bytes, against gguf-py's `Q4_K`.

    Random scale bytes are the point: any real writer leaves patterns (a
    sub-block whose 6-bit scale never reaches the high two bits), and a reader
    that mishandles the split high bits of sub-blocks 4..7 passes on those.
    """
    cases, _ = _golden()
    c = cases[("q4_k", "foreign")]
    blob = bytes.fromhex(c["blob"])
    q = _C._quantized_from_blob(blob, c["shape"], "q4_k")
    got = _bits(_flat(_C._dequantize(q)))
    assert got == bytes.fromhex(c["dequant_f32"])


def test_this_repositorys_transcription_agrees_with_llama_cpp_on_every_case():
    """`ggml_ref.py` against gguf-py, so the existing axis inherits the anchor.

    Every test in `test_shim.py`'s quantisation section is "candle ==
    ggml_ref". This makes it "candle == ggml_ref == llama.cpp" on these
    blocks, without touching those tests.
    """
    cases, _ = _golden()
    for (fmt, name), c in sorted(cases.items()):
        n = 1
        for d in c["shape"]:
            n *= d
        blob = bytes.fromhex(c["blob"])
        assert _bits(ggml_ref.DEQUANTIZERS[fmt](blob, n)) == bytes.fromhex(c["dequant_f32"]), (fmt, name)
        if "input_f32" in c:
            assert ggml_ref.QUANTIZERS[fmt](_f32_list(c["input_f32"])) == blob, (fmt, name)


def test_every_golden_case_still_exercises_what_it_was_built_for():
    """The check on the fixture. A case trimmed or regenerated into something
    easier must fail here, not quietly stop testing its edge."""
    cases, doc = _golden()
    assert len(doc["provenance"]["gguf_quants_sha256"]) == 64

    def d_of(c, i=0):
        off = i * gguf_ref.TYPE_SIZE[c["format"]]
        return struct.unpack("<e", bytes.fromhex(c["blob"])[off : off + 2])[0]

    def raw_d_bits(c, i=0):
        off = i * gguf_ref.TYPE_SIZE[c["format"]]
        return struct.unpack("<H", bytes.fromhex(c["blob"])[off : off + 2])[0]

    assert d_of(cases[("q4_0", "pos_extreme")]) < 0, "positive extreme must give a negative scale"
    assert d_of(cases[("q4_0", "neg_extreme")]) > 0
    zero = cases[("q4_0", "zero")]
    assert d_of(zero) == 0.0 and set(bytes.fromhex(zero["blob"])[2:]) == {0x88}
    opp = bytes.fromhex(cases[("q4_0", "opp_sign")]["blob"])[2:]
    nib = [b & 0xF for b in opp] + [b >> 4 for b in opp]
    assert nib[3] == 0 and nib[20] == 15, (nib[3], nib[20])
    assert (raw_d_bits(cases[("q4_0", "subnormal_d")]) & 0x7C00) == 0, "scale must be an f16 subnormal"
    assert cases[("q4_0", "two_rows")]["shape"] == [2, 64]
    ties = _f32_list(cases[("q8_0", "ties")]["input_f32"])
    assert max(abs(v) for v in ties) == 127.0 and 2.5 in ties and -126.5 in ties
    k = bytes.fromhex(cases[("q4_k", "foreign")]["blob"])
    highs = [k[i * 144 + 4 + j] >> 6 for i in range(3) for j in range(8)]
    assert len(k) == 3 * 144 and any(highs), "no high scale bits set -> the split branch is untested"


# --- 6-7: the quantised matmul against its exact value ----------------------


def _tie_free_gauss(rows, cols, seed, fmt):
    """Activations on which every reference quantiser has one answer.

    Exact .5 ties (half-even vs half-away) and opposite-sign maxima (first-wins
    vs lane-parallel max) are where llama.cpp's own `_ref` and SIMD paths
    disagree with each other; a fixture with either would grade a convention,
    not candle. Found ones are refused rather than nudged silently.
    """
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((rows, cols)) * 1.7).astype(np.float32)
    if fmt == "q4_0":
        assert gguf_ref.q8_0_ties(x) == 0 and gguf_ref.signed_max_ties(x, 32) == 0
    else:
        assert gguf_ref.q8_k_ties(x) == 0 and gguf_ref.signed_max_ties(x, 256) == 0
    return x


def _check_linear(fmt, rows, out_features, in_features, seed):
    w = (np.random.default_rng(seed + 1).standard_normal((out_features, in_features)) * 0.05).astype(np.float32)
    x = _tie_free_gauss(rows, in_features, seed, fmt)
    qw = _C._quantize(_dense(w.reshape(-1).tolist(), w.shape), fmt)
    blob = _C._quantized_blob(qw)
    got = np.array(_flat(_C._quantized_linear(_dense(x.reshape(-1).tolist(), x.shape), qw, None)))
    got = got.reshape(rows, out_features)

    if fmt == "q4_0":
        xq = ggml_ref.quantize_q8_0(x.reshape(-1).tolist())
        exact, abs_sum, ref_err = gguf_ref.exact_linear_q4_0(xq, blob, rows, out_features, in_features)
    else:
        xd, xq = gguf_ref.quantize_q8_k_ref(x)
        exact, abs_sum, ref_err = gguf_ref.exact_linear_q4_k(
            xd.reshape(rows, -1), xq.reshape(rows, -1, 256), blob, rows, out_features, in_features
        )
    bound = gguf_ref.linear_bound(fmt, in_features, abs_sum, ref_err)
    err = np.abs(got - exact)
    worst = float(np.max(err / bound))
    assert worst <= 1.0, (
        f"{fmt} {rows}x{in_features} @ {out_features}: |ours - exact| reaches {worst:.3g}x the derived "
        f"bound (K={gguf_ref.accumulation_k(fmt, in_features)})"
    )

    # The bound must refuse the nearest wrong matmul: same weights, but the
    # activation left in float32 (what CANDLE_DEQUANTIZE_ALL does).
    wdq = np.array(_flat(_C._dequantize(qw)), dtype=np.float64).reshape(out_features, in_features)
    no_act_quant = x.astype(np.float64) @ wdq.T
    refused = float(np.max(np.abs(no_act_quant - exact) / bound))
    assert refused > 1.0, (
        f"{fmt}: a matmul without activation quantisation sits inside the bound ({refused:.3g}x) -- "
        "the bound is too loose to tell the two apart"
    )
    return worst, refused


def test_q4_0_linear_is_within_the_counted_float32_bound_of_its_exact_value():
    """Q4_0 x Q8_0, three shapes: a gemv row, a short prefill, a wide k.

    One shape does not speak for a column (AGENTS.md §16), so the row count,
    the output count and the number of blocks all vary.
    """
    for rows, out_f, in_f, seed in ((1, 8, 64, 11), (3, 12, 512, 12), (2, 5, 1024, 13)):
        _check_linear("q4_0", rows, out_f, in_f, seed)


def test_q4_k_linear_is_within_the_counted_float32_bound_of_its_exact_value():
    """Q4_K x Q8_K on both of candle's aarch64 kernels.

    `out_features % 8 == 0` takes the repacked `vec_dot_8_q4k_q8k` path when
    the build has `+dotprod`; 12 does not and takes `vec_dot_q4k_q8k`. Both
    shapes run whichever the build has, so the bound is checked against what
    actually executed.
    """
    for rows, out_f, in_f, seed in ((1, 16, 256, 21), (3, 12, 512, 22), (2, 8, 1024, 23)):
        _check_linear("q4_k", rows, out_f, in_f, seed)


def test_the_bound_is_counted_and_not_a_free_parameter():
    """`K` must equal the count written beside it, and the bound must be the
    formula -- so raising either to make a red test green is itself red."""
    assert gguf_ref.accumulation_k("q4_0", 1024) == 1024 // 32 + 3
    assert gguf_ref.accumulation_k("q4_k", 1024) == 9 * (1024 // 256) + 3
    k = gguf_ref.accumulation_k("q4_0", 512)
    assert gguf_ref.gamma(k) == k * 2.0**-24 / (1 - k * 2.0**-24)
    one = np.ones(1)
    assert gguf_ref.linear_bound("q4_0", 512, one, 0 * one)[0] == gguf_ref.gamma(k)


def _main():
    if os.environ.get("CANDLE_DEQUANTIZE_ALL", "") not in ("", "0"):
        # Not a skip: the variable changes what candle computes, and the
        # matmul tests exist to notice. Say so, then let them run red.
        print("note: CANDLE_DEQUANTIZE_ALL is set; the matmul tests are expected to fail")
    return 1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")],
        suite="test_q4ref",
    ) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
