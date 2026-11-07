"""The CPU quantised matmul over many rows is the one-row matmul, bit for bit (issue #177).

candle's `k_quants::matmul` ran one barrier-synchronised parallel region per
activation row, each one streaming the whole weight matrix. The tiled form
(one parallel region, each worker walking every row for its own column range)
reorders *which (row, column) pair is computed when*, never *how* one is: each
output element is still one `vec_dot` over the same quantised blocks. So the
claim is exact, not a tolerance:

* *rows*: `_quantized_linear(x)` for an `m`-row `x` equals, bit for bit, the
  `m` one-row calls `_quantized_linear(x[i:i+1])`, for q4_0, q8_0 and q4_k, at
  row counts that do and do not fill a tile and column counts that leave a
  1-, 2- and 3-column tail after the 4-column groups.
* *lossless*: on operands whose quantisation is exact (scale 1, integer
  weights, activation blocks with absmax 127), the batched result equals a
  float64 integer matmul exactly. A tiling that dropped, duplicated or
  misplaced a row or column fails here even if it did so consistently.

What this cannot find: speed. Whether the tiled form is faster is the bench's
claim (tests/bench/bench_llama_cpp.py), measured alone.
"""

import os
import random
import struct

import _C
import _skip

_FORMATS = ("q4_0", "q8_0", "q4_k")
# rows: 1 (gemv), below one tile, one tile + 1, a prefill that is not a tile multiple
_ROWS = (2, 5, 17, 130)
# out features: tails of 0, 1, 2 and 3 columns after the 4-column groups
_COLS = (8, 9, 10, 11)


def _dense(values, shape):
    return _C._tensor_from_flat(list(values), list(shape), dtype=_C.float32)


def _flat(t):
    out, stack = [], [t.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _bits(values):
    return struct.pack(f"<{len(values)}f", *values)


def _no_dequantize_all():
    for var in ("CANDLE_DEQUANTIZE_ALL", "CANDLE_DEQUANTIZE_ALL_F16"):
        assert os.environ.get(var, "") in ("", "0"), (
            f"{var} is set: candle would multiply by dequantised weights and this "
            "file would measure a dense matmul. Refusing rather than passing."
        )


def _check_rows(fmt):
    _no_dequantize_all()
    rnd = random.Random(177)
    k = 512  # a multiple of every format's block (q4_k: 256)
    checked = 0
    for n in _COLS:
        w = [rnd.gauss(0.0, 0.05) for _ in range(n * k)]
        qw = _C._quantize(_dense(w, [n, k]), fmt)
        for m in _ROWS:
            x = [rnd.gauss(0.0, 1.3) for _ in range(m * k)]
            batched = _flat(_C._quantized_linear(_dense(x, [m, k]), qw, None))
            rows = []
            for i in range(m):
                rows += _flat(_C._quantized_linear(_dense(x[i * k:(i + 1) * k], [1, k]), qw, None))
            if _bits(batched) != _bits(rows):
                bad = [j for j in range(m * n) if _bits([batched[j]]) != _bits([rows[j]])]
                raise AssertionError(
                    f"{fmt} m={m} n={n}: {len(bad)} of {m * n} elements differ between the "
                    f"batched matmul and one-row calls; first at row {bad[0] // n}, "
                    f"col {bad[0] % n}: {batched[bad[0]]!r} vs {rows[bad[0]]!r}"
                )
            checked += m * n
    print(f"    qgemm: {fmt}: batched == one-row calls bit for bit, {checked} elements "
          f"over m in {_ROWS} x n in {_COLS}")


def test_q4_0_batched_rows_equal_one_row_calls_bit_for_bit():
    _check_rows("q4_0")


def test_q8_0_batched_rows_equal_one_row_calls_bit_for_bit():
    _check_rows("q8_0")


def test_q4_k_batched_rows_equal_one_row_calls_bit_for_bit():
    _check_rows("q4_k")


def test_lossless_operands_give_the_exact_integer_matmul():
    _no_dequantize_all()
    rnd = random.Random(1770)
    k = 64
    for fmt in ("q4_0", "q8_0"):
        for n in _COLS:
            blob = bytearray()
            dense = []
            for _ in range(n):
                row = []
                for _ in range(k // 32):
                    if fmt == "q8_0":
                        qs = [rnd.randint(-127, 127) for _ in range(32)]
                        blob += struct.pack("<e", 1.0) + struct.pack("<32b", *qs)
                        row += qs
                    else:
                        nib = [rnd.randint(0, 15) for _ in range(32)]
                        blob += struct.pack("<e", 1.0) + bytes(
                            (nib[j] & 15) | ((nib[j + 16] & 15) << 4) for j in range(16))
                        row += [v - 8 for v in nib]
                dense.append(row)
            qw = _C._quantized_from_blob(bytes(blob), [n, k], fmt)
            for m in _ROWS:
                xs = []
                for _ in range(m):
                    r = [rnd.randint(-126, 126) for _ in range(k)]
                    r[0], r[32] = 127, -127
                    xs.append(r)
                got = _flat(_C._quantized_linear(_dense([float(v) for r in xs for v in r], [m, k]), qw, None))
                want = [float(sum(a * b for a, b in zip(xs[i], dense[j]))) for i in range(m) for j in range(n)]
                assert got == want, (
                    f"{fmt} m={m} n={n}: the lossless matmul is not exact; first difference at "
                    f"{next(j for j in range(m * n) if got[j] != want[j])}"
                )
    print(f"    qgemm: q4_0 and q8_0 lossless operands give the exact integer matmul "
          f"over m in {_ROWS} x n in {_COLS}")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_qgemm")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
