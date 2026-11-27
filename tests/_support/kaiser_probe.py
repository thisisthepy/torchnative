"""Which arithmetic upstream's `kaiser_window` uses on this machine (#447).

Upstream's kernel computes, at `float` for every dtype but `float64`,

    calc_i0(beta * sqrt(abs(1 - pow((i - alpha) / alpha, 2)))) / calc_i0(beta)

and the shim reproduces it with `1 - r*r` rounded twice, which agrees on macOS.
On the aarch64 Linux runner every `periodic=False` window disagreed in the
last bits (run 45934950589) while `periodic=True` agreed. One explanation fits:
gcc folds `pow(r, 2)` to `r * r` before its default `-ffp-contract=fast`
contracts, so `1 - r*r` becomes one `fma(-r, r, 1)`; with an integral `alpha`
(`periodic=True`) `r*r` is exact and the two agree. This probe asks upstream
and rebuilds each window under both hypotheses in exact float32 arithmetic
(an exactly rounded fma, the C library's `expf`), and prints how many elements
each one reproduces. It names a hypothesis only when one matches every element.

    python tests/_support/kaiser_probe.py     # needs upstream torch; prints one line

`run.sh` prints it in the `GATE ORACLE` header. Measurement only.
"""

import ctypes
import ctypes.util
import struct
from fractions import Fraction

# Cephes' i0 coefficients, as `ATen/native/Math.h` has them (the shim's I0E_A/I0E_B).
I0E_A = [
    -4.41534164647933937950e-18, 3.33079451882223809783e-17, -2.43127984654795469359e-16,
    1.71539128555513303061e-15, -1.16853328779934516808e-14, 7.67618549860493561688e-14,
    -4.85644678311192946090e-13, 2.95505266312963983461e-12, -1.72682629144155570723e-11,
    9.67580903537323691224e-11, -5.18979560163526290666e-10, 2.65982372468238665035e-9,
    -1.30002500998624804212e-8, 6.04699502254191894932e-8, -2.67079385394061173391e-7,
    1.11738753912010371815e-6, -4.41673835845875056359e-6, 1.64484480707288970893e-5,
    -5.75419501008210370398e-5, 1.88502885095841655729e-4, -5.76375574538582365885e-4,
    1.63947561694133579842e-3, -4.32430999505057594430e-3, 1.05464603945949983183e-2,
    -2.37374148058994688156e-2, 4.93052842396707084878e-2, -9.49010970480476444210e-2,
    1.71620901522208775349e-1, -3.04682672343198398683e-1, 6.76795274409476084995e-1,
]
I0E_B = [
    -7.23318048787475395456e-18, -4.83050448594418207126e-18, 4.46562142029675999901e-17,
    3.46122286769746109310e-17, -2.82762398051658348494e-16, -3.42548561967721913462e-16,
    1.77256013305652638360e-15, 3.81168066935262242075e-15, -9.55484669882830764870e-15,
    -4.15056934728722208663e-14, 1.54008621752140982691e-14, 3.85277838274214270114e-13,
    7.18012445138366623367e-13, -1.79417853150680611778e-12, -1.32158118404477131188e-11,
    -3.14991652796324136454e-11, 1.18891471078464383424e-11, 4.94060238822496958910e-10,
    3.39623202570838634515e-9, 2.26666899049817806459e-8, 2.04891858946906374183e-7,
    2.89137052083475648297e-6, 6.88975834691682398426e-5, 3.36911647825569408990e-3,
    8.04490411014108831608e-1,
]

_libm = ctypes.CDLL(ctypes.util.find_library("m") or None)
_libm.expf.restype = ctypes.c_float
_libm.expf.argtypes = [ctypes.c_float]


def f32(x):
    """Round a float (already representable at f64) to float32, once."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_exact(q):
    """Round an exact Fraction to float32, to nearest with ties to even."""
    guess = f32(float(q))
    bits = struct.unpack("<I", struct.pack("<f", guess))[0]
    candidates = [guess]
    for step in (-1, 1):
        b = bits + step if guess >= 0 else bits - step
        if 0 <= b < 0x7f800000 or 0x80000000 <= b < 0xff800000:
            candidates.append(struct.unpack("<f", struct.pack("<I", b))[0])
    best = min(candidates, key=lambda c: (abs(Fraction(c) - q),
                                          struct.unpack("<I", struct.pack("<f", c))[0] & 1))
    return best


def fma32(a, b, c):
    return f32_exact(Fraction(a) * Fraction(b) + Fraction(c))


def chbevl(x, coeffs):
    # clang and gcc both contract `x * b1 - b2` (the shim's measured model).
    b0, b1, b2 = f32(coeffs[0]), 0.0, 0.0
    for c in coeffs[1:]:
        b2, b1 = b1, b0
        b0 = f32(fma32(x, b1, -b2) + f32(c))
    return f32(0.5 * f32(b0 - b2))


def calc_i0(v):
    x = abs(v)
    if x <= 8.0:
        y = f32(f32(x / 2.0) - 2.0)
        return f32(_libm.expf(x) * chbevl(y, I0E_A))
    t = f32(f32(32.0 / x) - 2.0)
    return f32(f32(_libm.expf(x) * chbevl(t, I0E_B)) / f32(x ** 0.5))


def window(n, periodic, beta, fused):
    built = n + 1 if periodic else n
    alpha = f32((built - 1) / 2.0)
    beta32 = f32(beta)
    den = calc_i0(beta32)
    out = []
    for i in range(built):
        r = f32(f32(f32(i) - alpha) / alpha)
        inside = fma32(-r, r, 1.0) if fused else f32(1.0 - f32(r * r))
        arg = f32(beta32 * f32(abs(inside) ** 0.5))
        out.append(f32(calc_i0(arg) / den))
    return out[:n]


def main():
    import torch

    cases = [(16, False, 12.0), (64, False, 12.0), (16, False, 5.0), (16, True, 12.0), (64, True, 12.0)]
    parts, verdict = [], set()
    for n, periodic, beta in cases:
        got = torch.kaiser_window(n, periodic, beta).tolist()
        plain = sum(a == b for a, b in zip(got, window(n, periodic, beta, False)))
        fused = sum(a == b for a, b in zip(got, window(n, periodic, beta, True)))
        parts.append(f"({n},{periodic},{beta}) unfused {plain}/{n} fused {fused}/{n}")
        if plain == n and fused != n:
            verdict.add("unfused")
        elif fused == n and plain != n:
            verdict.add("fused")
        elif plain != n and fused != n:
            verdict.add("neither")
    name = verdict.pop() if len(verdict) == 1 else ("mixed: " + ",".join(sorted(verdict)) if verdict else "both")
    print(f"GATE ORACLE kaiser: 1 - r*r is {name} | " + "; ".join(parts))


if __name__ == "__main__":
    main()
