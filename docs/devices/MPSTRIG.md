# Transcendentals on mps: sin cos exp log tanh sigmoid erf (issue #101)

Measured 2026-10-25 on the Apple Silicon Mac mini, `uptime` at the start:
`10:26 up 32 days, 7:03, 1 user, load averages: 6.37 7.79 8.30`. The load was
not quiet; the numbers below are errors, not timings, so load does not move them.
Harness and grades: `tests/devices/mps/test_mpstrig.py` (93 cells: 7 ops x
3 dtypes x 2 to 4 shapes and value ranges per op). Fast math ON is the build's
default; OFF is `CANDLE_METAL_ENABLE_FAST_MATH=0` (Relaxed math mode with precise
floating-point functions, `kernel.rs` `get_compile_options`).

## Method

* Truth: upstream torch on cpu in float64, on the dtype-rounded input.
* Compared: the shim on mps, upstream on mps, upstream on cpu, each in the
  dtype under test, in separate subprocesses (each asserts which torch it got).
* Error in ulps of the dtype at the truth (non-finite results must match by
  class), plus max absolute error.
* Tolerance (AGENTS.md section 16): per dtype, p90 of upstream-cpu's f32/f16/bf16
  error against its float64 result, floored at 8 ulp. The p90 came out 0.53
  (float32), 0.43 (float16), 0.44 (bfloat16), so the tolerance is the 8 ulp floor
  for all three.
* Ranges: `unit` [-1,1], `mid` [-10,10], `wide` [-80,80], `large` [-1e4,1e4]
  (sin/cos only), for log: (1e-3,1e3), (1e-6,1e-3) and [-50,0] (nan/-inf class).
  Shapes: 3, 4096, 64x64, 37x129.

## Result: worst cell per (dtype, op), max ulp and (max abs)

| dtype | op | upstream cpu | upstream mps | shim, fast math ON | shim, fast math OFF |
|---|---|---|---|---|---|
| float32 | sin | 0.71 | 2.75 | **1.6e6 (7.6e-4)** | 2.75 |
| float32 | cos | 0.75 | 2.55 | **1.3e7 (7.5e-4)** | 2.55 |
| float32 | exp | 0.90 | 1.15 | **58.7 (rel 3.6e-6)** | 1.15 |
| float32 | log | 0.55 | 0.77 | **8.1e3 (1.4e-6)** | 0.77 |
| float32 | tanh | 1.00 | 1.99 | 1.99 | 1.99 |
| float32 | sigmoid | 1.85 | 2.27 | **58.7** | 2.27 |
| float32 | erf | 2.0e3 (4.3e-7) | 0.93 | **4.1e3 (5.4e-7)** | **2.0e3 (4.1e-7)** |
| float16 | sin | 0.50 | 0.50 | **4.9e3 (1.0)** | **4.9e3 (1.0)** |
| float16 | cos | 0.50 | 0.50 | **4.7e3 (1.0)** | **4.7e3 (1.0)** |
| float16 | exp log tanh sigmoid erf | <= 0.89 | <= 0.5 | <= 0.65 | <= 0.89 |
| bfloat16 | sin cos | 0.5 | 0.5 | 1.9 | 0.5 |
| bfloat16 | exp log tanh sigmoid erf | <= 0.5 | <= 0.5 | <= 0.5 | <= 0.5 |

Bold means diverges (above 8 ulp). Counts of the 93 cells: fast math ON, 71
agree and 22 diverge; fast math OFF, 82 agree and 11 diverge. The 22 and the 11
are named in `DIVERGES` and `OFF_DIVERGES` in the test, with a floor under each
measured error; the test fails if one starts to agree or the sets change.

Notes on reading the table:

* ulp error near zero is harsh: float32 sin on [-1,1] has an absolute error of
  2.8e-7 (ON) but 1.5e3 ulp, because the outputs near 0 are tiny. The absolute
  error on `wide` is 5.8e-6 and on `large` 7.6e-4 (ON). #13's 1.0e-6 against the
  9.5e-7 floor was a range of this size; the conclusion there (diverges) holds
  for ON and is reversed by OFF. matrix.md section 6 graded sin/cos agreeing
  from a shape that hides the error; it should cite this record.
* The oracle matters little here. Upstream mps is within 3 ulp of truth for
  every cell; upstream cpu is within 2 ulp except erf. Both
  upstream backends agree with truth where the shim (ON) does not, so
  "upstream mps as the oracle" does not rescue the shim.
* float32 erf: upstream cpu itself has 4.3e-7 absolute error (2.0e3 ulp near
  zero), upstream mps does not (5.5e-8). The shim's erf (candle's polynomial in
  `utils.metal`) has the same class of error as cpu with either math mode. The
  switch does not fix it.
* float16 sin/cos: error up to 1.0 absolute at |x| up to 1e4, 3e-2 at |x| <= 80,
  9e-4 even at |x| <= 1, unchanged by the math mode. The half `sin`/`cos` kernel
  loses accuracy beyond the switch (not investigated further; cause not
  established by this round).
* bfloat16 sin/cos: 1.9 ulp ON, 0.5 OFF; both agree within 8.
* exp float16/bfloat16 overflow to inf in step with upstream (class match).

## Verdict

* Fast math OFF fixes every float32 cell of sin, cos, exp, log and sigmoid and
  bfloat16 sin/cos (all become within 3 ulp of truth, like upstream mps). It
  does not fix float32 erf or float16 sin/cos.
* Recommendation (the decision is the maintainer's; `kernel.rs` was not changed):
  turn fast math OFF for the transcendental kernels at least (a global
  `CANDLE_METAL_ENABLE_FAST_MATH=0` default, or precise variants for these
  ops), and fix the two remaining kernels separately (erf, half sin/cos), for
  example by computing in float32 on the device and rounding to half. Cost of OFF
  was not measured here (no timing, machine loaded).
* Recorded by name until then: the `DIVERGES` set in `test_mpstrig.py`.

## Closed (2026-10-25)

Fast math is off by default in the candle fork (PR #137; `CANDLE_METAL_ENABLE_FAST_MATH=1` turns it on).
The 11 cells that still diverged are fixed in `candle-metal-kernels/src/metal_src/unary.metal`:

- **float32 erf**: A&S 7.1.26 (1.5e-7 absolute) is replaced by the Maclaurin series below
  |x| = 0.5 and `1 - erfc` with Numerical Recipes' Chebyshev `erfcc` above it. Worst cell
  `float32|erf|unit|64x64` is now 2.4 ulp (was 2.0e3; upstream mps 0.93, upstream cpu 2.0e3).
- **float16 / bfloat16 sin, cos**: computed in float32 and rounded once, instead of Metal's
  half-precision `sin`/`cos`.

With the default build all 93 cells agree within the derived tolerance (8 ulp).
`gelu_erf` uses the same `erf`, so it moves with it.
