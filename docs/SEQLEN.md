# `float32` prefill vs upstream as a function of sequence length

**One-line result.** The gap is not one thing. Fitting the measured curve gives

```
gap(S)  =  0.50 · S  +  9.1e-4 · S²      ms      (SmolLM2-135M, f32 prefill)
             ^^^^^^^      ^^^^^^^^^^
             per-token    attention
```

which reproduces every measured point from `S=32` to `S=1024` to within 6%. At the
`S=128` the brief names, **81% of the gap is the linear term and 19% is the quadratic
one**; the quadratic only takes over past `S≈550`. So the growing ratio is mostly a
*constant per-token surcharge* that upstream does not pay, and attention is the
smaller half of the story at the length that matters.

---

## 0. What these numbers may not be used for

- **Host only** (M1, macOS 15, 8 cores). No Android or iOS measurement here.
- Upstream is the **`torch` 2.13.0 macOS arm64 wheel** in `/Volumes/macMini/caches/spike-venv`:
  `BLAS_INFO=accelerate`, `USE_MKLDNN=OFF`. A differently-built upstream would not
  give these numbers.
- The machine is not idle even when nothing else is scheduled — a window server, a
  user app and two Android emulators sit on it. Load was **2.4 – 5.5** throughout.
  The control below is what makes the readings usable, not the quiet.

---

## 1. The curve

`SmolLM2-135M`, `dtype=float32`, deterministic token ids (`(i*7919+13) % 49152`),
2 warmups then 5 timed passes, **minimum of 5 within a process, then minimum across
3 alternating rounds** (`up, shim, up, shim, up, shim`). Harness
`/Volumes/macMini/caches/f32len-scratch/sweep.{py,sh}`.

| S | upstream (ms) | ours (ms) | **ratio** | absolute gap (ms) |
|---:|---:|---:|---:|---:|
| 6 | 35.46 | 37.50 | **1.06×** | 2.0 |
| 32 | 37.68 | 55.51 | **1.47×** | 17.8 |
| 128 | 75.79 | 154.35 | **2.04×** | 78.6 |
| 512 | 231.25 | 728.66 | **3.15×** | 497.4 |
| 1024 | 465.02 | 1928.46 | **4.15×** | 1463.4 |

`S=1024` is 2 rounds × 3 timed passes rather than 3 × 5, for time.

### 1.1 The control, and the spread

Two extra shim processes were run back to back against each other after the three
rounds, with nothing else changed:

| S | ctl_c1 | ctl_c2 | ctl ratio |
|---:|---:|---:|---:|
| 6 | 37.75 | 37.81 | 1.002 |
| 32 | 55.57 | 55.36 | 0.996 |
| 128 | 155.12 | 154.38 | 0.995 |
| 512 | 730.38 | 728.72 | 0.998 |

**The control reads 1.00 to within 0.5%.** Round-to-round spread on the real
measurement is of the same size (shim `S=128`: 154.35 / 154.76 / 154.63, i.e. 0.3%;
upstream `S=128`: 76.38 / 75.79 / 76.14, i.e. 0.8%). The effects in the table are
6% to 315%, so every row except `S=6` is far outside the noise. **`S=6` at 1.06× is
about 12× the control spread**, so it is real but small — and it agrees with
`docs/BIND.md`, which measured 1.02–1.13× there across several rounds.

### 1.2 The shape, which is the finding

Marginal cost per additional token, taken between adjacent rows:

| interval | upstream ms/token | ours ms/token |
|---|---:|---:|
| 6 → 32 | 0.085 | 0.686 |
| 32 → 128 | 0.397 | 1.030 |
| 128 → 512 | 0.405 | 1.496 |
| 512 → 1024 | 0.456 | 2.343 |

**Upstream's marginal cost is flat at ~0.40 ms/token** from `S=32` up to `S=512`,
rising only 13% at `S=1024`. Prefill on this model is, for upstream, essentially
linear in `S` over this whole range — the `S²` attention term is a small correction
even at 1024.

**Ours rises monotonically, 0.69 → 2.34.** A marginal cost that itself grows linearly
in `S` is a quadratic total. Least-squares over `S = 32…1024` on the gap gives

| term | coefficient | what it is |
|---|---:|---|
| `a · S` | **0.497 ms/token** | per-token work we do and upstream does not |
| `b · S²` | **9.10e-4 ms/token²** | grows with attention's area |

with residuals: `S=32` predicted 16.8 vs 17.8 measured, `S=128` 78.1 vs 78.6,
`S=512` 493.1 vs 497.4, `S=1024` 1463.9 vs 1463.4. **Under 6% everywhere, and under
1% for `S ≥ 128`.**

Split at each length:

| S | linear part | quadratic part | quadratic share |
|---:|---:|---:|---:|
| 32 | 15.9 | 0.9 | 6% |
| 128 | 63.6 | 14.9 | **19%** |
| 512 | 254.6 | 238.5 | 48% |
| 1024 | 509.1 | 954.3 | 65% |

**So at `S=128` the answer is "mostly not attention".** A brief that chased only the
quadratic term would be chasing 15 ms of a 79 ms gap. The crossover is at
`S = a/b ≈ 546`.

### 1.3 Numerics baseline

Logits sha256 (over the little-endian f32 bytes of the flattened `[1,S,49152]`
tensor). **These are the values that must not move.** The `S=128` entry matches
`docs/DTYPE_PERF.md` §6.1 exactly, which is the cross-check that this harness is
measuring the same thing the previous round did.

| S | ours (f32) |
|---:|---|
| 6 | `b9fc5553ee1bf6a2…` |
| 32 | `331668f36da02f21…` |
| 128 | `00159a9dbd308eda…` |
| 512 | `07c2797dabc4552e…` |
| 1024 | `eda1e173727bb7f5…` |

Upstream's differ (`S=128`: `71e46824c0c40f15…`) and are expected to — accumulation
order is not contractual across implementations. The contract here is that *ours*
does not change.

---

## 2. Decomposition — the linear term is `pow`, entirely

### 2.1 What the sampler says

`sample` on a shim process spinning `S=512` prefill, 20 s at 1 ms
(`sample_shim_512.txt`). Main thread: **13715 samples**. Inclusive counts:

| symbol | samples | share of main thread |
|---|---:|---:|
| `aten::pow_tensor_scalar` | **4624** | **33.7%** |
| ↳ `aten::pow_from_pairs` | 4495 | 32.8% |
| ↳↳ **`pow` (in `libsystem_m.dylib`)** | **4245** | **31.0%** |
| `candle::cpu_backend::ReduceIndex::map` | 1577 | 11.5% |
| `libBLAS` sgemm inner kernels (all addresses) | ~1400 | 10.2% |
| `candle::cpu_backend::copy_strided_src_f` | 850 | 6.2% |
| `VVEXPF` (in `libvMisc.dylib`) | 786 | 5.7% |

The single hottest thing in a `float32` prefill is **a scalar `pow()` call out of
libm, one per tensor element.** The call chain places it six nested `nn.Module`
calls deep, which is `LlamaRMSNorm`:

```python
# transformers/models/llama/modeling_llama.py:65
variance = hidden_states.pow(2).mean(-1, keepdim=True)
```

SmolLM2-135M has 30 layers, so that runs **61 times per forward**
(`30 × 2 + 1` final norm).

### 2.2 What the code does

`rust/torch_c/src/aten.rs`, `pow_tensor_scalar` → `side_from_tensor` →
`pow_from_pairs`. For a `float32` tensor and any scalar exponent it makes
**six full passes over the data**:

1. `flat.to_dtype(F64)` — widen every element to `f64`
2. `.to_vec1::<f64>()` — copy into a `Vec<f64>`
3. `PowSide::as_f64()` → `v.clone()` — **copy it again**
4. `(0..n).map(|i| b[i % b.len()].powf(e[i % e.len()]))` — one **libm `pow`** call
   per element, plus two integer `%` per element for the broadcast cycling
5. `Tensor::from_vec(values)` — copy into tensor storage
6. `.fast_to(storage)` — narrow back to `f32`

Step 4 is 94% of it (4245 of 4495 samples). The other five are why the remaining
6% is not free either.

### 2.3 The microbenchmark, at the shape *and layout* the model produces

`hidden_states` in `LlamaRMSNorm` is `[1, S, 576]`, `float32`, **contiguous** —
printed by the harness (`powbench.py`) rather than assumed, because this
repository has twice been misled by a microbench at a layout the model never
makes (`docs/DTYPE_PERF.md` §2 vs §4). All rows below read `contig=True`.

| S | shape | contig | upstream `pow(2)` | ours `pow(2)` | ours **`x*x`** | ratio pow |
|---:|---|:--:|---:|---:|---:|---:|
| 6 | `(1,6,576)` | True | 0.0013 | 0.0457 | 0.0016 | **35×** |
| 32 | `(1,32,576)` | True | 0.0024 | 0.2584 | 0.0027 | **108×** |
| 128 | `(1,128,576)` | True | 0.0118 | 1.0628 | 0.0082 | **90×** |
| 512 | `(1,512,576)` | True | 0.0314 | 4.2598 | 0.0271 | **136×** |
| 1024 | `(1,1024,576)` | True | 0.0497 | 8.5849 | 0.0519 | **173×** |

ms, min of 5 rounds after 2 warmups. **Our own `x*x` already matches upstream**
(0.0271 vs upstream's `x*x` 0.0289 at `S=512`) — so this is not a kernel-quality
problem anywhere except in `pow`'s own path. Upstream is fast because ATen
special-cases small integral exponents in `pow_tensor_scalar` and multiplies.

### 2.4 The sum, reconciled against the model-level gap

Weighted by the call count — 61 `pow(2)` per forward — against the fitted terms
of §1.2:

| S | `61 × (ours − upstream)` | fitted **linear** term | model gap | pow share of gap |
|---:|---:|---:|---:|---:|
| 6 | 2.7 | 3.0 | 2.0 | 133% |
| 32 | 15.6 | 15.9 | 17.8 | 88% |
| 128 | **64.1** | **63.6** | 78.6 | **82%** |
| 512 | **257.9** | **254.6** | 497.4 | **52%** |
| 1024 | **520.6** | **509.1** | 1463.4 | **36%** |

**The middle two columns agree to within 2–10% at every sequence length.** This is
the reconciliation the brief asked for, and it is not a near-miss: an independent
microbenchmark of one op, multiplied by a call count derived from the model's
layer count, reproduces a term that was fitted from model-level wall clock alone.

**So the linear term of the gap is `aten.pow.Tensor_Scalar`, and nothing else
material.** At `S=6` the prediction (2.7 ms) slightly exceeds the measured gap
(2.0 ms), which is the regime where `docs/BIND.md`'s Python-layer residual and
this cancel within the noise; at `S ≥ 32` it is clean.

What this does **not** explain is the quadratic term — 14.9 ms at `S=128`,
238.5 ms at `S=512`, 954 ms at `S=1024`. That is dealt with in §4.

---

## 3. The fix — square by multiplying

One function in `rust/torch_c/src/aten.rs`, `pow_square_fast_path`, taken before
`pow_tensor_scalar` falls into the generic path:

```rust
if exponent.as_f64() != 2.0 { return Ok(None); }
if !matches!(t.dtype(), DType::F32 | DType::F64) { return Ok(None); }
if PyDtype::new(tag).storage(op)? != t.dtype() { return Ok(None); }
t.mul(t).map(Some)
```

**`f16`/`bf16` are excluded on purpose.** For those, candle's reduced-precision
multiply is not obviously one correctly-rounded step, and `docs/DTYPE_PERF.md`
owns the `bfloat16` checksum. Leaving them on the old path means that checksum
is *untouched* rather than merely expected to hold.

### 3.1 Why the answer cannot move — exactness, not tolerance

An `f32` significand is 24 bits, so `b × b` needs at most 48 and `f64` has 53.
**The exact square is representable in the old path's `f64` intermediate**, so a
correctly-rounded libm `pow(b, 2.0)` returns it with no rounding at all, and the
closing `fast_to(F32)` is then a *single* rounding of the exact product. IEEE-754
`f32` multiplication is defined as the correctly-rounded exact product — the same
rounding of the same value. There is no double-rounding step because the
intermediate was exact, and no range problem either: the smallest `f32` subnormal
squared is ~2e-90, comfortably normal in `f64`, and anything overflowing `f32`
gives `inf` on both paths.

### 3.2 Three Rust tests, and the proof they can fail

`cargo test --release` goes **10 → 13**. `squaring_matches_the_libm_round_trip_bit_for_bit`
transcribes the old path (`f32 → f64 → libm pow → f32`) and compares bit patterns
against `b*b` over signed zeros, both subnormal extremes, `f32::MAX` (which
overflows to `inf`), `±inf`, a 4096-point sweep needing real rounding, and a
200-point geometric sweep across the exponent range.
`squaring_matches_libm_for_f64_too` checks the weaker `f64` claim — the one that
rests on this platform's libm rather than on representability — over 4700 values.

**Tampered, it fails.** Multiplying the fast path's result by `1.0000001` and the
reference's exponent by `0.0000001`:

```
test aten::pow_square_tests::squaring_matches_the_libm_round_trip_bit_for_bit ... FAILED
  assertion `left == right` failed: input 3 (-1e0) squared: fast 1e0 vs libm round-trip NaN
test result: FAILED. 12 passed; 1 failed
```

### 3.3 Model level — old vs new vs upstream, alternated

`old, new, upstream` × 3 rounds with the artefact swapped on disk and the swap
`cmp`-verified before every run, then a **new-vs-new control**. Minimum across
the 3 rounds. The two artefacts are confirmed distinct (`cmp`) and the
`pow_square_fast_path` symbol is present in exactly one of them (`nm`: new 1,
old 0).

| S | old | **new** | upstream | old ratio | **new ratio** | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 37.64 | **34.46** | 35.34 | 1.065× | **0.975×** | 3.2 |
| 32 | 55.42 | **39.19** | 37.83 | 1.465× | **1.036×** | 16.2 |
| 128 | 154.73 | **89.74** | 75.94 | 2.037× | **1.182×** | 65.0 |
| 512 | 728.52 | **464.48** | 231.04 | 3.153× | **2.011×** | 264.0 |

**The `S=128` case the brief opened with goes 1.84–2.04× → 1.18×, and `S=6` now
runs *faster* than upstream.**

Control (`new` vs `new`, two fresh processes):

| S | 6 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| ratio | 1.000 | 1.000 | 1.000 | 0.999 |

### 3.4 The saving is the predicted saving

§2.4 predicted the model-level effect from a microbenchmark of one op times a
call count. That prediction was made before the fix existed:

| S | predicted `61 × (old_pow − new_pow)` | **measured model-level saving** |
|---:|---:|---:|
| 128 | 64.3 | **65.0** |
| 512 | 258.2 | **264.0** |

**Within 1% and 2%.** The decomposition was not a plausible story that happened to
point at a fast op — it quantitatively predicted the outcome.

### 3.5 Numerics — unchanged, at every length

Every `old` and `new` run above printed the logits sha256. **All four pairs are
identical**, and `S=128` still equals `docs/DTYPE_PERF.md` §6.1's recorded value:

| S | old | new | same |
|---:|---|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | `b9fc5553ee1bf6a2…` | ✅ |
| 32 | `331668f36da02f21…` | `331668f36da02f21…` | ✅ |
| 128 | `00159a9dbd308eda…` | `00159a9dbd308eda…` | ✅ |
| 512 | `07c2797dabc4552e…` | `07c2797dabc4552e…` | ✅ |

That is 24 forward passes' worth of agreement (3 rounds × 4 lengths × 2
artefacts), not a single spot check.

### 3.6 What the fit says happened

Refitting `gap(S) = a·S + b·S²` on the **new** gaps (`S=128`: 13.80,
`S=512`: 233.44):

| term | before | after |
|---|---:|---:|
| linear `a` | 0.497 ms/token | **−0.008 ms/token** |
| quadratic `b` | 9.10e-4 ms/token² | **9.07e-4 ms/token²** |

**The linear term is gone — to zero, not merely reduced — and the quadratic
coefficient is unchanged to within 0.3%.** That is the strongest available
confirmation that §2's decomposition was right about *which* term `pow` was:
removing it moved one coefficient to zero and left the other exactly where it was.

### 3.7 `S=1024`, and `bfloat16` for free

| dtype | S | old | new | upstream | new ratio |
|---|---:|---:|---:|---:|---:|
| `f32` | 1024 | 1928.46 | **1405.00** | 465.02 | 4.15× → **3.02×** |
| `bf16` | 128 | 180.34 | **113.79** | 368.7 (§DTYPE_PERF §3) | **3.24× *faster*** |

`S=1024` saved 523.5 ms against a predicted 520.5 — **0.6%**.

**`bfloat16` got 1.58× faster even though the fast path refuses `bf16`.** That is
not luck and not a leak: `LlamaRMSNorm` upcasts *first* —

```python
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
```

— so the `pow` is a `float32` `pow` in every model, whatever the model's dtype.
The `bf16` logits sha256 is `7ff8e9334449b147…`, alternated old/new/old/new, and
it still equals the value `docs/DTYPE_PERF.md` §6.1 recorded.

---

## 4. What remains — the quadratic term is SDPA, and mostly one reduction

### 4.1 SDPA on the tensors the model actually passes

Not reconstructed: `sdpabench.py` monkeypatches
`F.scaled_dot_product_attention`, runs a real prefill, keeps the **first call's
actual arguments**, and times with exactly those. Shapes and contiguity printed:

```
q  (1, 9, S, 64)  k (1, 3, S, 64)  v (1, 3, S, 64)   float32
is_causal=True  scale=0.125  enable_gqa=True  attn_mask=None
30 sdpa calls per forward
```

(9 query heads, 3 kv heads, head_dim 64 — GQA. Upstream reports `q` as
non-contiguous and `v` contiguous; the shim reports the reverse. Same shapes,
same values, different stride bookkeeping.)

| S | upstream/call | ours/call | ratio | **× 30 = per forward** | remaining model gap | **share** |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.132 | 0.519 | 3.9× | **+11.6 ms** | 13.80 | **84%** |
| 512 | 1.427 | 7.898 | 5.5× | **+194.1 ms** | 233.44 | **83%** |

**So the quadratic term is SDPA, and the sum reconciles at both lengths.**

### 4.2 The sign flip the brief predicted

The brief noted `_softmax` had measured 0.15× (6.8× *faster* than upstream) at a
6-token prompt and asked whether that survived at 128. **It does not.** At the
score shape attention actually produces, `[1, 9, S, S]` contiguous:

| op, `[1,9,512,512]` f32 contiguous | upstream | ours | ratio |
|---|---:|---:|---:|
| `sum.dim_IntList(-1)` | 0.092 | 0.136 | 1.5× |
| `exp.default` | 0.631 | 1.083 | 1.7× |
| **`max.dim(-1)`** | 0.349 | **5.577** | **16.0×** |
| `amax.default(-1)` | 0.089 | *not implemented* | — |
| **`_softmax(-1)`** | 0.994 | **10.601** | **10.7×** |

An op that was 6.8× fast at `S=6` is 10.7× slow at `S=512`. That is precisely the
shape of a growing gap, and it is why the `S=6` reading in `docs/BIND.md` could
not have found this.

### 4.3 The specific thing, which is one reduction

`sample` on the new build at `S=1024` (main thread 13701 samples), top of stack:

| symbol | samples | share |
|---|---:|---:|
| **`candle::cpu_backend::ReduceIndex::map`** | **3323** | **24.3%** |
| `VVEXPF` (Accelerate vectorised exp) | 1404 | 10.2% |
| `libBLAS` sgemm kernels | ~2200 | 16% |
| `copy_strided_src_f` | 853 | 6.2% |
| `binary_map` Mul / Div / Sub | 1151 | 8.4% |
| `candle::cpu_backend::ReduceSum::map` | 204 | 1.5% |

**`ReduceIndex` and `ReduceSum` run over the same tensor, the same number of
times, and differ by 16×.** `ReduceIndex` is candle's *index-tracking*
reduction — the one behind `max`/`min`/`argmax` — and it computes and then
discards an argmax the softmax never wanted. `ReduceSum` has a vectorised path;
`ReduceIndex` is a scalar loop.

Upstream's own numbers say the same thing from the other side: its `amax`
(0.089) is **3.9× faster than its own `max.dim`** (0.349) for exactly this
reason — no indices. **The shim has no `amax` kernel at all**, so anything
needing a maximum *value* has only the index-producing path available.

There are also no `flash.rs` symbols anywhere in the profile. `flash.rs` is the
*reference* implementation (`reference_enabled()`), not the default; the default
SDPA materialises the `[1, 9, S, S]` score matrix and drives it through candle
ops — which is what every symbol in that table is.

---

## 5. What is left, and what it would take

**Fixed here:** the entire linear term, 0.497 ms/token, bit-identically.

**Not fixed:** the quadratic term, `≈9.05e-4 · S²` ms — 13.8 ms at `S=128`,
233 ms at `S=512`, 940 ms at `S=1024`. §4 attributes 83–84% of it to SDPA and
localises the largest single item inside SDPA to the softmax's max-reduction
running on candle's index-tracking `ReduceIndex`.

Sizing the next step from the measured numbers: if our max-reduction reached
upstream's `amax` (5.577 → 0.089 ms at `[1,9,512,512]`), `_softmax` would fall
from 10.601 to roughly 5.1 ms — **about half the softmax gap**, with the rest in
`exp` (1.7×) and the strided copies. Because a maximum involves no rounding, that
change is numerically free in a way the `pow` fix had to argue for.

Two reasons it is not done here:

1. **It is a candle-level concern.** The fast reduction has to exist before
   anything can call it, and `candle_core`'s `ReduceIndex` is a dependency, not
   this crate. The in-crate alternative is an `amax` kernel plus routing
   softmax/SDPA to it — a real change to the attention path rather than a
   guarded fast path in a leaf function.
2. **The remaining `exp` and copy costs are not obviously separable** from how
   SDPA materialises its scores, and a rewrite that stops materialising them is
   a different piece of work with its own numerics argument (softmax
   accumulation order is exactly what `docs/GENERATE.md` §6 warns about).

**The honest summary is that the growing gap had two terms, one of them is now
gone, and the other is named, measured, reconciled to 83–84%, and localised to a
single reduction — but closing it is a change to the attention path, not to a
leaf.**

---

## 6. Reproducing this

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-f32len
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
```

Harnesses live in `/Volumes/macMini/caches/f32len-scratch/`:

| file | what it does |
|---|---|
| `sweep.py` / `sweep.sh` | model-level wall time + logits sha256 across `S`, alternating |
| `ab.sh` | old/new/upstream × 3 rounds with a verified artefact swap, plus control |
| `spin.py` | prefill in a loop so `sample` can attach |
| `powbench.py` | `pow(2)` at the real RMSNorm shape, printing contiguity |
| `sdpabench.py` | SDPA on tensors captured from a real forward |
| `redbench.py` | the reductions, at the real score shape |

Gates, all exit 0 on the new artefact:

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh      242          (unchanged)
$PY tools/golden/compare.py                    3302/3302 ops=133, pending 2
$PY tools/golden/compare.py --self-test        PASS 13 comparators x 11 fault modes
$PY rust/torch_c/pytests/verify_schemas.py     4331/4331
( cd rust/torch_c && cargo test --release )    13           (was 10, +3)
```
