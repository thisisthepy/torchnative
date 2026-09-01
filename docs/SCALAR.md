# The reduced-float scalar rule

**One-line result.** When an op folds a Python number into a `float16` or
`bfloat16` tensor, upstream reads that number at **one of two precisions**, and
which one is a property of *the individual kernel*, not of the `.Scalar` family:
`mul` and `div` read it at `opmath_type` (`float`), `add` and `sub` read it
narrowed to the tensor's own dtype. This shim narrowed for all of them.

**The second result is that no recorded digest moved** — not `float32`, which
was the control, and not `bfloat16` or `float16`, which were expected to. That
is not luck and it is not the change failing to bite: a `TorchDispatchMode` over
a real SmolLM2-135M `bfloat16` prefill shows the forward never calls a changed
op with a scalar that separates the two rules. §4 has the log and the
demonstration that the numerics *do* move where such a call exists.

The defect was not found by reading kernels. It was found because a **sabotage
fault failed to fail** (docs/TRAIN.md §5, S4): scaling dropout's *input* instead
of its *mask* is detectable upstream and was not detectable here, because
`mul.Scalar`'s narrowing made the two shapes bit-identical.

---

## 0. What this document is, and what it may not be used for

- Upstream is `torch` 2.13.0, the macOS arm64 wheel in
  `/Volumes/macMini/caches/spike-venv` — the same build every other measurement
  in this repository uses.
- Every "upstream does X" below is a **measurement**, run over 420 values per
  dtype against two models built out of upstream's own arithmetic, not a reading
  of ATen source. Where a kernel line is quoted it is quoted as corroboration
  after the fact.
- Comparison is **bit-exact on the packed bytes**. The whole effect is one
  representable step, and the golden harness's `bfloat16` tolerance is `6e-2` —
  three orders of magnitude too loose to see any of this. Every case added by
  this round uses `_exact_value_check`.

---

## 1. Establishing the rule

Two models, both computed with upstream's own `float32` arithmetic so that the
*only* difference between them is the scalar's precision:

```
narrow   the scalar is first rounded to the tensor's dtype, then used
widen    the scalar is kept at float32 -- opmath_type<bfloat16> == float
```

For each op, each dtype and each of ten scalars, upstream's answer is compared
against both. `/tmp/scalar_probe2.py` is the probe; the inputs are 420 values
spanning `[-4, 4]` plus hand-picked awkward ones (`±0.3`, the two `bfloat16`
neighbours of `0.3`, signed zeros, integers).

**A third column matters as much as the two models: `same`.** For a good many
ops the two models are *identical* — the scalar's only role is to be stored or
compared, and storing it into the output narrows it whichever road you took.
Those ops cannot be got wrong this way, and saying so is what stops the next
reader from "fixing" them.

| upstream reads the scalar at | ops (measured, `float16`/`bfloat16`) |
|---|---|
| **`opmath_t` — widen** | `mul.Scalar`, `div.Scalar`, `floor_divide`, `div.*_mode` (both modes), `leaky_relu`, `elu`, `celu`, `softshrink`, `softplus` (both `beta` and `threshold`), `addcmul`/`addcdiv` `value`, `lerp` weight, `norm` `p` |
| **`scalar_t` — narrow** | `add.Scalar`, `sub.Scalar`, `rsub.Scalar`, `add`/`sub` `alpha`, `pow.Tensor_Scalar` exponent, `pow.Scalar` base, `remainder.Scalar`, `fmod.Scalar`, `hardshrink` `lambd`, every comparison (`eq`/`ne`/`lt`/`le`/`gt`/`ge`) |
| **structurally insensitive** | `clamp` (both bounds), `clamp_min`/`clamp_max`, `threshold`, `hardtanh`, `masked_fill`, `where.Scalar*`, `fill_`, `full`, `scalar_tensor`, `nan_to_num` |

`hardshrink` narrows and `softshrink` widens. `clamp` is insensitive and
`threshold` is insensitive, but `leaky_relu` — which looks like the same shape of
kernel — widens. **There is no principle here that survives contact with the
table**; the rule really is per kernel, and upstream's own source says the same
thing when you look afterwards: `softshrink_kernel` writes
`lambd.to<opmath_t>()` and `hardshrink_kernel` writes `lambd.to<scalar_t>()`,
two files apart.

### 1.1 Why `clamp` and friends cannot be got wrong

`clamp(bf16 x, min=0.3)` looks like it should be sensitive and is not. The
scalar reaches the output in only two ways: as a comparison against a value that
is already `bfloat16`, or as the returned value itself — and the return narrows
it. Take `x = 0.298828125`, the `bfloat16` neighbour below `0.3`:

```
narrow:  min = bf16(0.3) = 0.30078125,  max(x, min) = 0.30078125
widen:   0.298828125 < 0.3, so the result is 0.3, stored as bf16 -> 0.30078125
```

The two roads meet at the store. That holds for every value, which is why the
`same` column is 10/10 for those ops in every dtype. **They are listed above so
that nobody adds cases for them believing the cases prove something.**

### 1.2 `float32` is not in this story, and that is load-bearing

At `float32` the widen model and the narrow model are the *same computation*:
`opmath_type<float>` is `float`, so "narrow the scalar to the tensor's dtype" and
"widen the scalar to `opmath_type`" both say `float`. Every arithmetic op above
reads `same` 10/10 at `float32`.

**So a `float32` result that moves under this change has not been fixed, it has
been broken.** §4 uses that as the control.

Two exceptions, and neither is the scalar rule: `pow.Tensor_Scalar` keeps a
`double` exponent at `float32` (§3.3), and `celu` disagrees on 2 of 10 scalars
for its own reasons. Both are recorded here so the next reader does not
rediscover them as this rule.

---

## 2. `mul.Scalar` — the reported defect

```
bfloat16   [3, 5, 7] * 0.3    upstream [0.8984375,     1.5, 2.09375     ]
                              shim     [0.90234375,    1.5, 2.109375    ]
float16    [3, 5, 7] * 0.3    upstream [0.89990234375, 1.5, 2.099609375 ]
                              shim     [0.900390625,   1.5, 2.099609375 ]
```

`5 * 0.3` agrees in both dtypes and `7 * 0.3` agrees in `float16`. **A case set
that happened to pick those values would have found nothing**, which is the
shape of this whole document.

Fixed in `arith_scalar` by building the scalar operand at `opmath_in(storage)`
for `Mul` and `Div` and leaving `Add`/`Sub` narrowing as they were. The fix is a
three-line branch; finding out that it was only those two took the table in §1.

### 2.1 The cases that could not fail, and the ones that replace them

`mul_scalar_cases` had three scalars: `2.0`, `0.0`, `-1.5`. **All three are
exactly representable in `float16` and `bfloat16`**, so narrowing them is the
identity and the builder passed under either implementation — 6587/6587 green
over an op that had been wrong since it was written.

The replacement (`_scalar_rule_cases`) needs two things at once, and dropping
either one makes every case pass again:

* the **scalar** must not be representable in the tensor's dtype;
* the **tensor values** must be, so the scalar's rounding is the only thing
  under test.

Separating power of each candidate, measured over `[3, 5, 7, 11, 13, 96, -3, -5]`:

| scalar | narrowed to `bfloat16` | separates | narrowed to `float16` | separates |
|---|---|---:|---|---:|
| `0.3` | `0.30078125` | **5/8** | `0.300048828125` | **3/8** |
| `0.7` | `0.69921875` | 1/8 | `0.7001953125` | **5/8** |
| `1.3` | `1.296875` | **5/8** | `1.2998046875` | **4/8** |
| `0.1` | `0.10009765625` | 1/8 | `0.0999755859375` | 4/8 |
| `0.5` | `0.5` | **0/8** | `0.5` | **0/8** |
| `2.0` | `2.0` | **0/8** | `2.0` | **0/8** |

`0.3`, `0.7` and `1.3` are carried because between them they separate in *both*
reduced dtypes; `0.1` is not, because its `bfloat16` column is the same 1/8
near-miss that let `div.Scalar` pass for months (docs/TRAIN.md §4 —
`bfloat16` rounds both roads of `1 / 0.3` to `3.328125`). `0.5` and `2.0` are
kept **as controls**: they pass under either rule, and a run in which only the
controls pass is a run that has stopped testing anything.

Red before the fix, green after:

```
FAIL aten.mul.Scalar :: (dtype=float16,  scalar=0.3) -- 3/8 elements differ
FAIL aten.mul.Scalar :: (dtype=float16,  scalar=0.7) -- 5/8 elements differ
FAIL aten.mul.Scalar :: (dtype=float16,  scalar=1.3) -- 4/8 elements differ
FAIL aten.mul.Scalar :: (dtype=bfloat16, scalar=0.3) -- 5/8 elements differ
FAIL aten.mul.Scalar :: (dtype=bfloat16, scalar=0.7) -- 1/8 elements differ
FAIL aten.mul.Scalar :: (dtype=bfloat16, scalar=1.3) -- 5/8 elements differ
     the four control cases passed, before and after
```

### 2.2 `mul_.Scalar` is **not** the same op, and upstream is the reason

The in-place spelling was measured too, and it does not follow `mul.Scalar`:

```
bfloat16 [3,5,7,11], scalar 0.3
  torch.ops.aten.mul.Scalar        0.8984375 ...   widen
  torch.ops.aten.mul_.Scalar       0.90234375 ...  NARROW
  x * 0.3        (python)          0.8984375 ...   widen
  x *= 0.3       (python)          0.8984375 ...   widen
  torch.ops.aten.div_.Scalar       widen
```

**`torch.ops.aten.mul_.Scalar` is the only spelling of a scalar multiply in
upstream that narrows**, and `div_.Scalar` — the same shape of op, one letter
away — does not. Checked over 4096 values × 4 scalars × 2 dtypes; it is not a
tail or a vectorisation edge.

The shim reproduces it as measured, which means `mul_.Scalar` keeps the
narrowing this document removes everywhere else. That is deliberate and it has a
cost, stated plainly: a caller who writes `x *= 0.3` on a `bfloat16` tensor gets
the *narrowed* answer here and the *widened* answer upstream, because upstream's
Python `*=` does not reach `mul_.Scalar` at all — a `TorchDispatchMode` over
`x *= 0.3` reports `aten.mul_.Tensor`. The shim's parser reports `mul_.Scalar`,
so the two land on different upstream kernels for the same source line.

Golden cases pin the narrowing (`aten.mul_.Scalar`, five scalars × two dtypes,
`_exact_value_check`) so that the asymmetry is asserted rather than merely
present. Closing it properly is a resolver change, above this file, and is
listed in §6.

---

## 3. The rest of the family

`mul.Scalar` was the reported defect. It was not the only one, and the point of
establishing the rule in §1 rather than fixing two call sites is that **three
more ops were wrong and two of them were wrong in the opposite direction.**

The differential that found them is shim-vs-upstream, bit-exact, over every
`Scalar`-taking op `_aten_implemented()` advertises, at four dtypes and ten
scalars (`/tmp/shim_diff.py`). Run twice: once with scalars that are *not*
representable in the reduced dtypes and once with scalars that are. **An op
that disagrees on both is not this defect** — it has an ordinary precision
problem, and §5 lists the three that turned out to be exactly that.

| op | before | after |
|---|---:|---:|
| `mul.Scalar` | 6/10 bf16, 8/10 f16 | **0** |
| `floor_divide.Scalar` | 3/10 at *every* dtype incl. `float64` | **0** |
| `div.Scalar_mode` (floor) | 3/10 bf16, 3/10 f16 | **0** |
| `div.Scalar_mode` (trunc) | 3/10 bf16 | **0** |
| `pow.Tensor_Scalar` | 8/10 bf16, 8/10 f16 | **0** |
| `pow.Scalar` | 10/10 bf16, 10/10 f16, 10/10 f32 | **0** |
| `mul_.Scalar`, `div_.Scalar`, `add_`, `sub_`, `rsub`, `remainder`, six comparisons, `clamp`, `clamp_min`, `masked_fill`, `where.ScalarOther`, `fill_`, `leaky_relu` | 0 | 0 |

`leaky_relu` is worth naming among the ones that were already right: it widens,
and it was already widening — `leaky_relu_cases` happens to pass slopes of
`0.01` and `0.1`, neither representable, so that builder has been asserting this
rule by accident since it was written. It is the one place in this repository
where the blind spot was avoided, and not on purpose.

### 3.1 `div` was already fixed, and the fix survives a closer look

docs/TRAIN.md §4 fixed `div.Scalar`/`div_.Scalar` before this round. Re-checked
here against a model of upstream's branch rather than against the old cases:
upstream computes `opmath_t(1) / opmath_t(scalar)` — the reciprocal in
**`float`** — and `div_scalar_reduced_float` writes exactly
`1.0f32 / (scalar as f32)`. A model that takes the reciprocal in `f64` and
narrows disagrees with both on 2 of 10 scalars, so the `f32` in that line is
load-bearing and not incidental.

### 3.2 Floor and truncating division — where one ULP becomes one unit

`div_floor_kernel` and `div_trunc_kernel` carry the same reduced-float scalar
branch `div_true_kernel` does. The shim narrowed the divisor into the tensor's
dtype *and* narrowed every intermediate step of `div_floor_floating` with it.
Both are wrong, and here the consequence is not a rounding step:

```
bfloat16   [40, 43, 61, -49] // 0.3   upstream [133, 143, 203, -164]
                                      before   [133, 143, 202, -163]
float16    [7, 14, -49]      // 0.7   upstream [10, 20, -71]
                                      before   [ 9, 20, -71]
```

**A floor turns a fractional error into an integer one.** Whether it does so at
any particular value is erratic, though, which matters for the cases — see §4.2.

The same change surfaced a second defect that has nothing to do with scalars.
`floor_divide.Scalar` computed `floor(a / b)`; `div.Scalar_mode(floor)`
computed upstream's `fmod`-based `div_floor_floating`. `div_floor_float`'s own
doc comment has said since it was written that `floor(a / b)` "is the plausible
implementation and it is wrong", and one of the two spellings of floor division
in this file was doing it anyway:

```
float64  -3.0 // 0.3    upstream -11.0    before -10.0
```

`-3.0 / 0.3` in `f64` is `-10.000000000000002`, and the two algorithms take
different views of that. They now share one function, so the two keys cannot
drift apart again; a pytest asserts they agree as well as what they agree on.

### 3.3 `pow` narrows, and `float32` narrows too

The other half of §1's split, and the reason the rule has to be established per
kernel rather than argued from `mul`. `pow_tensor_scalar_kernel` converts the
exponent to the dispatched `scalar_t`; the shim handed `pow_from_pairs` the
`f64` the parser produced.

```
float16   [3] ** 0.3     upstream 1.390625        before 1.3896484375
bfloat16  0.3 ** [3]     upstream 0.0272216796875 before 0.0269775390625
```

**And `float32` narrows as well** — `opmath_type<float>` is `float`, so there is
no widening anywhere in this kernel and the parser's `f64` is never the value
upstream uses. That is the one place in this change where a `float32` result
moves. §4 shows it does not reach any recorded digest, and why.

**What `pow` at `float32` does *not* get is a case**, and that is deliberate.
Upstream's `float32` `pow` answers different bits for the same element
depending on the tensor's length — SLEEF's vectorised `powf` against libm's on
the tail:

```
float32  [3, 5, 7, 11, 13, 96, 2, 0.5] ** 0.3   differs from the same elements
                                                one at a time on 4 of 8
float32  0.3 ** [..., 0.5, ...]                 1.1401753425598145 in the vector
                                                1.140175461769104  alone
```

A golden case at `float32` would be pinned to whichever road an 8-element
tensor happens to take. That is the shape of a test that passes for the wrong
reason, so the `pow` scalar-rule cases run at `float16` and `bfloat16` only and
this paragraph is what stands in for them. The residual `float32`/`float64`
disagreements that remain (1 of 7 exponents, ~1 ULP) are the same
vectorisation/libm difference: on the pair that separates them, `5.0 ** -1.5`,
this shim's `f64` road is the **correctly rounded** `float32` answer and
upstream's is one step below.

---

## 4. What moved, and what did not

Every number here is an A/B against the artefact built from `develop` at
`f83f94c` (`base_C.dylib`, kept on disk and `cmp`-verified before each run),
with the same harness docs/SEQLEN.md §1 uses.

### 4.1 The digests — none of them moved

| dtype | S | recorded | base | new |
|---|---:|---|---|---|
| `float32` | 6 | `b9fc5553ee1bf6a2…` (SEQLEN §1.3) | same | **same** |
| `float32` | 32 | `331668f36da02f21…` | same | **same** |
| `float32` | 128 | `00159a9dbd308eda…` (also DTYPE_PERF §6.1) | same | **same** |
| `float32` | 512 | `07c2797dabc4552e…` | same | **same** |
| `float32` | 1024 | `eda1e173727bb7f5…` | same | **same** |
| `bfloat16` | 6 | `8ef1550ea33c4f3d…` (TRAIN §6) | same | **same** |
| `bfloat16` | 32 | `b81325c83a0a3d15…` | same | **same** |
| `bfloat16` | 128 | `7ff8e9334449b147…` | same | **same** |
| `bfloat16` | 512 | `9ab1e82f01378e38…` | same | **same** |
| `float16` | 6 / 32 / 128 / 512 | not previously recorded | `d48534af3d22e7f0…` / `1c5f53ebc584babe…` / `38a0b21c39ea24f6…` / `0eca60265a8c734e…` | **same** |

The `float32` row is the control §1.2 asks for and it holds — including through
the `pow` change, which does touch `float32`.

### 4.2 Why the `bfloat16` digests did not move either

Not luck. A `TorchDispatchMode` over a real `bfloat16` SmolLM2-135M prefill
(`S=32`) lists **every** op that receives a Python number:

```
  30  aten._scaled_dot_product_flash_attention_for_cpu   (0.0, 0.125)
  61  aten.add.Tensor                                    (1e-05,)
   2  aten.mul.Tensor                                    (1.0,)
  61  aten.pow.Tensor_Scalar                             (2,)
       ... plus arange/cat/slice/transpose/unsqueeze, all integer axes
```

**`mul.Scalar` is not called at all**, `pow`'s exponent is `2`, the attention
scale is `0.125`, and `add`'s `1e-05` goes to the `.Tensor` overload and to the
half of the family that still narrows. Every scalar in that list is either an
integer or exactly representable in `bfloat16`, so no changed kernel can move.

**That is a statement about this model, not about the change.** Where a
separating scalar does reach `mul.Scalar`, the numerics move and they move onto
upstream's answer. The SDPA *math* backend is such a place: for `E=8` it applies
`sqrt(1/sqrt(8)) = 0.5946035575013605` to the query (docs/TRAIN.md §4). At the
tensors it actually passes:

```
mul.Scalar(bfloat16 query [1,2,8,8], 0.5946035575013605)
    base   36 of 128 elements differ from upstream   (first: -0.2490234375 vs -0.25)
    new     0 of 128                                  BIT-EXACT
```

Run the whole composite through and it still disagrees with upstream — on 47 of
128 elements after, 44 before. **That is not a regression and it is not this
op**: the same composite at `float32`, where nothing in this change applies,
disagrees with upstream on 57 of 128 by ~1e-7, and `base` and `new` are
identical there. The math backend has a pre-existing 1-ULP disagreement of its
own (accumulation order in the `bmm`/softmax); which elements happen to coincide
at 1 ULP is arbitrary, and the maximum absolute error against upstream is
unchanged at one `bfloat16` step.

### 4.3 The cost — none that is measurable

`bfloat16` `S=128` prefill, 3 alternating rounds, 2 warmups + 5 timed passes,
minimum within a process then minimum across rounds.

| | round 1 | round 2 | round 3 | **min** |
|---|---:|---:|---:|---:|
| base | 107.52 | 106.60 | 107.10 | **106.60** |
| new | 107.73 | 106.55 | 107.67 | **106.55** |
| upstream | 382.19 | 383.09 | 385.09 | **382.19** |

```
new / base            106.55 / 106.60 = 0.9995      <- no cost
control (new vs new)  107.24 / 108.35 = 1.010       <- the noise floor
upstream / new        382.19 / 106.55 = 3.59x faster than upstream
```

**The difference between base and new is 0.05 ms on 107, twenty times inside the
control's own spread.** Widening the scalar costs one extra `full` at `f32`
instead of at `bf16` per call, and nothing in this model calls it.

docs/DTYPE_PERF.md §3 recorded `2.04×` faster than upstream (180.7 against
368.7); the shim's own `bfloat16` `S=128` has since come down to 106.6 in other
rounds, so `3.59×` is that figure re-measured on today's artefact and not a
change produced here — `base` reads the same.

---

## 5. Found while establishing the rule, and *not* fixed

The differential in §3 turns up three more disagreements with upstream. None of
them is the scalar rule, and the way that was decided is worth stating: **each
one still disagrees when the scalar is exactly representable**, so narrowing
cannot be what causes it.

| op | disagrees at | with an exactly-representable scalar? | what it looks like |
|---|---|---|---|
| `softplus` | every dtype **including `float64`**, with the *default* `beta=1, threshold=20` | yes, 7/7 | the `log1p(exp(x))` formula, computed in the storage dtype where upstream uses `opmath` — `bfloat16 softplus(-3)` is `0.048583984375` upstream and `0.0458984375` here |
| `norm.ScalarOpt_dim` | `bfloat16` 8/10, `float16` 8/10, `float32` 1/10 — but **`p=2` agrees exactly in all four dtypes** | yes, 4/7 | the accumulation of `|x|^p`, and `powf` for fractional `p`; the same class as docs/KERNELS26.md §9.3 |
| `leaky_relu` | one case, and only on the **sign bit of a zero** with a negative slope | yes | `leaky_relu(0.0, -1.5)` — the scalar handling itself is correct |

Fixing `softplus` or `norm` means changing where those kernels accumulate, which
is a different change with its own digest question, and folding it into a
correctness commit about scalars is exactly the thing docs/TRAIN.md §5 says not
to do. They are recorded here rather than filed away; `softplus`'s is the larger
of the two, since it is wrong at `float64` where nothing about reduced precision
applies.

---

## 6. Open

* **`x *= 0.3` disagrees with upstream by one step at reduced precision**, for
  §2.2's reason: upstream's Python `*=` reaches `mul_.Tensor` and widens, this
  shim's parser reaches `mul_.Scalar` and upstream's `mul_.Scalar` narrows. The
  two are pinned as they are measured. Closing it means teaching the resolver
  upstream's "numbers as tensors" rule, which lives above `aten.rs`; the same
  gap `floor_divide.Scalar`'s doc comment already describes for a different op.
* **The narrowing half of the family has no golden coverage.**
  `aten.add.Scalar` and `aten.sub.Scalar` are implemented in `aten.rs` but are
  not in `_aten_implemented()`, so `CASE_BUILDERS` has nowhere to hang a builder
  and the only check on them is `test_add_and_sub_scalar_still_narrow_and_did_
  not_follow_mul`. Sabotage F3 is caught by two smoke tests and **zero golden
  cases**.
* **`softplus` and `norm`**, per §5.

---

## 7. Sabotage

Six faults in `aten.rs`, each the most plausible wrong shape for what this round
changed, plus a re-run of docs/TRAIN.md §5's S4 — the one that started it. Every
one was applied to the source, **rebuilt**, and run through
`tools/golden/compare.py` and `pytests/test_shim.py`.

| # | fault | golden | smoke |
|---|---|---:|---|
| F1 | `mul.Scalar` narrows its scalar again (the reported defect, put back) | **6 FAIL** | 2 FAIL |
| F2 | the fix over-applied — `mul_.Scalar` widens too | **6 FAIL** | 1 FAIL |
| F3 | the fix applied to "the `.Scalar` family" — `add`/`sub` widen as well | **0** | 2 FAIL |
| F4 | the floor/trunc scalar narrows again, both spellings | **14 FAIL** | 1 FAIL |
| F5 | `pow` keeps the parser's `f64` scalar | **12 FAIL** | 1 FAIL |
| F6 | `floor_divide` back to `floor(a / b)` | **2 FAIL** | 1 FAIL |

**Two of these were caught by nothing on the first attempt, and the cases were
wrong, not the faults.**

* **F4 failed 5 golden cases and 0 smoke tests.** The shared
  `_SCALAR_RULE_VALUES` (`[3, 5, 7, 11, 13, 96, -3, -5]`) barely separate the two
  rules under a floor — upstream's `fmod`-based algorithm reaches the same
  integer from either divisor at small magnitudes, so `bf16(3) // 0.3` is 9 both
  ways and the pytest asserting it **could not fail**. Re-measured: at
  `bfloat16` the divisor `0.3` separates 4 of `[7, 14, 40, 43, 48, 61, 100, -49]`
  and at `float16` it separates *none* — the two dtypes need different scalars,
  which is why `0.1` is in the floor family's list and not in the shared one.
  With the measured values: 5 → **14** golden, 0 → 1 smoke.
* **F3 was caught by one pre-existing smoke test and not by the new one.** The
  new test asserted `add_`/`sub_`, which go through `arith_inplace_scalar`; F3
  changes `arith_scalar`, which owns the *out-of-place* `add.Scalar`/
  `sub.Scalar`. And its operands (`[1, 3, 5, 7]`) do not separate the rules for
  `add` at all — `0.3` and `0.30078125` are under half a `bfloat16` ULP apart at
  those magnitudes. Re-measured to `[-0.546875, -0.5, -0.9375, -7.96875,
  -0.421875, -1.1875, 0.0625, 0.1875]`, which separate 3/8 for `add` and 2/8 for
  `sub`/`rsub` in `bfloat16`.

**F3's golden column stays at 0, and that is a real gap, not a blind spot in the
fault.** `aten.add.Scalar` and `aten.sub.Scalar` are not in
`_aten_implemented()`, so `CASE_BUILDERS` has nowhere to register them; the two
smoke tests are the whole coverage. §6 carries it.

### 7.1 S4 re-run — the fault that started this, and it fails now

docs/TRAIN.md §5 recorded S4 ("dropout scales the input by `1/(1-p)` instead of
the mask") as **0 golden, 0 smoke**, and named `mul.Scalar`'s narrowing as the
reason: the two shapes were bit-identical here where upstream separates them.

`bootstrap.py` belongs to another change, so the fault was not applied to it.
Instead both shapes were computed **with the shim's own kernels**, on the
dropout cases' own 240 values and a fixed mask, and asked whether they are still
identical:

```
                          base           new         upstream
float32   p=0.25       0 of 168      0 of 168      0 of 168
float32   p=0.7        0 of 168      0 of 168      0 of 168
float16   p=0.25       0 of 168     52 of 168     52 of 168
float16   p=0.7        0 of 168     46 of 168     46 of 168
bfloat16  p=0.25       0 of 168     50 of 168     50 of 168
bfloat16  p=0.7        0 of 168     54 of 168     54 of 168
```

**The shim now separates the two shapes on the same elements upstream does, and
still not at all in `float32` — which is also upstream.** The dropout cases
compare bit-for-bit against upstream over these exact 240 values, so ~30% of the
survivors would differ and S4 is now caught. TRAIN.md §5's closing sentence —
"the dropout cases were widened anyway, so that they **will** catch S4 the day
`mul.Scalar` is fixed" — is discharged, and the widening was necessary: on the
all-ones input it replaced, the two shapes coincide by construction in every
dtype.

---

> Standing check (docs/DOCWATCH.md) — the claims above that have a single
> ground truth:
> <!-- DOCWATCH: op-implemented aten.mul.Scalar -->
> <!-- DOCWATCH: op-implemented aten.mul_.Scalar -->
> <!-- DOCWATCH: op-implemented aten.floor_divide.Scalar -->
> <!-- DOCWATCH: op-implemented aten.div.Scalar_mode -->
> <!-- DOCWATCH: op-implemented aten.pow.Tensor_Scalar -->
> <!-- DOCWATCH: op-implemented aten.pow.Scalar -->
> <!-- DOCWATCH: symbol-in-file tools/golden/cases.py _scalar_rule_cases present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_mul_scalar_reads_the_scalar_at_opmath_not_narrowed present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_add_and_sub_scalar_still_narrow_and_did_not_follow_mul present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_pow_narrows_its_scalar_where_mul_widens_it present -->
>
> §6's first bullet says `aten.add.Scalar` and `aten.sub.Scalar` are implemented
> in `aten.rs` but absent from `_aten_implemented()`, which is why golden has no
> builder for them. That is checkable both ways, and both halves are asserted so
> the bullet cannot go stale in either direction:
> <!-- DOCWATCH: op-not-implemented aten.add.Scalar -->
> <!-- DOCWATCH: op-not-implemented aten.sub.Scalar -->
