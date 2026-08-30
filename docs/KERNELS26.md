# The kernels that stopped six architectures

docs/ARCH26.md widened the sweep from 20 architectures to 26 and found the six new ones stopped
**not by a missing name but by a missing kernel** — `aten.rs`, `tensor.rs`, `dtype.rs` and
`flash.rs` were all out of territory that round, so it named the walls and left them. This document
is that round's follow-up: the kernels themselves.

**Written incrementally, one kernel at a time**, for the reason ARCH26.md gives for the same
practice — a kill mid-task should leave the finished parts behind rather than nothing.

## 0. Method, and what "landed" means here

The acceptance test is **architectures forwarding**, not kernels passing their own cases. So the
sweep is re-run after every kernel, and each section below records the count and, when the count
does not move, **which wall replaced the one that was removed**. A kernel that moves the first wall
forward without moving the count is progress that has to be visible as progress, not folded into a
batch total at the end.

The sweep is `/tmp/k26/sweep26.py` — the 20 of docs/ARCH20.md (`/tmp/arch7/sweep.py`'s list and
toy config, unchanged) plus the six toy configs of ARCH26.md §1-5, transcribed from that
document's own scratch scripts. It is run twice: once with no `PYTHONPATH` (upstream, the oracle)
and once with the vendored tree (the shim).

**Upstream is 26/26**, measured first, before anything here was written:

```
llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 stablelm olmo phi mixtral
bert bloom cohere falcon gpt_bigcode mamba persimmon
deberta deberta_v2 vits zoedepth sew_d sam3_video          TOTAL 26/26
```

One caveat carried forward from ARCH26.md §5, and it is the architecture's property rather than a
shortcut: **`sam3_video`'s top-level `forward` does not take a tensor.** It takes a stateful
`Sam3VideoInferenceSession` advanced one video frame at a time, and `transformers` ships no toy
`ModelTester` for it (only `@slow` integration tests against the real checkpoint). Its
`tracker_model` is session-shaped too — `m.tracker_model(pixel_values)` raises
`'Tensor' object has no attribute 'get_obj_num'` **on upstream**, measured, which is how that was
established rather than assumed. So "sam3_video forwards" here means its `detector_model` — the
SAM3 detector, a plain `pixel_values`/`input_ids`/`attention_mask` in, `pred_logits` out call —
forwards, which is the deepest tensor-shaped forward this architecture has.

### The baseline, every gate, before any edit

```
pytests/run.sh                261 ok, 0 FAIL                        exit 0
tools/golden/compare.py       4290/4290, ops=139, pending=1          exit 0
compare.py --self-test        13 comparators x 11 fault modes        exit 0
verify_schemas.py             4353/4353                              exit 0
sweep26 (shim)                20/26                                  exit 0
```

and the six failures, each exactly the wall ARCH26.md recorded:

| architecture | first wall at baseline |
|---|---|
| `deberta` | `torch.sqrt(...)` — no overload table entry |
| `deberta_v2` | `torch.sqrt(...)` — no overload table entry |
| `vits` | `TensorBase.set_: expected a torch.UntypedStorage, got Parameter` |
| `zoedepth` | `aten.convolution.default: only 1-D convolution ... got 4-D` |
| `sew_d` | `TensorBase(...) takes an existing tensor to re-wrap` (legacy `torch.Tensor(int)`) |
| `sam3_video` | `TensorBase.__mod__` |

---

## 1. `aten.sqrt.default`

**Sweep after: 20/26 (unchanged). What moved: `deberta` and `deberta_v2` both advanced from
`torch.sqrt` to `TensorBase.repeat`.**

That is the honest way to report it. `sqrt` was the highest-value item on the list because it was
the first wall for two architectures, but it was not the *only* wall for either, and §6/§8 of
ARCH26.md said so in advance — `aten.repeat.default` is listed there for both. The count moves when
the last wall for an architecture falls, and attributing the count to whichever kernel happened to
be last would be the wrong story.

### 1.1 Why it was a kernel and not a spelling

ARCH26.md §1.2 established this and it was re-checked rather than inherited:
`_dispatch_has_kernel_for_dispatch_key('aten::sqrt', 'CompositeImplicitAutograd')` is `False`, and a
`TorchDispatchMode` trace of `torch.sqrt(x)` fires exactly one op. There was nothing to wire a name
to.

The asymmetry — `rsqrt` present since RMSNorm, `sqrt` absent — is why two architectures stopped
before any weight multiplied. `DebertaLayerNorm.forward` computes its own layer norm by hand,
`(h - mean) / torch.sqrt(var + eps)`, instead of calling `nn.LayerNorm`; `deberta_v2` uses a real
`nn.LayerNorm` but its `scaled_size_sqrt` computes an attention temperature through `torch.sqrt`
**unconditionally**, before the `if self.relative_attention:` branch.

### 1.2 What upstream actually does, measured

Run on upstream 2.13.0 with no shim on the path, before writing anything:

```
sqrt dtype promotion            float64->float64  float32->float32  float16->float16  bfloat16->bfloat16
                                int64->float32    int32->float32    int16->float32
                                uint8->float32    bool->float32

sqrt sign and domain, float32   sqrt(-1.0)  = nan        bits 7fc00000
                                sqrt(-0.0)  = -0.0       bits 80000000     <-- sign of zero survives
                                sqrt( 0.0)  =  0.0       bits 00000000
                                sqrt( inf)  =  inf       bits 7f800000
                                sqrt(-inf)  = nan        bits 7fc00000     <-- NOT -inf
                                sqrt( nan)  = nan
                                sqrt( 2.0)  = 1.4142135381698608
```

So the rule is `unary_float`'s — a float keeps its own width, anything else becomes the default
float — and the domain is IEEE's, with **no refusal anywhere**: a negative input answers NaN rather
than raising. Both halves are asserted, because "raises on a negative" is the plausible wrong guess
and nothing in either DeBERTa's forward has a negative variance to reveal it.

**`sqrt(-0.0) = -0.0` is the property this kernel exists to get right and the one no value
comparison can see**, because `-0.0 == 0.0` is true in Python and `abs(-0.0 - 0.0)` is `0.0`. It is
checked on the sign bit, through `math.copysign`, in both the golden case
(`_signed_zero_check`) and the pytest.

### 1.3 What it is written on, and what it is deliberately not

`Unary::Sqrt` in `aten.rs`'s existing `unary_float` family, which is candle's own `Tensor::sqrt`.
Two alternatives were considered and both are wrong for measured reasons:

- **`pow(x, 0.5)` as a `bootstrap.py` composite.** ARCH26.md §1.2 rejected it as inventing a
  computation path; there is a second, harder reason. candle's `Tensor::pow` is
  `exp(exponent * log(base))` — `aten.rs`'s own `pow_result_tag` comment records this and exists
  because of it — and that expression answers **NaN for `sqrt(+inf)`**, where upstream answers
  `+inf`. The golden case `sqrt(dtype=..., shape=(1,)) [+inf -> +inf]` fails for exactly that.
- **`x.powf(0.5)`.** Gets the values right, but is a second rounding on `float16`/`bfloat16`:
  candle's `powf` widens, exponentiates and narrows, where `sqrt` is one correctly-rounded
  operation. The four reduced-float rows of the golden builder are what would catch it.

`sqrt` joins the family rather than getting its own function the way `rsqrt` has one, because it
*is* one candle call — `rsqrt` is separate only because it is two (`sqrt` then `recip`).

### 1.4 Reachable by name, not only by dispatch key

The brief's standing warning, and ARCH20.md §9's whole inventory: **golden compares by dispatch
key and cannot see a kernel with no way in.** `deberta`'s wall was the *spelling* `torch.sqrt(...)`,
so a kernel with no `overloads.json` entry would have left that wall exactly where it was while
every golden case passed.

| table | entry | what it opens |
|---|---|---|
| `overloads.json` | `aten::sqrt.out`, `aten::sqrt` | `torch.sqrt(x)` |
| `methods.json` | `aten::sqrt` | `x.sqrt()` |

`sqrt.out` is in the table with **no kernel behind it**, exactly as `rsqrt.out` already was, so that
`torch.sqrt(x, out=y)` refuses by the right name rather than falling through to "no table entry for
this op". Asserted as a refusal, not left implicit.

Three routes are tested, and they fail independently:

1. `_C._aten_dispatch("aten.sqrt.default", ...)` — the kernel, in-process against the bare artefact
   (`test_sqrt_is_a_leaf_kernel_with_ieee_domain_and_sign`).
2. `_C._shim_overloads` / `_C._shim_methods` — the two tables, asserted by content
   (`test_sqrt_is_reachable_by_name_not_only_by_dispatch_key`).
3. `torch.sqrt(x)` and `x.sqrt()` through a **real `import torch`** in a subprocess against the
   vendored tree (`test_kernels26_road_through_the_vendored_tree`), including `deberta_v2`'s
   `scaled_size_sqrt` and `DebertaLayerNorm.forward` transcribed verbatim.

### 1.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 261 | **264** (+3: two `sqrt` tests, one road test) |
| `compare.py` | 4290/4290, ops=139 | **4339/4339, ops=140** (+49 cases) |
| `compare.py --self-test` | 13 comparators | **14** (+`_signed_zero_check`), 0 problems |
| `verify_schemas.py` | 4353/4353 | **4359/4359** (+6: `sqrt` in both tables, `.out` included) |
| sweep26 (shim) | 20/26 | **20/26** |

Two existing tests carried hardcoded counts and both had to move; both are counts of *upstream's*
answer, not of this shim's, which is why they are worth having:

- `test_core_ops_and_op_tags_agree`: **84 → 85**. `torch.ops.aten.sqrt.default.tags` is
  `['core', 'pointwise', 'pt2_compliant_tag']`, read off the op rather than inferred from `rsqrt`.
- `test_schema_text_survives_the_round_trip_through_the_transcribed_tables`: **221 → 223**. **+2,
  not +3** — `sqrt` went into both tables and `overloads.json`'s `aten::sqrt|default` and
  `methods.json`'s are one identity; the second is `aten::sqrt|out`.

### 1.6 Sabotage check

Three faults, each aimed at a different claim, each rebuilt and re-run in full. The rule this
round is enforcing: **a check that cannot fail is not a check.**

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| S1 | `sqrt(x)` then `affine(1.0, 0.0)` — adds `+0.0`, which turns `-0.0` into `+0.0` and changes **nothing else** | **4335/4339, 4 failed** | 2 FAIL | the sign-of-zero claim is load-bearing. Exactly the 4 signed-zero cases (one per float dtype) fail, and every other `sqrt` case still passes — a value comparison alone would have seen none of it |
| S2 | always promote to the default float, so `float16` in gives `float32` out | **4306/4339, 33 failed** | 1 FAIL | the dtype rule is checked at every width, not just at `float32` |
| S3 | **kernel left intact and correct; the `sqrt` entry removed from `overloads.json` and `methods.json`** | **4339/4339, 0 failed, exit 0** | 3 FAIL | the point of §1.4. Golden is *completely green* on a kernel no model can reach, and re-running the sweep puts `deberta` straight back on its original wall: `torch.sqrt(...) -- overload resolution has no table entry for this op`. The three tests that do fail are the two table tests and the vendored-tree road |

S3 is the one worth keeping. It is the shape of gap that hid four times in this project's history,
and it is invisible to the harness that exists to catch gaps.

A fourth fault was tried first and is recorded because its *failure mode* is informative: making
`sqrt` keep the input dtype outright (`int64` in, `int64` out) does not produce a wrong answer, it
produces `PanicException: not yet implemented: no unary function for i64` from candle. A wrong
promotion here cannot be silent, which is a smaller claim than the tests make but worth knowing.

---

## 2. `aten.repeat.default`

**Sweep after: 22/26 (+2). `deberta` and `deberta_v2` both forward.**

This is the kernel that moved the count, and `sqrt` is the one that made it reachable — neither
alone gets either architecture through. §8 of ARCH26.md listed `repeat` for four of the six
(`deberta`, `deberta_v2`, `sew_d`, `sam3_video`), which is why it was worth taking even though it
was the first wall for none of them.

### 2.1 Tiling is not broadcasting

`expand` produces a view whose strides are zero; `repeat` materialises. `[1,2,3].repeat(2, 3)` is
`(2, 9)` and not `(2, 3, 3)` — **the last repeat multiplies the existing dimension and the earlier
ones are new leading dimensions.** Measured, along with the rest:

```
[1,2,3].repeat(2)          -> (6,)      [1,2,3,1,2,3]
[1,2,3].repeat(2,3)        -> (2, 9)
[[1,2],[3,4]].repeat(2,3)  -> (4, 6)
tensor(5).repeat(3)        -> (3,)      0-D gains a dimension
tensor(5).repeat([])       -> ()        an empty repeat list is legal on a 0-D
[1,2,3].repeat(0)          -> (0,)      NOT a no-op
m.repeat([2])              raises: Number of dimensions of repeat dims can not be smaller
                                    than number of dimensions of tensor
m.repeat([2,-1])           raises: Trying to create tensor with negative dimension -2: [4, -2]
dtype                      passed through unchanged, bool included -- no promotion
result                     always a fresh copy; writing into it does not reach the source
```

The negative-repeat message reports the **product** (`2 * -1` on a size-2 axis) and the whole
computed output shape, not the repeat the caller passed. Transcribed rather than paraphrased,
because the number in the message is not the number that was handed in.

### 2.2 candle has a `Tensor::repeat` and it is not the one called

Its entire body is `for (idx, &repeat) in repeats.iter().enumerate() { if repeat > 1 { cat } }`,
and that loop disagrees with upstream in three places:

| | candle | upstream |
|---|---|---|
| a repeat of `0` | skipped — `if repeat > 1` — so `[1,2,3].repeat(0)` is `(3,)` | `(0,)`, a genuinely empty dimension |
| `len(repeats) < rank` | takes the `self.clone()` branch and concatenates along axes that no longer line up | raises |
| every repeat `1` | returns `self.clone()`, and **a candle clone is an `Arc` clone** | materialises |

The third is the dangerous one and it is not visible in any result. `x.repeat(1, 1)` would share
storage with `x`, so `x.repeat(1,1).fill_(0)` would zero `x` — the exact defect docs/VIEWS.md §6
records for `_to_copy`, wearing a new hat: correct values, corrupted input, every golden case green
because they all read the *result*.

So the tiling is written out in `repeat_default`: resolve the output shape first (which is where
both refusals live and where `0` is honoured), tile with `Tensor::cat`, and end with an explicit
`.copy()` that is load-bearing rather than defensive.

### 2.3 Reachable by name

**`repeat` is a member and not a free function.** `hasattr(torch, "repeat")` is `False` on 2.13.0 —
there is `Tensor.repeat` and the unrelated `torch.repeat_interleave`, and nothing else. So the entry
goes in `methods.json` only, and `test_sqrt_is_reachable_by_name_not_only_by_dispatch_key` asserts
`"repeat" not in _C._shim_overloads` — inventing a `torch.repeat` would be adding a name upstream
does not have.

Both call shapes are exercised through a real `import torch`: `x.repeat(2, 3)` (varargs) and
`x.repeat([2, 3])` (list).

### 2.4 The numeric comparison ARCH26.md could not make

ARCH26.md §1.4 checked `deberta`/`deberta_v2`'s `@torch.jit.script` helpers standalone under
`PYTORCH_JIT=1` and `PYTORCH_JIT=0` and found them identical, but recorded honestly that this was
corroborating evidence rather than the acceptance-grade check, and that **"the real answer to 'do
the numbers drift' is: cannot be measured until `aten.sqrt.default` lands."**

It lands here, so it was measured. Same toy config, `torch.manual_seed(0)`, a 6-token forward, run
once on upstream and once through the shim, dumping the full `state_dict` as well as the output —
because "the initialisers agree" is itself a claim and not something `manual_seed` should be
trusted for:

| | `deberta` | `deberta_v2` |
|---|---|---|
| weight tensors compared | 37 | 45 |
| weight tensors differing | **0** | **0** |
| max weight abs diff | **0.0** (bit-identical) | **0.0** (bit-identical) |
| output elements | 192 | 192 |
| bit-identical output elements | 46 | 57 |
| max abs diff | 4.77e-07 | 3.58e-07 |
| **max relative diff** | **1.61e-07** | **1.21e-07** |

Both are at float32 epsilon (1.19e-07) — accumulation-order noise, not drift. **There is no
TorchScript-versus-eager divergence in either architecture**, which is the question ARCH26.md left
open, now answered on a real forward rather than on standalone helper functions.

The weight half being *bit-identical* is the stronger half of that result: it means the shim's RNG
reproduces upstream's initialisation exactly for both architectures, so the output difference is
attributable to the forward alone and to nothing upstream of it.

### 2.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 264 | **265** (+1) |
| `compare.py` | 4339/4339, ops=140 | **4481/4481, ops=141** (+142 cases) |
| `verify_schemas.py` | 4359/4359 | **4363/4363** (+4) |
| sweep26 (shim) | 20/26 | **22/26** |
| `test_core_ops_and_op_tags_agree` | 85 | **86** — `repeat`'s tags are `['core', 'pt2_compliant_tag']`, core but *not* `pointwise`, read off the op |
| schema identities | 223 | **224** (+1: `methods.json` only, no `.out`) |

### 2.6 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| R1 | call candle's `Tensor::repeat` — the naive implementation | **4451/4481, 30 failed** | 3 FAIL | the zero-repeat rows, the rank refusal and the aliasing entry are each load-bearing |
| R2 | **everything correct except the final `.copy()`** | **4481/4481, 0 failed, exit 0** | 2 FAIL | the same lesson as §1.6's S3 from the other direction. Golden is *completely green* while `x.repeat(1)` silently zeroes its own input; only `test_which_ops_share_storage_with_their_input_and_which_do_not` and the vendored-tree road can see it |

R2 is why the `.copy()` has a comment saying it is load-bearing. Without the aliasing-table entry
added in the same change, dropping it would have been a free 4481/4481.

---

## 3. `aten.remainder.Scalar` and `aten.remainder.Tensor`

**Sweep after: 22/26 (unchanged). What moved: `sam3_video` advanced from
`TensorBase.__mod__` to `aten.div.Scalar_mode` — two lines later, in the same
`Sam3ViTRotaryEmbedding.__init__`.**

`modeling_sam3.py:428` builds a rotary position grid as `flattened_indices % end_x` for the x axis
and `torch.div(flattened_indices, end_x, rounding_mode="floor")` for the y axis. The first of those
is this kernel; the second is `div.Scalar_mode`, which §6 leaves named and unwritten.

### 3.1 The sign of the divisor, which is the whole op

**`remainder` follows the sign of the divisor; `fmod` follows the sign of the dividend.** Measured
on upstream 2.13.0, and this table is the acceptance criterion:

```
        remainder   fmod
 7,  3      1        1     agree
 7, -3     -2        1     DISAGREE
-7,  3      2       -1     DISAGREE
-7, -3     -1       -1     agree
```

They agree in exactly half the quadrants. **A case set built from positive operands, or one that
varies only the dividend's sign, passes an `fmod` implementation completely.** Both the golden
builder and the pytest run all four quadrants on both overloads and on both categories, and the
pytest asserts the *disagreement* explicitly — for opposite-sign pairs it requires
`remainder(a, b) != math.fmod(a, b)`, computed in Python rather than taken from the shim.

The implementation is upstream's own correction, transcribed:

```
mod = fmod(a, b);
if (mod != 0) && ((b < 0) != (mod < 0)) { mod += b }
```

### 3.2 Three corners that fall out of that guard, all measured

- **`remainder(-0.0, 3.0)` is `-0.0`.** `fmod(-0.0, 3.0)` is `-0.0`, and `-0.0 != 0.0` is *false*,
  so the correction never fires and the negative zero survives. **Python's own `-0.0 % 3.0` is
  `+0.0`** — so "spell it the way Python spells it" is wrong here, and wrong invisibly, because
  `-0.0 == 0.0`. Checked on the sign bit by `_signed_zero_check`, the comparator §1 added.
- **Division by zero splits by category.** A float divisor of `0.0` gives NaN with no raise; an
  integral one raises `RuntimeError('ZeroDivisionError')` — upstream's message is that bare string.
  Same op, same shape of input, two different kinds of answer.
- **Infinite divisors.** `remainder(5.0, inf)` is `5.0`, `remainder(5.0, -inf)` is `-inf`, and
  `remainder(-5.0, inf)` is `inf`. All three are the correction firing on a finite `fmod`; none of
  them is special-cased.

One divergence from C rather than from `fmod`: **`i64::MIN % -1` panics in Rust** (the quotient
overflows) where upstream answers `0`. `wrapping_rem` is used for exactly that pair, and a golden
case plus a pytest pin it — a panic here would cross the FFI boundary as a `PanicException`, which
is not a refusal.

### 3.3 Dtype, and the one deliberate gap

`remainder.Tensor` follows `torch.promote_types` **exactly** — checked cell by cell over the eight
storable numeric dtypes rather than assumed from `mul`, with no disagreements. `remainder.Scalar`
follows the wrapped-number rule: an int scalar never widens a tensor of any category, a float scalar
floats an integral one.

**The scalar is narrowed into the result dtype before the arithmetic**, and that is observable:
`remainder(uint8(200), -3)` is `200`, because `-3` becomes `253` in `uint8` and `200 % 253` is
`200`. Building the scalar as a 0-D tensor at the result storage reproduces the narrowing instead of
restating it.

**Bool is refused on both overloads, but only one of those refusals is upstream's.**

| | upstream | here |
|---|---|---|
| `remainder.Tensor(bool, bool)` | raises `NotImplementedError: "remainder_cpu" not implemented for 'Bool'` | **same message, verbatim** |
| `remainder.Scalar(bool, 2)` | computes, `int64` | refuses |
| `remainder.Scalar(bool, 2.0)` | computes, `float32` | refuses |
| `remainder.Scalar(bool, True)` | raises | refuses |

The `Scalar` row is this shim's gap and it is deliberate, for the reason `arith_tag` already gives
for refusing `bool_tensor * 2`: the rule is a fast-path ladder keyed on the **Python type** of the
scalar, and `scalar_arg` has already erased `True` into `Scalar::Int(1)` by the time the kernel
sees it — so reproducing the third row is not possible from where the kernel stands, and
implementing the first two without it would compute where upstream raises. Two golden cases carry
it as `expect="c_error"`, so the harness watches the gap rather than filing it away.

### 3.4 Reachable by name

| table | entry | what it opens |
|---|---|---|
| `overloads.json` | `remainder.{Scalar_out, Tensor_out, Tensor, Scalar}` | `torch.remainder(x, y)`; the two `.out` forms refuse by name |
| `methods.json` | `remainder` → `[Tensor, Scalar]` | `x.remainder(y)` |
| `methods.json` | `__mod__` → `[Tensor, Scalar]` | **`x % y`**, which is how every caller actually spells it |

Order is load-bearing and asserted: the resolver picks `.Tensor` when handed a tensor and `.Scalar`
otherwise, so a table listing only one would silently answer the wrong overload for half the calls
`x % y` can make.

**`__rmod__` is deliberately absent, and asserted absent.** `3 % x` is
`aten.remainder.Scalar_Tensor`, a distinct overload with its own promotion rule, and it is not
implemented. Adding `__rmod__` without the kernel would make `3 % x` refuse by a name that resolves
to nothing.

### 3.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 265 | **266** (+1) |
| `compare.py` | 4481/4481, ops=141 | **4682/4682, ops=143** (+201 cases) |
| `verify_schemas.py` | 4363/4363 | **4376/4376** (+13) |
| sweep26 (shim) | 22/26 | **22/26** |
| `test_core_ops_and_op_tags_agree` | 86 | **88** — both overloads are `['core', 'pointwise', 'pt2_compliant_tag']`; +2 rather than +1 because this counts overloads, and two overloads of one op need not agree |
| schema identities | 224 | **228** (+4: all four `overloads.json` entries; `remainder`/`__mod__` add none) |

### 3.6 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| M1 | drop the correction on both paths — plain `fmod`/`wrapping_rem`, the sign of the **dividend** | **4623/4682, 59 failed** | 2 FAIL | the sign rule is checked in every quadrant, on both overloads and both categories. An implementation that "looks right" and is written on `fmod` fails 59 cases |
| M2 | **Python's own `%`**: `a - floor(a/b)*b`. Gets every sign quadrant *right* | **4666/4682, 16 failed** | 2 FAIL | the corner cases are the ones doing the work. All 18 sign rows still pass; what fails is the 4 signed-zero cases (`-0.0` becomes `+0.0`) and the 12 infinity rows (`5.0 % inf` becomes NaN) |

M2 is the interesting one: it is the *plausible* wrong implementation — it satisfies the headline
property the op is named for — and it is caught only by rows a "does it follow the divisor's sign"
case set would never contain.

---

## 4. The legacy `torch.Tensor(int)` constructor — the decision, and why

**Sweep after: 22/26 (unchanged). What moved: `sew_d` advanced from the constructor to
`TensorBase.set_` — the same wall `vits` is on.**

The brief asked for a verdict rather than a patch: *is upstream's behaviour worth reproducing, or
is refusing right and `sew_d` simply unsupportable?*

**Verdict: reproduce it.** Three grounds, and the grounds matter more than the conclusion.

1. **It is not a new computation path.** A `TorchDispatchMode` trace of `torch.Tensor(3)` on 2.13.0
   fires exactly one op — `aten.empty.memory_format` — which this shim already implements and
   already golden-compares. So this is a *constructor spelling* over an existing kernel,
   structurally identical to ARCH26.md §3.1's `torch.conv2d` over `aten.convolution.default`. It is
   not the kind of invention `sqrt`-as-`pow(x, 0.5)` would have been, and that distinction is the
   one this round keeps having to make.
2. **The three forms are distinguishable at the type level**, which is what the old refusal already
   did — it extracted a `TensorBase` and refused everything else. Nothing has to be guessed:

   ```
   TensorBase(existing)   re-wrap, sharing the candle tensor
   TensorBase(2, 3)       uninitialised storage of that shape
   TensorBase([3, 4])     build from data -- a (2,) tensor of 3.0 and 4.0
   ```

3. **Refusing costs a family, not an architecture.**
   `nn.Parameter(torch.Tensor(config.hidden_size).uniform_())` is the `masked_spec_embed` idiom and
   it is constructed **unconditionally** in `__init__` whether or not `apply_spec_augment` is set —
   so no toy config avoids it. `sew_d` is where ARCH26.md §4 found it; `wav2vec2`, `sew`, `hubert`,
   `unispeech` and `wavlm` carry the same line.

### 4.1 What was reproduced, and the one divergence

```
TensorBase(3)           -> (3,)         TensorBase(3, 4)  -> (3, 4)
TensorBase()            -> (0,)         not ()  -- measured
TensorBase(0)           -> (0,)
dtype                   the default float, read at call time, so it follows
                        set_default_dtype exactly as upstream's does
TensorBase(-1)          raises: Trying to create tensor with negative dimension -1: [-1]
```

**The bytes are zeros and upstream's are uninitialised.** That is a real divergence, and it is the
one `aten.empty.memory_format` already has — its golden comparator is `_dtype_shape_only_check`,
whose docstring is "there is no correct value to diff". Reading a `torch.Tensor(n)` before writing
it is undefined upstream, so this narrows undefined behaviour rather than disagreeing about defined
behaviour, and the real caller writes it immediately with `.uniform_()`.

### 4.2 The form that stays refused, and why that is the interesting one

**`torch.Tensor([3, 4])` builds from data.** It is `tensor([3., 4.])`, a `(2,)` tensor — **not** a
`(3, 4)` empty one. It looks exactly like a size list and is not one.

That is the trap in implementing this at all: a constructor that accepted a sequence as a shape
would silently answer `(3, 4)` zeros where upstream answers two numbers, and nothing downstream
would raise. So it refuses, the message says *which* form it is refusing, and the test asserts the
refusal with a note explaining that answering it would be worse than refusing.

It could be implemented — it needs `_tensor_new_from_data`, a module-level function `PyTensorBase`
has no handle to — but no measured caller reaches it, and this round does not add unreached
surface.

### 4.3 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 266 | **267** (+1) |
| `compare.py` | 4682/4682, ops=143 | **4682/4682** (unchanged — a constructor spelling, no new dispatch key) |
| `verify_schemas.py` | 4376/4376 | **4376/4376** (unchanged — not a table entry) |
| sweep26 (shim) | 22/26 | **22/26** |

The two unchanged rows are the expected result and are worth stating: this adds no aten key and no
schema, so the only gate that can see it is the pytest — which is the same reason ARCH26.md §7's
four `bootstrap.py` names moved none of these numbers either.

---

## 5. `aten.set_.source_Tensor` and `aten.set_.default`

**Sweep after: 22/26 (unchanged). What moved: `vits` and `sew_d` both advanced past `set_` — and
what they advanced *to* is the finding in §5.3.**

### 5.1 The two forms of `set_` now have opposite aliasing behaviour, on purpose

```
a.set_(storage, offset, size, stride)   COPIES   (unchanged, docs/CKPT.md §4)
a.set_(b)                               ALIASES  (new)
a.set_()                                empties in place, keeping the dtype
```

That is not an inconsistency. The storage form has to copy: candle owns its memory and an
`UntypedStorage` is bytes this shim holds separately, so there is nothing to alias. The tensor form
does not have to: `Repr::Dense` **is** a candle tensor, and a candle clone is an `Arc` clone of the
same storage — so `replace_with` gives upstream's semantics for free, and `a.set_(b)` followed by a
write into `b` is visible through `a`, exactly as upstream. Both directions are asserted side by
side, because "one of these copies and the other does not" is the kind of claim that rots quietly.

The order of the two arms is load-bearing: **the tensor check comes first**, because an
`nn.Parameter` extracts as a `TensorBase` and would otherwise fall into the storage arm — producing
precisely the message ARCH26.md §2 recorded (`expected a torch.UntypedStorage, got Parameter`).

### 5.2 A fourth spelling of the dtype set

Upstream refuses a dtype mismatch:
`Could not set tensor of type long long to a tensor of type float`. Those names are the plain C++
type names and are **a fourth naming**, distinct from all three this crate already carries:

| dtype | `TorchDType::name()` | `c10_name` | `scalar_type_name` | **`set_type_name`** |
|---|---|---|---|---|
| int64 | `int64` | `int64_t` | `Long` | **`long long`** |
| int16 | `int16` | `int16_t` | `Short` | **`short`** |
| uint8 | `uint8` | `uint8_t` | `Byte` | **`unsigned char`** |
| float32 | `float32` | `float` | `Float` | `float` |

Not derivable from any of the others; every row was read off a real `RuntimeError` by provoking the
mismatch across all ten storable dtypes. The pytest asserts the **exact whole string**, not a
substring, because a substring match would pass on any of the four tables.

Silently adopting the source's dtype instead of refusing would make `set_` a `to()` with no
conversion — and the parametrize machinery would then swap a float parameter for an integer one
without complaint.

### 5.3 The finding: upstream swallows the next `NotImplementedError`

This is the part worth carrying forward. With `set_` working, `vits` and `sew_d` both fail with:

```
TypeError: _WeightNorm.forward() missing 1 required positional argument: 'weight_v'
```

which names no kernel at all and points at a line 200 frames from the cause.
`ParametrizationList.__init__` (`torch/nn/utils/parametrize.py:163`) computes:

```python
for module in reversed(self):
    if hasattr(module, "right_inverse"):
        try:
            new = module.right_inverse(new)
        except NotImplementedError:
            pass                      # <-- "we assume right_inverse is the identity"
```

`_WeightNorm.right_inverse` calls `torch.norm_except_dim`, which this shim raises
`NotImplementedError` for — **and upstream's own `except NotImplementedError: pass` eats it.** `new`
stays the original tensor, so `is_tensor` becomes `True`, `ntensors` becomes `1`, and `forward`
calls a two-argument function with one argument.

So the wall is `torch.norm_except_dim`, and the error message says nothing about it. Two things
follow:

1. **`NotImplementedError` is not a safe refusal type everywhere.** This shim uses it as its
   standard "not implemented" signal, and there is at least one place in the vendored tree where
   upstream treats it as a *control-flow* signal meaning "this optional method is absent". Anywhere
   that pattern occurs, a missing kernel becomes a confusing error somewhere else entirely.
2. **It is a worked example of this round's own trap** — "if you touch a refusal, re-read what it
   claims". Reading only the message would have sent the next round after
   `ParametrizationList.forward`.

### 5.4 What `weight_norm` actually costs, measured

Traced on upstream, so the remaining bill is a number rather than an estimate:

| name | fires | status |
|---|---|---|
| `torch.norm_except_dim(w, 2, dim)` | `view` → **`aten.norm.ScalarOpt_dim`** → `view` | `norm.ScalarOpt_dim` missing |
| `torch._weight_norm(v, g, dim)` | **`aten._weight_norm_interface.default`** (returns `(output, norms)`) | missing |

ARCH26.md §6 said `weight_norm` costs two kernels. **It costs three**, and the third
(`norm.ScalarOpt_dim`) was invisible to that document's method because `norm_except_dim` is a
composite and the trace that found `_weight_norm_interface` ran on a *forward*, not on
`register_parametrization`. §6 below carries it forward with the rest.

### 5.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 267 | **268** (+1) |
| `compare.py` | 4682/4682, ops=143 | **4682/4682** (unchanged) |
| `verify_schemas.py` | 4376/4376 | **4376/4376** (unchanged) |
| sweep26 (shim) | 22/26 | **22/26** |

**Golden cannot see this kernel at all, and that is structural rather than an omission.** `set_` is
a `TensorBase` method implemented directly in `tensor.rs`; it does not go through `_aten_dispatch`,
so it has no dispatch key for `compare.py` to reach and no `CASE_BUILDERS` entry is possible. The
pytest is the only gate, which is exactly why it asserts the aliasing in both directions and the
refusal string in full.

---

## 6. `aten.convolution.default`, the 2-D case

**Sweep after: 22/26 (unchanged). What moved: `zoedepth` went from the convolution refusal, through
`TensorBase.expand_as` (fixed here), to `torch.conv_transpose2d` — two walls further into the same
model.**

### 6.1 The brief's question: was it in reach?

**Yes, and it was the small piece rather than the large one.** The brief sized it as possibly
"larger than the rest combined" and said to leave it if so. It is not, for one reason that could
only be found by looking: **candle already carries `Tensor::conv2d`, with the same
`(padding, stride, dilation, groups)` signature `conv1d` has.** So the 2-D case is the same thin
wrapper written twice, not a second kernel — about forty lines of argument handling, of which the
convolution itself is one call.

What it is *not* is a general 2-D convolution, and the difference is the next section.

### 6.2 What it does and does not do

```
4-D input, symmetric stride/padding/dilation      implemented
1-element stride/padding/dilation on a 4-D input  implemented -- expands to both axes
asymmetric (stride_h != stride_w, etc.)           REFUSED by name
transposed / non-zero output_padding              refused, as before
```

**candle's `conv2d` takes one scalar per argument**, so it is symmetric only. torch allows
`(stride_h, stride_w)` to differ. An asymmetric call reaching a symmetric kernel would convolve with
the wrong geometry and produce a *wrong output shape in one axis only* — which a set of square test
cases cannot show. So it refuses, the message names which argument disagreed, and three golden cases
carry the gap as `expect="c_error"`.

Nothing measured needs it: `Dinov2`'s patch embedding, which is what ARCH26.md §3.2 stopped on, is
`nn.Conv2d(3, hidden, kernel_size=16, stride=16)` — square kernel, square stride, no padding.

**One measurement corrected a first attempt.** The kernel initially refused a 1-element
`stride`/`padding`/`dilation` on a 4-D input; the golden case written for it as `both_error` failed,
because **torch computes it** — `expand_param_if_needed` broadcasts a single value to every convolved
axis, so `padding=[2]` on a 4-D input pads both axes and gives `(1, 3, 7, 7)`. The kernel now
expands, and both spellings are live cases. The mirror rule is also upstream's and also asserted: a
2-element stride on a 3-D input **raises upstream**, so refusing it is not a shim restriction.

The non-square rows are the ones a square-only case set cannot fail on: `(1, 2, 4, 7)` input and
`(2, 1, 3, 2)` weight both appear, so an implementation that swapped the height and width axes has a
case that shows it.

### 6.3 `TensorBase.expand_as`, found behind it

With 2-D convolution working, `zoedepth` stopped on `TensorBase.expand_as` — **a name gap, not a
kernel gap**, and checked as such rather than assumed: `aten::expand_as` is
`CompositeImplicitAutograd` upstream, and a `TorchDispatchMode` trace of `x.expand_as(y)` fires
exactly one op, `aten.expand.default(x, y.shape)`, which this shim already implements and already
golden-compares. Added in `bootstrap.py` as `self.expand(list(other.shape))` — through `.expand`
rather than through `dispatch`, so it inherits that method's `-1` handling and rank rules and stays
a **view** (upstream's `expand_as` shares storage, measured by `data_ptr()`).

This is ARCH26.md §7's `torch.conv2d` fix in a new place, and it is the third time this round that
the wall behind a kernel turned out to be a spelling.

### 6.4 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 268 | **268** (the new assertions joined the existing road test) |
| `compare.py` | 4682/4682, ops=143 | **4709/4709, ops=143** (+27 cases, same op) |
| `verify_schemas.py` | 4376/4376 | **4376/4376** (unchanged — same schema) |
| sweep26 (shim) | 22/26 | **22/26** |

`ops covered` does not move because this widens an op that was already counted, which is the honest
shape of the change: `convolution.default` was never *absent*, it was 1-D only.

### 6.5 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| C1 | delete the symmetry refusal and silently use axis 0 for both axes — the exactly-plausible mistake | **4706/4709, 3 failed** | 1 FAIL | all three asymmetric cases fire, and the harness's message is the right one: *"gap appears CLOSED: both sides now succeed, promote this case to expect=match and diff real values."* The `c_error` mechanism catches a refusal being dropped, not only a value being wrong |


## 7. Job 2: the blocked transposed copy

docs/SEQLEN.md §8.12 named this as the one clean kernel win left in SDPA and sized it at
**1.15 ms of a 13.64 ms per-call gap at `S=1024` — 8% of the SDPA gap, 7% of the model gap.** It is
the only entry in that table which is bit-identical by construction rather than by tolerance, and
that is why it is the one taken.

### 7.1 What was slow, and why it is safe to change

`sdpa_flash_cpu`'s default branch computes `k.transpose(2, 3).contiguous()`. candle's
`copy_strided_src` walks a transposed layout **one element at a time**, recomputing a
multi-dimensional index per element and reading `head_dim` floats away from the previous one — a
cache miss per element once the source leaves L2.

**Every output element is a copy of exactly one input element.** There is no arithmetic, so there is
no summation order to reassociate and no rounding to move; the only thing a blocked traversal
changes is the order in which the same assignments happen. That is the whole safety argument, and it
is a construction argument rather than a measurement.

**The trap recorded beside it in §8.5 is the opposite change and was not made.** Simply *dropping*
the `contiguous` is 5% faster and moves the `S=6` digest, because it lets Accelerate take a
transposed GEMM with a different accumulation order. This makes the copy faster; it does not remove
it. The `contiguous` call is still there — `transposed_contiguous` is a drop-in for it.

### 7.2 The layout, confirmed against a real forward before anything was timed

docs/DTYPE_PERF.md §2-vs-§4 is the standing warning that this repository has twice been misled by a
microbench at a layout the model never produces. So `k` is captured out of a real
`SmolLM2-135M` prefill by monkeypatching `F.scaled_dot_product_attention`, and the shape **and
strides** are printed before any timing:

```
S=512    k        shape=(1, 3, 512, 64)   stride=(98304, 32768, 64, 1)   contig=True   f32   393216 bytes
         k.T(2,3) shape=(1, 3, 64, 512)   stride=(98304, 32768, 1, 64)   contig=False
S=1024   k        shape=(1, 3, 1024, 64)  stride=(196608, 65536, 64, 1)  contig=True   f32   786432 bytes
         k.T(2,3) shape=(1, 3, 64, 1024)  stride=(196608, 65536, 1, 64)  contig=False
```

That transposed layout is **exactly** the pattern `transposed_plan` recognises: `stride[-2] == 1`,
`stride[-1] == dims[-2]` (64), and the two leading strides contiguous over the pair's area
(`32768 = 512·64`, `98304 = 3·512·64`). 30 SDPA calls per forward.

`TensorBase.stride` is not implemented in this shim, so the strides above are read from the
**upstream** capture of the same tensor; both sides run the identical capture script, and the shim
side prints `contig=False` for the view and `contig=True` for `k`, which is the part it can answer.

### 7.3 The measurement

Alternating old/new with a `cmp`-verified artefact swap each time
(`/Volumes/macMini/caches/k26-scratch/ab.sh`), 3 rounds, 5 timed repetitions each, **minimum and
spread both reported**. `old` is this same tree with the kernel present but not wired in — so the
only difference between the two artefacts is the call site, and the other five kernels this round
added cannot confound it.

Machine: alone, but carrying a windowing server and a user application; `uptime` printed at each
round and was 2.24–3.08 throughout.

**The copy itself**, `aten.contiguous.default` on the captured view:

| | old (min of 3 rounds) | new | ratio | upstream |
|---|---:|---:|---:|---:|
| `S=512` | 0.2123 ms, 1.85 GB/s | **0.0404 ms, 9.73 GB/s** | **5.25x** | 0.0314 ms, 12.5 GB/s |
| `S=1024` | 0.4232 ms, 1.86 GB/s | **0.0875 ms, 8.99 GB/s** | **4.84x** | 0.0582 ms, 13.5 GB/s |

Spread on the new numbers is 0.0002–0.0008 ms; on old, 0.0005–0.0040 (one round showed 0.28, and
that round's minimum is the one reported). **Relative to upstream the copy goes from 6.8x slower to
1.29x at `S=512`, and from 7.3x to 1.50x at `S=1024`.**

**Per SDPA call**, same capture:

| | old | new | delta | spread (old / new) |
|---|---:|---:|---:|---|
| `S=512` | 4.6303 / 4.6330 / 4.6699 | **4.1324 / 4.1356 / 4.1470** | **−0.50 ms (−10.8%)** | 0.04–0.13 / 0.02–0.06 |
| `S=1024` | 17.3902 / 17.4377 / 17.4881 | **16.3991 / 16.4371 / 16.4519** | **−1.01 ms (−5.8%)** | 0.25–0.41 / 0.09–0.57 |

The `S=1024` figure is **1.01 ms against SEQLEN.md's sizing of 1.15 ms** — the sizing was right.

**The control**, the same build against itself, run last in every sweep:

| | run A | run B | ratio |
|---|---:|---:|---:|
| copy `S=512` | 0.0403 | 0.0405 | **1.005** |
| copy `S=1024` | 0.0875 | 0.0872 | **0.997** |
| SDPA `S=512` | 4.1449 | 4.1254 | **0.995** |
| SDPA `S=1024` | 16.4403 | 16.7473 | **1.019** |
| model `S=512` f32 | 349.61 | 349.64 | **1.000** |
| model `S=1024` f32 | 930.56 | 930.93 | **1.000** |
| model `S=512` bf16 | 433.63 | 433.07 | **0.999** |
| model `S=1024` bf16 | 1091.45 | 1092.52 | **1.001** |

Every control reads within 2% of 1.00 and every measured effect is larger than its control, at both
lengths and in both dtypes. The one control that is not within 1% (SDPA `S=1024`, 1.019) is still a
third of the 6.1% effect it sits beside, and that repetition carried a 32 ms outlier which the
minimum discards but the spread reports.

### 7.4 Model level, and the ratios

`f32`, min of 3 per length, 2 alternating rounds:

| S | old | new | upstream | ratio before | **ratio after** |
|---:|---:|---:|---:|---:|---:|
| 6 | 34.27 | 33.91 | 35.38 | 0.97x | 0.96x |
| 32 | 38.75 | 37.89 | 38.15 | 1.02x | 0.99x |
| 128 | 85.45 | 80.43 | 75.57 | 1.13x | **1.06x** |
| 512 | 364.27 | 350.17 | 229.31 | **1.589x** | **1.527x** |
| 1024 | 961.21 | 929.35 | 460.22 | **2.089x** | **2.019x** |

The two "before" ratios reproduce the brief's stated baseline exactly (1.58x, 2.09x), which is the
cross-check that this harness is measuring the same thing.

`bf16`, same protocol:

| S | old | new | delta |
|---:|---:|---:|---:|
| 512 | 446.86 | **433.70** | −2.9% |
| 1024 | 1122.01 | **1091.50** | −2.7% |

### 7.5 Bit-identity, proved by the digests

**All ten prefill digests are unchanged**, at every length docs/SEQLEN.md §1.3 records, in both
dtypes, on both sides of the change:

| S | `f32` | `bf16` |
|---:|---|---|
| 6 | `b9fc5553ee1bf6a2` | `8ef1550ea33c4f3d` |
| 32 | `331668f36da02f21` | `b81325c83a0a3d15` |
| 128 | `00159a9dbd308eda` | `7ff8e9334449b147` |
| 512 | `07c2797dabc4552e` | `9ab1e82f01378e38` |
| 1024 | `eda1e173727bb7f5` | `b4d9440df61212b1` |

Every `f32` row matches SEQLEN.md §1.3 and the `bf16` `S=128` row matches SEQLEN.md §4's
`7ff8e9334449b147`. Upstream's `S=128` digest came back `71e46824c0c40f15`, the value §1.3 records
for it — so the harness is reading the same model the previous rounds read.

### 7.6 Where it is wired, and the second call site

Two places, and the second is what makes the copy directly measurable:

1. `sdpa_flash_cpu`'s `k.transpose(2, 3)` — the site SEQLEN.md sized.
2. **`aten.contiguous.default`** — so `x.contiguous()` on any transposed view gets it, and so the
   microbench in §6.3 is timing the kernel rather than timing candle on both sides. (The first
   attempt at this measurement wired only site 1, and the copy row read 0.2123 vs 0.2120 — an
   accidental null control that showed the harness was not reaching the code under test.)

`contiguous.default`'s aliasing contract is unchanged and still asserted by
`test_which_ops_share_storage_with_their_input_and_which_do_not`, which has it in **both** tables:
it shares storage with a contiguous input (the fast exit returns candle's handle clone) and is
independent of a strided one (it copies). The fast exit is the first thing `transposed_contiguous`
checks, which is what keeps that true.

### 7.7 Sabotage check

| # | fault | `cargo test` | `run.sh` | golden | digests |
|---|---|---|---|---|---|
| T1 | ragged-edge bug: `r0 + BLOCK.min(cols)` instead of `(r0 + BLOCK).min(cols)` — correct inside a full block, overruns on the remainder | **1 FAILED** (panics at `3x5`) | — | — | — |
| T2 | **in-bounds and silent**: `dst[r*rows+c] = row[cols-1-r]` — right shape, right dtype, wrong values | **4 FAILED** | **3 FAIL**, including the real llama forward and `do_sample` | **4645/4682, 37 failed** | **all moved**: `S=6` `5948eac9…`, `S=128` `9954d220…`, `S=512` `a6f101ec…` |

T2 is the one that matters. It is the silent failure mode — a tensor of exactly the right shape and
dtype with the wrong numbers in it — and it is caught at four independent levels, **including the
prefill digests**. That is the proof the digests in §6.5 are a real gate on this change and not a
formality: a change to this kernel that was not bit-identical would move them, and this one does
not.

Sizes in the Rust test straddle the 32-element block on both axes in both directions
(`33x31`, `31x33`, `129x65`) and include the real `(512, 64)` SDPA shape, so an edge bug on either
side has a shape that shows it. A separate test copies `-0.0`, both infinities, both NaNs and a
subnormal and compares `to_bits()`, because `==` cannot see the first or the last.

---

## 8. Where every architecture stands, and what is left

### 8.1 The sweep, kernel by kernel

**20/26 → 22/26.** Which kernel moved which wall, in the order they were done:

| after | sweep | `deberta` / `deberta_v2` | `vits` | `sew_d` | `zoedepth` | `sam3_video` |
|---|---:|---|---|---|---|---|
| *(baseline)* | 20/26 | `torch.sqrt` | `set_` | `Tensor(int)` | `convolution` 4-D | `__mod__` |
| `sqrt` | 20/26 | → `repeat` | — | — | — | — |
| `repeat` | **22/26** | **PASS** | — | — | — | — |
| `remainder` | 22/26 | — | — | — | — | → `div.Scalar_mode` |
| `Tensor(int)` | 22/26 | — | — | → `set_` | — | — |
| `set_` | 22/26 | — | → `norm_except_dim`\* | → `norm_except_dim`\* | — | — |
| `convolution` 2-D | 22/26 | — | — | — | → `expand_as` | — |
| `expand_as` | 22/26 | — | — | — | → `conv_transpose2d` | — |

\* surfacing as `TypeError: _WeightNorm.forward() missing 1 required positional argument` — see §5.3.

**Only `repeat` moved the count**, and it could only move it because `sqrt` came first. Every other
kernel moved a first wall without closing an architecture, which is what §0 said this table would
have to show rather than crediting the batch.

### 8.2 What is left, with reasons

**Left because the next wall is a different kernel, not because the work was hard.** Each of these
was reached, named and left:

| architecture | next wall | size |
|---|---|---|
| `vits`, `sew_d` | **`aten.norm.ScalarOpt_dim`** then **`aten._weight_norm_interface.default`** | two kernels, both small; §5.4 has the traces. `norm_except_dim` is a composite over the first, `torch._weight_norm` fires exactly the second |
| `zoedepth` | **`aten.convolution.default`, `transposed=True`** (`torch.conv_transpose2d`) | candle has `conv_transpose2d`, so it is *probably* the same wrapper again — but the weight-layout convention of a transposed convolution is `(in, out/groups, kH, kW)` rather than `(out, in/groups, kH, kW)`, and getting that backwards produces a plausible tensor. It needs its own measurement round, not an extension of §6 |
| `sam3_video` | **`aten.div.Scalar_mode`** / `div.Tensor_mode` (`rounding_mode="floor"`) | one kernel, small. It is the `y_positions` line two characters from the `% end_x` §3 fixed |

**Behind those**, from ARCH26.md §6's operator sweep, re-checked against `all_implemented()` rather
than against `_aten_implemented()` — which matters, because three of that table's entries turned out
to *already exist*, parked in `IMPLEMENTED_AWAITING_GOLDEN`:

```
already implemented, contrary to ARCH26.md §6:  aten.zeros.default   aten.add.Scalar
                                                aten.masked_fill.Tensor
```

Genuinely still missing, none of them attempted here:

```
aten.sigmoid.default        aten.sign.default          aten.erf.default
aten.log2.default           aten.clamp_min.default     aten.clamp_min.Tensor
aten.flip.default           aten.leaky_relu.default    aten.ones_like.default
aten.randn_like.default     aten.all.default           aten.div.Tensor_mode
aten.div.Scalar_mode        aten.avg_pool2d.default    aten.native_group_norm.default
aten.upsample_bilinear2d.default   aten.norm.ScalarOpt_dim
aten._weight_norm_interface.default                    aten.fmod.{Tensor,Scalar}
aten.remainder.Scalar_Tensor  (and with it `__rmod__`)
aten.set_.source_Tensor_storage_offset
```

The first six of those are one-line members of families that already exist here (`unary_float` for
`sigmoid`/`erf`/`log2`, the comparison family for `clamp_min`) and are the cheapest remaining work.
`native_group_norm`, `avg_pool2d` and `upsample_bilinear2d` are real kernels with their own
measurement rounds.

**Left deliberately, with the reason rather than the size:**

- **`torch.Tensor([3, 4])`**, the build-from-data constructor (§4.2). It looks like a size list and
  is not one; refusing is better than answering `(3, 4)` zeros where upstream answers two numbers.
- **`remainder.Scalar` on a `bool` tensor** (§3.3). Upstream's promotion there is a fast-path ladder
  keyed on the scalar's *Python type*, and `scalar_arg` has erased that by the time the kernel runs.
  Two `c_error` golden cases watch it.
- **asymmetric 2-D convolution** (§6.2). candle's `conv2d` is symmetric-only; three `c_error` cases
  watch it.
- **`sqrt.out`, `remainder.{Scalar,Tensor}_out`** — in `overloads.json` with no kernel, so the `out=`
  spelling refuses by the right name instead of falling through to "no table entry for this op".

### 8.3 One correction to docs/ARCH26.md

ARCH26.md §6 says `weight_norm` costs **two** kernels. **It costs three.** The third,
`aten.norm.ScalarOpt_dim`, was invisible to that document's method for two compounding reasons:
`torch.norm_except_dim` is a *composite* (so the op name never appears in the source), and the trace
that found `_weight_norm_interface` ran on a **forward**, while `norm_except_dim` is called from
`register_parametrization` at **construction** time. That is ARCH26.md §6's own stated blind spot —
the one it demonstrated on `sam3_video` — appearing a second time in the same document.

### 8.4 Final gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   268 ok, 0 FAIL                    exit 0   (was 261)
$PY tools/golden/compare.py                 4709/4709, ops=143, pending 1     exit 0   (was 4290/4290, ops=139)
$PY tools/golden/compare.py --self-test     14 comparators x 11 fault modes   exit 0   (was 13)
$PY rust/torch_c/pytests/verify_schemas.py  4376/4376                         exit 0   (was 4353/4353)
( cd rust/torch_c && cargo test --release ) 28 passed                         exit 0   (was 24)
sweep26 (shim)                              22/26                             exit 0   (was 20/26)
sweep26 (upstream)                          26/26                             exit 0
```

Prefill digests on the final artefact, re-run after every change above:

```
f32   S=6 b9fc5553ee1bf6a2   S=32 331668f36da02f21   S=128 00159a9dbd308eda
      S=512 07c2797dabc4552e   S=1024 eda1e173727bb7f5
```

all five unchanged from docs/SEQLEN.md §1.3.
