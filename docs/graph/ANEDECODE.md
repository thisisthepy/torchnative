# ANEDECODE — a decode step on the Neural Engine, and what actually decides

[`../platform/RELEASE_0_1_0b4.md`](../platform/RELEASE_0_1_0b4.md) §3 shipped
this as a measured limit:

> **A decode step reaches the Neural Engine on none of its Linears.** Per
> shape, batch 1 against batch 128: 576→192, 576→576, 576→1536 and 1536→576
> are all **CPU** at batch 1 and NeuralEngine at 128; `lm_head` is CPU at
> both. The unit is *supported* for all of them and CoreML prefers the CPU at
> batch 1 — and a decode step is batch 1 by definition. **This arm is a
> prefill story, not a decode story.**

The table is right and this round reproduced it exactly before changing
anything. The *explanation* attached to it — that one token through a 576-wide
projection is not enough work, [`NPU2.md`](NPU2.md) §8.2 — is **not what the
measurements below support**, and the difference matters because it points at a
different fix.

Two things decide, and neither is "how much arithmetic".

1. **The form.** `ios16.linear` at batch 1 is CPU-preferred at every size
   tried. A 1x1 `ios16.conv` over a rank-4 `(1, C, 1, 1)` tensor is not.
2. **The program, not the operation.** The threshold is roughly **4.7M weights
   in the compiled program**. This project compiles one program per leaf, so a
   projection can never reach it alone.

Everything here is `MLComputePlan`'s **`preferred`** column. `supported` said
NeuralEngine for all five shapes throughout, while all five ran on the CPU;
that is the whole reason §3 could be written, and it is why `supported` is not
used as evidence anywhere below.

---

## 1. `linear` cannot reach the unit at batch 1, at any size

The first thing to eliminate, because without it every result below could be a
size effect wearing a layout's clothes. A single rank-2 `ios16.linear`,
float16, batch 1, `ComputeUnit.ALL`:

| in→out | weights | `preferred` |
|---|---|---|
| 576→192 | 0.11M | CPU |
| 576→576 | 0.33M | CPU |
| 576→1536 | 0.88M | CPU |
| 576→4096 | 2.36M | CPU |
| 576→8192 | 4.72M | CPU |
| 576→16384 | 9.44M | CPU |
| 576→49152 | 28.3M | CPU |

And it does not help to give it company. 64 chained 576→576 linears in **one**
program — 21M weights, four times the threshold a chain of convs crosses at —
are CPU-preferred on every one of the 64 operations.

So `linear` at batch 1 is not a matter of degree. It is excluded.

## 2. The same arithmetic as a 1x1 conv is not excluded

Identical dot products, rank-4 `(1, C, 1, 1)` input, `mb.conv` with a 1x1
kernel — Apple's `ml-ane-transformers` layout:

| in→out | weights | `linear` | `conv` 1x1 |
|---|---|---|---|
| 576→192 | 0.11M | CPU | CPU |
| 576→576 | 0.33M | CPU | CPU |
| 576→1536 | 0.88M | CPU | CPU |
| 576→4096 | 2.36M | CPU | CPU |
| 576→8192 | 4.72M | CPU | **NeuralEngine** |
| 576→16384 | 9.44M | CPU | **NeuralEngine** |
| 576→49152 | 28.3M | CPU | **NeuralEngine** |

Apple's guidance is confirmed, and narrowed: the rank-4 1x1-conv form is
**necessary** and not sufficient. It moves the crossover from "never" to
"about 4.7M weights"; it does not remove one.

## 3. The threshold is weights in the program, not arithmetic

Three controls separate the candidates.

**Not the aspect ratio, and not out-channels.** Four shapes at the same 4.72M
weights, conv, batch 1 — 576→8192, 8192→576, 2304→2048, 2048→2304 — are
**all** NeuralEngine. 4096→576 (2.36M) is CPU. Only the product moves it.

**Not MACs.** A 576→576 conv over `(1, 576, 1, S)` holds 0.33M weights whatever
`S` is, and does 0.33M·S MACs. At S=128 that is 42M MACs — nine times the
threshold — and it is **CPU-preferred**, as it is at every S from 1 to 128. So
the quantity is the weight set, not the work.

**Not the operation.** Chained 576→576 convs, batch 1, all identical:

| convs in one program | weights | `preferred` |
|---|---|---|
| 1 | 0.33M | CPU |
| 2 | 0.66M | CPU |
| 4 | 1.3M | CPU |
| 8 | 2.7M | CPU |
| 16 | 5.3M | **NeuralEngine** (all 16) |
| 32, 64 | 10.6M, 21M | **NeuralEngine** |

Nothing about any operation changed between rows 4 and 5. **"Which unit runs
this leaf" has no answer that is a property of the leaf.** A lowering that
emits one program per leaf has already answered it, and answered CPU.

That is a fact about *this project's lowering strategy*, not about the
hardware, and it is the finding with the most left in it.

## 4. A real decode step, and a negative that has to be withdrawn

SmolLM2-135M's decoder-layer projections (q 576→576, k/v 576→192, o 576→576,
gate/up 576→1536, down 1536→576 = **3.54M weights per layer**) as 1x1 convs, at
batch 1, with the `silu`, `mul` and `add` a decode step really contains:

| layers | weights | `preferred` |
|---|---|---|
| 1 | 3.54M | CPU — every op |
| 2 | 7.08M | **NeuralEngine — every op** |
| 3, 4 | 10.6M, 14.2M | **NeuralEngine — every op** |

Every operation includes the `silu`, `mul` and `add`. That **withdraws** a
negative from `RELEASE_0_1_0b4.md` §3:

> `layer_norm`, `relu`, `gelu`, `softmax`, `max_pool` and `batch_norm` are
> ANE-*supported* at float16 and never ANE-*preferred*, at every size tried.

They were measured as **one-operation programs**, which hold no weights at all
and therefore cannot clear §3's threshold by construction. "At every size
tried" varied the tensor, which §3 above shows is not the variable. In a
program that does clear the threshold they are preferred on the unit like
everything else. The rejection of those leaf types was decided on a number that
was measuring the wrong thing.

One SmolLM2 layer is *just* under the threshold, which is unlucky and is also
the whole shape of the remaining work: **two layers per program would do it.**

## 5. Negative results, recorded because they are the first things to try

**No `computeUnits` setting forces the unit.** 576→576 linear, batch 1:

| setting | `preferred` | `supported` |
|---|---|---|
| `ALL` | CPU | CPU, GPU, NeuralEngine |
| `CPU_AND_NE` | CPU | CPU, NeuralEngine |

`CPU_AND_NE` removes the GPU from the column it was never chosen from and
leaves `preferred` exactly where `ALL` left it. `MLComputePlan` has no forcing
knob: `computeUnits` **bounds** the choice, it does not make it. The same holds
for the conv form — `CPU_AND_NE` changed no `preferred` value at any shape
measured.

**Static versus flexible input shape changes nothing.** Measured rather than
argued, because "we already use the favourable case" is an argument:

| program | shape | `preferred` |
|---|---|---|
| `conv` 576→49152 | static `(1, 576, 1, 1)` | NeuralEngine |
| `conv` 576→49152 | symbolic last dim `(1, 576, 1, s)` | NeuralEngine |
| `linear` 576→49152 | static `(1, 576)` | CPU |
| `linear` 576→49152 | symbolic batch `(b, 576)` | CPU |

A flexible dimension neither costs the conv form the unit nor buys the linear
form it. Note that `mb.TensorSpec` takes an integer or a MIL symbol and
**rejects `RangeDim` outright** — that is a frontend `ct.TensorType` concept
and does not reach a program built through `mb.program`, so "enumerated
shapes" is not a setting this lowering has to offer.

**`lm_head` is not different in kind — it is the opposite of hard.** §3 of the
release notes singled it out as CPU at batch 128 *as well*, which reads as a
shape the unit struggles with. It is the reverse: at 28.3M weights it is the
only projection large enough to clear the threshold **on its own**, so in the
conv form it is the one shape that reaches the unit at batch 1. Its oddity at
batch 128 is `linear`'s, not the shape's.

## 6. What was implemented, and what it is worth

`_CoreMLLinear._compile_for` emits the 1x1 conv form **at batch 1 only**.

| shape at batch 1 | before | after |
|---|---|---|
| 576→192 | CPU | CPU |
| 576→576 | CPU | CPU |
| 576→1536 | CPU | CPU |
| 1536→576 | CPU | CPU |
| 576→49152 (`lm_head`) | CPU | **NeuralEngine** |

One shape of five. It is the largest Linear in SmolLM2-135M — 28.3M of the
model's 162M parameters — and it is the half a **per-leaf** lowering can
deliver; §3 is why the other four are not reachable without changing the unit
of compilation, and that change is not made here.

**The batch-1 bound is measured, not cautious.** Above batch 1 the conv form is
*worse*. Sabotaging the implementation to apply it at every batch, and
re-reading the plan at batch 128:

| shape at batch 128 | `linear` (shipped) | `conv` (sabotaged) |
|---|---|---|
| 576→192 | NeuralEngine | **CPU** |
| 576→576 | NeuralEngine | **CPU** |
| 576→1536 | NeuralEngine | NeuralEngine |
| 1536→576 | NeuralEngine | NeuralEngine |

Applying it unconditionally would trade two measured prefill wins for nothing.
`_compile_for` is keyed by batch and the two forms never meet.

**It costs no accuracy.** At 576→576 the conv output is **bit-for-bit** the
linear output. At `lm_head` the two differ by 1.3e-03 relative *because they now
run on different units*, which is inside the 1.5e-03 float16 grade
`RELEASE_0_1_0b4.md` §3 already named — **not** a widened tolerance — and the
conv form is the **closer of the two to float32** (4.2e-04 against 1.3e-03).
Both pick the same argmax, which is the token. The float16-versus-float32
question is unchanged by this round and remains as §3 states it.

## 7. What is not done

* **The other four projections.** They need two decoder layers in one program
  (§4), which means lowering a *subgraph* rather than a leaf. This project
  lowers leaves. That is the next round and it is not small: it needs the
  residual stream, the norms and the KV cache inside the program.
* **The KV-cache decode loop is unverified end to end.** Everything here is
  `MLComputePlan` on the projections a decode step performs, at the shapes it
  performs them. Whether the win survives a real `generate()` — where the cache
  concatenation and the attention matmul are not leaves and stay in eager torch
  — is not measured, and the per-leaf tensor round trip is very likely to eat
  it. **No speed claim is made here, in either direction.**
* **No timings at all.** The host was under load average 38 for this round's
  duration, with other agents building. CLAUDE.md's rule on solitary
  measurement applies and a number taken under that load would be worthless.
  `preferred` is a scheduler decision and is not load-sensitive; latency is.
* **The alternative, if §7's first bullet proves too expensive:** make the
  batch dimension stop being 1. Speculative or multi-token decoding puts 4–8
  candidate tokens through the same projections, and §1's crossover for
  `linear` already sits between batch 1 and 16 for three of the four shapes
  (`NPU2.md` §8.2). That reaches the unit without changing the unit of
  compilation at all.

## 7a. bfloat16 is not excluded from the conv path

The rewrite changed the op a batch-1 leaf emits, and `test_bf16ane.py` had a
test asserting that op by name. It went red reading
`{'preferred': ['CPU'], 'supported': [...], 'ops': ['ios16.conv']}`, which is
easy to read as *the conv rewrite is a regression for bfloat16* — the rewrite
demonstrably applied, and the unit came out CPU.

It is not a regression, and the shape is why. That test's leaf is
576 → 1536: **0.88M weights**, an order of magnitude under §3's ~4.7M program
threshold. A float16 leaf of exactly that shape is CPU-preferred at batch 1
too — `test_anedecode.py::test_the_four_projection_shapes_are_still_cpu_at_batch_one`
asserts it. So the CPU verdict is the threshold, not the dtype, and the old
assertion was about the lowering's internal form rather than about bfloat16.

Asked at a size where the question can be answered, the control is
unambiguous. A bfloat16 `Linear` of 576 → 49152 (28.3M weights), widened to
float16 by the same path every bfloat16 checkpoint takes, compiled at batch 1:

| batch-1 form | `ops` | `preferred` |
|---|---|---|
| rewrite on | `ios16.conv` | **NeuralEngine** |
| rewrite off | `ios16.linear` | CPU |

The second row is measured, not assumed: it is what the test reports when the
rewrite is disabled on purpose. bfloat16-sourced weights reach the unit
through `ios16.conv` at the size where anything does.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bf16ane.py test_a_bfloat16_linear_is_not_excluded_from_the_conv_path present -->

## 7b. The defect that made this module report as a subprocess exit

All nine tests in `test_anedecode.py` failed identically with
`RuntimeError: npu subprocess exited 1`, which names the exit status and
nothing else. The cause was in the first ten lines of the spliced script:
`test_shim._STDOUT_GUARD` used `io`, `os` and `sys`, and required the splicing
script to have imported all three — a contract recorded only in a comment.
This module imported `json`, `os` and `sys`. The guard died in
`sys.stdout = io.StringIO()` with `NameError`, before any measurement ran.

Every CoreML fixture in this repository splices that guard, so the trap was
one missing import away for each of them. The guard now carries its own
`import io, os, sys`; three redundant stdlib imports cost nothing and no
script can get it wrong.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_anedecode.py test_the_stdout_guard_carries_its_own_imports present -->

## 8. Reproducing

`rust/torch_c/pytests/test_anedecode.py` is every measurement in this document,
as assertions on `preferred`. It skips by name where coremltools or the
vendored shim is absent.

## 9. Split the way CLAUDE.md §5.3 asks

| | |
|---|---|
| features added | 1 — the batch-1 conv form in `_CoreMLLinear` |
| defects fixed | 1 — `_STDOUT_GUARD` required imports it did not make (§7b) |
| tests added | 11 — 10 in this new module, 1 bfloat16 control in `test_bf16ane.py` |
| docs corrected | 4 — `RELEASE_0_1_0b4.md` §3, this document's account of `NPU2.md` §8.2, and two `test_bf16ane.py` tests whose assertions were about the old form (§7a) |
| removed | 0 |

One shape of five moved. The larger result is §3 and §4: the reason the other
four did not is the per-leaf program, and that had been attributed to the size
of a token.
