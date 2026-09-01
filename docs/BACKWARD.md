# A training step: the tape, the gradients, and where they stop agreeing

`docs/AUTOGRAD.md` §6.6 ended with a recommendation and an order of work:

> **Build the tape.** Reverse-walk a captured, Core-ATen-lowered trace, with a `grad` map keyed on
> the trace's existing value identities, and derivative rules written against Core ATen only.

Steps 1 and 2 of that order landed in `docs/LOSS.md` — a real loss, and `native_dropout` so a
`.train()` forward captures. This is step 3, and it reached step 4 and step 5 as well.

**The target was one optimiser step on a real model that moves the weights the way upstream moves
them.** Not full autograd. The smallest thing that is honestly a training step:

```
forward with labels -> loss -> backward -> p.grad populated -> SGD step -> weights moved
```

Environment: `/Volumes/macMini/caches/spike-venv/bin/python`, torch 2.13.0, transformers 5.15.1,
worktree at `develop` `a39d0b4`. Upstream is the oracle for every gradient below.

### Answers, before the evidence

| question | answer |
|---|---|
| Does a training step run? | **Yes, on the whole of SmolLM2-135M.** 1862 nodes, 272 of 272 parameters get a gradient, SGD moves all 272 (§4) |
| Do the gradients agree with upstream? | **All 134,515,008 of them were compared.** Median relative L2 **8.8e-05**, element sign agreement **99.9987%** (§4.2) |
| Is SDPA's backward the wall `docs/AUTOGRAD.md` §5.1 predicted? | **No.** It is the one op there that has no Core ATen decomposition — but a *rule* is not a kernel, and the tape recomputes the attention it needs. §3.4, and §4.4 measures that removing SDPA entirely changes the residual by 1% |
| Where does the 8.8e-05 come from, then? | **The forward, not the tape.** The same tape on the same model in `float64` agrees to **8.5e-07** (§4.4) |
| Are the rules right in isolation? | Every one of the **56** is checked against central differences in `float64`, an oracle that shares no code with them; a decoder-shaped `nn.Linear`/MLP case is **bit-identical** to upstream (§2, §5) |
| Did a check fail? | **Yes, and it was the check.** A finite-difference probe on the real model disagrees with the tape by 600× — and with **upstream's own autograd** by the same 600× (§6) |
| How much of upstream's machinery did this need? | **None of `torch/csrc/autograd`.** No `VariableType` wrappers, no `AutogradMeta`, no engine, no version counters. One reverse walk of a list, and 56 rules that are compositions over ops that already existed (§1) |

Written incrementally, one stage at a time, for the reason `docs/KERNELS26.md` §0 gives.

### The baseline, every gate, before any edit

```
pytests/run.sh                296 ok, 0 FAIL, DOCWATCH 95/95      exit 0
tools/golden/compare.py       7447/7447, ops=166, pending=1       exit 0
compare.py --self-test        19 comparators x 11 fault modes     exit 0
verify_schemas.py             4475/4475                           exit 0
sweep26   (shim, .eval())     26/26                               exit 0
sweeptrain (shim, .train())   26/26                               exit 0
```

---

## 1. What the tape is, and what it did not need

`rust/torch_c/src/tape.rs`, 1644 lines, and the shape of it is the point:

```
replay   the forward, keeping every intermediate     (PyCaptureTrace::run)
seed     the declared outputs' gradients
walk     the node list backwards, once, accumulating into a map keyed on the trace's own Ref
```

Three properties come from **capture** rather than from anything in the new file, and they are
exactly the properties upstream spends `torch/csrc/autograd` on:

| property | how upstream gets it | how this gets it |
|---|---|---|
| single assignment | `ADInplaceOrView` + a version counter per tensor | capture refuses in-place ops (docs/CAPTURE.md §4), so nothing recorded is ever overwritten |
| a graph to walk | a generated `VariableType` wrapper per op allocates a `Node` and links it | the record **is** the graph; the recorder is one line at one door |
| a reverse order | ready queue, dependency counts, `GraphTask`, 1862 lines of `engine.cpp` | the region is straight-line, so it is the forward order reversed |

`docs/AUTOGRAD.md` §2.5 listed four layers this shim would have to provide. It provided **one and a
half**: the walk, and a `.grad` slot. `VariableType`'s 163 wrappers, `AutogradMeta`, the engine's
device threads and reentrancy, and version counters are all absent and none of them was missed.

### 1.1 The rules are compositions, and that is what dissolves §4's bill

`docs/AUTOGRAD.md` §4 counted the derivative requirement as **25 formulas needing their own
kernel**, reaching 24 distinct backward ops of which **4 must be hand-written**. That is the bill
for a *kernel* set. It is not the bill for a *rule* set, for one reason that only became clear once
the backward existed:

> **The backward runs outside a capture region.** It may use ops capture would refuse to record, it
> may mutate, and it may recompute a value instead of reading a saved one.

So `_softmax_backward_data` is not needed — `g - out * rowsum(g * out)` is. `nll_loss_backward` is
not needed — a `scatter` into a zero buffer is.
`_scaled_dot_product_flash_attention_for_cpu_backward`, the one op on SmolLM2's path with neither a
CPU kernel here nor a Core ATen decomposition, is not needed either: §3.4.

**Zero new aten kernels were written for this document.** `ops=166` before and after, and
`tools/golden/compare.py` is unchanged at 7447/7447 — which is the check that says so, because a
new kernel would have had to appear there.

### 1.2 The surface

```python
trace.backward(inputs, grad_outputs=None, wrt_constants=None)
    -> {"inputs": [Tensor | None, ...], "constants": [Tensor | None, ...]}

trace.differentiable(wrt_constants=None)
    -> {"nodes": int, "nodes_on_a_gradient_path": int,
        "covered": {op: count}, "missing": {op: count}}

_C._tape_rules() -> [op, ...]        # the 56 rules, readable from Python
```

`constants` is where a model's weights are: capture burns in every tensor it was not handed as an
input (docs/CAPTURE.md §2), which is the same split `ExportedProgram.graph_signature` makes between
user inputs and lifted parameters. So *"the gradient of the loss with respect to this parameter"* is
*"the gradient at this constant"*, matched by object identity through `trace.constant_values`.

`differentiable()` exists so that **"what stops this model" is answerable without running a backward
and reading an exception.** Every wall table in this document is generated from it. Hand-kept op
lists have gone stale in this repository five times.

### 1.3 The one thing that is paid twice

`PyCaptureTrace` keeps the *shape* of every intermediate, not the intermediate — capture drops its
keepalives at `_capture_end` (docs/CAPTURE.md §6). So `backward()` **replays the forward first**, to
materialise the activations it needs, and a backward therefore costs two forwards rather than one.
That is a deliberate default: holding every activation for the life of every trace would be wrong
for the traces that are never differentiated. Measured on SmolLM2-135M at S=8, the replay plus the
whole reverse walk is **0.4 s**.

`run()` is a refactor of `replay()`, not a second interpreter — `replay` is now `run` plus a
projection onto the declared outputs. A backward that materialised activations its own way would be
differentiating a different forward from the one `replay` proves equal to eager.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs RULE_OPS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs sdpa_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs nll_loss_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs crate::tape::backward present -->

---

## 2. Stage one: a `nn.Linear`, and a gradient that is bit-identical

The smallest thing that is a gradient. `nn.Linear(8, 4)` on a `[3, 8]` input, `.sum()` as the loss,
weights built from a deterministic generator so both sides see identical bytes with no shared RNG.

The trace is three nodes and it is worth reading, because it is the whole design in miniature:

```
%0 = aten.t.default(%c0)                    ->  [8, 4]
%1 = aten.addmm.default(%c1, %in0, %0)      ->  [3, 4]
%2 = aten.sum.default() {'self': %1}        ->  []
constants  [[4, 8], [4]]      <- the weight and the bias
```

`%c0` and `%c1` are the parameters. Two things follow immediately: the arguments arrive **all
keyword** for one node and **all positional** for the next (docs/CAPTURE.md §2 measured that; the
vendored tree binds by name before dispatching), so every rule binds against a schema's parameter
names rather than reading `args[0]`; and a gradient for a parameter is a gradient at a constant.

Against upstream, comparing **bit patterns** rather than with a tolerance:

| | n | elements differing bit-for-bit | max abs |
|---|---:|---:|---|
| `grad weight` | 32 | **0** | 0.0 |
| `grad bias` | 4 | **0** | 0.0 |
| `grad input` | 24 | **0** | 0.0 |

The loss itself is *not* bit-identical (4.8e-07) — the shim's `sum` reduces serially and upstream's
does not — and the gradients are, because `sum`'s derivative never touches the sum.

---

## 3. Stage two: a loss, an activation, and attention

Four more cases, each adding one thing that a `Linear` does not have. Weights and inputs are again
built from the same deterministic generator on both sides. `|d|/max|g|` is the largest elementwise
disagreement scaled by the largest gradient in that tensor — a plain relative error on a gradient
whose smallest entries are near zero says more about those entries than about the rule.

| case | what it adds | worst bit-diff | worst `|d|` | worst `|d|/max|g|` |
|---|---|---:|---|---|
| `linear_sum` | — | 0 of 60 | 0.0 | **0.0** |
| `linear_ce` | `cross_entropy`, i.e. `_log_softmax` + `nll_loss_forward` | 20 of 40 | 2.98e-08 | 1.27e-07 |
| `mlp_ce` | a hidden layer and `silu` | 85 of 128 | 5.96e-08 | 2.06e-07 |
| `mlp_ce_ignore` | `ignore_index=-100` on one row | 57 of 128 | 2.98e-08 | 1.03e-07 |
| `rmsnorm` | `pow`/`mean.dim`/`rsqrt`, and a `[6]` weight broadcast against a `[2, 3, 6]` | 19 of 36 | 4.77e-07 | 7.87e-08 |
| `attention` | SDPA, causal, **grouped-query 9:3** | 246 of 360 | 7.15e-07 | 8.54e-07 |
| `attention_mask` | SDPA with an additive `attn_mask` | 87 of 96 | 2.38e-07 | 4.49e-07 |

Everything is inside one `float32` ULP of the gradient's own scale. The bit-diff column is there to
say that these are *not* bit-identical and that the tolerance is doing work; §5's `float64` check is
what says the rules are right rather than merely close.

### 3.1 `nll_loss_forward`'s gradient reads the op's **second** result

`docs/LOSS.md` §3.1 found that `total_weight` is not decoration and that every caller in
`transformers` drops it. The backward is the caller that does not: for `reduction=Mean` the divisor
is `total_weight`, and **not** the number of rows. The two are equal for every unweighted call with
nothing ignored, which is every case anyone writes first — §7's T5 is that fault and the first
version of the test could not catch it.

The other two things the rule has to get right, both from the same section read backwards:

* an ignored row contributes nothing, and
* its target cannot be scattered with, because it is routinely out of range (`-100`). It is sent to
  column 0, which is only safe *because* its value is already zero. Zeroing after clamping rather
  than before is §7's T6.

### 3.2 The broadcast that is invisible until it is not

`reduce_to` — undo broadcasting by summing the axes it expanded — is the one piece of arithmetic
every elementwise rule needs and the easiest to leave out, because it does nothing whenever the two
operands already have the same shape. An RMSNorm is where it shows: `weight * hidden` is a `[576]`
against a `[1, 8, 576]`. §7's T3 removes it; without the `rmsnorm` and `mul.Tensor` broadcast cases
it fails nothing.

### 3.3 The `float32` residual is not where a `float32` test can see it

None of the seven cases above separates a *summation order*, for the reason `docs/LOSS.md` §5.3
gives about the forward: at `float32` the shim's serial reductions and upstream's vectorised ones
differ by less than any tolerance a small case can set. §5 is where that is answered, in `float64`,
against an oracle that is not upstream at all.

### 3.4 SDPA's backward: a rule, not a kernel

`docs/AUTOGRAD.md` §5.1 named `_scaled_dot_product_flash_attention_for_cpu_backward` as *"the only
genuinely new kernel on this list"* and §4.2 as one of four with neither a CPU kernel here nor a
Core ATen decomposition. **It is still true that the kernel does not exist, and it stopped nothing**,
because the tape needs a derivative and not a kernel:

```
  P  = softmax(scale * q k^T + mask)        out = P v
  dv = P^T dout                             dP  = dout v^T
  dS = P * (dP - rowsum(dP * P))
  dq = scale * dS k                         dk  = scale * dS^T q
```

Every op in that already existed. Three details are not optional and each is a measured property of
the forward rather than a choice:

* **`is_causal` is upper-left aligned**, not bottom-right — `aten.rs`'s own comment measured that
  on a (q=2, kv=5) pair. The rule rebuilds the mask the same way.
* **Grouped-query attention is inside the kernel**, so key and value are repeated to the query's
  head count before anything touches them, and the gradient has to be **summed back down over each
  group**. Getting this wrong yields a `[1, 9, ...]` gradient where a `[1, 3, ...]` was wanted, or —
  worse, and this is what §7's T8 does — the right shape holding one group's contribution instead of
  the sum of three.
* `dropout_p > 0` is refused by name, because the forward refuses it too and a gradient would need a
  draw that was never made.

What it costs is a second attention and one `[B, H, T, S]` probability matrix per layer — upstream's
fused kernel exists to avoid exactly that, so this is correctness bought with memory. §4.5.

<!-- DOCWATCH: op-implemented aten._scaled_dot_product_flash_attention_for_cpu.default -->
<!-- DOCWATCH: op-not-implemented aten._scaled_dot_product_flash_attention_for_cpu_backward.default -->
<!-- DOCWATCH: op-not-implemented aten.nll_loss_backward.default -->
<!-- DOCWATCH: op-not-implemented aten._log_softmax_backward_data.default -->
<!-- DOCWATCH: op-not-implemented aten.embedding_dense_backward.default -->

---

## 4. Stage three: the whole of SmolLM2-135M

Real weights from the HF cache, `float32`, `.train()`, the deterministic ids `(i*7919+13) % 49152`,
`labels=ids`, `S=8`, 134,515,008 parameters — the recipe `docs/AUTOGRAD.md` §5 and `docs/LOSS.md` §4
use. The **entire** `model(input_ids=ids, labels=ids)` call is captured, embedding included.

```
captured <CaptureTrace 1862 nodes, 1 inputs, 333 constants, 1 outputs>   0.1 s
loss 12.871352195739746           (upstream 12.871366500854492, docs/LOSS.md §4)
parameters that are trace constants: 272 of 272
nodes 1862, on a gradient path 1723, distinct ops 20, missing rules {}
backward 0.4 s
params with a grad: 272 of 272
```

The twenty ops a gradient flows through, from `differentiable()` rather than by hand:

```
211 aten.t.default           211 aten.matmul.default    272 aten.mul.Tensor
120 aten.add.Tensor          120 aten.cat.default       121 aten.slice.Tensor
120 aten.transpose.int        92 aten.view.default       61 aten.add.Scalar
 61 aten.mean.dim             61 aten.pow.Tensor_Scalar  61 aten.rsqrt.default
 60 aten.contiguous.default   60 aten.neg.default        30 aten.reshape.default
 30 aten.silu.default         30 aten._scaled_dot_product_flash_attention_for_cpu.default
  1 aten.embedding.default     1 aten._log_softmax.default   1 aten.nll_loss_forward.default
```

### 4.1 Two things the model taught that the small cases could not

**`cat` has an empty-tensor exemption and `DynamicCache` relies on it.** A 1-D size-0 operand is
skipped rather than checked against the concatenation axis, so layer 0 concatenates a `[0]` past key
with a `[1, 3, 8, 64]` present one. A rule that assumed every operand had the output's rank panicked
on the first real model it saw. It is handled by name now, and a non-empty operand of the wrong rank
is refused rather than guessed at.

**Token ids are constants too, and they are not differentiable.** With every constant a gradient
target, the reverse walk reached the `constant_pad_nd` that builds `transformers`' shifted labels
and asked for a derivative of an integer. The fix is upstream's own rule stated one step earlier: a
non-floating value is dropped from the wanted set whichever way it was named — which is exactly what
`requires_grad_` refuses on a non-floating tensor upstream. `constant_pad_nd` has a rule anyway,
because `F.pad` is genuinely on the path of any `labels=` forward whose *activations* are padded.

### 4.2 The gradient comparison: all 134,515,008 elements

There is no way to write a tensor out of this shim — `torch.save`, `Tensor.numpy` and
`untyped_storage` all refuse — so the comparison is done the other way round: **upstream writes its
gradients to a `.safetensors` file and the shim loads it**, and every number below is computed
inside the shim process against upstream's own bytes.

| | value |
|---|---|
| tensors compared | **272 of 272** |
| elements compared | **134,515,008** |
| relative L2 `‖u−s‖/‖u‖`, median over tensors | **8.780e-05** |
| relative L2, worst tensor | 3.031e-04 (`model.layers.24.input_layernorm.weight`) |
| worst single element | 3.933e-03 (`model.layers.0.self_attn.v_proj.weight`) |
| **elements agreeing in sign** | **134,513,262 / 134,515,008 = 0.999987** |
| tensors where upstream is exactly zero and the shim is not | **0** |

### 4.3 The step, and whether the weights move the way upstream moves them

`p.grad = ...` for all 272, then `torch.optim.SGD(lr=0.1).step()` — the real optimiser, through the
real `torch.optim`, which `docs/LOSS.md` §6 got as far as running vacuously.

```
tensors whose weights moved: 272 of 272
```

| | relative L2 (median) | worst | element sign agreement |
|---|---|---|---|
| the step `w_after − w_before` | **9.632e-05** | 3.032e-04 | 134,511,760 / 134,515,008 = **0.999976** |
| the weights after the step | 1.043e-07 | 1.233e-04 | 134,515,006 / 134,515,008 = **1.000000** |

**The weights move, and they move where upstream moves them**: 99.9976% of the 134.5M step
components point the same way, and the post-step weights agree to a median relative 1.0e-07.

### 4.4 Where the 8.8e-05 comes from, and where it does not

Two attributions, both measured rather than argued.

**It is not the SDPA composition.** Running the same model with `attn_implementation="eager"` —
which replaces the one fused op with `matmul`/`softmax`/`matmul` and removes §3.4's rule from the
path entirely — moves the residual by about one percent:

| | nodes on a gradient path | distinct ops | gradient rel-L2 (median) | sign agreement |
|---|---:|---:|---|---|
| `sdpa` | 1723 | 20 | 8.780e-05 | 0.999987 |
| `eager` | 2053 | 23 | 8.701e-05 | 0.999987 |

**It is the forward's arithmetic.** The same tape, the same model, all 30 layers, in `float64`:

| dtype | gradient rel-L2 (median) | worst |
|---|---|---|
| `float32` | 8.780e-05 | 3.031e-04 |
| **`float64`** | **8.451e-07** | 4.511e-05 |

A hundredfold drop from changing nothing but the arithmetic width says the rules are not what the
`float32` number is measuring. `docs/SDPA.md` §3 already recorded that this shim's flat attention
disagrees with upstream's blocked kernel on 3562 of 4096 `float32` elements *in the forward*, and
`docs/LOSS.md` §5.4 that `_log_softmax` at vocabulary width exceeds the harness's own `float32`
tolerance; a backward inherits both and amplifies them by the network's condition number.

*(The `float64` run needs its loss computed explicitly rather than through `labels=`. `transformers`'
`fixed_cross_entropy` calls `logits.float()`, so a `float64` model returns a **`float32`** loss on
both sides — which is upstream's behaviour, not a defect here, and it hid the whole experiment until
the fd probe in §6 turned up a loss quantised to `float32` ULPs.)*

### 4.5 The costs, stated

| | |
|---|---|
| capture | 0.1 s, 1862 nodes, 333 constants held by reference |
| backward, including the replay it needs | **0.4 s** at S=8 |
| the extra forward | §1.3 — a backward is two forwards, by design |
| SDPA's rule | a `[B, H, T, S]` probability matrix per layer, recomputed. §3.4 |
| the embedding's rule | a `[vocab, tokens]` one-hot — 1.5 MB at S=8, **200 MB at S=1024** |

The embedding one is the one to fix, and the fix is named rather than done. The natural spelling is
a scatter-add into a zero buffer, i.e. `index_put_(accumulate=True)`, and **this shim refuses that**
— it is one line in `index_put_inplace`'s write loop (`o[dest] += s[src]`). It is left because the
refusal it would make stale lives in a document this round was told not to edit, and a stale refusal
is the failure this repository has had five times. The one-hot is correct for repeated tokens
without an accumulating scatter, because a column of it names exactly one row; it is the memory that
is wrong, not the answer.

---

## 5. The rules, against an oracle that is not upstream

Comparing against upstream is necessary and it is not sufficient, for a reason specific to this
file: a derivative rule here is a short expression over ops that already work, so a test that
recomputed the same expression would agree with a rule that had its operands the wrong way round.
Every one of the **56** rules is therefore also checked against **central differences in
`float64`**, which shares no code with the rule at all.

```
test_every_tape_rule_agrees_with_central_differences_in_float64
    56 cases, |tape - (f(x+h) - f(x-h))/2h| / max|fd| < 1e-5 for every element
```

`float64` with `h = 1e-6` puts truncation near 1e-13 and cancellation near 2e-9 on the O(1)
functions used, so the bound has two orders of margin.

Three things about the case construction, each of which was a hole first:

* **`sum(t)` is not enough to scalarise.** Its gradient is all ones through any shape op, which is
  exactly what a transposed, reversed or mis-sliced rule still returns. Every case ends
  `sum(exp(t))` so that each input element has a different gradient.
* **`sum(exp(log_softmax(x)))` is identically 1**, so its gradient is zero for every input *and
  every rule*, correct or not. The case weights the log-probabilities first. A test asserting
  "the gradient is not all zeros" is what caught it.
* **Symmetry hides pads and divisors.** `slice(x, 0, 1, 3)` of a `[4, 3]` leaves one row of padding
  on each side, so swapping them is invisible; an `nll_loss` with nothing ignored has
  `total_weight == rows`, so a mean that divides by the wrong one is invisible. Both cases are
  deliberately asymmetric now, and §7 is where that shows.

### 5.1 The rule table cannot go stale, structurally

`RULE_OPS` is a second list of op names, which is the exact shape of the failure `docs/AUDIT.md`
found six times. So it is not read by a human: `_C._tape_rules()` exposes it,
`test_the_tape_has_a_gradient_case_for_every_rule_it_claims` asserts that the case table equals it
in both directions, and `differentiable()` reports against it. Adding a rule without a gradient case
fails the suite; so does leaving a case for a rule that was removed.

### 5.2 An op with no rule is refused by name

`docs/DESIGN.md` §6, applied to a derivative — and it matters more here than for a kernel, because a
wrong gradient looks exactly as plausible as a right one and the program keeps running.

```
NotImplementedError: torch._C tape: no derivative rule for aten.topk.default -- a gradient
reached it, and the tape refuses to guess. Add a rule in tape.rs and a gradient case in
pytests/test_shim.py; trace.differentiable() lists every op in a trace that would need one
```

`differentiable()` names it *before* a backward is run, which is what makes "what stops this model"
answerable without reading an exception.

---

## 6. A check that failed, and it was the check

The obvious way to validate a backward on a real model is finite differences, and it was run:
perturb one weight of SmolLM2 by ±1e-6 in `float64`, recompute the loss, compare. It disagreed with
the tape by a factor of **600**.

```
model.layers.0.self_attn.q_proj.weight[0,0]   tape 3.66e-04   central difference 6.53e-02
model.norm.weight[0]                          tape 2.94e-02   central difference 2.94e-02
```

The final norm agrees to eight digits; everything inside the transformer layers does not. Before
believing either side, the same probe was run **on upstream**, against upstream's own autograd:

```
upstream, same model, same weights, same h
model.layers.0.self_attn.q_proj.weight[0,0]   autograd 3.41e-04   central difference 2.09e-01
model.norm.weight[0]                          autograd -3.5588e-02  central difference -3.5588e-02
```

**Upstream's autograd fails the probe in the same place and by the same order.** So the probe is not
an oracle for this function at this scale, and the conclusion it invited — "the tape is wrong deep in
the network" — was not available. What made it usable at all in §5 is exactly what is missing here:
small, well-scaled, smooth functions where the linear term dominates the quadratic one at
`h = 1e-6`.

This is recorded rather than dropped because the failure mode it belongs to is the one this
repository keeps meeting from the other side. `docs/AUTOGRAD.md` §5.4's lesson was *a criterion I
wrote decided the answer*; this is the same lesson with the sign flipped — a criterion I wrote was
about to condemn code that was right, and the thing that stopped it was running the criterion
against a known-good implementation first.

---

## 7. Sabotage: 17 faults

Every one applied to `rust/torch_c/src/tape.rs`, **rebuilt**, and run through the five tape tests.

| # | fault | caught |
|---|---|---|
| T1 | `addmm`: `mat1 @ g` instead of `mat1^T @ g` | ✅ shape mismatch at `mm` |
| T2 | `mul.Tensor`: operands swapped, so `grad_self = g * self` | ✅ |
| T3 | `reduce_to` short-circuited — every broadcast left unreduced | ✅ |
| T4 | `_log_softmax`: the `exp(out) * rowsum(g)` term dropped, gradient passed through | ✅ |
| T5 | `nll_loss`: `reduction=Mean` divides by the row count, not `total_weight` | ✅ *(after §5's third bullet)* |
| T6 | `nll_loss`: the ignored row not zeroed, so its clamped column 0 collects a real gradient | ✅ *(same)* |
| T7 | SDPA: `dS = P * dP`, without the softmax Jacobian's row-sum | ✅ |
| T8 | SDPA: the grouped-query fold replaced by "take group 0" | ✅ *(after §5's third bullet)* |
| T9 | `slice`: the two pads swapped | ✅ *(same)* |
| T10 | `cat`: the offset never advances, so every operand reads the first piece | ✅ |
| T11 | `silu`: the naive `g * sigmoid(x)` | ✅ |
| T12 | `embedding`: a write instead of a sum — `index_put_(accumulate=False)`, which loses a repeated token's second contribution | ✅ |
| T13 | `rsqrt`: the sign of the exponent rule | ✅ |
| T14 | `mean.dim`: the division by the reduced extent dropped | ✅ |
| T15 | `detach`: a gradient flows through the op whose purpose is that none does | ✅ |
| T16 | `where.self`: the two branches swapped | ✅ |
| T17 | `contiguous`/`clone` answered through `alias` instead of returning `g` | **❌, and correctly** |

**Sixteen of seventeen.** T17 cannot be caught and it is right that it cannot: `contiguous`, `clone`,
`alias` and `lift_fresh` all have the identity as their derivative, so substituting one for another
is not a different computation. It is in the table because the *forward* ops differ, and only the
derivative makes them coincide — the same shape as `docs/LOSS.md` §8.2's D2.

**Four of the sixteen could not fail when they were first run** — T5, T6, T8, T9 — and none of them
was a defect in the rule. Each was a case that could not separate the fault from the fix: an
`nll_loss` with nothing ignored, an attention case where the input was the query alone so the
grouped-query fold never entered the checked gradient, and a slice whose two pads were equal. The
cases were made asymmetric and the faults then failed. This is the pattern `docs/LOSS.md` §5.2 and
§8.3 record twice, met twice more.

### 7.1 What this suite still cannot see

* **Nothing here compares against upstream.** The tape tests use finite differences, because
  `pytests/test_shim.py` runs against bare `_C` with no upstream torch in the process. The
  upstream comparison is §2, §3 and §4, and those are measurements in this document rather than
  tests in `pytests/` — so §4's 8.8e-05 can move without anything going red.
* **`tools/golden/compare.py` cannot see the tape at all.** It compares *ops* by dispatch key, and a
  derivative rule is not an op. That is why the rule table is pinned by a test instead.
* **No `float32` case separates a summation order** — the same statement `docs/LOSS.md` §5.3 makes,
  for the same reason, and it is why §4.4's attribution had to be done in `float64`.
* **Nothing checks memory.** §4.5's numbers are arithmetic on shapes, not measurements.
* **One backward, once.** No accumulation across steps, no `zero_grad` between them, no second
  optimiser, no convergence.

---

## 8. What this round did not do

| | why |
|---|---|
| `Tensor.backward()` | It differentiates *whatever produced this tensor*, which needs a node per op and a flag that propagates — `docs/AUTOGRAD.md` §6's `VariableType` half. The tape differentiates a *recorded region* and needs neither. The refusal stands and the test that pins it is unchanged |
| `requires_grad` propagation | Same boundary. Still inert, still nothing reads it, and `test_the_autograd_boundary_is_where_autograd_md_says_it_is` still passes unmodified |
| `torch.autograd.grad`, hooks, `create_graph` | None is on the path of a federated or test-time-adaptation step, which is what README §2 and §3 describe |
| double backward | The tape can in principle record its own backward, but the backward runs outside a capture region today (§1.1) — which is exactly what makes the rules cheap. The two would have to be reconciled |
| Adam and AdamW | `docs/LOSS.md` §6.4's four items are still open: `torch.is_complex` (a name), `lerp_.Scalar`, `addcmul_`, `addcdiv_`. **SGD needed none of them** and that is what a first training step wanted |
| `index_put_(accumulate=True)` | §4.5. One line, and it would make a refusal stale in a document this round could not edit |
| `native_layer_norm`'s derivative | Not on SmolLM2's path (RMSNorm is `mean.dim` + `rsqrt` + `mul`, which are rules). `gpt2`/`bert` would need it |
| anything on device | Desktop macOS only. A tape walker generates no code, so `docs/DESIGN.md` §5's iOS W^X constraint does not obviously apply — but that is reasoning, not a measurement |
| training more than one step | §7.1's last bullet |

---

## 9. Gates

| gate | before | after |
|---|---|---|
| `pytests/run.sh` | 296 ok, 0 FAIL | **302 ok, 0 FAIL** |
| `run.sh` DOCWATCH | 95/95 | **109/109** (14 new markers, all in this document) |
| `tools/golden/compare.py` | 7447/7447, ops=166, pending 1 | **7447/7447, ops=166, pending 1** |
| `compare.py --self-test` | 19 comparators × 11 fault modes | **unchanged** |
| `verify_schemas.py` | 4475/4475 | **4475/4475** |
| sweep26 (`.eval()`) | 26/26 | **26/26** |
| sweeptrain (`.train()`) | 26/26 | **26/26** |

`ops=166` is unchanged **on purpose**: no kernel landed (§1.1), so nothing here could have moved it,
and a change in that number would have meant the rules had stopped being compositions.

### 9.1 The forward did not move

`docs/SEQLEN.md` §1.3's prefill logits sha256 over real SmolLM2-135M, re-measured on the final
artefact. A backward that moves a forward result is a bug.

| S | f32 | | bf16 | |
|---:|---|:--:|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | ✅ | `8ef1550ea33c4f3d…` | ✅ |
| 32 | `331668f36da02f21…` | ✅ | `b81325c83a0a3d15…` | ✅ |
| 128 | `00159a9dbd308eda…` | ✅ | `7ff8e9334449b147…` | ✅ |
| 512 | `07c2797dabc4552e…` | ✅ | `9ab1e82f01378e38…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | ✅ | — | |

All nine equal `docs/LOSS.md` §10.1 and `docs/TRAIN.md` §6.

### 9.2 The six new tests

```
test_the_tape_has_a_gradient_case_for_every_rule_it_claims
test_every_tape_rule_agrees_with_central_differences_in_float64
test_the_tape_refuses_an_op_it_has_no_rule_for_by_name
test_a_gradient_reaches_a_burned_in_constant_and_only_the_ones_asked_for
test_the_tape_seeds_a_one_only_for_a_scalar_and_says_so_otherwise
test_grad_is_a_real_slot_now_and_takes_only_a_tensor_or_none
```

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_every_tape_rule_agrees_with_central_differences_in_float64 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_tape_has_a_gradient_case_for_every_rule_it_claims present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_grad_is_a_real_slot_now_and_takes_only_a_tensor_or_none present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _set_grad present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs _shim_grad present -->

---

## 10. `.grad` is a slot now, and why that reverses a decision

`docs/AUTOGRAD.md` §7 argued explicitly against this, and the argument was right at the time:

> making `.grad` writable while nothing writes to it would move the shim from "honestly reports no
> gradient" to "has a slot that is always empty"

The antecedent is gone. The tape writes gradients, so the slot is not always empty, and what would
be dishonest *now* is a `torch.optim` step that silently skipped all 272 parameters because the slot
it reads cannot be filled — which is precisely what `docs/LOSS.md` §6.3 had to report.

Two things did **not** change with it, and they are what keeps the reversal narrow:

* **Nothing fills it implicitly.** `CaptureTrace.backward()` *returns* gradients and the caller
  assigns them, the same shape `torch.optim.sgd.sgd` already had.
* **A clone does not inherit one.** A gradient belongs to the leaf it was accumulated into, and
  upstream gives a non-leaf no `.grad` at all.

---

## 11. Every command in this document

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-tape
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
SHIM="PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY"     # VENDOR.md wall 3

# §2, §3  the small cases, both sides, then the element-wise comparison
$PY   /tmp/tape/run_up.py   <case> /tmp/tape/up_<case>.json
$SHIM /tmp/tape/run_shim.py <case> /tmp/tape/shim_<case>.json
$PY   /tmp/tape/cmp.py      <case>

# §4  the whole model: gradients, the step, and the weights
$PY   /tmp/tape/smol_up2.py   sdpa 8 sdpa8      # writes upstream's grads/weights
$SHIM /tmp/tape/smol_shim2.py sdpa 8 sdpa8      # loads them and compares in-process
$PY   /tmp/tape/smol_up2.py   eager 8 eager8    # §4.4, the SDPA attribution
$SHIM /tmp/tape/smol_shim2.py eager 8 eager8
$PY   /tmp/tape/clean64.py up 0    &&  $SHIM /tmp/tape/clean64.py shim 0   # §4.4, float64

# §6  the probe that failed, on both implementations
$SHIM /tmp/tape/clean64.py shim 4 fd
$PY   /tmp/tape/fdup.py

# §7  sabotage: 17 faults, each rebuilt
$PY /tmp/tape/sab.py            # or /tmp/tape/sab.py T5 T8 for one

# §9  gates
PYTHON=$PY sh rust/torch_c/pytests/run.sh
$PY tools/golden/compare.py  ;  $PY tools/golden/compare.py --self-test
$PY rust/torch_c/pytests/verify_schemas.py
$SHIM /tmp/k26/sweep26.py /tmp/tape/ev   ;  $SHIM /tmp/train/sweeptrain.py /tmp/tape/tr
$SHIM /tmp/loss/seqlen.py f32            ;  $SHIM /tmp/loss/seqlen.py bf16
```

The scratch harnesses live under `/tmp/tape/` and are reproduced nowhere else; every number they
produce is quoted above with the command that made it.
