# Test-time adaptation: Tent on a real checkpoint, and the delta underneath it

`docs/BACKWARD.md` ended with a training step that runs — forward, loss, tape,
SGD, 272 parameters moved the way upstream moves them — reached through
`torch._C._capture_begin(...)` and `trace.backward(inputs)`. That is a capture
API. The README advertises a different one:

```python
from torchnative import adapt
model = adapt.wrap(model, method=adapt.Tent())
model.online()
```

This document is that API, made real on `SmolLM2-135M`, and the measurements
that say it does something rather than merely completing.

Environment: `/Volumes/macMini/caches/spike-venv/bin/python`, torch 2.13.0,
transformers 5.15.1, worktree at `develop` `55b6a7e`,
`HF_HOME=/Volumes/macMini/caches/hf-home`. Upstream is the oracle throughout.

### Answers, before the evidence

| question | answer |
|---|---|
| Which of README §2 and §3 was built? | **§3, test-time adaptation.** §2 cannot be measured on this stack and §1 says why |
| Does the adaptation change the model? | **Yes, and by the amount it is supposed to.** Prediction entropy on unlabelled text falls **4.1604 → 2.9828** over ten steps (§3) |
| Does it transfer, or only fit the batch it saw? | Entropy on a **held-out** sentence the loop never adapted on falls **3.7237 → 2.9439** (§3) |
| Could a do-nothing loop have produced that? | **No.** Wrong sign takes it *up* to 7.4062; `lr=0` holds it at 4.16039658 to every printed digit; a detached objective is **refused by name** rather than running vacuously (§4) |
| Does upstream agree? | Upstream's own autograd runs the same Tent step. Adapted weights agree to a median relative **1.5e-06** with **100.0000%** element sign agreement over 35,136 numbers (§6) |
| Is the delta abstraction real, or is `Tent` a special case? | `Tent` is **40 lines and holds no state**. Keeping, measuring, reverting and shipping live once, on `Delta` (§2) |
| Applied, kept, reverted? | Base weights are **bit-identical after a revert** — all 272 parameters, not only the 61 covered (§5) |
| What could the tape not carry? | **Any `nn.LayerNorm` model.** `aten.native_layer_norm.default` has no derivative rule, so Tent refuses on `gpt2`/`bert` while working on every RMSNorm architecture (§8) |

Written incrementally, one stage at a time, for the reason `docs/KERNELS26.md`
§0 gives.

### The baseline, every gate, before any edit

```
pytests/run.sh                302 ok, 0 FAIL, DOCWATCH 159/159    exit 0
tools/golden/compare.py       7447/7447, ops=166, pending=1       exit 0
compare.py --self-test        19 comparators x 11 fault modes     exit 0
verify_schemas.py             4475/4475                           exit 0
sweep26   (shim, .eval())     26/26                               exit 0
sweeptrain (shim, .train())   26/26                               exit 0
```

---

## 1. Why test-time adaptation and not federated learning

The brief allowed either and asked for the reasoning to be checked rather than
inherited. It checks out, and the deciding argument is not the one that was
offered.

The offered argument was *federated needs `torch.distributed` at world_size > 1,
which refuses by name*. That is true — `ProcessGroupLocal.__init__` refuses a
world larger than one, `TCPStore` refuses because there is no socket peer,
`send`/`recv` refuse, and `ProcessGroupGloo` is deliberately **absent** so that
the vendored tree's own `_GLOO_AVAILABLE` probe reads False
(`docs/DISTRIBUTED.md` §4.2, §4.3). But "the transport refuses" is only an
argument against a *distributed* demonstration. Federated averaging can be
simulated in one process, and most FL research code does exactly that.

So the real question is what a one-process simulation would prove, and the
answer is nothing:

* **README §2 forbids it.** *"Federated averaging **is** collective
  communication, so this is built on `torch.distributed` rather than beside
  it."* An in-process aggregator is precisely "beside it". Building one would
  be the §5.2 failure in CLAUDE.md — choosing a design that makes the stated
  structure unreachable and then reporting the substitute as the thing.
* **At world_size 1, FedAvg is the identity.** `docs/DISTRIBUTED.md` §4.1 says
  a single-rank reduction *is* the identity and that this is a fact rather than
  a stub. So a "federated round" run through the transport that does exist is
  arithmetically indistinguishable from one local training step — and a test of
  it would pass for a correct aggregator, a broken aggregator, and no aggregator
  at all. That is CLAUDE.md §5.5's verification that cannot fail.

Test-time adaptation has neither problem: one device, no transport, no
aggregation, and — the property that decided it — **an objective whose value is
a number that has to move in a direction**. Entropy either goes down or it does
not, so the loop can be caught doing nothing.

What federated needs, concretely, is therefore not "aggregation code":

| | |
|---|---|
| a second rank | `ProcessGroupLocal` refuses `world_size != 1`; there is no backend that does not |
| a rendezvous | `TCPStore` refuses; `HashStore` is process-local |
| a delta on the wire | `torch.save` refuses at `PyTorchFileWriter.write_end_of_file`, and there is no other tensor serialiser here |

The third is the interesting one, because it is *not* about distribution: a
delta that cannot be written to bytes cannot be sent anywhere, and that is the
same wall that stops a delta surviving a process restart. `Delta.persist` and
`Delta.publish` refuse today, each naming the check that would make the refusal
stale (§2).

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py ProcessGroupLocal present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/delta/__init__.py publish present -->

---

## 2. The abstraction: a delta, and a method that declares three things

`docs/DESIGN.md` §3 is explicit that the central type is not an adaptation
method:

> 적응 방법들을 관통하는 것은 **베이스 가중치 위의 델타**입니다 — TTA 가 적응시킨
> 파라미터와 FL 의 로컬 업데이트가 같은 물건이고 **수명과 행선지만 다릅니다.**

So `torchnative.delta.Delta` owns everything about a weight change and
`torchnative.adapt.Method` owns nothing. A method declares three things:

```python
class Tent(Method):
    stage = STAGE_NARROW_BACKWARD          # DESIGN.md §3 axis 1
    def select(self, model):   ...         # which parameters move
    def objective(self, outputs): ...      # what scalar is descended
```

That is the whole of `Tent` apart from docstrings — 40 lines, no state, no
`reset()`, no base copy, no serialisation. The second method inherits all of
that by not writing it.

`torchnative/delta/__init__.py` **imports no torch at all** — it calls methods
on the tensors it is handed and holds no module. That is not tidiness: a delta
is what federated averaging would send, so it has to be constructible and
readable where the model layer is not, and an import at module scope would have
made it the model layer's dependent.

### 2.1 `stage` is declared per method, not per directory

`docs/DESIGN.md` §10 says `adapt/` must not split into directories by
differentiation requirement, because normalisation calibration sits on **both**
sides of the line: recomputing statistics needs no backward, updating the same
layer's affine parameters by a loss does. The declaration is therefore a class
attribute, and `wrap()` reads it before anything runs — so a build without a
backward refuses a stage-1 method at wrap time rather than at the first step.

Stage 2 (full autograd through an inner update) is refused permanently, which
is §3's own decision, and the refusal names a check rather than a fact:

```
Check: torch.ones(1, requires_grad=True).sum().backward(). If that returns
instead of refusing, an autograd exists that this refusal predates.
```

### 2.2 Lifetime: three questions, answered by running

`docs/DESIGN.md` §3 withdrew one invented set of lifetime names
(`Ephemeral · Session · Persistent · Shared`) and one borrowed set
(`ttadapters`' `ScenarioType`, which turned out to be an *evaluation* protocol),
and said the names get chosen once a real integration shows the usage. **This is
that integration, and it did not need names.** It needed three answers, and
`Delta` gives each of them by doing the thing or refusing:

| §3's question | on `Delta` | today |
|---|---|---|
| can this delta be discarded, and at what cost | `revert()`, `nbytes` | **yes**, and the cost is measured in §5 |
| does it survive a process restart | `persist()` | **refuses** — no tensor serialiser |
| can it leave the device | `publish()` | **refuses** — no world larger than one |

A label would have had to be invented for the two that refuse, and it would
have described a capability nothing here has. Two refusals that each name a
runnable check are the honest shape until one of them stops refusing.

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/delta/__init__.py Delta present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/delta/__init__.py revert_by_subtraction present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/adapt/__init__.py Tent present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/adapt/__init__.py STAGE_NARROW_BACKWARD present -->

### 2.3 What `Tent.select` picks, and why by class name

Tent moves the affine parameters of the normalisation layers. There is no base
class to test for — `nn.LayerNorm`, `LlamaRMSNorm`, `T5LayerNorm` and
`BatchNorm1d` share nothing — so the rule is *class name contains "norm"* **and**
the module carries `weight` or `bias` as its own parameter. The second half is
what stops a container named for normalisation from selecting its children's
weights.

On `SmolLM2-135M` that is **61 of 272 parameters**, 35,136 numbers: the
`input_layernorm` and `post_attention_layernorm` of all 30 layers plus
`model.norm`. It is identical, name for name, to the list upstream's script
selects independently (`selected 61; identical to upstream's list: True`).

**What is deliberately not implemented**: Tent also puts normalisation layers
into batch-statistic mode, because the paper's models are BatchNorm ones. This
selects and updates affine parameters only. On a model whose normalisation has
no running statistics — LayerNorm, RMSNorm, so every transformer — the two
coincide. The class docstring names the check
(`any(hasattr(m, "running_mean") ... for m in model.modules())`) rather than
claiming no such model exists.

---

## 3. Tent on SmolLM2-135M: the entropy curve

Real weights from the HF cache, `float32`, `.eval()`. Unlabelled test data is a
real sentence through the real tokenizer, 29 tokens; the model is never shown a
label. `lr = 1e-3`, plain `torch.optim.SGD`, ten steps.

A second sentence, 15 tokens, is a **held-out probe**: it is scored before and
after and is never adapted on.

```
adapt : "The quick brown fox jumps over the lazy dog. On-device test-time
         adaptation lets a shipped model meet data its training set never had."
probe : "Paris is the capital of France, and the Seine runs through it."
```

| step | shim | upstream | \|d\| |
|---:|---|---|---|
| 0 | **4.16039658** | 4.16005039 | 3.46e-04 |
| 1 | 3.95863104 | 3.95819712 | 4.34e-04 |
| 2 | 3.80226469 | 3.80176568 | 4.99e-04 |
| 3 | 3.66015100 | 3.65960956 | 5.41e-04 |
| 4 | 3.52872062 | 3.52816343 | 5.57e-04 |
| 5 | 3.40793467 | 3.40737152 | 5.63e-04 |
| 6 | 3.29765630 | 3.29707599 | 5.80e-04 |
| 7 | 3.19748116 | 3.19687915 | 6.02e-04 |
| 8 | 3.11022496 | 3.10941410 | 8.11e-04 |
| 9 | 3.03639579 | 3.03583193 | 5.64e-04 |
| after the last step | **2.98279548** | 2.98224974 | 5.46e-04 |

**Entropy falls monotonically, by 28%, and it tracks upstream's own curve to
within 8.1e-04 at every step** — a relative 1.4e-04, which is the size of this
stack's `float32` forward residual and not a divergence of the two trajectories.

The held-out probe:

| | before | after |
|---|---|---|
| shim | 3.72368073 | **2.94390273** |
| upstream | 3.72342634 | 2.94364214 |

**The adaptation transfers.** The probe sentence shares no content with the
adaptation sentence and was never stepped on; its entropy falls by 21%. That
distinguishes "the model adapted" from "the loop memorised one batch", and it is
the distinction an adaptation API is for.

Cost, on this machine: **10 steps in 4.2 s**, i.e. 0.42 s per step at S=29 —
one capture, one replay and one reverse walk each, which is `docs/BACKWARD.md`
§1.3's two-forwards-per-backward.

---

## 4. The three controls, because a loop that does nothing passes

Each is the same code path with one thing changed, on the same checkpoint and
the same sentence.

### 4.1 Wrong sign — the objective negated

```python
class AntiTent(adapt.Tent):
    def objective(self, outputs):
        return -super().objective(outputs)
```

| | step 0 | step 9 | after |
|---|---|---|---|
| entropy | 4.16039658 | 7.04619408 | **7.40623283** |
| held-out probe | 3.72368073 | | **5.28927040** |

**Entropy rises by 78% and the probe rises with it.** So the fall in §3 is
produced by the gradient's direction, and not by the model drifting toward some
low-entropy attractor that any perturbation of the norm weights would reach.

### 4.2 No step — `lr = 0`

```
step  0 objective 4.16039658
step  1 objective 4.16039658
step  2 objective 4.16039658
step  3 objective 4.16039658
step  4 objective 4.16039658
step  5 objective 4.16039658
delta <Delta over 61 parameters, |d|=0, base 140544 B, value 140544 B>
tensors whose weights moved: 0 of 61
```

Identical to every printed digit, six times. This is the control for the other
direction: it says the forward is deterministic across steps, so the §3 curve is
the *update* and not capture, replay, or a cache changing the answer between
calls. `sha256` over the covered weights is unchanged, and `Delta.norm()` is
exactly 0.

### 4.3 Detached objective — the loop that silently does nothing

```python
class DetachedTent(adapt.Tent):
    def objective(self, outputs):
        return super().objective(outputs.logits.detach())
```

This is the shape of the bug the whole section exists for: an adaptation loop
that runs, completes, reports a step, and cannot possibly have changed anything.
It **refuses at step 0**:

```
RuntimeError: torchnative.adapt: the objective produced a gradient for none of
the 61 selected parameters, so a step would run and change nothing. The usual
cause is an objective computed on a detached tensor: the tape's rule for
aten.detach.default is to stop, which is what detach is for.
Check: trace.differentiable(wrt_constants=[...]) -- 'nodes_on_a_gradient_path'
is 0 when nothing connects.
```

It is a refusal rather than a flat curve because there is no reading under which
a caller wanted it. The same guard fires for the other two ways to get an
inert loop — a method that selects no parameters, and a selected parameter that
this forward did not use — and both refuse with the check to run.

---

## 5. Lifetime: applied, kept, reverted — and what the base copy buys

Ten Tent steps, then the three operations, with `sha256` over the little-endian
`f32` bytes of the covered weights (the same construction `docs/SEQLEN.md` uses
for logits, because no tensor can be written out of this shim any other way).

```
base sha     19024cf129df3240        61 tensors, 35136 numbers
adapted sha  ba20583bf87c432b        tensors whose weights moved: 61 of 61
```

| operation | sha | bit-identical | elements differing | worst \|d\| |
|---|---|---|---:|---|
| `revert()` | **19024cf129df3240** | **yes** | 0 / 35136 | 0 |
| `revert_by_subtraction()` | aecce6d3366f1dda | no | **2 / 35136** | 5.821e-11 |
| `apply()` after a revert | c1d8caadcae67777 | no | **3 / 35136** | 5.821e-11 |

And the revert is checked over the **whole model**, not only what the delta
covers:

```
all 272 parameters after revert() identical to before online(): True
```

That last line is the one that matters, because a delta that reverted its own 61
tensors while something else had moved the other 211 would report success.

### 5.1 Two floating-point facts, and what they decide

**`(w + d) − d ≠ w`.** `revert_by_subtraction` is the revert that needs no base
copy — subtract the offset back off — and it lands 2 elements of 35,136 away
from the base. That is the measurement that justifies `Delta` holding a base
copy at all. If it had been bit-identical, the copy would have been waste.

**`base + (w − base) ≠ w`.** Re-applying a *kept* delta after a revert lands 3
elements of 35,136 away from the weights it was recorded from. So "keep the
delta" and "keep the weights" are not the same operation, by a measurable
amount, and this is the number to quote when a lifetime policy chooses between
them. 5.8e-11 on weights of order 1 is one `float32` ULP at that magnitude
carried through a `float64`-free subtraction; it is small, and it is not zero,
and a revert that has to be exact cannot be built on it.

### 5.2 What the base copy costs

`docs/DESIGN.md` §9 item 5 records the problem this is supposed to solve:
`ttadapters` holds a full weight copy in `base_state` so that `reset()` has
something to restore. A delta narrows the copy to what it covers:

| | bytes | |
|---|---:|---|
| `Delta.base` (61 tensors) — what a revert needs | 140,544 | 137 KiB |
| `Delta.value` (61 tensors) — what a send would need | 140,544 | 137 KiB |
| both | 281,088 | 275 KiB |
| the whole model, tied weights counted once | 538,060,032 | 513 MiB |
| **ratio, base alone** | **3828x** | |
| **ratio, base + value** | **1914x** | |

So `reset()` on a Tent-adapted SmolLM2 costs 137 KiB rather than 513 MiB. The
narrowing is not free in generality — a method that adapts every parameter
gets no reduction, and correctly so, because then the delta *is* the model.

---

## 6. Against upstream: the same Tent step with upstream's own autograd

Upstream runs the identical recipe — same checkpoint, same tokenized sentence,
same 61 parameters, same `torch.optim.SGD`, same `lr` — with
`loss.backward()` and its real autograd, and writes its base weights, its
step-0 gradients and its final weights to `.safetensors`. The shim loads those
bytes and compares in-process, which is the only direction that works
(`docs/BACKWARD.md` §4.2: nothing can be written *out* of this shim).

**After ten steps, over all 35,136 adapted numbers:**

| | value |
|---|---|
| tensors compared | 61 of 61 |
| relative L2 `‖u−s‖/‖u‖`, median over tensors | **1.512e-06** |
| relative L2, worst tensor | 2.615e-05 (`model.layers.0.input_layernorm.weight`) |
| worst single element | 8.410e-05 (`model.layers.29.post_attention_layernorm.weight`) |
| **elements agreeing in sign** | **35136 / 35136 = 1.000000** |

The *delta* — the quantity ten steps of adaptation actually produced, which is
three orders of magnitude smaller than the weights it sits on — agrees to a
median relative **1.253e-03** with sign agreement 0.999573.

### 6.1 The residual is the objective's arithmetic, not the tape

`docs/BACKWARD.md` §4.2 measured 8.780e-05 for a cross-entropy gradient over all
272 parameters. Tent's is larger, and the attribution is a measurement rather
than an argument: **the same tape, the same forward, the same 61 parameters,
one step, two objectives.**

| objective | value, rel. to upstream | gradient rel-L2, median | worst tensor |
|---|---|---|---|
| cross-entropy (`labels=ids`) | 2.12e-05 | **4.875e-04** | 2.761e-03 |
| **entropy** (Tent's) | 8.32e-05 | **1.388e-03** | 5.768e-03 |

Changing only the objective moves the gradient residual by 2.8x and the
objective's own *forward* value by 3.9x, so it is arithmetic in the objective
and not the reverse walk. The mechanism is visible in the seed:

```
dH/dlogits    max|element| 1.344e+00   max|row sum| 1.051e-03   ratio 7.82e-04
dCE/dlogits   max|element| 9.861e-01   max|row sum| 2.794e-04
```

`dH/dx_i = −p_i(log p_i + H)` sums to exactly zero over a row, analytically —
`Σp_i log p_i = −H`. So entropy's seed is a **cancellation across 49,152
columns** where cross-entropy's is not, and a `float32` evaluation of it carries
7.8e-04 of relative residual before the reverse walk begins. That number, times
the network's condition number, is the 1.4e-03.

Two further things that keep this honest:

* The 61 tensors Tent adapts are, by construction, the tensors
  `docs/BACKWARD.md` §4.2 already found worst — its worst-tensor entry is
  `model.layers.24.input_layernorm.weight` at 3.031e-04. Comparing only the norm
  weights is comparing the hard subset, and 4.875e-04 for cross-entropy on that
  subset is consistent with 8.780e-05 over all 272.
* **Both objectives disagree with upstream in sign on exactly 10 elements**,
  which looks like a shared cause and is not: the two sets of 10 are
  **disjoint**, and in each case upstream's own gradient there is near zero
  (≤1.044e-04 for entropy, ≤4.533e-06 for cross-entropy) against gradients whose
  scale is 1e-2. They are sign flips on numbers that have no sign.

---

## 7. The forward did not move

`docs/SEQLEN.md` §1.3's prefill logits sha256 over real SmolLM2-135M, measured
twice at every length: once on the model, and once on the same model **inside
`adapt.wrap(...)` in its default offline state**. An adaptation API that moves a
plain forward is a bug, and a wrapper that is not the identity while disarmed is
the specific way that bug would arrive.

| S | f32 | | bf16 | |
|---:|---|:--:|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | ✅ | `8ef1550ea33c4f3d…` | ✅ |
| 32 | `331668f36da02f21…` | ✅ | `b81325c83a0a3d15…` | ✅ |
| 128 | `00159a9dbd308eda…` | ✅ | `7ff8e9334449b147…` | ✅ |
| 512 | `07c2797dabc4552e…` | ✅ | `9ab1e82f01378e38…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | ✅ | — | |

All nine equal `docs/BACKWARD.md` §9.1, `docs/LOSS.md` §10.1 and
`docs/TRAIN.md` §6, **and `plain == wrapped` at every one of the nine.**

---

## 8. What the tape could not carry

Four walls, each with the check that would say it had fallen.

### 8.1 `nn.LayerNorm` models cannot take a Tent step

`aten.native_layer_norm.default` has no derivative rule. `docs/BACKWARD.md` §8
predicted exactly this and said why it was not needed there — RMSNorm is
`mean.dim` + `rsqrt` + `mul`, which are rules, so SmolLM2's path never reaches
it. Tent walks into it immediately, because a `LayerNorm` model's affine
parameters are *behind that op*.

```
torchnative.adapt: this model cannot take a stage-1 step -- 1 op(s) on the
gradient path have no derivative rule: ['aten.native_layer_norm.default'].
Check: torch._C._tape_rules() is the list that does exist, and
trace.differentiable() produced this one.
```

It is refused **before** the backward, by `trace.differentiable()`, with the
whole list rather than whichever missing rule the reverse walk reached first.
So the wall is *"this architecture family"*, not *"this model"*: every RMSNorm
transformer works, `gpt2` and `bert` do not.

This is a rule, not a kernel — `native_layer_norm` itself is implemented and
`sweep26` forwards through it — so closing it is one arm in `tape.rs` and one
gradient case, and `rust/torch_c/src/` was out of scope for this round.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs aten.native_layer_norm.default absent -->
<!-- DOCWATCH: op-implemented aten.native_layer_norm.default -->

### 8.2 `use_cache=False` cannot be captured

Passing `use_cache=False` to a `transformers` forward reaches a `torch.diff`
that has no entry in the overload table:

```
not implemented in torch._C shim: torch.diff(...) -- overload resolution has no
table entry for this op (rust/torch_c/src/overloads.json)
```

So an adaptation step runs on the default cache path, which is what
`docs/BACKWARD.md` §4 also did. It costs nothing here — the cache is built and
dropped inside one traced forward, and §4's `lr=0` control shows the forward is
identical across steps — but it is a real restriction on how the model may be
called, and it is a table entry rather than a kernel.

<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json diff absent -->

### 8.3 A delta cannot be written down

`torch.save` refuses at `PyTorchFileWriter.write_end_of_file`, and there is no
other way to get tensor bytes out of this shim. That is the same wall for two of
`docs/DESIGN.md` §3's three lifetime questions — surviving a process and leaving
the device both begin with serialisation — and it is why `Delta.persist` and
`Delta.publish` refuse rather than carrying a lifetime label.

### 8.4 `Tensor.backward()` still refuses

Unchanged, and the API is honest about it: `Adapted.step` captures a region and
walks a tape, which is why an adaptation step inherits every refusal capture
makes (`docs/CAPTURE.md` §4). The class docstring says so rather than hiding it
behind a fallback nobody asked for. `docs/BACKWARD.md` §8's reasoning stands: a
`Tensor.backward()` needs a node per op and a propagating flag, and nothing in
test-time adaptation wants one.

---

## 9. Sabotage: 13 faults

Every one applied to `torchnative/adapt/__init__.py` or
`torchnative/delta/__init__.py`, then the eight tests re-run. No rebuild is
needed — these are Python — which is exactly why the fault set is larger than a
Rust round's.

| # | fault | caught by |
|---|---|---|
| S1 | `Tent.objective`: sign flipped, so entropy is maximised | ✅ `..._reduces_prediction_entropy_...` |
| S2 | `Tent.objective`: the entropy summed over the wrong dim | ✅ same |
| S3 | `Delta.revert`: subtract the offset instead of restoring the base | ✅ `..._reverts_the_base_weights_bit_for_bit` |
| S4 | `Delta.record`: store the weight rather than the offset | ✅ two tests |
| S5 | `Adapted.step`: the optimiser never steps | ✅ three tests |
| S6 | `Adapted.step`: the "no gradient reached anything" guard removed | ✅ `..._cannot_move_anything_is_refused` |
| S7 | `_is_normalisation`: every module counts as a normalisation layer | ✅ three tests |
| S8 | `Adapted.forward`: an **offline** wrapper steps anyway | ✅ eight tests |
| S9 | `Delta.over`: the base aliases the live parameter instead of copying | ✅ `..._reverts_the_base_weights_...` |
| S10 | `Delta.apply`: the offset is subtracted | ✅ same |
| S11 | `Adapted.step`: the pre-backward missing-rule check removed | ✅ `..._refuses_a_layer_norm_model_...` |
| S12 | `_tensor_inputs`: positional tensors are not declared as trace inputs | **❌, and correctly** |
| S13 | `Adapted.online`: re-snapshots the base on every call | ✅ *(after the case below)* |

**Twelve of thirteen.**

**S13 could not fail when it was first run**, and the fault was not in the rule
— it was that every test called `online()` exactly once, so a second snapshot
had nothing to be a second snapshot *of*. Worse, the suite would have stayed
green in the most misleading way available: the delta would revert perfectly to
the weights the first round left behind, and every other assertion would still
hold. A case that arms, steps, arms again, steps again and then reverts to the
**original** base was added, and the fault then failed. This is
`docs/BACKWARD.md` §7's pattern and `docs/LOSS.md` §5.2's, met once more.

**S12 cannot be caught and it is right that it cannot.** With one trace per
step, declaring the tensor arguments and letting capture burn them in are the
same computation: nothing replays the trace with a different input, so the guard
the declaration installs is never consulted. Checked rather than argued — under
S12 the entropy history, the final entropy and every adapted weight are
**bit-identical** to the unsabotaged run. `_tensor_inputs` stays because a trace
that is a function of nothing is not the object `docs/CAPTURE.md` §2 describes,
and because the first caller that reuses a trace across steps will need it; but
nothing in this document's tests is entitled to claim it.

### 9.1 What this suite still cannot see

* **It runs on a 24-token toy, not on SmolLM2.** The real-checkpoint numbers in
  §3–§6 are measurements in this document, not tests in `pytests/`, for the
  reason `docs/BACKWARD.md` §7.1 gives — the suite does not download a
  checkpoint. So §6's 1.512e-06 can move without anything going red.
* **One method.** The claim that `Delta` generalises across methods is
  structural (`Tent` holds no state) and not yet demonstrated by a second one.
* **Nothing checks memory or time.** §5.2's bytes are arithmetic on shapes and
  §3's 0.42 s/step is a single wall-clock reading on a loaded machine.
* **No convergence claim.** Ten steps at one learning rate on one sentence. The
  curve is monotone over those ten; nothing here says it stays monotone, and
  `lr=16` on the toy model is measurably not.

---

## 10. What this round did not do

| | why |
|---|---|
| `torchnative.nn.federated` | §1. It needs a second rank, a rendezvous and a serialiser, and all three refuse. `Delta.publish` is the seam it will attach to |
| a second adaptation method | The abstraction is built for one; §9.1's second bullet is honest that one method does not prove it |
| `native_layer_norm`'s derivative rule | §8.1. It is one arm in `tape.rs`, which was out of scope this round |
| momentum, Adam, a learning-rate schedule | `torch.optim.SGD` was enough for a curve, and `docs/LOSS.md` §6.4's four missing ops still gate Adam |
| a stage-0 method | `wrap` refuses one by name. A method that recomputes statistics needs no capture and no tape, so it is a different step function, not a flag on this one |
| lifetime **names** | §2.2. This integration needed three answers and no names, and inventing a fourth set after §3 discarded two would be the same mistake a third time |
| adapting on a stream | The loop adapts on one batch and is scored on a held-out one. A real device sees a stream, and nothing here says what happens after a thousand steps |
| anything on device | Desktop macOS only, as with `docs/BACKWARD.md` |

---

## 11. Gates

| gate | before | after |
|---|---|---|
| `pytests/run.sh` | 302 ok, 0 FAIL | **310 ok, 0 FAIL** |
| `run.sh` DOCWATCH | 159/159 | **173/173** (14 new markers, all in this document) |
| `tools/golden/compare.py` | 7447/7447, ops=166, pending 1 | **7447/7447, ops=166, pending 1** |
| `compare.py --self-test` | 19 comparators × 11 fault modes | **unchanged** |
| `verify_schemas.py` | 4475/4475 | **4475/4475** |
| sweep26 (`.eval()`) | 26/26 | **26/26** |
| sweeptrain (`.train()`) | 26/26 | **26/26** |
| prefill sha256, f32 × 5 and bf16 × 4 | — | **9/9 unchanged, and 9/9 equal through the wrapper** (§7) |

`ops=166` is unchanged **on purpose**: nothing in this round touched
`rust/torch_c/src/`, and the whole of `torchnative.adapt` and
`torchnative.delta` is Python over the capture and tape surfaces
`docs/BACKWARD.md` built. A change in that number would have meant an
adaptation API had needed a kernel, which would have been news.

<!-- DOCWATCH: count smoke_ok ge 310 -->
<!-- DOCWATCH: count golden_ops_covered ge 166 -->
> The line above was `eq 166` and failed the moment an unrelated round added two ops. The
> claim it is backing is *"this round added none"*, and a shared global count cannot express
> that — only that it did not go **down**. A marker asserting equality on a number other work
> legitimately moves fails on somebody else's commit, which is the crying-wolf failure
> `docs/DOCWATCH.md` warns about, arriving in a marker rather than in the checker.
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_tent_reduces_prediction_entropy_and_upstream_agrees present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_a_delta_reverts_the_base_weights_bit_for_bit present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_tent_refuses_a_layer_norm_model_by_naming_the_missing_rule present -->

### 11.1 The eight new tests

```
test_tent_reduces_prediction_entropy_and_upstream_agrees
test_the_wrong_sign_and_a_zero_step_do_not_reduce_entropy
test_an_adaptation_step_that_cannot_move_anything_is_refused
test_a_delta_reverts_the_base_weights_bit_for_bit
test_a_delta_names_a_check_for_the_destinations_it_cannot_reach
test_tent_refuses_a_layer_norm_model_by_naming_the_missing_rule
test_a_method_declares_its_differentiation_stage_and_wrap_reads_it
test_an_offline_wrapper_is_the_model_it_wraps
```

They run in the two-interpreter shape the checkpoint, capture and distributed
roads use: a subprocess on the vendored tree does the adaptation, and **this**
process runs the identical ten Tent steps on upstream torch with
`loss.backward()`. The model source is one string executed on both sides, so the
two cannot drift apart; the step is the only thing that differs.

---

## 12. Every command in this document

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-adapt
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
SHIM="PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY"    # VENDOR.md wall 3

# §3, §5, §6  Tent on SmolLM2: upstream writes, the shim loads and compares
$PY   /tmp/adapt/tent_up.py   1e-3 10 t1
$SHIM /tmp/adapt/tent_shim.py tent 1e-3 10 t1
$SHIM /tmp/adapt/tent_shim.py tent 1e-3  1 t1       # the one-step gradient

# §4  the three controls
$SHIM /tmp/adapt/tent_shim.py anti     1e-3 10 t1
$SHIM /tmp/adapt/tent_shim.py lr0      0.0   6 t1
$SHIM /tmp/adapt/tent_shim.py detached 1e-3  3 t1

# §6.1  the attribution: one tape, two objectives
$PY   /tmp/adapt/attr_up.py   ;  $SHIM /tmp/adapt/attr_shim.py

# §7  the forward, plain and through the wrapper
$SHIM /tmp/adapt/seqlen_adapt.py f32  ;  $SHIM /tmp/adapt/seqlen_adapt.py bf16

# §9  sabotage: 13 faults, no rebuild
$PY /tmp/adapt/sab.py            # or /tmp/adapt/sab.py S12 S13 for one

# §11  gates
PYTHON=$PY sh rust/torch_c/pytests/run.sh
$PY tools/golden/compare.py  ;  $PY tools/golden/compare.py --self-test
$PY rust/torch_c/pytests/verify_schemas.py
$SHIM /tmp/k26/sweep26.py /tmp/adapt/ev1  ;  $SHIM /tmp/train/sweeptrain.py /tmp/adapt/tr1
```

The scratch harnesses live under `/tmp/adapt/` and are reproduced nowhere else;
every number they produce is quoted above with the command that made it.
