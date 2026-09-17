# VOICE5 — Higgs Audio v2, 5.4B and autoregressive, end to end; and a tie that no tolerance can cross

Worktree `work/higgs` on develop `0276ae7`. Upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv/bin/python`) is the oracle throughout, in its own process
with `PYTHONPATH` stripped.

`docs/architectures/VOICE4.md` ran BigVGAN — 112M parameters, no sampler, one forward — and closed on
"one model is one model". This round takes the model VOICE4.md §1 **rejected by name**.

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| Which model? | **Higgs TTS 2** (`bosonai/higgs-audio-v2-generation-3B-base`), **5,377,281,024 parameters**, plus its 600 MB audio tokenizer. Autoregressive, eight codebooks, delay pattern. | §1 |
| Did it reach the end? | **Yes.** `from_pretrained` → `generate(do_sample=False)` → `processor.batch_decode` → **53,760 samples = 2.24 s of speech**, under the shim. | §3 |
| Does the waveform agree? | **2.065e-06 relative**, against a tolerance of **8.860e-06** derived from upstream's own float32-vs-float64 error. Pearson r = 0.999999999999. | §6 |
| Do the logits agree? | Yes — **every comparable step is within 1.4× upstream's own bfloat16-vs-float32 error**, against a 4× bar. | §6.2 |
| Do the two greedy trajectories match? | **No, and this is the round's finding.** They agree for six steps and then split at a position where **upstream's own top-two logits are bit-identical in bfloat16**. | §6.3 |
| Which side is right at that tie? | **The shim.** float32 ranks the two apart and picks the token the shim picked. | §6.3 |
| What were the walls? | Three. `aten.norm.ScalarOpt_dim` and `aten._weight_norm_interface.default` **with no meta kernel**, and `UntypedStorage.copy_` refusing a *fresh* storage. | §4, §5 |
| How many new operators? | **Zero. Two meta kernels, one defect fixed, one refusal narrowed.** | §8 |

---

## 1. Why this model, and why VOICE4's rejection no longer holds

VOICE4.md §1 dismissed this family in one line:

> **Higgs TTS 2/3, Qwen3-TTS, Dia2, Chroma** — 1.7B–4B parameters and autoregressive with
> sampling. The download alone is tens of gigabytes, and `generate` is nondeterministic, which
> `AGREE.md` §2 excludes from its denominator for exactly this reason.

Three objections. Each was checked rather than assumed:

| objection | status now |
|---|---|
| "cannot be scored" | **False, and was already false.** `rust/torch_c/pytests/agree2_scores.json` records `higgs_audio_v2` at `rel` 2.854e-07 against an `oracle_rel` of 3.111e-07 over n=16384 logits, `self_repeat_rel` 0.0, `missing: []`, `unexpected: []`. That is a *shrunk config with random weights* (`rust/torch_c/pytests/agree_sweep.py`'s stated limit), so it says the arithmetic agrees, not that the model runs — which is precisely the gap this round closes. |
| "`generate` is nondeterministic" | **A property of the call, not the model.** This checkpoint's `generation_config.json` ships `do_sample: true, temperature: 1.0, top_k: 50, top_p: 0.95`; every one is overridden here. §2 lists what was pinned. |
| "tens of gigabytes" | **Stands, and is a constraint rather than a reason to stop.** 11.5 GB + 806 MB, onto the external disk. §2.1. |

**Two claims that `docs/kernels/KERNELS26.md` and `docs/kernels/SCATTER.md` still carry were checked against the code and are
stale.** `docs/kernels/KERNELS26.md` records `higgs_audio_v2_tokenizer` tripping at construction on
`nn.Buffer(torch.Tensor([True]))`; it does not — measured, it answers `tensor([1.])`,
`torch.float32`. `docs/kernels/SCATTER.md` cites `docs/architectures/ARCH100.md` for `masked_scatter` being open for this model;
`aten.masked_scatter.default` is in `_aten_implemented()` and has been. Neither was the wall.

---

## 2. What was pinned, and what was substituted

```
model          bosonai/higgs-audio-v2-generation-3B-base      5,377,281,024 params
audio tokenizer bosonai/higgs-audio-v2-tokenizer             24 kHz, 8 codebooks, 320x downsample
prompt         the model card's own "Single-speaker smart voice" example, verbatim
               system + scene + user,  add_generation_prompt=True  ->  57 input ids
generation     do_sample=False, num_beams=1, max_new_tokens=64, use_cache=True,
               torch.manual_seed(0)
dtype          language model bfloat16;  audio tokenizer float32
output         (1, 64, 8) codes  ->  53,760 samples  =  2.24 s at 24 kHz
```

**`add_generation_prompt=True` is load bearing and was not obvious.** Without it — the first thing
this round tried — greedy decoding emits the audio-stream BOS and EOS back to back and stops after
ten steps with 0.04 s of near-silence (absmax 1.4e-04). Sampling did the same, which is what
identified the prompt rather than the sampler as the cause. The generated audio is only a claim
about the shim if the model was asked properly, so this is recorded rather than quietly fixed.

**`bfloat16`, not `float32`, for the language model, and the reason is the machine.** 5.4B
parameters in float32 is 21.5 GB on a 16 GB host; measured, it swaps to **40 s per token**
(8 tokens in 321 s) against 0.3 s per token in bfloat16. float32 is therefore used where it is
affordable — as the **oracle** for eight steps (§6.2), and for the whole audio decoder — and
bfloat16 is the pinned condition for the 64-step run. §7 says what that costs.

### 2.1 Disk

`HF_HOME` was unset, which puts the HuggingFace cache on the **internal** disk with 12 GB free
against an 11.5 GB checkpoint. It was set to **`/Volumes/macMini/caches/hf`** on the external disk
before anything was downloaded.

```
before   /  12 GiB free     /Volumes/macMini  57 GiB free
after    /  12 GiB free     /Volumes/macMini  45 GiB free      (12.4 GB of model)
```

The internal disk did not move. No smaller variant was needed; there is no smaller Higgs TTS 2.

### 2.2 The one substitution

**`torchaudio` is not installed** in the spike venv, and `HiggsAudioV2TokenizerModel` is gated
behind `@requires(backends=("torchaudio",))`, so the *processor will not load* without it.
`transformers` decides availability with `importlib.util.find_spec`, so a module on `sys.path`
satisfies the gate.

The stub is written so it cannot flatter anything: **every entry point raises**, and records the
call. `torchaudio` is reachable only from the tokenizer's ENCODE path
(`torchaudio.functional.resample`, 24 kHz → 16 kHz for the semantic model); this round decodes
only. `test_the_torchaudio_stub_was_never_called` asserts the recorded call list is empty, so the
claim "the decode path never reaches it" is checked rather than argued. §7 says what the absence
costs — it is why voice cloning was not run.

---

## 3. It reached the end

```
[shim] model 5377281024 params        63.3 s
[shim] generate 85.6 s                audio_sequences (1, 64, 8)
[shim] decode 0.8 s                   53760 samples  absmax 0.733  rms 0.1047
```

Upstream, for comparison, generates the same 64 steps in 18.1 s. The shim is **4.7× slower** and
that is the whole of the cost; nothing in the path was skipped, stubbed or routed around.

---

## 4. The wall, which was VOICE4's wall in a new place

Loading the processor stopped with:

```
TypeError: _WeightNorm.forward() missing 1 required positional argument: 'weight_v'
```

**That message names the wrong thing, and the way it does is worth more than the fix.**

`torch.nn.utils.parametrizations.weight_norm` builds the audio tokenizer's decoder convolutions,
and `from_pretrained` builds under `accelerate.init_empty_weights` — **on the meta device**.
`_WeightNorm.right_inverse` calls `torch.norm_except_dim`, which lands on
`aten.norm.ScalarOpt_dim`, which had no meta kernel. And `ParametrizationList.__init__` runs
`right_inverse` inside:

```python
try:
    new = module.right_inverse(new)
except NotImplementedError:
    pass                       # <- the shim's "no meta kernel" disappears here
```

So the refusal was **swallowed**, `new` stayed the un-inverted tensor, the list recorded
`is_tensor = True` — and `forward()`, which takes two originals, was then called with one.
`test_the_weight_norm_wall_reports_itself_and_not_a_typeerror` is the regression test: it is not
enough for the meta kernels to exist, `weight_norm` on a meta module has to actually answer.

This is `VOICE4.md` §4.1 again with different names. **Both ops were in `_aten_implemented()` the
whole time**, with dense kernels and golden cases; that list means "has a dense kernel and a golden
case", and a meta tensor has no values to compare, so meta support is invisible to it by
construction.

### 4.1 The two meta kernels

Both follow `docs/devices/META.md` §7.1: the dtype and shape rules are the **dense kernel's**, called rather
than restated.

| kernel | shape | dtype |
|---|---|---|
| `aten.norm.ScalarOpt_dim` | `reduced_dims`, with `keepdim`. An **empty `dim` list means every axis** — the opposite of the usual reading, and the dense kernel's own documented rule. `p` decides values, never shape, so it is read and discarded. | the input's, unchanged, `float16`/`bfloat16` included. Non-floating raises `norm(): input dtype should be either floating point or complex`, the dense kernel's wording. |
| `aten._weight_norm_interface.default` | a pair: `out` is `v`'s shape; `norms` keeps `dim` and reduces every other axis keepdim — the dense kernel's own `for d in 0..rank { if d != axis }` loop, which for a rank-1 `v` reduces nothing. | `out` is `v`'s; `norms` is `float32` for a `float16`/`bfloat16` input, which is the dense kernel's `norm_tag` and the reason upstream's dense `norms` come back widened. |

`_compare` drives them through `torch.ops.aten.<op>.<overload>` on both sides. That spelling is
not cosmetic: **`torch.norm(x, 2, [0])` does not reach `aten.norm.ScalarOpt_dim`** — it decomposes
to `aten.linalg_vector_norm.default`, a different op with a different meta kernel. The first
version of this round's test called `torch.norm` and was measuring something else.

### 4.2 Where upstream's meta kernel disagrees with upstream's dense kernel

Six cases, all measured on 2.13.0. In each the shim follows **its own dense kernel**, per META.md
§7.1's rule that a meta kernel may not promise a dtype, or accept a call, that the dense kernel
would refuse. They are pinned by name in
`test_where_upstreams_meta_and_dense_kernels_disagree_the_shim_follows_dense` rather than dropped
from the comparison:

| case | upstream meta | upstream dense | this shim (meta and dense) |
|---|---|---|---|
| `float16` `v` | `norms` `float16` | `norms` `float32` | `float32` |
| `dim=1` on a 3-D `v` | accepted, `[1, 3, 1]` | `INTERNAL ASSERT FAILED (dim == 0 \|\| dim == v.dim() - 1)` | refused by name |
| `v` `float32`, `g` `float64` | accepted | `expected scalar type Float but found Double` | refused, upstream's wording |
| rank-1 `v` | `norms` `[1]` | `norms` `[4]` | `[4]` |
| `dim=-1` on a 3-D `v` | accepted, `[1, 1, 1]` | refused (raw `-1`, before normalisation) | accepted, `[1, 1, 5]` — the shim normalises `-1` to the last axis in **both** its kernels. A pre-existing divergence, not introduced or removed here; Higgs uses `dim=0`. |
| integral / boolean `v` | `RuntimeError` (through its `linalg.vector_norm` decomposition) | `RuntimeError` | `NotImplementedError` — a `RuntimeError` **subclass**, carrying upstream's dense wording |

### 4.3 A dense defect the meta work found

`torch.norm_except_dim(v, 2, 0)` on a **rank-1** `v` answered `[1]` where upstream answers `[4]`.
The meta kernel was right and the dense answer it was being compared against was wrong.

The cause is the empty-list ambiguity above. A rank-1 `v` exempts its only axis, so `dims` came out
`[]` — and `[]` does not mean "reduce nothing" to `aten.norm.ScalarOpt_dim`, it means **every
axis**. Upstream's C++ never has the ambiguity because it passes no dim list at all:
`v.view(v.size(0), -1).norm(pow, 1)`, a reduction over a trailing axis of extent one. That
construction is reproduced in `bootstrap.py` rather than special-cased to `abs`, because the
reduction of a single element is `|x|` only for some `pow` (`pow=0` counts non-zeros instead).

No existing case covered it: `test_shim.py`'s `norm_except_dim` cases and
`tools/golden/cases.py`'s are all rank-2 and rank-3.

---

## 5. The second wall, which is not an operator at all

```
NotImplementedError: torch._C shim: UntypedStorage.copy_ -- this shim's storages are
filled once, by the reader that delivers their bytes, and are read-only afterwards
```

reached from `AutoProcessor.from_pretrained`. `transformers.ProcessorMixin.__repr__` calls
`to_dict()`, which `copy.deepcopy`s every attribute — and `HiggsAudioV2Processor` **holds the audio
tokenizer model**. So the user path deep-copies an entire network before generating anything, and
`__repr__` is evaluated eagerly inside an f-string, so no log level avoids it.

`Tensor.__deepcopy__` → `UntypedStorage.clone()` →
`type(self)(self.nbytes(), device=self.device).copy_(self)`: **a storage allocated one line
earlier, which has never been filled.** Filling it *is* "filled once, by the reader that delivers
the bytes" — the rule the refusal was enforcing was not being broken. So exactly that case is
accepted and every other keeps its refusal:

```
destination is meta            refused -- it has no bytes at all
destination is a SNAPSHOT      refused -- a write would be invisible to its tensor
destination already filled     refused -- filled once
destination shares its buffer  refused -- visible to some holders of the alias, not others
source is not a storage        refused
sizes differ                   refused, with upstream's wording
fresh, unshared, unfilled      FILLED
```

`test_the_read_only_storage_refusal_survives_the_deepcopy_fix` asserts the snapshot refusal is
still there, because the risk of this change is widening it, and
`test_a_deep_copy_is_independent_of_its_original` asserts the copy does not alias — a `copy_` that
aliased would pass every shape and dtype check and still be wrong.

One line behind it, `TensorBase.set_` refused a **`TypedStorage`**, which is what
`torch/_tensor.py:234` passes. It is now unwrapped to its `_untyped_storage`; the dtype it carries
is **not** adopted, so the size and itemsize checks below are unchanged.

---

## 6. The measurement

`rust/torch_c/pytests/higgs_e2e.json` is the record; `higgs_e2e.py` regenerates it (nine runs,
about fifteen minutes, not wired into `run.sh`).

### 6.1 The waveform

Both sides decode **upstream's own generated codes** in float32; upstream additionally decodes them
in float64 to supply the oracle. This is the half that is deterministic, so it is the half where a
waveform tolerance means something.

```
waveform length                    53760 samples, both sides            (2.24 s at 24 kHz)
scale (max |upstream f32|)         0.689036

shim  vs  upstream f32             2.0653e-06      <- the result
upstream f32  vs  upstream f64     2.2150e-06      <- the tolerance source
shim  vs  upstream f64             2.4234e-06

TOLERANCE = max(1.186e-06, 4 x oracle)  8.8602e-06      PASS
ratio (shim's error / upstream's)  1.094
pearson r                          0.999999999999
bit-identical samples              1290 / 53760
rms   shim 0.11360748   upstream 0.11360748
```

The rule is `docs/numerics/AGREE.md` §2's, carried over with its citation exactly as VOICE4.md §5.1 carried it:
the floor is AGREE.md's 1.186e-06 and the factor of 4 is AGREE.md's own. Here the model's own
oracle error is 2.2e-06, so the per-model derivation still decides the answer — AGREE.md's bare
population floor of 1.186e-06 would have flagged a result that is *nearer* the float64 truth than
one upstream float32 run is to another.

### 6.2 The logits, and the oracle one precision up

The language model runs in bfloat16, so its oracle is **upstream in float32** — the same
construction as float32-vs-float64, applied to the precision actually in use. While two runs have
produced the same tokens, their logits are functions of the same input and are comparable; past
the first differing token they are not, and this population stops exactly there rather than
averaging across the line.

| step | scale | oracle (up bf16 vs up f32) | shim vs up bf16 | shim vs up f32 | ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.485 | 7.136e-03 | 0.000e+00 | 7.136e-03 | 1.00 |
| 1 | 10.858 | 1.168e-02 | 2.878e-03 | 1.168e-02 | 1.00 |
| 2 | 16.308 | 3.528e-03 | 3.833e-03 | 4.912e-03 | 1.39 |
| 3 | 15.337 | 4.706e-03 | 4.075e-03 | 6.202e-03 | 1.32 |
| 4 | 19.371 | 7.017e-03 | 6.453e-03 | 4.380e-03 | 0.62 |
| 5 | 15.610 | 9.423e-03 | 1.201e-02 | 9.978e-03 | 1.06 |
| 6 | 18.344 | 9.159e-03 | 1.022e-02 | 8.512e-03 | 0.93 |

Every ratio is inside AGREE.md's 4×, and two are **below 1** — the shim nearer the float32 answer
than upstream's own bfloat16 is. In absolute terms the disagreements are **one to three bfloat16
ulps** (the ulp at magnitude 10 is 0.0625).

### 6.3 Where it stops, and why no tolerance reaches it

The two greedy runs produce **identical text tokens** and identical audio codes for six steps, and
then split. At **(step 6, codebook 5)**:

```
upstream bfloat16   logit at token 75 = 10.25     logit at token 764 = 10.25     <- EXACTLY EQUAL
upstream float32    logit at token 75 = 10.1923   logit at token 764 = 10.2696
upstream picks 75   (argmax breaks the tie by index)
the shim picks 764  (one ulp separates them on its side)
float32 picks 764
```

**The tie is an artefact of the dtype, and at it the shim is right.** A one-ulp difference — well
inside the agreement measured in §6.2 — decides which of two bit-identical logits wins, and from
the next step the two runs are continuing different sequences. So the waveforms of the two *full*
64-step runs are not comparable and are not compared: 85 of 512 codes match, and the relative
difference of those waveforms is 1.35, which is a statement about two different utterances and not
about arithmetic.

This is the honest end of the autoregressive half, and it is a property of **greedy decoding in
bfloat16**, not of either implementation. Three things follow, and all three are stated rather than
worked around:

* it is **not** fixable by widening a tolerance — there is no tolerance on `argmax`;
* it **would** likely go away in float32 (where the two logits differ by 7.7e-02, ~4000 ulps), which
  this machine cannot hold for 5.4B parameters (§2);
* it is **not** evidence of a defect, and the float32 oracle is what says so — the disagreement is
  a tie-break, and upstream loses it.

---

## 7. What is not closed

* **The full 64-step trajectory does not match**, for the reason in §6.3. What is proven is: the
  shim reaches audio, its logits agree within upstream's own bfloat16 error, and its decoder
  agrees on a waveform within upstream's own float32 error. What is *not* proven is that a long
  greedy generation reproduces upstream token for token in bfloat16 — and the measurement says
  upstream does not reproduce *itself* across dtypes there either.
* **float32 for the language model.** 21.5 GB on a 16 GB host. The oracle runs are 8 steps; a
  64-step float32 pair was not attempted because at 40 s/token it is ~45 min per side and would
  have been swapping against other work on the machine, which `CLAUDE.md` warns contaminates
  exactly this kind of number.
* **Voice cloning / the ENCODE path.** `_extract_semantic_features` resamples 24 kHz → 16 kHz
  through `torchaudio.functional.resample`, which is not installed (§2.2). The reference-audio
  conversations in the model card are therefore unreached. This is the same shape as VOICE4.md
  §7's mel front end, and it is a *transcribable* gap rather than a candle-level one: the repo
  already closed `sinc`, `kaiser_window` and `i0` for exactly this family of resamplers.
* **`aten.linalg_vector_norm.default` still has no meta kernel** — it is what `torch.norm` with a
  `dim` list actually decomposes to (§4.1), so `torch.norm(x, 2, [0])` on a meta tensor still
  refuses. Higgs does not need it; naming it is the point.
* **The `dim=-1` divergence** in `_weight_norm_interface` (§4.2, row 5) is pre-existing in the
  dense kernel and was not changed.
* **GraalVM native image.** Not attempted; out of this round's scope.

---

## 8. Numbers

```
test_higgs.py                 19 tests
new meta kernels               2   (aten.norm.ScalarOpt_dim, aten._weight_norm_interface.default)
new operators                  0
defects fixed                  1   (norm_except_dim on a rank-1 v)
refusals narrowed              2   (UntypedStorage.copy_ into a fresh storage; set_ of a TypedStorage)
golden cases                   unchanged -- a meta kernel has no values to compare, and both ops
                               were already golden-covered as dense kernels
nullifications run             9, all caught -- §9
```

## 9. Nullification

Every kernel, every fix and every claim was broken deliberately and the suite re-run.

| # | nullification | caught by | verdict |
|---|---|---|---|
| 1 | meta `norm.ScalarOpt_dim` arm deleted | the norm kernel test, `norm_except_dim`, **and** the `weight_norm`-on-meta regression test | **red (3)** |
| 2 | meta `_weight_norm_interface` arm deleted | the wni kernel test, the divergence test, the `weight_norm` regression test | **red (3)** |
| 3 | meta `wni` follows upstream's *meta* dtype (`float16` norms) instead of its own dense | the divergence test | **red** |
| 4 | `norm_except_dim`'s rank-1 fix reverted | `test_norm_except_dim_the_live_caller_answers_what_upstream_answers` | **red** |
| 5 | `UntypedStorage.copy_` back to an unconditional refusal | both deepcopy tests | **red (2)** |
| 6 | `copy_` allowed to write through a **snapshot** storage | `test_the_read_only_storage_refusal_survives_the_deepcopy_fix` | **red** |
| 7 | `_ORACLE_FACTOR` widened from 4.0 to 400.0 | `test_the_tolerance_would_actually_reject_a_wrong_waveform` **and** `test_the_tolerance_is_read_off_upstreams_own_error_not_chosen` | **red (2)** |
| 8 | `higgs_e2e.json`'s `logits.per_step` emptied | the logits test and the not-empty test | **red (2)** |
| 9 | the recorded tie perturbed so the two logits are no longer equal | `test_the_greedy_trajectory_diverges_at_an_exact_bfloat16_TIE` | **red** |

**Nullification 7 is VOICE4.md §6.1's uncaught one, and it is caught here.** The reason it is
caught is not the rejection test on its own: that test computes `tol` from the factor, so scaling
the factor scales the bar and the relative assertions stay true — the same tautology. What catches
it is the pair of **absolute** pins: `_ORACLE_FACTOR == 4.0` with AGREE.md's citation on it, and
`abs(tol - 8.8602e-06) < 1e-09`. A derivation that is only checked against itself cannot fail; the
number has to be nailed to something outside the derivation.

<!-- DOCWATCH: op-implemented aten.norm.ScalarOpt_dim -->
<!-- DOCWATCH: op-implemented aten._weight_norm_interface.default -->
<!-- DOCWATCH: op-implemented aten.masked_scatter.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs reduced_dims present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs refuse_duplicate_dims present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py norm_except_dim present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_higgs.py test_higgs_reached_audio_under_the_shim present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_higgs.py test_the_weight_norm_wall_reports_itself_and_not_a_typeerror present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_higgs.py test_the_tolerance_would_actually_reject_a_wrong_waveform present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_higgs.py test_the_greedy_trajectory_diverges_at_an_exact_bfloat16_TIE present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_higgs.py test_the_read_only_storage_refusal_survives_the_deepcopy_fix present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_higgs.py test_the_torchaudio_stub_was_never_called present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/higgs_e2e.py CONVERSATION present -->
<!-- DOCWATCH: json-key rust/torch_c/pytests/higgs_e2e.json waveform present -->
<!-- DOCWATCH: json-key rust/torch_c/pytests/higgs_e2e.json trajectory present -->
<!-- DOCWATCH: json-key rust/torch_c/pytests/higgs_e2e.json pinned present -->
<!-- DOCWATCH: json-key rust/torch_c/pytests/higgs_e2e.json logits present -->
