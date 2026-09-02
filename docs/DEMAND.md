# DEMAND — regenerating the work queue by running real models

Measurement round only. **Nothing in this round changed source** — `rust/torch_c/{aten.rs,tensor.rs,dtype.rs,flash.rs}`,
`bootstrap.py`, `tools/golden/cases.py`, `torchnative/src/main/torch/` (vendored) are all untouched.
`git status --short` in the worktree is empty throughout. Every finding below is a *candidate*
for the next round, not something applied here.

docs/ARCH26.md (26 architectures, `transformers` 5.15.1, toy `AutoConfig`s) is the existing
denominator. This round goes **wider**: 18 new architectures/tasks the 26 do not cover, plus one
feature path (`TorchnativeConfig` quantised load) explicitly asked for, run against **upstream
torch as the oracle**, not against each other.

---

## 0. The ranked list — build these next

Ranked by **how many distinct models hit each wall as their first blocker** in this round. Ties
broken by real-world reach (how many architecture *families*, not just the one model tried, use
the same code path).

| rank | gap | models that hit it (this round) | kind |
|---|---|---|---|
| 1 | `torch.batch_norm` / `aten.native_batch_norm.default` | `resnet`, `mobilenet_v2` (2) | **missing kernel** — leaf op upstream (`_dispatch_has_kernel_for_dispatch_key(..., "CompositeImplicitAutograd")` is `False`), absent from `aten.rs`/`overloads.json`/`methods.json` entirely, no spelling to wire even once a kernel exists. Blocks **every BatchNorm-based CNN backbone** (ResNet, MobileNetV2/V3, and by the same code shape EfficientNet, RegNet, DenseNet) — the single most generic vision-CNN primitive tried this round. Note 2-D `aten.convolution.default`, ARCH26's blocker for `zoedepth`, is **no longer the wall** — both CNNs got past their stem `conv2d` and stopped one layer later, at the norm. |
| 2 | legacy `torch.Tensor(...)` constructor (`TensorBase.__new__`, `tensor.rs::py_new`) | `pegasus` (1 new; `sew_d` already recorded this family in ARCH26 §4) | **structural** — forbidden-file territory even outside this round's rules: the refusal fires inside PyO3's `#[new] fn py_new`, and the class that could grow a Python-level override (`torch.Tensor(TensorBase)`) is the vendored tree. Two *distinct* call shapes now confirmed blocked: `torch.Tensor(int)` (allocate-uninitialised, `sew_d`) and `torch.Tensor(ndarray)`/`torch.FloatTensor(ndarray)` (build-from-data, `pegasus`, in `PegasusSinusoidalPositionalEmbedding.create_weight`). Both trace to `aten.lift_fresh.default` upstream — **the same primitive `torch.tensor()`/`new_tensor()` already use** — so the missing piece is purely the constructor-slot dispatch, not new arithmetic. |
| 3 | `torch.full_like` / `aten.full_like.default` | `t5`, `switch_transformers` (2) | **missing kernel** — leaf upstream, sibling `aten.full.default` already implemented and golden-compared. Both hits are the same line, `T5Attention._relative_position_bucket`'s `torch.full_like(relative_position_if_large, num_buckets - 1)` — shared code across the whole T5 family (`t5`, `mt5`, `long_t5`, `umt5`, `switch_transformers`, `t5gemma` all inherit or duplicate it), so this one kernel is worth more than 2 votes. |
| 4 | `torch.as_tensor` (`.generate()` path) | `whisper` (1, but only after its **forward** already matched upstream — see §2) | **missing spelling**, cheapest class in this table. `torch.as_tensor(data)` and `torch.Tensor(ndarray)` (rank 2) both trace to `aten.lift_fresh.default`, and `torch.tensor()`/`Tensor.new_tensor()` already wire that primitive (ARCH26 §7). `generation_whisper.py`'s `_retrieve_init_tokens` is the hit here, but `transformers/generation/utils.py` calls `torch.as_tensor` in several shared helpers (`_prepare_attention_mask_for_generation`, stopping-criteria bookkeeping) that `t5`/`bart`/`mbart`/`pegasus`'s `.generate()` never reached this round because each stopped earlier for its own reason (ranks 2, 3, 5). This is a wall several already-run models are queued up behind, not just whisper's. |
| 5 | `TensorBase.new_zeros` / `aten.new_zeros.default` | `bart` (1) | **missing kernel**, but same shape as an already-solved sibling: `aten.new_ones.default` is implemented (`aten.rs:9416`, `new_ones_default`) and `new_zeros` is not registered at all — not in `aten.rs`, not in `methods.json`. `BartForConditionalGeneration.forward`'s `shift_tokens_right` is generic code duplicated across the whole BART-derived family (`bart`, `marian`, `blenderbot`, `mvp`) — `mbart` and `pegasus` use *different* shift-right code and hit ranks 2/8 instead, which is itself informative: this family is not as uniform as it looks from one member. |

**Honorable mentions, one vote each, not in the top five but each closes a distinct model:**
`aten.linalg_vector_norm.default` (`sentence_embed`'s `F.normalize`, missing kernel — a **distinct**
leaf from the already-implemented `aten.norm.ScalarOpt_dim`, confirmed by
`_dispatch_has_kernel_for_dispatch_key` returning `False` for it too); `torch.meshgrid` (`swin`'s
relative-position-index construction, missing **spelling** — `CompositeImplicitAutograd` upstream,
decomposing purely into `aten.view.default` + `aten.expand.default`, both already implemented, so
this is a same-shape fix to ARCH26 §7's `conv2d`/`new_tensor` composites); `aten.squeeze.default`
(`mbart`'s `shift_tokens_right`, **the GOLDEN.md blind-spot shape**: `squeeze` is *declared* in both
`overloads.json` and `methods.json` with three overloads — `squeeze()`, `squeeze.dim`,
`squeeze.dims` — but `aten.rs`'s dispatch `match` only has an arm for `squeeze.dim`; the no-arg
overload looks present in the name tables and is not reachable); `torch.linspace` (`convnext`'s
stochastic-depth rate schedule, missing kernel, leaf upstream — the generic "N vision backbones
with a `drop_path_rate` schedule" pattern, same family as `swin`'s missing `meshgrid` in that both
are vision-transformer/CNN construction-time utilities rather than forward-path arithmetic).

---

## 1. Method

- `transformers` 5.15.1, `torch` 2.13.0 (upstream, the vendored/shim build's origin), toy
  `AutoConfig`s (small hidden size, 1–2 layers, few heads, tiny vocab) — coverage, not model
  quality, same recipe as ARCH20.md/ARCH26.md.
- Every script printed `"shim" if hasattr(torch._C, "_aten_implemented") else "upstream"` as its
  first line, both sides, every run. Transcribed literally in every log; no run of the 18
  architectures × 2 sides (36 runs) plus 8 numeric dumps × 2 sides (16 runs) plus the quant-load
  run (2 runs) printed the wrong label.
- Every run wrapped in a 120s `SIGALRM` per model. **No hang observed** — every run (upstream and
  shim, forward and construction failures alike) returned in well under a second, toy configs
  being tiny. `float8_e4m3fn` specifically (the brief's named hang case) was not exercised this
  round — no target below asked for a non-default dtype.
- Each side run as its own subprocess (`PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1`
  for the shim; `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL` for upstream) — never both `torch`
  variants in one interpreter.
- Built via the instructed pipeline: `cargo build --release` in `rust/torch_c` with
  `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-sweep`,
  `TORCH_C_ARTEFACT=.../release/lib_C.dylib` exported before every gate run (docs/GOLDEN.md §6's
  own trap — a build without the matching artefact env var silently measures a stale binary).
  `bash vendor/install_shim.sh` after.
- Scratch scripts under `/tmp/sweep/` (not committed, matching ARCH26/KERNELS26 convention):
  `check_common.py` (18-architecture pass/fail sweep, shared by both sides via `PYTHONPATH`),
  `dump_model.py` + `compare_dumps.py` (numeric comparison for the 8 that forward), `quant_build_ckpt.py`
  + `quant_load_test.py` (the `TorchnativeConfig` path).
- **"Matches upstream" method**: `torch.manual_seed(0)` immediately before model construction on
  both sides (KERNELS26.md §2.4's method — the shim's RNG must reproduce upstream's init bit for
  bit for this to mean anything, and it does, see §2 below), then a forward on identical
  hand-built input tensors (`torch.manual_seed(1)` before any randomly-generated input, so both
  sides see the same pixels/waveform), comparing every `state_dict` tensor and the full output by
  `.tolist()`.

## 2. Model table

"loads" = model class constructs from a toy `AutoConfig`. "forwards" = a real forward call
returns. "matches upstream" = numeric comparison done in §1's method; "n/a" where forward itself
did not succeed, so there is nothing to diff yet — GOLDEN.md's whole point, restated by the
brief: a model that runs and disagrees is worse than one that refuses, so this column is never
guessed.

| model (task) | loads | forwards | matches upstream | refusal (exact) |
|---|---|---|---|---|
| `bert` (`BertForMaskedLM`, `AutoModelForMaskedLM`) | yes | yes | **yes** — 28/28 weights identical, max weight diff 1.9e-06 (accumulation noise), output max abs diff 8.9e-08 (float32 eps 1.19e-07) | — |
| `roberta` (`RobertaForMaskedLM`) | yes | yes | **yes** — 28/28 weights identical, max abs diff 7.5e-08 | — |
| `albert` (`AlbertForMaskedLM`) | yes | yes | **yes** — 30/30 weights identical, max abs diff 7.5e-08 | — |
| `vit` (`ViTForImageClassification`) | yes | yes | **yes** — 24/24 weights identical, max abs diff 3.7e-08 | — |
| `clip` (`CLIPModel`, text+vision) | yes | yes | **yes** — 46/46 weights identical, max abs diff 4.5e-08 | — |
| `wav2vec2` (`Wav2Vec2ForCTC`) | yes | yes | **yes, with a caveat** — 32/32 weights identical, max abs diff 2.3e-07 (still float32-eps scale); the *relative*-diff figure (2.3e-02) looked alarming in isolation and is a near-zero-denominator artefact, not drift — flagged rather than silently dropped, per the brief's own "a model that runs and produces different numbers is worse than one that refuses" |
| `qwen2_moe` (`Qwen2MoeForCausalLM`, small MoE) | yes | yes | **yes** — 19/19 weights identical, max abs diff 4.5e-08 | — |
| `whisper` (`WhisperForConditionalGeneration`, forward only) | yes | yes | **yes** — 51/51 weights identical, max abs diff 1.2e-07 | — |
| `whisper` (`.generate()`) | yes | forward yes, **generate no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.as_tensor(...)` — `generation_whisper.py:1608`, `_retrieve_init_tokens` |
| `t5` (`T5ForConditionalGeneration`) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.full_like(...)` — `modeling_t5.py:258`, `_relative_position_bucket` |
| `switch_transformers` (MoE-T5, `AutoModelForSeq2SeqLM`) | yes | **no** | n/a | same as `t5`: `torch.full_like`, same shared code |
| `bart` (`BartForConditionalGeneration`) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: TensorBase.new_zeros` — `modeling_bart.py:62`, `shift_tokens_right` |
| `mbart` (`MBartForConditionalGeneration`) | yes | **no** | n/a | `NotImplementedError: aten op not implemented in torch._C shim: aten.squeeze.default` — `modeling_mbart.py:77`, `shift_tokens_right` (different shift-right code than `bart`'s) |
| `pegasus` (`PegasusForConditionalGeneration`) | **no** (construction) | n/a | n/a | `NotImplementedError: torch._C shim: TensorBase(...) ... building from data, torch.Tensor(ndarray), is a third form and is not implemented` — `modeling_pegasus.py:91`, `PegasusSinusoidalPositionalEmbedding.create_weight` |
| `resnet` (`ResNetForImageClassification`) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.batch_norm(...)` — stem `BatchNorm2d` after a **successful** 2-D `conv2d` |
| `mobilenet_v2` (`MobileNetV2ForImageClassification`) | yes | **no** | n/a | same as `resnet`: `torch.batch_norm` |
| `convnext` (`ConvNextForImageClassification`) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.linspace(...)` — `modeling_convnext.py:219`, stochastic-depth rate schedule (construction time, not forward) |
| `swin` (`SwinForImageClassification`) | **no** (construction) | n/a | n/a | `NotImplementedError: not implemented in torch._C shim: torch.meshgrid(...)` — `modeling_swin.py:354`, `_create_relative_position_index` |
| `sentence_embed` (`bert` + mean-pool + `F.normalize`, the sentence-transformers shape) | yes | forward yes, **normalize no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch._C._linalg.linalg_vector_norm` — `torch/nn/functional.py:6100`, `normalize` |
| `TorchnativeConfig("q8_0")` load (toy 2-layer `llama`, self-built checkpoint) | yes | yes | not re-diffed this round (docs/HFQUANT.md already established bit-identical vs. `quantize_`) | — dense and quantised loads both forward under the shim |

**7 of 18 architectures forward and match upstream** (`bert`, `roberta`, `albert`, `vit`, `clip`,
`wav2vec2`, `qwen2_moe`); **1 forwards but its `.generate()` does not** (`whisper`); **10 do not
forward** (`t5`, `switch_transformers`, `bart`, `mbart`, `pegasus`, `resnet`, `mobilenet_v2`,
`convnext`, `swin`, `sentence_embed`'s normalize step). The quantised-load feature path works.

**Combined with ARCH26's 26** (of which 22 forward per docs/KERNELS26.md), the widened denominator
this project has actually run is **44 architectures/tasks**, of which **29 forward** — narrower
than "26/26" or any round-specific number implies on its own, which is exactly why this round
exists.

## 3. What was not run, and why

- **Real checkpoints and real preprocessing.** Every vision/audio target used hand-built tensors
  (`torch.zeros`/`torch.randn` of the right shape) in place of `AutoImageProcessor`/
  `AutoFeatureExtractor` output. `pillow`, `sentencepiece`, `protobuf` were installed (small, pure
  dependencies, none of which touch `torch`) to keep tokenizer/config paths open, but no `soundfile`,
  `librosa`, `torchvision`, `torchaudio`, or `accelerate` was installed — real audio decoding and
  real image loading were never exercised, only the tensor-in/tensor-out model forward. A model
  that needs a working `WhisperFeatureExtractor` (log-mel spectrogram from a raw waveform, which
  wants `numpy`+`scipy`, present, but conventionally routes through `torchaudio` for resampling) was
  not checked end-to-end from an audio file.
- **`sentence-transformers` the package.** Not installed — it pulls `scikit-learn`/`scipy` and,
  more importantly, its own `torch` dependency pin, which is exactly the "transformers can replace
  torch" trap this round's brief calls out by name. `sentence_embed` above approximates the
  library's actual `SentenceTransformer.encode()` (mean-pooling + `F.normalize`, the standard
  recipe for `all-MiniLM`-class models) using plain `AutoModel`, which is architecturally identical
  to `bert` — so the *forward* half of this target adds no new evidence beyond `bert`'s own row; only
  the `normalize` tail is new information.
- **Object detection (DETR-shaped models).** Considered (would add more `batch_norm`/`conv2d`
  votes through a ResNet backbone plus new box-regression ops) and skipped for time — the toy-config
  recipe for DETR's `num_queries`/`d_model`/backbone composition is not as mechanical as the
  `AutoConfig.for_model` targets used here and would have cost disproportionate setup time for one
  more `batch_norm` vote already established by two other models.
- **`float8_e4m3fn` / other exotic dtypes.** Not attempted. The brief names this as a known hang
  case; none of this round's 19 runs touched a non-default dtype, so the per-run 120s alarm was
  never tested against a real hang, only against ordinary failures (which all resolved in
  sub-second time).
- **A real HF Hub checkpoint end-to-end** (tokenizer + real weights + generation against a known
  reference transcript/translation) — network access to `huggingface.co` was confirmed working
  (`curl`, and Python via `certifi`'s CA bundle — the venv's default `ssl` context could not verify
  TLS without it, worth flagging for anyone else scripting downloads here), and `HF_HOME` was
  pointed at `/Volumes/macMini/caches/hf` per the brief, but no download was actually made: every
  target that reached construction used a **self-built toy config**, and the one real-weights test
  (`TorchnativeConfig`) used a **self-built local checkpoint** (`save_pretrained` on upstream, 2-layer
  toy `llama`, ~100 KB) rather than a Hub download, to keep the round finishing quickly and to avoid
  a licence-gated or multi-hundred-MB checkpoint eating the external disk's 80 GB headroom
  unnecessarily. No checkpoint was skipped for being too large or gated — none was attempted.

## 4. Gates — tree unchanged, pasted in full

```
$ PYTHON=$PY sh rust/torch_c/pytests/run.sh
343 ok
SELF-TEST: PASS -- 20 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
DOCWATCH: PASS -- 257/257 evaluated marker(s) hold

$ $PY tools/golden/compare.py
SUMMARY: 7763/7763 cases passed, 0 failed, ops covered=168, pending case builders=1

$ $PY tools/golden/compare.py --self-test
SELF-TEST: PASS -- 20 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised

$ $PY rust/torch_c/pytests/verify_schemas.py
SUMMARY: 4487/4487 table entries matched upstream, 0 failed
```

All four numbers match the brief's expected baseline (343 ok, DOCWATCH 257/257, 7763/7763
ops=168) exactly. `git status --short` was empty before, during (checked between models), and
after this round — nothing in `rust/torch_c/src/`, `bootstrap.py`, `tools/golden/cases.py`, or the
vendored tree moved.

---

## The ranking, as checks

Every op named above is asserted **absent**, so the day one of them lands this document
fails and has to be re-ranked rather than quietly describing a closed gap. That is the
failure mode this file exists to break: the last demand list went stale and the queue
was fed by ad-hoc probes instead, one of which misclassified fifteen names.

<!-- DOCWATCH: op-not-implemented aten.native_batch_norm.default -->
<!-- DOCWATCH: op-not-implemented aten.full_like.default -->
<!-- DOCWATCH: op-not-implemented aten.new_zeros.default -->
<!-- DOCWATCH: op-not-implemented aten.linalg_vector_norm.default -->
<!-- DOCWATCH: op-not-implemented aten.squeeze.default -->
