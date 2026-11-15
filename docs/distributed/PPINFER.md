# PPINFER: pipeline-parallel generation of a real `transformers` model on `torchnative`

BACKEND.md §9 step 10, issue #295, SPEC S6.13. Written 2026-11-13 on `feat/dist-pp-inference` (stacked on #230
point-to-point, #246 TP inference and `develop`).

**Grade: builds.** Every file below passed `python3 -m py_compile` and nothing else; the test file's three child
sources were also compiled as Python. No suite, gate or probe ran in the round that wrote it (AGENTS.md §10: agents
do not run suites; the gate runs in CI). Nothing here is *reaches* or *agrees* until CI runs
`tests/distributed/test_ppinfer.py` (AGENTS.md §16). Read every "does" below as "is written to".

Provenance. Pipeline parallelism as the uneven-device tool is D11 and loading only a rank's layer subset is D12, both
**decided (leader), 2026-11-08**, after the owner wrote "나머지는 너의 결정에 따를게". The owner's goal, verbatim
(2026-11-08): "분산 추론을 torch.distributed와 호환되게 구현하고, 한 모델을 여러 디바이스에서 추론 혹은 학습 가능하게
구성해줘." Route B (a thin stage runner rather than `torch.distributed.pipelining`) was an agent proposal in
BACKEND.md §6.2 and this round took it. **Everything else in this document is an agent proposal, not confirmed**:
the module and class names, the `split` argument, which stage drives `generate`, the message layout and tags, the
loader's name and arguments, the traffic record, and the list of refusals.

## 0. Answer first

| Question | Answer (read, not run) |
|---|---|
| What was built | `torchnative.distributed.pipeline.PipelineStage` (one rank's stage and its part of `generate`), `split_layers` (validates a split, takes a placement callable), `torchnative.transformers.load_layer_subset` (the D12 loader option), and `record_traffic` on the `torchnative` backend (what crossed ranks, read from the backend) |
| Can the installed loader skip non-local layers without `device_map`? | **Yes, if the model it builds has only those layers** (§2). It builds on `meta`, opens safetensors lazily, and materialises only keys the model's own `state_dict()` names. It has no option to build such a model; the loader option supplies exactly that and nothing else |
| Whose `generate` runs | transformers' own, on the last stage, with its `forward` wrapped for the call (§1). Token choice, logits processors, stopping and finished-row padding are transformers' code |
| Shim changes | **None in Rust or `bootstrap.py`.** One Python addition to `torchnative/python/torchnative/distributed/backend/__init__.py`: `record_traffic()` (§4) |
| Tests | `tests/distributed/test_ppinfer.py`, 12 tests, registered `BOTH` in `.github/scripts/gate_suites.py` (§5) |

## 1. The runner

```python
# agent proposal, not confirmed: the API
import torch.distributed as dist
import torchnative.distributed                      # registers backend="torchnative"
from torchnative.distributed.pipeline import PipelineStage

dist.init_process_group("torchnative")             # one rank per stage
stage = PipelineStage.from_pretrained(path, split=[3, 1], dtype=torch.float32)
tokens = stage.generate(input_ids, max_new_tokens=16)   # the same call on every rank
```

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/pipeline.py PipelineStage present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/pipeline.py split_layers present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/pipeline.py PipelineDesync present -->

**The split.** `split` is a list of per-stage layer counts (`[3, 1]`, uneven allowed, D11), a list of `(start, stop)`
ranges that tile `0..num_layers` in order, or a callable `split(num_layers, world_size)` returning either. The
callable is the hook for the capability-weighted placement of issue #296. One rank runs one stage, and stage *i* is
rank *i* of the default group.

**What each stage holds** (the loader, §2). Stage 0 keeps `embed_tokens`; the last stage keeps the final `norm` and
`lm_head`; every stage keeps only its own decoder layers. A stage that is not last has `norm` replaced by
`nn.Identity()`, because the base model's forward calls `self.norm` unconditionally and the next stage needs the
un-normed hidden state. With tied embeddings the first and last stages each keep the one shared tensor under both
names.

**Who drives.** The last stage calls transformers' `generate(input_ids, attention_mask=ones, past_key_values=its
cache, do_sample=False, num_beams=1, ...)`. For the duration of that call its `forward` is an instance attribute
(`functools.wraps` of the real bound method, so `inspect.signature` still sees `logits_to_keep` and
`position_ids`) that, each step:

1. sends the step's token ids to stage 0;
2. receives the hidden state from the stage before it;
3. calls the real `forward(input_ids=None, inputs_embeds=hidden, **the rest)`.

The other stages serve in a loop: receive (stage 0: token ids from the last stage; any other: a hidden state from
the stage before), run the real base model's `forward` over their own layers with their own cache, and send the
hidden state on. When `generate` returns, the last stage sends the final sequences down the chain, so every rank
returns the same tensor.

**Positions and masks** are not sent. Every stage rebuilds them from `(batch, past + length)` the way
`_prepare_position_ids_for_generation` builds them for an all-ones mask (`mask.cumsum(-1) - 1`, sliced to the
step). That is exact for unpadded input and wrong for padded input, which is why a padded mask refuses.

**Why this is bit-for-bit, by reading.** Each layer receives the same bytes it would receive in one process (the
wire carries raw element bytes, P2P.md §2), with the same position ids, the same mask (built by the same
`create_causal_mask` from the same lengths) and the same rotary embedding (recomputed from the same positions). The
last stage computes logits with transformers' own `logits_to_keep=1`, as the single-process `generate` does. So the
claim is not "close": it is equality, and the test asserts equality.

## 2. The loader option (D12) and what the installed loader does

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/transformers/layer_subset.py load_layer_subset present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/transformers/layer_subset.py StageLayers present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/transformers/__init__.py load_layer_subset present -->

Read off transformers 5.15.1 in `.caches/spike-venv` (read-only):

1. `from_pretrained` builds the model inside `get_init_context`, which includes `torch.device("meta")`
   (`modeling_utils.py:3796`), so constructing layers that are then dropped costs no memory.
2. For safetensors it opens each file with `safe_open(..., framework="pt")` and stores `get_slice(k)` for every key
   (`modeling_utils.py:4470-4472`). A slice reads nothing.
3. `convert_and_load_state_dict_in_model` (`core_model_loading.py:1608-1700`) materialises a slice only when its
   renamed key is in the model's own meta `state_dict()`; any other key goes to `unexpected_keys` and is never
   sliced. The only read is `_materialize_copy`'s `tensor[...]` (`core_model_loading.py:1217-1222`).
4. Missing keys are allocated with `empty_like` and initialised (`_move_missing_keys_from_meta_to_device`,
   `_initialize_missing_keys`); non-persistent buffers (the rotary `inv_freq`) are recomputed the same way.

So the installed loader skips non-local layers **without `device_map`**, provided the model it constructs has only
the local layers. `load_layer_subset(path, layers=(start, stop), **from_pretrained_kwargs)` supplies that one piece:
it calls the real `from_pretrained` on a subclass of the real model class whose `__init__` runs the real `__init__`
(on meta) and then prunes, and afterwards sets the object's class back to the real class, so what comes back is
`Qwen3ForCausalLM`. Every byte is read by transformers' own loader.

**Key names.** The kept layers live in `StageLayers`, an `nn.ModuleList` that registers its children under their
**global** index, so `state_dict()` names (`model.layers.2.mlp.up_proj.weight`) equal the checkpoint's and nothing is
renamed. Non-local checkpoint keys are added to the instance's `_keys_to_ignore_on_load_unexpected`, so the load
report does not list them as surprises.

**The cache.** The kept layers' `layer_idx` attributes are renumbered to local positions, `config.num_hidden_layers`
becomes the local count and `config.layer_types` is sliced. `DynamicCache(config=model.config)` then has one entry per
local layer, and `cache.get_seq_length()` (which the mask builder and the position fallback read at index 0) is this
stage's own length. Without the renumbering a stage owning layers 2 and 3 would read layer 0 of its cache, which it
never writes, and build the wrong mask. The global indices are on `model._torchnative_stage["global_layer_idx"]`.

**A missing key refuses.** The loader takes transformers' loading info and refuses by name if any key the subset
needs is missing or mismatched, rather than letting transformers initialise it randomly and say so only in a log
line.

## 3. Refused by name (AGENTS.md §18)

| Refusal | Raised by | Why |
|---|---|---|
| `do_sample=True` | `PipelineStage.generate` | not built; only greedy tokens are checked against the single-process run |
| `num_beams > 1` | same | beam reordering would have to reorder every stage's cache in step |
| any `generate` keyword outside `_ALLOWED` (`streamer=`, `assistant_model=`, `output_scores=`, ...) | same | not built |
| `use_cache=False` | same | every stage keeps its own cache; that is the point of the split |
| a padded `attention_mask` | same (last stage) | positions and masks are rebuilt from lengths alone |
| `cache_implementation` other than dynamic, `num_return_sequences > 1` in the generation config | same | not built |
| one stage | `split_layers` | a pipeline of one stage is the model itself |
| an empty stage, a gap, an overlap, a reordering, a range outside the model | `split_layers`, `load_layer_subset` | a stage is one contiguous non-empty range |
| a stage count that is not the world size | `split_layers` | one rank runs one stage |
| a group other than the default group | `PipelineStage` | `new_group` is BACKEND.md step 8 |
| a model whose loaded range or embed/head flags are not what the split assigns its rank | `PipelineStage` | a misassembled pipeline would compute garbage |
| no `model.layers` `nn.ModuleList`, no `embed_tokens`/`norm`/`lm_head`, no integer `layer_idx` to renumber | `load_layer_subset` | not a model shape this loader understands |
| layer types other than full and sliding attention; `num_kv_shared_layers` | same | their cache is not one key/value pair per layer |
| encoder-decoder models | same | decoder-only causal LMs only |
| `device_map`, `tp_plan`, `distributed_config`, `device_mesh`, `state_dict`, `gguf_file`, quantisation, `export` | same | none was built or tested with a subset |
| a checkpoint missing a key the subset needs | same | transformers would initialise it randomly |

A stage whose cache length disagrees with the position a message carries raises `PipelineDesync` naming both stages
and both lengths, rather than computing on a cache that drifted.

## 4. What crosses a boundary, and how the test reads it

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/__init__.py record_traffic present -->

Every message is preceded by a 4-element int64 header `[kind, batch, length, past]` on tag 7301. Per step:

| Edge | Payload | Tag |
|---|---|---|
| stage s to s+1 | hidden state `(batch, length, hidden_size)`, the model's dtype | 7302 |
| last stage to stage 0 | the step's token ids `(batch, length)`, int64 | 7303 |
| once at the end, down the chain | the final sequences `(batch, prompt + new)`, int64 | 7304 |

Nothing vocabulary-sized crosses. `length` is the prompt length at the first step and 1 after.

`ProcessGroupTorchnative.record_traffic()` (and the module function `record_traffic(group=None)`) starts appending one
`(direction, peer, tag, dtype, shape)` tuple per `send` and per posted `recv` and returns the live list. It is off
until asked. The test compares each rank's list with the exact ordered list the protocol implies, so a runner that
shipped logits, sent a step twice or skipped a header fails there, by the backend's account rather than the
runner's.

## 5. The tests, and what would make each fail

The model: Qwen3 (the owner's test family, 2026-10-03), `vocab 96, hidden 48, intermediate 60, 4 layers, 12 heads over
6 kv heads, head_dim 8`, untied, written once by an upstream child with `torch.manual_seed(0)` and every parameter from
`N(0, 0.3)`, saved under `.scratch/ppinfer-*` inside the repository; plus the same with `tie_word_embeddings=True`.
The prompt is a batch of two unpadded rows of 8 tokens; `max_new_tokens = min_new_tokens = 6`, so every run makes
exactly six tokens whatever the random weights favour.

Every world runs twice: on the shim over `backend="torchnative"`, and on upstream torch over gloo with the same runner
(appended to `sys.path` after upstream `torch` is imported). The first compares with the shim's own single-process
`generate`, the second with upstream's.

<!-- DOCWATCH: symbol-in-file tests/distributed/test_ppinfer.py test_two_stages_generate_the_single_process_tokens_bit_for_bit present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_ppinfer.py test_three_stages_generate_the_single_process_tokens_bit_for_bit present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_ppinfer.py test_only_hidden_states_and_token_ids_cross_ranks_as_read_from_the_backend present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_ppinfer.py test_load_layer_subset_reads_and_allocates_only_the_requested_layers present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_ppinfer.py test_sending_a_stale_hidden_state_changes_the_tokens present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_ppinfer.py test_a_stage_that_drops_its_cache_updates_is_caught_twice present -->

| Test | Asserts | Fails if |
|---|---|---|
| `test_two_stages_generate_the_single_process_tokens_bit_for_bit` | split 2:2: every rank's sequences equal the single-process sequences; six steps per rank | any boundary, position, mask or cache detail differs |
| `test_three_stages_generate_the_single_process_tokens_bit_for_bit` | split 1:2:1, given as ranges | a middle stage is wrong; the chain's stop does not reach every rank |
| `test_an_uneven_three_to_one_split_generates_the_same_tokens` | split 3:1 (D11) | an uneven split is mishandled |
| `test_a_tied_embedding_checkpoint_split_one_to_three_generates_the_same_tokens` | tied checkpoint, split 1:3 | the shared embedding is lost on either end |
| `test_each_stage_cache_holds_only_its_own_layers` | entry count equals the stage's layers; each at `prompt + new - 1`; key shape `(2, 6, 13, 8)`; local `layer_idx` | a stage caches other stages' layers, or a cache did not grow |
| `test_only_hidden_states_and_token_ids_cross_ranks_as_read_from_the_backend` | shim only: the backend's record equals the exact protocol list; refused calls add nothing | anything else crosses |
| `test_each_pipeline_rank_reads_only_its_layers_from_the_checkpoint` | per rank: materialised keys equal exactly the subset's; allocated parameters equal them; real class; no `meta` parameter | a rank reads or allocates other layers |
| `test_load_layer_subset_reads_and_allocates_only_the_requested_layers` | single-process, eight ranges (untied and tied); values equal the file's bytes; the instrument recorded every key on a full load | the loader reads too much, loads the wrong layer into a slot, or the instrument is blind |
| `test_what_is_not_built_refuses_by_name` | twelve loader and split refusals; four `generate` refusals on every rank; the placement callable is honoured | a refusal is weakened or a message stops naming itself |
| `test_sending_a_stale_hidden_state_changes_the_tokens` | sabotage 1 | the token comparison could not see a wrong boundary |
| `test_a_stage_that_drops_its_cache_updates_is_caught_twice` | sabotage 2, both ways | see below |
| `test_every_child_ran_the_torch_it_claims` | shim children loaded the shim; upstream children upstream torch, with only `torchnative` from the tree | a child silently imported the other torch (AGENTS.md §15.2) |

**The reads are measured, not inferred.** The child wraps the `safe_open` that `transformers.modeling_utils` calls and
logs every slice that is materialised (`tensor[...]`). The read set must equal the subset's keys exactly. The
instrument is checked in the same child: a full `from_pretrained` under it must record every key in the file.

**The nullifications** (AGENTS.md §17.5), run inside the children on both sides:

1. *Stale hidden state.* Stage 0's `_send_hidden` is replaced so that, from the second decode step on, it sends the
   previous decode step's hidden state (same shape). The tokens must differ from the single-process run, and the
   prompt plus the first two new tokens, which precede the first stale send, must not.
   `test_sending_a_stale_hidden_state_changes_the_tokens` holds it.
2. *A stage drops its cache updates.* A serving stage's cache returns each step's keys and stores nothing. With the
   runner's position check in place, that stage must raise `PipelineDesync` naming `stage 1 (layers (1, 3))` (a
   separate world of three on the shim, since the error ends that world). With the check also disabled, the tokens
   must differ. `test_a_stage_that_drops_its_cache_updates_is_caught_twice` holds both, so the check and the token
   comparison each catch it on their own.

What these tests cannot see (AGENTS.md §17.4): padded batches, sampling, beam search, dtypes other than float32, a
checkpoint sharded over several files, a model family other than Qwen3, memory actually saved (reads and allocated
parameters are asserted, resident memory is not), speed, overlap between stages (the runner is strictly sequential
per token), and any device other than CPU.

## 6. Not done, and open

1. **Nothing ran.** CI's first run is the measurement. The first places it could fail, by reading: `str(dtype)` on the
   shim printing something other than `torch.int64`/`torch.float32` (the traffic test would name it); a transformers
   path that reads a slice other than through `tensor[...]` (the instrument test would name it); `del model.forward`
   on the shim's `nn.Module`.
2. **Device ranks.** `_isend` stages a non-CPU tensor through `.cpu()` and receives with `.to(stage.device)`, so a
   stage on `vulkan`, `mps` or `npu` is shaped for; it is not tested and not claimed. BACKEND.md step 10's "one rank on
   vulkan/npu where the device exists" is a device test for the coordinating session.
3. **No overlap.** One token at a time traverses the chain; micro-batching or overlapping prefill is not built.
4. **Sub-groups.** A pipeline over part of the world waits for `new_group` (BACKEND.md step 8).
5. **Placement.** `split` accepts the callable that issue #296 will provide; no placement is computed here.
