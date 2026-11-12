# Fusion over a captured decode step: front end, dynamic length, in-place KV, handoff

Issue #68. Written 2026-11-13 on `feat/fusion-capture` (from `develop` at 80f23ce), reading PR #208
(`feat/dynamo-guards-manager` at 5d836f5) for `torchnative.compile`.

**Every design choice in this document is an agent proposal, not confirmed by the owner.** Where a
line rests on the owner's words, it quotes them with the date and the place they are recorded.
Everything else is marked as a proposal, and §8 lists the questions the owner has to answer before
an implementation follows.

**Nothing in this document was run.** The round that wrote it was not allowed to build, run a
model or run a suite (AGENTS.md §10). Upstream behaviour is read from source, with the file and
line given; the `torchnative.compile` behaviour is read from #208's code. The capture of one
decode step is written as a suite (`tests/export/test_fusion_capture.py`, §7) whose expected
values are derived by reading and are **not yet observed**.

---

## 0. At a glance

| Question | Proposal (agent, not confirmed) | Section |
|---|---|---|
| Which front end captures the graph? | **Dynamo, through `torchnative.compile` (#208).** `capture.rs` stays the NPU delegates' substrate; `torch.export` stays the second way | §2 |
| At which level is the graph partitioned? | **Stage 1: the Dynamo graph as the backend receives it** (torch-level ops, mutations in place). Stage 2: the functionalized ATen graph, once AOTAutograd reaches on the shim | §2.3 |
| Dynamic sequence length | **Decode is served by one static-shaped graph (StaticCache shape). Kernels are symbolic in the length with a recorded range; buckets only where a kernel needs static tiling** | §3 |
| In-place KV update | **A mutation is a region barrier and runs through `_aten_dispatch` in program order.** Functionalization plus re-inplacing is stage 2 | §4 |
| Region rule | Elementwise, reduction and view chains grow a region; GEMM, gather, mutation and unknown ops end it; SDPA is its own region. Implemented as `torchnative/export/regions.py` | §5 |
| Handoff to TypedPython | **The region as its own FX graph** (`regions.extract`), lowered by TypedPython's front end into its IR. A `@kernel` source per region is the alternative | §5.3 |
| Run-time dispatch | **One fused op per region, entering through `_aten_dispatch` under its own name.** No bypass | §5.4 |
| What refuses | Ops without a rule and random ops are listed by name; a region missing from the bundle, a size outside its range and a bundle version mismatch refuse by name | §6 |
| Decode step captured, regions listed | Written as a suite; **expected values not observed**; CI and #208 decide | §7 |
| Open owner decisions | Nine, each with options | §8 |

---

## 1. What is asked, and what the owner has already said

The issue asks for the mobile replacement of `torch.compile`'s main gain, fusion: capture a model's
forward, hand fusable regions to TypedPython's kernel compiler, bundle the kernels, and dispatch them
at run time. Its done-when is a design note that says how dynamic sequence length and in-place KV
updates are captured, compared with Inductor, AOTInductor and ExecuTorch, plus one decode step
captured with its fusable regions listed.

The constraints below are the owner's, and each one shapes a choice later in this document.

| Constraint | Source | What it rules out |
|---|---|---|
| "model.to를 하기 위해서 특정 디바이스들은 torch.compile이 트리거 되는 걸로 정의하자는거야" | owner, 2026-11-08, `docs/INTENT.md` §5 | a fusion path that `.to()` does not reach through `torch.compile` |
| "입력 크기가 바뀌면 재컴파일을 해야 한다는 측면에서도 논리적으로 그게 맞는거 같아" | owner, 2026-11-08, `docs/INTENT.md` §5 | nothing; it states that a new shape recompiles (§3 asks whether that stays the rule for the length) |
| "파이썬 분기 처리 불가 >> 내가 봤을 때 이건 좋은 일이 아니야. 이럼 onnx같이 된다는거잖아" | owner, 2026-11-08, `docs/INTENT.md` §5 | a capture that records one traced path, so `capture.rs` alone as the front end |
| "torch.export도 가능하게는 해주고" | owner, 2026-11-08, `docs/INTENT.md` §5 | `torch.export` as the path `.to()` depends on (it is the second way) |
| "abi3 문제는 내가 동의한거 맞아" | owner, 2026-11-07, `docs/INTENT.md` §7 | PEP 523, hence upstream `torch.compile` as is |
| Kernels compile at build time; iOS forbids run-time code generation | maintainer decision 2026-10-04, python-multiplatform `typedpython.md` §5 and §5.1 | Triton and Inductor's run-time codegen on the device |
| Kernels land as torch ops or `kernels` bundle entries; the registration path is still to be agreed with torchnative | maintainer decision 2026-10-04, `typedpython.md` §8 item 4 | a separate kernel runtime |
| No offline or CI `.pte` pipeline for the Android NPU | owner, 2026-11-07, `docs/INTENT.md` §5 | it is about the NPU back end, not GPU fusion; §8 Q1 asks whether it extends here |
| Not an ahead-of-time export pipeline as the user-facing model | `docs/INTENT.md` §7 | the user ever holding a fused artefact |

`typedpython.md` is python-multiplatform's
<https://github.com/thisisthepy/PythonMultiplatform/blob/develop/docs/design/typedpython.md>.

The issue's own wording ("build-time fusion") is quoted as the issue's text. The issue was opened
from the account `b-re-w`; this round cannot tell whether the owner wrote it, so "build-time" is not
cited as an owner decision. §8 Q1 asks it directly.

---

## 2. The front end

### 2.1 Three ways to get a graph, and what each refuses today

| | `capture.rs` (`_capture_begin`/`_capture_end`) | `torchnative.compile` (#208) | `torch.export` |
|---|---|---|---|
| Where it intercepts | the one door, `_aten_dispatch` | Dynamo's bytecode translator (`ConvertFrame`), entered without PEP 523 | Dynamo plus FakeTensor plus functionalization |
| What it records | ATen ops of one executed path | torch-level FX graphs, one per graph break, with guards | one functional ATen graph |
| Python control flow | **refused**: `_local_scalar_dense` poisons the trace (`docs/graph/CAPTURE.md` §4) | **kept**: guards plus graph breaks plus resume functions | refused (no graph breaks) |
| In-place mutation | **refused** (`is_mutating`, `capture.rs`; `docs/graph/CAPTURE.md` §4) | **kept** as nodes (`index_copy_`, `add_`) in the graph | functionalized, lifted as mutated buffers |
| Dynamic shapes | **refused**: guards fix every shape (`docs/graph/CAPTURE.md` §4) | **static only**: #208 sets `automatic_dynamic_shapes = False` and `assume_static_by_default = True` (`torchnative/compile/__init__.py`, `_ensure_shim_ready`); a new shape recompiles | `Dim(min, max)` |
| Random | refused (`multinomial`, `randint`) | traced by Dynamo | traced |
| Reaches real models today | yes, for straight-line regions | 2-layer toy LM and `.to(device.npu)` in #208's suite (`tests/export/test_compile_nopep523.py`); not yet on a `transformers` model in any suite | **0 of 40** architectures (`docs/graph/EXPORT6.md` §1.2) |

### 2.2 Proposal: Dynamo through `torchnative.compile`

**Proposal (agent, not confirmed): the fusion front end is a `torchnative.compile` backend.** The
backend Dynamo calls with `(gm, example_inputs)` partitions `gm` (§5), replaces each region with one
fused call (§5.4) and returns the rewritten module's `forward`.

Why this one:

- `.to(device)` already goes there (owner, 2026-11-08). A second capture path would be a second
  front end, and the accelerator rule is one fixed front end (AGENTS.md §20).
- It keeps Python control flow. The owner rejected the one-path capture in those words, and an LLM
  decode step has branches (`if past_key_values is not None`, mask selection, cache lazy init).
- It records the KV update as it is (§4), where `capture.rs` refuses it and `torch.export` does not
  reach the model at all.

Why not the others:

- **`capture.rs`** refuses exactly what a decode step does: mutation, a changing length, and branches.
  Lifting those refusals would rebuild Dynamo below the door. It stays the substrate the NPU
  delegates consume (S5.1), and it keeps a property fusion needs: the fused op of §5.4 goes through
  the door, so `capture.rs` still records it.
- **`torch.export`** is 0 of 40, has no graph breaks, and the owner named it the second way, never the
  path `.to()` depends on.

### 2.3 Two stages of the graph

The graph Dynamo hands a backend is **torch-level and not functionalized**: `call_function` on
`torch.rsqrt`, `operator.add`, `F.silu`, `call_method` on `"view"`, `"index_copy_"`. Upstream's
Inductor never partitions this graph; it first runs AOTAutograd, which functionalizes and decomposes
to ATen.

- **Stage 1 (proposal): partition the Dynamo graph directly.** It is what #208 delivers today. The
  partitioner reads op names, and the torch-level name is the ATen base name in almost every case
  (`regions.op_name` maps `aten.add.Tensor`, `operator.add`, `torch.add` and `x.add` to `add`).
  Mutations stay nodes and become barriers (§4.3).
- **Stage 2 (proposal): partition the functionalized ATen graph** once AOTAutograd runs on the shim.
  Whether it does is **unverified**: the shim fills
  `torch._C._functionalization_reapply_views_tls`, but this round did not check the rest of the
  functionalization surface `FunctionalTensorMode` needs.

The partitioner is written to read both name spaces, so moving from stage 1 to stage 2 changes the
input, not the rule.

### 2.4 What `torchnative.compile` does today that fusion must not inherit

Read from #208 at 5d836f5, not run:

- **A Dynamo exception runs the model eagerly.** `attach_to_module` catches exceptions whose type name
  contains `Dynamo`, `Unsupported` or `Internal`, increments `stats["fail"]` and calls the original
  `forward`. A counter nobody reads is a silent fallback in the sense of S1.4.
- **The recompile limit runs the frame eagerly.** Upstream logs a warning at `recompile_limit`
  (8, `torch/_dynamo/config.py:121`) and marks the frame `RUN_ONLY`
  (`torch/_dynamo/convert_frame.py:1986-2036`). With static shapes and a `DynamicCache`, the past
  length changes every token (§3.2), so the ninth decode step of a frame would run eager with only a
  log line. **Unverified on the shim**; the decode suite asserts the per-step recompile (§7).
- The backend names `"eager"`, `"inductor"` and `"npu"` all map to `gm.forward`.

These are #208's to decide. They are listed here because a fused graph inherits them: a fused
backend behind a silent eager fallback would report fusion that did not happen.

---

## 3. Dynamic sequence length

### 3.1 How upstream does it

| | How a length becomes dynamic | What the artefact is | What happens outside the range |
|---|---|---|---|
| **Dynamo** | `assume_static_by_default = True` (`torch/_dynamo/config.py:175`): every dim starts static. With `automatic_dynamic_shapes` (`:185`, on by default upstream, **off in #208**), a shape guard failure recompiles once with the wobbled dims as symbols. `mark_dynamic(t, dim, min=, max=)` (`torch/_dynamo/decorators.py:1236`) marks one ahead of time with a range. Guards then test `SymInt` relations, and 0 and 1 are specialized | a guarded cache of graphs per code object | a guard fails and the frame recompiles, up to `recompile_limit`, then runs eager (§2.4) |
| **Inductor** | takes Dynamo's symbols; generated kernels take the sizes as arguments | kernels generated at run time (JIT) | Dynamo's guards |
| **AOTInductor** | `torch.export` with `Dim(name, min, max)`; one compiled library whose kernels are symbolic in those dims | **one artefact for the whole range** | the generated wrapper checks each dim against its range and throws (`torch/_inductor/codegen/cpp_wrapper_cpu.py:740-763`), but **only when `AOTI_RUNTIME_CHECK_INPUTS` is set** (`:822-834`); by default nothing is checked |
| **ExecuTorch** | `torch.export` with an upper bound; `ConstraintBasedSymShapeEvalPass` turns symbolic shapes into concrete upper bounds "so we can properly do memory planning" (`exir/passes/sym_shape_eval_pass.py:257`, ExecuTorch 1.4.1 in `.caches/qnn-venv`) | one program; buffers planned at the upper bound; tensors resized within it at run time | refused at run time by the resize check (from upstream documentation, **not read in this round**) |

### 3.2 What a decode step actually varies

The length a decode step sees depends on the cache, not on the model:

- **`DynamicCache`**: `torch.cat([self.keys, key_states], dim=-2)` (`transformers/cache_utils.py:144`,
  5.15.1). The key and value tensors grow by one every token, so **the shape changes every step**.
  With #208's static shapes that is one compile per token until the recompile limit.
- **`StaticCache`**: keys and values are allocated at `max_cache_len` and written with
  `index_copy_(2, cache_position, key_states)` (`cache_utils.py:480`). Query length is 1 and the cache
  length is fixed, so **no shape changes between decode steps**; the position is data
  (`cumulative_length`, a tensor, advanced with `add_` at `:477`). One graph serves every step.
- **`torchnative.kernels.kv_cache.PreallocatedCache`** (#180) writes in place but returns a narrowed
  view `self._kbuf[..., :end, :]` and keeps the length in a Python int, so under Dynamo it behaves
  like `DynamicCache` for shapes (a new length per step) and like `StaticCache` for mutation. Read,
  not run.
- **Prefill** varies with the prompt length, whatever the cache.

### 3.3 What a build-time artefact must do

TypedPython compiles kernels at build time (`typedpython.md` §5.1). So at run time a captured region
can only be matched to a kernel that already exists. Two shapes are possible:

| | Bucketed | Symbolic with a range |
|---|---|---|
| Kernels | one per region per bucket | one per region |
| Run time | pad the length up to the bucket | pass the length as an argument |
| Waste | padding (up to 2x with power-of-two buckets) | none in the data; possibly slower code without static tiling |
| Fits | GEMM and attention tiles that need static sizes, autotuned per bucket (`typedpython.md` §5.7) | elementwise and reduction regions, where the length is only an outer loop bound |

**Proposal (agent, not confirmed):**

1. **Decode is served by one static-shaped graph**: the StaticCache shape, where nothing about the
   shape moves. That needs no symbolic shapes at all, and it is how upstream's own `generate`
   compiles (`cache_implementation="static"`). Whether compiled devices should switch the cache is
   §8 Q3.
2. **Region kernels are symbolic in the length, with a recorded range.** In the regions a decoder
   produces (§5, §7) the length is an outer loop bound: RMSNorm reduces over the hidden or head dim,
   the MLP is elementwise, rotary is elementwise. Only attention reads the length as a reduction
   extent, and it takes `kv_len` as an argument.
3. **Buckets only where a kernel needs static tiling** (attention and GEMM tiles if the autotuner
   asks for them), recorded in the bundle beside the range.
4. On the Dynamo side, prefill needs the length dimension symbolic (`mark_dynamic` on the sequence
   dim, or automatic dynamic) rather than one compile per prompt length. #208 forces both off today,
   and the owner's statement on recompiling (§1) is the reason it may stay that way; §8 Q4 asks.

The bundle records each region's range. A length outside it refuses by name (§6), unlike
AOTInductor's default, which does not check.

---

## 4. In-place KV updates

### 4.1 What upstream does with a mutation

- **AOTAutograd functionalization.** A mutation of an input becomes a functional op whose result is
  returned as an extra output; a run-time epilogue writes it back with `copy_`
  (`torch/_functorch/_aot_autograd/runtime_wrappers.py:164`, the write-back at `:716-723`).
  `index_copy_` decomposes through `_index_copy` (`torch/_decomp/decompositions.py:3424`) into an
  `index_put` on a clone.
- **Inductor** asks for the opposite placement: `keep_inference_input_mutations=True`
  (`torch/_inductor/compile_fx.py:3067`) keeps the write-back inside the graph as `aten.copy_` onto the
  placeholder. Its re-inplacing pass then turns `index_put` back into `index_put_`
  (`torch/_inductor/fx_passes/reinplace.py:373-374`) when no later node can observe the old value and a
  `copy_` in the epilogue says the caller wants the mutation (`:460-476`). The result is a store
  straight into the cache buffer, with no clone.
- **ExecuTorch.** Its llama example keeps the cache as registered buffers and writes them with
  `index_copy_` or indexed assignment (`examples/models/llama/attention.py:71-111`). `torch.export`
  lifts those as mutated buffers, and `insert_write_back_for_buffers_pass` inserts the `copy_` that
  reflects them (`exir/passes/insert_write_back_for_buffers_pass.py`, `_insert_copy`). Memory planning
  then gives the buffer one allocation (from upstream documentation, **not read in this round**).
- `transformers` marks `StaticCache` tensors with `torch._dynamo.mark_static_address`
  (`cache_utils.py:449`) so their address is constant across calls, which is what CUDA graphs need.

### 4.2 What the cache choice means for capture

| | `DynamicCache` | `StaticCache` |
|---|---|---|
| KV update | `torch.cat`: functional, a new tensor | `index_copy_`: in place |
| In the Dynamo graph | a `cat` node, fusable with the rotary that feeds it | an `index_copy_` node per K and V per layer, plus `add_` on `cumulative_length` per layer |
| Shapes between steps | change every token | fixed |
| Compiles (static shapes) | one per token, then eager after the limit (§2.4) | one for all steps |
| Bytes moved per step | the whole cache, per layer (O(n) per token, O(n²) per generation) | one row per layer |

Fusing the `cat` into the K region saves a launch, not the copy. **The cache, not the capture, decides
whether decode is cheap.** That is the reason §3.3 proposes the StaticCache shape for compiled
devices.

### 4.3 Proposal: a mutation is a barrier (stage 1)

**Proposal (agent, not confirmed):** in the stage 1 graph a mutation node is not fused. It stays
where it is and runs through `_aten_dispatch` in program order, and **it closes every open region**,
not only the ones it reads.

The reason it must close every region: the Dynamo graph has **no edge** from
`index_copy_(cache, ...)` to a later read of `cache`; only the node order says the read comes after
the write. A region that straddled the write would compute some members on the wrong side of it.
`regions.py` implements the barrier, and `test_partition_keeps_kv_writes_outside_every_region`
checks a region that reads the cache before the write and one after it, with no edge between.

What stage 1 costs: the K rotary computes into a temporary and `index_copy_` copies one row into the
cache. That is one extra row of traffic per layer per step, not the cache.

Stage 2, after functionalization reaches on the shim, can do what Inductor does: the region ends in a
store into the cache slot (the re-inplaced `index_put_`), and the temporary disappears.

---

## 5. The region rule and the handoff

### 5.1 The rule

Implemented in `torchnative/python/torchnative/export/regions.py` (pure Python over `torch.fx`, no
new dependency). Every call node falls into exactly one category by its base op name:

| Category | Examples | In a region? | Why |
|---|---|---|---|
| elementwise | `add`, `mul`, `rsqrt`, `silu`, `to`, `where`, `cat`, `arange` | yes | one output element per input element (or an index computation): the fused kernel's body |
| reduction | `mean`, `sum`, `amax`, `softmax`, `layer_norm` | yes | a reduction plus its elementwise prologue and epilogue is one kernel (RMSNorm, softmax) |
| view | `view`, `transpose`, `unsqueeze`, `getitem` by slice, `contiguous` | yes, no work | an index map; it costs nothing inside a kernel |
| attention | `scaled_dot_product_attention` and its ATen spellings | its own region | it is a tiled kernel of its own (flash), not a chain |
| extern | `linear`, `matmul`, `embedding`, gather, allocation | no, it ends the regions it reads | a library kernel; TypedPython's tile GEMM is a separate kernel (`typedpython.md` §5.2) |
| mutation | `index_copy_`, `add_`, `setitem`, `iadd` | no, it ends **every** region | §4.3 |
| ignored | `size`, `_set_grad_enabled`, `_enter_autocast` | no | metadata and global state: no data |
| refused | no rule, or random (`dropout` in training, `rand`) | no, it ends every region, listed by name | §6 |

Regions grow in graph order. A fusable node joins the open region of its **most recent** producer and
closes every other open region it reads from. A node that is not fusable closes the open regions it
reads from. A closed region takes no new members. This keeps the contracted graph acyclic: an open
region has no consumer outside itself, so leaving it closes it, and no path can come back in.
`test_partition_joins_the_most_recent_producer_and_never_makes_a_cycle` checks that, and checks that
its cycle detector does fire on the naive grouping it replaces.

A region with two or more non-view ops is reported; one with a single op runs unfused; one of views
alone needs no kernel and is dropped. Every call node is accounted for in exactly one field
(`Partition.accounted()`), so nothing leaves the listing silently.

### 5.2 What the rule costs, by reading

- **It fragments rotary.** In Qwen3, `q * cos` reads the Q region and the `cos` region; the most
  recent producer is `cos.unsqueeze`, so the multiply joins the `cos` side and the Q norm region
  closes. `rotate_half` then starts its own region. Inductor fuses the whole rotary into the norm's
  kernel. The alternative rule, merging the open regions at a shared consumer, is also acyclic but
  grows layer 0's region across the model-level rotary computation and both norms. Neither was
  measured; the decode suite prints the regions so the first CI run shows the real fragmentation.
- **No horizontal fusion.** Independent chains (Q and K norms) are never merged.
- **Reductions over different axes can share a region.** The kernel compiler may have to split it.
- **SDPA takes no prologue.** Fusing rotary into attention's load is a later step.

These are engineering tunings, not owner decisions; they are listed so a reader does not mistake the
v0 rule for the final one.

### 5.3 The handoff to TypedPython

`typedpython.md` §5.4 fixes that the kernel IR is TypedPython's own and maps onto MLIR's `gpu`,
`vector` and `linalg` structure. It does not yet fix a serialised format; this round found no tier-2
kernel IR in python-multiplatform (its `typedpython/ir.py` is the tier-1 `@compiled` IR). So this
document does not invent one.

**Proposal H1 (agent, not confirmed): the handoff is the region as its own FX graph.**
`regions.extract(region)` returns a `torch.fx.Graph` whose placeholders are the region's inputs and
whose output is the values used outside it, with each node's `meta` copied (Dynamo's
`example_value` carries shape and dtype). It is upstream's own format, it is the format
`docs/graph/CAPTURE.md` §5 already maps one to one onto the capture trace, and it can be executed on
CPython as the reference the three-way comparison of `typedpython.md` §5.6 needs.
`test_partition_extract_hands_off_a_graph_that_computes_the_region` executes an extracted RMSNorm
region and requires it to equal the eager arithmetic bit for bit.

TypedPython's front end would lower each node to its IR: an elementwise op to the body of a
`linalg.generic`-like loop, a reduction to a reduction loop, a view to an index map.

Alternatives, for §8 Q5:

- **H2: a generated `@kernel` source per region**, valid Python that runs on CPython (`typedpython.md`
  §5.2). It fits TypedPython's "kernels are valid Python" principle best, but needs the tile-op
  vocabulary, which does not exist yet.
- **H3: emit TypedPython's IR directly**, once its format exists.

### 5.4 Bundling and dispatch

**Proposal (agent, not confirmed):**

- **The bundle** holds, per region: a key (the op sequence, dtypes, static dims, and which dims are
  symbolic with their ranges), the compiled kernel per target, and the IR and emitter versions
  (`typedpython.md` §5.5 already requires the versions, and the run time rejects a mismatch).
- **At run time** the `torchnative.compile` backend partitions the graph, computes each region's key,
  looks it up, and rewrites the `GraphModule`: each found region becomes one `call_function` node.
- **That node enters `_aten_dispatch`** under its own name (for example
  `torchnative_fused.<key>.default`), and the door's arm for that namespace runs the kernel on the
  device layer (`vulkan.rs`, candle-metal). **No bypass** (AGENTS.md §11): the fused call is one more
  op through the one door, so `capture.rs` records it, the dispatch counters count it, and its
  refusals carry its name.
- **`torch.library` custom ops** are the upstream-shaped alternative, and `typedpython.md` §5.6 names
  them. On the shim today, `torch.library.impl` registrations are recorded and dispatch nothing
  (`docs/design/REGISTRATIONS.md` §0: 1549 registrations at import, none dispatched), so that route
  first needs the shim to honour `torch.library`, and its kernels would still have to land in the
  door. §8 Q6.

---

## 6. What refuses by name (S1.4)

| Case | Proposal | Where |
|---|---|---|
| An op with no rule | listed in `Partition.refused` as `(node, op, reason)`; it ends every open region and runs unfused through the door | `regions.classify` |
| A random op (`rand`, `dropout` in training) | refused: a fused kernel would need eager's generator state | `regions.classify` |
| A region whose key is not in the bundle | refuse by name, naming the key and the bundle; or run it unfused and list it (§8 Q2) | proposal |
| A length outside the region's range | refuse by name, naming the dim, the value and the range | proposal |
| A bundle whose IR or emitter version does not match | refuse by name (`typedpython.md` §5.5) | proposal |
| A Dynamo exception or the recompile limit in `torchnative.compile` | must surface by name, not only in `stats` (§2.4) | #208 |

A mutation is not a refusal. It is correct as it is; it only cannot be inside a region (§4.3).

A listing is the refusal at partition time, in the same shape `capture.rs` uses: record the reason,
let the program finish on the path it was on, and make the claim (here: "this graph is fused") only
where the reasons are visible. Whether an op without a rule should instead raise at `.to(device)` is
§8 Q2.

---

## 7. One decode step, captured (the suite)

`tests/export/test_fusion_capture.py`. Two halves:

- **The partitioner on hand-built FX graphs** (six tests): a decoder layer, the extracted RMSNorm
  executed against eager arithmetic, the KV barrier, a refusal by name, the most-recent rule and the
  cycle detector's own negative control, and the look-alike spellings (`operator.and_`, `iadd`,
  `getitem` by tensor). One more test reads the shim's own names for `F.linear`, `F.silu`,
  `F.scaled_dot_product_attention` and the door's OpOverloads, because the rule reads those names.
  These need only the vendored tree.
- **The decode step**: the tiny Qwen3 of `tests/models/test_qwen3.py` (`_TINY_CFG`: two layers, GQA
  4/2, `head_dim` 16, SDPA, random weights, no download) is prefilled eagerly, then three decode steps
  run through `torchnative.compile.Engine` with a recording backend, beside an eager twin with its
  own cache. Each graph of the first step is partitioned. It runs once with `DynamicCache` and once
  with `StaticCache`, each in a fresh subprocess that asserts `hasattr(torch._C, "_aten_implemented")`.

The expected listing, per decode step with L = 2, **derived by reading `modeling_qwen3.py` and
`cache_utils.py` against the rule, not observed**:

| Expectation | DynamicCache | StaticCache |
|---|---|---|
| attention regions | L | L |
| RMSNorm regions (`pow`, `mean`, `rsqrt`) | 4L + 1 | 4L + 1 |
| MLP regions, exactly `silu, mul` | L | (not asserted) |
| mutations | none | `index_copy_` 2L times, `add_` L times |
| refused | none | (printed, not asserted: the mask path is not read closely enough) |
| new compiles on steps 2 and 3 | at least one each (the length is a shape) | none |
| compiled logits vs eager twin | bit-equal every step | bit-equal every step |

Without `torchnative.compile` (develop before #208) the decode tests **fail**, naming #208. They do
not skip, because a skipped capture would read as a capture nobody needed.

How to read a red first run: a wrong count is a finding about the rule or about this reading of
`transformers`, and the printed region list says which. It is reported and the expectation is
corrected from the observation, with the observation quoted; it is not edited until green.

---

## 8. Open decisions for the owner

Each question is open. None has been answered; nothing below is implemented beyond the partitioner
and its tests.

1. **When are fused kernels compiled, given that capture happens at the first `forward`?**
   Constraint: iOS forbids run-time code generation (`typedpython.md` §5.1), and `.to(device)`
   captures at run time (owner, 2026-11-08). Options:
   (a) a build-time step runs the same capture on the app's model, collects region keys, and
   TypedPython compiles them into the app's bundle (a new pipeline and a new artefact kind: needs
   approval; the 2026-11-07 "no offline or CI `.pte` pipeline" was about the NPU, and whether it
   extends here is the question);
   (b) a fixed kernel library for known patterns (RMSNorm, rotary, SwiGLU, attention) shipped with
   torchnative, matched at run time, no per-app step;
   (c) fusion on desktop only (JIT allowed there, `typedpython.md` §5.1), none on mobile for now.
2. **A region that cannot be fused, or a key missing from the bundle: refuse or run unfused?**
   Options: (a) run it unfused through the door and list it by name (the partition's current
   behaviour); (b) raise at `.to(device)`; (c) (a) by default, (b) under a strict flag.
3. **Which KV cache does `generate` use on a compiled device?** Options: (a) whatever the user passed
   (`DynamicCache` by default: a compile per token, §4.2); (b) switch to `StaticCache` for compiled
   devices, as upstream's `cache_implementation="static"`; (c) a static-shaped variant of
   `PreallocatedCache` (#180).
4. **Is the sequence length symbolic in `torchnative.compile`?** The owner said a new input size
   recompiles (2026-11-08). Options: (a) keep static shapes, bucket prefill lengths; (b) mark the
   sequence dim dynamic (`mark_dynamic` with a range) and compile symbolic kernels (§3.3); (c)
   automatic dynamic after the first recompile, as upstream.
5. **The handoff format to TypedPython.** Options: H1 the region's FX graph (proposed, §5.3); H2 a
   generated `@kernel` source; H3 TypedPython's IR directly, once it exists.
6. **How a fused kernel is registered** (`typedpython.md` §8 item 4 left this to be agreed with
   torchnative). Options: (a) a `torchnative_fused` arm in `_aten_dispatch`; (b) `torch.library`
   custom ops, after the shim honours them, still landing in the door; (c) `kernels` bundle resolver
   entries.
7. **Stage 1 or wait for stage 2?** Options: (a) partition the Dynamo graph now, mutations as barriers;
   (b) first make AOTAutograd functionalization reach on the shim, then partition ATen.
8. **Bucketed or symbolic kernels** (§3.3). Options: (a) symbolic by default, buckets where tiling
   needs them (proposed); (b) buckets everywhere (simpler kernels, more of them).
9. **Do the NPU back ends consume these regions?** CoreML, QNN and OpenVINO take whole subgraphs
   (AGENTS.md §20). Options: (a) fusion targets the GPU and CPU device layer only; (b) the partition
   also feeds NPU subgraph selection.

**No SPEC row is added** until these are answered: a row would state a behaviour the owner has not
approved (AGENTS.md rule 5, §7).

---

## 9. What was not verified

- **No model, no suite, no build was run.** `python -m py_compile` passed on the two Python files
  this round wrote; that is the whole of what was checked by running anything.
- Every decode expectation in §7 is a reading of `transformers` 5.15.1, not an observation.
- Whether Dynamo, entered through #208's engine, traces `Qwen3ForCausalLM.forward` with either cache
  at all. #208's own suite covers a 2-layer toy LM, not a `transformers` model.
- Whether the shim's names for torch-level callables are the base op names
  (`test_op_name_reads_the_shims_own_spellings` checks it on the first run).
- Whether AOTAutograd functionalization runs on the shim (stage 2).
- ExecuTorch's resize check and memory planning of mutated buffers are from upstream documentation;
  only the files cited in §3.1 and §4.1 were read.
- The rotary fragmentation of §5.2 is derived by hand.

What this design cannot find: speed. Whether a region is worth fusing on a given GPU is a measurement
(`typedpython.md` §5.8), and nothing here measures.

---

## 10. Files

- `torchnative/python/torchnative/export/regions.py`: the partitioner and `extract`.
  <!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/export/regions.py partition present -->
- `tests/export/test_fusion_capture.py`: §7's suite.
- `.github/scripts/gate_suites.py`: the suite classified for both CI runners.
