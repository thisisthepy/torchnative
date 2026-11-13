# Fusion over a captured decode step: front end, dynamic length, in-place KV, handoff

Issue #68. Written 2026-11-13 on `feat/fusion-capture` (from `develop` at 80f23ce), reading PR #208
(`feat/dynamo-guards-manager` at 5d836f5) for `torchnative.compile`.

**Two choices in this document are the owner's**, both relayed by the leader on 2026-11-13:

- **§8 Q1 is answered (b): a fixed pattern library matched at run time.** torchnative ships a fixed
  set of fused kernels (patterns), and the compile backend matches captured regions against that
  library at run time. There is no per-model build-time capture pipeline.
- **A symbolic sequence length is approved** (§8 Q8, and the kernel half of Q4): kernels take the
  length as a symbolic dimension, not bucketed shapes, where the pattern allows.

The leader relayed the decisions in these words; the owner's own wording was not quoted to this
round. Each place that rests on them cites "owner, 2026-11-13, relayed by the leader". **Every other
design choice in this document is an agent proposal, not confirmed by the owner**, and §8 lists the
questions still open. Where a line rests on the owner's earlier words, it quotes them with the date
and the place they are recorded.

**Nothing in this document was run.** The round that wrote it was not allowed to build, run a
model or run a suite (AGENTS.md §10). Upstream behaviour is read from source, with the file and
line given; the `torchnative.compile` behaviour is read from #208's code. The capture of one
decode step is written as a suite (`tests/export/test_fusion_capture.py`, §7) whose expected
values are derived by reading and are **not yet observed**. The pattern matcher of §5.5 has its own
suite (`tests/export/test_fusion_patterns.py`), written before the matcher and also **not yet run**.

---

## 0. At a glance

| Question | Answer, and whose | Section |
|---|---|---|
| When are fused kernels compiled? | **Decided (owner, 2026-11-13, relayed by the leader): a fixed pattern library shipped with torchnative, matched at run time.** No per-model build-time capture pipeline, no per-model kernel bundle | §3.3, §5.5 |
| Dynamic sequence length | **Decided (owner, 2026-11-13, relayed by the leader): symbolic, not bucketed, where the pattern allows.** Agent proposal: decode served by the StaticCache shape. Open: how `torchnative.compile` makes the length symbolic (Q4) | §3 |
| Which front end captures the graph? | Agent proposal: **Dynamo, through `torchnative.compile` (#208).** `capture.rs` stays the NPU delegates' substrate; `torch.export` stays the second way | §2 |
| At which level is the graph partitioned? | Agent proposal: **stage 1, the Dynamo graph as the backend receives it** (torch-level ops, mutations in place). Stage 2: the functionalized ATen graph, once AOTAutograd reaches on the shim | §2.3 |
| In-place KV update | Agent proposal: **a mutation is a region barrier and runs through `_aten_dispatch` in program order.** Functionalization plus re-inplacing is stage 2 | §4 |
| Region rule | Agent proposal: elementwise, reduction and view chains grow a region; GEMM, gather, mutation and unknown ops end it; SDPA is its own region. Implemented as `torchnative/export/regions.py` | §5.1 |
| The first patterns, and how a region matches | Agent proposal: `rms_norm`, `add_rms_norm`, `silu_mul`, `rotary`, `sdpa_kv`; a structural match over the region (op sequence, operand wiring, outputs, symbolic shapes, dtypes). Implemented as `torchnative/export/patterns.py`, matcher only | §5.5 |
| A region that matches no pattern | Runs unfused, each op through `_aten_dispatch`, and is **listed** in the match report (implemented). Listing versus refusing is the owner's open Q2 | §5.5, §6 |
| Run-time dispatch | Agent proposal: **one op per matched region, `torchnative_fused.<pattern>`, through `_aten_dispatch`.** No bypass. Not wired | §5.4 |
| Handoff to TypedPython | Changed by Q1: no region of a user's model is handed to a compiler. Each library pattern gets one kernel, written and compiled when torchnative is built; `regions.extract` stays as the reference oracle. How those kernels are written is Q5 (open) | §5.3 |
| What refuses | Ops without a rule and random ops are listed by name; a region with no pattern is listed (Q2); a static dim given a symbol and missing meta are listed; a length outside a kernel's range and a pattern the door does not implement refuse by name (proposal) | §6 |
| Decode step captured, regions listed | Written as a suite; **expected values not observed**; blocked by #268 and #269 | §7 |
| Open owner decisions | Q1 and Q8 answered, Q4 partly; Q2, Q3, Q5, Q6, Q7, Q9 open | §8 |

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
| No offline or CI `.pte` pipeline for the Android NPU | owner, 2026-11-07, `docs/INTENT.md` §5 | it is about the NPU back end, not GPU fusion; Q1's answer below leaves no per-model pipeline for fusion either |
| Not an ahead-of-time export pipeline as the user-facing model | `docs/INTENT.md` §7 | the user ever holding a fused artefact |
| Fused kernels come from a fixed pattern library matched at run time (§8 Q1 (b)) | owner, 2026-11-13, relayed by the leader | a per-model build-time capture, a per-model kernel bundle, and per-region compilation |
| Symbolic sequence length: kernels take the length as a symbolic dimension, not bucketed shapes, where the pattern allows | owner, 2026-11-13, relayed by the leader | one kernel per length bucket where a pattern can take the length as a symbol |

`typedpython.md` is python-multiplatform's
<https://github.com/thisisthepy/PythonMultiplatform/blob/develop/docs/design/typedpython.md>.

The issue's own wording ("build-time fusion") is quoted as the issue's text. The issue was opened
from the account `b-re-w`; the round that first wrote this note could not tell whether the owner
wrote it, so "build-time" was not cited as an owner decision and §8 Q1 asked it. The answer (b) settles
it: nothing about a user's model is captured or compiled at build time. What is built at build time
is torchnative's own fixed kernel library, once, for every model.

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
backend Dynamo calls with `(gm, example_inputs)` partitions `gm` (§5), matches each region against
the fixed pattern library (§5.5), replaces each matched region with one fused call (§5.4) and returns
the rewritten module's `forward`.

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

### 3.3 What the fixed library does with the length

Two owner decisions apply here (2026-11-13, relayed by the leader): fused kernels come from a fixed
pattern library matched at run time, and the sequence length is symbolic, not bucketed, where the
pattern allows.

TypedPython compiles kernels at build time (`typedpython.md` §5.1). With a fixed library that build
is torchnative's own: each pattern's kernel is compiled when torchnative is built and ships in the
wheel, and no model is captured at build time. So at run time a region can only be matched to a
pattern that already exists, and one kernel has to serve every length a model brings. The symbolic
length is what makes that possible: a pattern kernel takes the length as an argument, not one kernel
per bucket.

Which dims the first patterns (§5.5) take symbolically, and which stay static, as declared in
`patterns.py` (the split is an agent proposal; that the length is symbolic is the owner's):

| Pattern | Symbolic (an argument) | Static (the kernel tiles over it) |
|---|---|---|
| `rms_norm`, `add_rms_norm` | every leading dim (`lead`: batch and length, as rows) | `N`, the normalised dim |
| `silu_mul` | `lead` | `I`, the intermediate width |
| `rotary` | `B`, `S` | `H`, `D` |
| `sdpa_kv` | `B`, `Sq`, `Skv` (the KV cache length) | `Hq`, `Hkv`, `D` |

"Where the pattern allows": every pattern of the first set allows it. In the norms, the MLP and
rotary the length is only an outer loop bound; in attention it is a reduction extent the kernel reads
as an argument (tiles over `kv_len` with a remainder tile). The dims that stay static are the model's
widths, which do not move between steps. A static dim that arrives as a symbol does not match, and the
report names the dim (§6). Bucketing remains only for a future pattern that cannot take a symbol;
none of the first set needs it. Whether a symbolic-length attention kernel is fast enough without
per-bucket autotuning is a measurement (`typedpython.md` §5.8), not made here.

**Agent proposals that remain (not confirmed):**

1. **Decode is served by one static-shaped graph**: the StaticCache shape, where nothing about the
   shape moves. It is how upstream's own `generate` compiles (`cache_implementation="static"`).
   Whether compiled devices should switch the cache is §8 Q3.
2. **How the length reaches the backend as a symbol** is not decided. The decision says the kernels
   take the length symbolically; it does not say whether `torchnative.compile` marks the sequence
   dim dynamic ahead of time (`mark_dynamic` with a range) or after the first recompile (automatic
   dynamic). #208 forces both off today. That half of §8 Q4 stays open. Until it is answered the
   matcher treats a concrete length and a symbolic one alike: it binds the length, records whether it
   was a symbol, and matches the same pattern either way (`test_patterns_*_matches_at_two_lengths_*`).
3. **A supported range per symbolic dim.** Each pattern kernel declares the range it supports (for
   `sdpa_kv`, the KV length up to the kernel's maximum), and a length outside it refuses by name,
   naming the dim, the value and the range, unlike AOTInductor's default, which does not check.
   Not implemented: no kernel exists yet to state a range.

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
v0 rule for the final one. Since the library is fixed (owner, 2026-11-13, relayed by the leader), a
pattern matches a whole region, so the first of them also decides whether `rotary` can match at all
(§5.5).

### 5.3 What becomes of the handoff to TypedPython

The first version of this note proposed handing each captured region to TypedPython, compiled for
that model (H1 the region's FX graph, H2 a generated `@kernel` source per region, H3 TypedPython's IR).
All three assumed per-model compilation. The fixed library (owner, 2026-11-13, relayed by the leader)
removes it: **no region of a user's model is handed to a compiler, at build time or at run time.**
What remains:

- **The kernels are written per pattern, not generated per region.** Each library pattern gets one
  kernel per target, compiled when torchnative is built. Agent proposal, not confirmed: written as
  TypedPython `@kernel` source (valid Python that runs on CPython, `typedpython.md` §5.2), which is
  H2 moved from "generated per region" to "written per pattern". It still needs the tile-op
  vocabulary, which does not exist yet. §8 Q5 asks.
- **The pattern declaration is the contract.** `patterns.py` states each pattern's steps, input
  shapes (static and symbolic dims) and dtypes. A kernel implements exactly that computation, and the
  matcher hands it only regions that compute exactly that.
- **`regions.extract` stays, as the reference.** It returns a region as its own `torch.fx.Graph`
  with each node's `meta` copied, executable on CPython. A matched region's extract is the oracle for
  its pattern kernel in `typedpython.md` §5.6's three-way comparison.
  `test_partition_extract_hands_off_a_graph_that_computes_the_region` executes an extracted RMSNorm
  region and requires it to equal the eager arithmetic bit for bit.

### 5.4 Library versioning and dispatch

**Agent proposal, not confirmed**, rewritten for the fixed library:

- **No per-model bundle.** The library is part of torchnative: `patterns.py` and the compiled kernels
  ship in the same wheel, so their version is torchnative's version. The earlier "bundle version
  mismatch" refusal has no per-model artefact left to check. What can still disagree is a pattern
  name the matcher knows and the door's arm does not implement (a Python tree ahead of its
  extension); that refuses by name at dispatch.
- **At run time** the `torchnative.compile` backend partitions the graph, matches each region
  (§5.5), and rewrites the `GraphModule`: each matched region becomes one `call_function` node whose
  operands are the pattern's inputs. The region's views stay ordinary view ops in front of it, so
  the node receives exactly the tensors the pattern's steps read. **Not wired**: it needs the
  owner's answers to Q2, Q6 and Q7, and the decode step cannot be traced until #268 and #269 are
  fixed.
- **That node enters `_aten_dispatch`** under the pattern's name (for example
  `torchnative_fused.rms_norm.default`), and the door's arm for that namespace runs the kernel on the
  device layer (`vulkan.rs`, candle-metal). **No bypass** (AGENTS.md §11): the fused call is one more
  op through the one door, so `capture.rs` records it, the dispatch counters count it, and its
  refusals carry its name.
- **`torch.library` custom ops** are the upstream-shaped alternative, and `typedpython.md` §5.6 names
  them. On the shim today, `torch.library.impl` registrations are recorded and dispatch nothing
  (`docs/design/REGISTRATIONS.md` §0: 1549 registrations at import, none dispatched), so that route
  first needs the shim to honour `torch.library`, and its kernels would still have to land in the
  door. §8 Q6.

### 5.5 The pattern library and the matcher

The library is fixed (owner, 2026-11-13, relayed by the leader). **Which patterns it holds first,
and the matching rule, are agent proposals, not confirmed.** Implemented in
`torchnative/python/torchnative/export/patterns.py`, matcher only: pure Python over `torch.fx`, no
new dependency, no kernels, nothing dispatched.

#### The first set, and why these

The partitioner's regions decide what a pattern can be, because a pattern matches a whole region.
Read against the rule of §5.1, a decoder's decode step yields per layer: four RMSNorm regions (two of
them begin with the residual `add`, which joins the norm that reads it), one SDPA region, one
`silu, mul` region, and the rotary chains. These are the computations Qwen3 and the Llama family
share, and they are where decode's small ops are.

| Pattern | Computation (as Dynamo spells it) | Spellings | Source read |
|---|---|---|---|
| `rms_norm` | `h = x.to(f32)`; `h * rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)`; times the weight, with or without the cast back | 2 | `modeling_qwen3.py:59-64` (transformers 5.15.1) |
| `add_rms_norm` | `s = a + b`, then `rms_norm(s)`; `s` leaves the region too when something outside reads it | 2 | the residual adds of `Qwen3DecoderLayer` and the final norm |
| `silu_mul` | `silu(gate) * up` (SwiGLU) | 1 | `Qwen3MLP.forward`, `SiLUActivation` (`activations.py`) |
| `rotary` | `x * cos + cat((-x[..., D/2:], x[..., :D/2]), -1) * sin` | 1 | `rotate_half`, `apply_rotary_pos_emb` (`modeling_qwen3.py:140-170`) |
| `sdpa_kv` | `scaled_dot_product_attention(q, k, v, attn_mask=, dropout_p=0, scale=, is_causal=, enable_gqa=)` over a KV cache of symbolic length | 1 | `integrations/sdpa_attention.py:154-163` |

#### How a region matches

A region matches a spelling when all of these hold (the full statement is `patterns.py`'s
docstring):

1. **Op sequence.** Its work ops (members minus views), in graph order, are the spelling's steps.
2. **Structure.** Each step's operands are what the step names: a pattern input, an earlier step
   (through the region's own views), a half of the last dim, a constant or a literal. `add` and `mul`
   may take their operands either way round; a keyword the step does not name refuses the match. So a
   region with the right op sequence and the wrong wiring does not match.
3. **Outputs.** The values that leave the region are the spelling's outputs.
4. **Shapes, symbolic.** Each input's shape (Dynamo's `example_value`) fits its spec; same-named dims
   agree; a dim may be a symbol unless the pattern lists it as static (§3.3).
5. **Dtypes.** Inputs sharing a dtype variable agree, and the dtype is allowed (`float32`, `float16`,
   `bfloat16`). The RMSNorm upcast must be to `float32`.
6. **Relations.** `sdpa_kv` checks grouped-query heads (`Hq` a multiple of `Hkv`, and
   `enable_gqa` when they differ), the mask's broadcast dims and dtype, and no dropout.

Missing meta is listed, not guessed. Two pattern names matching one region is a library defect,
listed as `ambiguous`, never resolved by registry order. `check_registry` refuses, at import, a
pattern that reads an undefined name, uses an op the partitioner never puts in a region, or has an
input no step reads, because such a pattern could never match and would look like coverage.

#### No match, and dispatch

A region that matches no pattern **runs unfused: each of its ops through `_aten_dispatch` on its own**,
exactly as without fusion. It is **listed** in `FusionReport.unmatched` with the reason for each
candidate pattern, and every report carries that sentence (`FusionReport.UNMATCHED_RUNS`). This is
implemented as listing, not silence. Whether such a region should instead be refused is the
owner's **open Q2**; the listing is what Q2 (a) describes, chosen here only because it changes
nothing about what runs.

A matched region would dispatch as one op, `torchnative_fused.<pattern>`, through the one door
`_aten_dispatch` (§5.4). Not wired.

#### What the reading predicts on Qwen3, and the first finding

Per decode step with L layers and `DynamicCache`, reading `modeling_qwen3.py` and `cache_utils.py`
against §5.1 and the matcher. **Not observed**: the decode step cannot be traced until #268 and #269
are fixed.

| Region | Per step | Pattern |
|---|---|---|
| layer 0's input norm | 1 | `rms_norm` |
| `q_norm` (its trailing `transpose` is a view) | L | `rms_norm` |
| residual `add` + norm: post-attention, the input norm of layers 1 to L-1, the final norm | 2L | `add_rms_norm` (the final norm's sum is read by nothing else, so that output is optional) |
| SDPA | L | `sdpa_kv` |
| MLP | L | `silu_mul` |
| `k_norm`, then K's whole rotary, then (`DynamicCache` only) the cache `cat` | L | none: listed |
| Q's `rotate_half(q) * sin + q * cos` tail (`neg, cat, mul, add`) | L | none: listed |
| the model-level `cos`/`sin` computation plus layer 0's `q * cos` | 1 | none: listed |

**Rotary matches nowhere.** The most-recent-producer rule (§5.2) cuts Q's rotary in two (the
`q * cos` joins the `cos` side) and fuses K's rotary into the K norm. With a fixed library the shape of
a region decides whether any pattern can match, so §5.2's tuning is no longer only a speed question.
Options, agent proposals, none implemented: (a) a composite pattern `rms_norm_rotary` for the K side
as found; (b) a partition rule under which rotary forms its own region (cut at the norm's output, or
merge at the shared consumer); (c) match patterns inside a region and split it. `rotary` is declared
and tested on a graph where the chain is one region (a projection's views feeding rotary directly),
so it is ready once the partition yields that region.

By this reading 5L + 1 of the 7L + 2 regions in this table would match (with L = 2: 11 of 16).
Regions of the mask and position computations are not in the table: that path was not read closely
enough. The count is a prediction to be corrected from the first observed run, with the observation
quoted.

#### The suite

`tests/export/test_fusion_patterns.py`, 16 tests over hand-built FX graphs, each input carrying a
stand-in for Dynamo's `example_value`, and a stand-in for a `SymInt` (an object that is not an `int`):

- each pattern matches the region the partitioner finds, at two concrete lengths and a symbolic one,
  with the length bound as a symbol and the static dims bound to the same numbers;
- near-misses do not match and are listed with the reason: wrong op order, an extra op, a wrong
  dtype (in the meta, and the upcast constant), wrong wiring with an equal op sequence (a negative
  control asserts the sequences are equal, so only the structural check can reject them), a static
  dim given a symbol, no meta, an ambiguous library, SDPA with dropout or with GQA heads and no
  `enable_gqa`;
- three mutants of `rms_norm` (a constant, an edge, a dropped step) each make the positive check fail
  (AGENTS.md §17.5), and the optional-output mark is removed once to show the output rule fails
  without it;
- the registry checks itself and refuses a pattern with an undefined name.

Expected red before `patterns.py` existed: the whole suite, at the import of
`torchnative.export.patterns`. Not observed (the round could not run a suite, AGENTS.md §10).

What the matcher cannot find: whether a real Dynamo graph spells these ops as the hand-built graphs
do, view parameters other than the rotary halves (views a step does not name are passed through),
and anything about numbers, since there is no kernel to compare.

---

## 6. What refuses by name (S1.4)

| Case | Proposal | Where |
|---|---|---|
| An op with no rule | listed in `Partition.refused` as `(node, op, reason)`; it ends every open region and runs unfused through the door | `regions.classify` |
| A random op (`rand`, `dropout` in training) | refused: a fused kernel would need eager's generator state | `regions.classify` |
| A region that matches no library pattern | listed in `FusionReport.unmatched` with the reason per candidate; it runs unfused, each op through `_aten_dispatch`. Listing is implemented; refusing instead is the owner's open Q2 | `patterns.match_partition` |
| A region two pattern names match | listed as `ambiguous` (a library defect), never resolved by registry order | `patterns.match_region` |
| A region whose inputs carry no shape/dtype meta | listed, naming the input; not guessed | `patterns.match_region` |
| A static dim (hidden, head, intermediate) that arrives as a symbol | listed, naming the dim | `patterns.match_region` |
| A library pattern that could never match (undefined name, unfusable op, unread input) | `check_registry` raises `ValueError` at import | `patterns.check_registry` |
| A length outside a pattern kernel's supported range | refuse by name, naming the dim, the value and the range; no ranges exist until kernels do | proposal |
| A pattern name the door's arm does not implement | refuse by name at dispatch (§5.4). There is no per-model bundle, so no bundle version to mismatch | proposal |
| A Dynamo exception or the recompile limit in `torchnative.compile` | must surface by name, not only in `stats` (§2.4) | #208 |

A mutation is not a refusal. It is correct as it is; it only cannot be inside a region (§4.3).

A listing is the refusal at partition time, in the same shape `capture.rs` uses: record the reason,
let the program finish on the path it was on, and make the claim (here: "this graph is fused") only
where the reasons are visible. Whether an op without a rule, or a region with no pattern, should
instead raise at `.to(device)` is §8 Q2.

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

**The decode step cannot be traced yet.** #268 (the shim refuses `torch.tensor(..., dtype=int)`,
which `StaticCache` calls) and #269 (Dynamo tracing of the `DynamicCache` decode step reaches
`Tensor._base`, which the shim refuses) stop it before the backend sees a graph.

**What the 2026-11-13 decisions changed in these three tests:** only the message and docstring of
`test_decode_step_with_dynamic_cache_recompiles_every_step_under_static_shapes`. It still asserts a
new compile on every `DynamicCache` step, because `torchnative.compile` runs with static shapes
today, and it now says that this expectation flips once `torchnative.compile` makes the length
symbolic (approved; the means is open Q4). The fixed library changes nothing they assert: they
assert the partition, which the matcher reads and does not change. They do not assert pattern
matches; §5.5's table is the expectation for that, to be added once the step traces.

The matcher has its own suite, `tests/export/test_fusion_patterns.py` (§5.5).

---

## 8. Open decisions for the owner

Q1 and Q8 are answered and Q4 partly (owner, 2026-11-13, relayed by the leader). The others are
open. Nothing below is implemented beyond the partitioner, the matcher and their tests.

1. **When are fused kernels compiled, given that capture happens at the first `forward`?**
   **Answered (owner, 2026-11-13, relayed by the leader): (b), a fixed pattern library matched at
   run time.** torchnative ships a fixed set of fused kernels; the compile backend matches captured
   regions against it at run time; no per-model build-time capture pipeline. The options were:
   (a) a build-time step running the capture on the app's model and compiling its region keys into
   the app's bundle; (b) the fixed library; (c) fusion on desktop only.
2. **A region that cannot be fused, or matches no library pattern: refuse or run unfused?** Open.
   Options: (a) run it unfused through the door and list it by name; (b) raise at `.to(device)`;
   (c) (a) by default, (b) under a strict flag. The partition and the matcher list (`Partition.refused`,
   `FusionReport.unmatched`) and refuse nothing; that is the listing half of (a), and it is not an
   answer to this question.
3. **Which KV cache does `generate` use on a compiled device?** Options: (a) whatever the user passed
   (`DynamicCache` by default: a compile per token, §4.2); (b) switch to `StaticCache` for compiled
   devices, as upstream's `cache_implementation="static"`; (c) a static-shaped variant of
   `PreallocatedCache` (#180).
4. **Is the sequence length symbolic in `torchnative.compile`?** **Partly answered (owner,
   2026-11-13, relayed by the leader): the length is symbolic**, so (a) static shapes with bucketed
   prefill is out where the pattern allows. Open: how the backend receives the symbol: (b) mark the
   sequence dim dynamic ahead of time (`mark_dynamic` with a range), or (c) automatic dynamic after
   the first recompile, as upstream; and how that squares with the owner's 2026-11-08 "a new input
   size recompiles" (§1).
5. **How are the library's kernels written?** Reworded after Q1: no region is handed to a compiler,
   so the old H1/H2/H3 handoff no longer applies (§5.3). Options: (a) TypedPython `@kernel` source per
   pattern (proposed); (b) TypedPython's IR directly, once its format exists; (c) hand-written
   device kernels (Metal, Vulkan) per pattern, outside TypedPython.
6. **How a fused kernel is registered** (`typedpython.md` §8 item 4 left this to be agreed with
   torchnative). Options: (a) a `torchnative_fused` arm in `_aten_dispatch`; (b) `torch.library`
   custom ops, after the shim honours them, still landing in the door; (c) `kernels` bundle resolver
   entries.
7. **Stage 1 or wait for stage 2?** Options: (a) partition the Dynamo graph now, mutations as barriers;
   (b) first make AOTAutograd functionalization reach on the shim, then partition ATen.
8. **Bucketed or symbolic kernels** (§3.3). **Answered (owner, 2026-11-13, relayed by the leader):
   symbolic, not bucketed, where the pattern allows.** The options were (a) symbolic by default,
   buckets where tiling needs them; (b) buckets everywhere.
9. **Do the NPU back ends consume these regions?** CoreML, QNN and OpenVINO take whole subgraphs
   (AGENTS.md §20). Options: (a) fusion targets the GPU and CPU device layer only; (b) the partition
   also feeds NPU subgraph selection.

**No SPEC row is added yet.** Q1 and Q8 fix a direction, but no user-visible behaviour follows from
them until the matcher is wired into the backend, and that needs Q2, Q6 and Q7. A row now would state
a behaviour the owner has not approved (AGENTS.md rule 5, §7).

---

## 9. What was not verified

- **No model, no suite, no build was run.** `python -m py_compile` passed on the two Python files
  the first round wrote, and on the files the 2026-11-13 round wrote or changed (`patterns.py`,
  `test_fusion_patterns.py`, `test_fusion_capture.py`, `regions.py`, `gate_suites.py`); that is the
  whole of what was checked by running anything. In particular the matcher's 16 tests have never run:
  neither their red before `patterns.py` existed nor their green after.
- Whether Dynamo spells the five patterns' ops as `patterns.py` declares them (`to` with a
  positional dtype, `mean(-1, keepdim=True)`, `F.silu` inlined from `SiLUActivation`, the
  `rotate_half` slices with a constant half, SDPA's keywords). The §5.5 table is a reading.
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
- `torchnative/python/torchnative/export/patterns.py`: the pattern library's declarations and the
  matcher (§5.5).
  <!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/export/patterns.py match_partition present -->
- `tests/export/test_fusion_capture.py`: §7's suite.
- `tests/export/test_fusion_patterns.py`: §5.5's suite.
- `.github/scripts/gate_suites.py`: both suites classified for both CI runners.
