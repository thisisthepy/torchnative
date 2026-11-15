# Functionalization on the shim: what `run_decompositions` / `to_edge` need (#306, wall W3)

Status: agent proposal, not confirmed. The route below (section 3 (i), a real
wrapper as a `Repr` variant) was then chosen as a **leader decision
2026-11-15, agent proposal; not owner-confirmed** (issue #329, SPEC S5.11).
Nothing here is an owner decision.

Copied into the repository from the round's scratch note when step 1 of #329
was written. Two things changed since the note was written, and both are
recorded here rather than edited into the text below:

* **Step 0's premise was wrong** (measured in the round after the note, torch
  2.13.0): upstream's `_op_dk(DispatchKey.Functionalize, ...)` on *plain*
  tensors returns exactly the plain call's answer, so there is no refusal to
  add for plain tensors. The guard belongs in step 1 as a check on
  functional-tensor *arguments*: every operator that reaches the door with a
  functional argument refuses by name (`functional.rs::refuse_functional_args`).
  One case does need a refusal with plain arguments, and step 1 has it: with
  `Functionalize` in the TLS include set (inside `FunctionalTensorMode`'s
  `_ForceDispatchKeyGuard`) upstream wraps even a factory op's output, so
  `_op_dk(Functionalize, ...)` refuses there (`bootstrap.py` `op_dk`).
* The file names in section 2 (`trace_func.py`, `decomp_probe3.py`,
  `decomp3.log`, `jit_types.py`, `jit_up.json`) are the round's scratch
  scripts and are not in the repository. Step 1's measurements are in
  `tests/bindings/test_functional_tensor.py`, re-derived from upstream on
  every run.

The note as written:

Status: agent proposal, not confirmed. Nothing here is an owner decision. Written 2026-11-15
for draft PR #307 (branch `feat/exir-on-shim`). Every upstream number below was measured in
this round with `trace_func.py` (in the round's scratch directory) on torch 2.13.0 (`.caches/spike-venv`) and on
torch 2.14.0 + executorch 1.4.1 (`.caches/qnn-venv`); the shim facts were measured on the main
checkout's old develop build.

## 1. Where the wall is

With the mode-stack fix of this round (W1) and last round's `resize_` fix (W2) applied, the
Linear+ReLU probe (`decomp_probe3.py --stack --resize`, `decomp3.log`) stops at

    torch._to_functional_tensor(...) -- overload resolution has no table entry
    collect_metadata_analysis.py:183 _to_fun -> functional_utils.py:44 to_fun
    -> functional_tensor.py:303 FunctionalTensor.to_functional

`exir.to_edge(ep)` cannot avoid this: `executorch/exir/program/_program.py:1433` calls
`program.run_decompositions(table)` unconditionally for every program, and
`run_decompositions` always re-traces through `aot_export` (no fast path for a program that
is already Core ATen and functional; `torch/export/exported_program.py:357` only chooses
between the joint and the non-joint re-trace).

## 2. The exact surface, measured on upstream

Calls during one `run_decompositions()` of `Sequential(Linear(8, 4), ReLU)` (counts per call,
2.13.0; 2.14.0 is identical except where noted). "mut" is the same module with a buffer
`count.add_(1)` in `forward`.

### 2.1 Real machinery (each one changes or creates functionalization state)

| name | plain | mut | what it does upstream |
|---|---|---|---|
| `torch._to_functional_tensor` | 8 | 8 | wraps a tensor in a C++ `FunctionalTensorWrapper` with a fresh `FunctionalStorageImpl` |
| `torch._from_functional_tensor` | 15 | 20 | unwraps: the wrapper's current value |
| `torch._functionalize_sync` | 7 | 12 | applies pending updates from aliased views to this wrapper |
| `torch._mirror_autograd_meta_to` | 16 | 16 | copies `requires_grad` and autograd meta between outer and inner |
| `torch._functionalize_unsafe_set` | 2 | 2 | swaps the wrapper's value for another wrapper's |
| `torch._functionalize_enable_reapply_views` | 28 | 36 | TLS setter, returns the old value; changes how views are replayed |
| `torch._disable_functionalization` | 14 | 18 | TLS: drops `Functionalize` from the included key set |
| `OpOverload._op_dk(DispatchKey.Functionalize, ...)` | per op | per op | **the per-op functionalization kernel**: mutation to out-of-place plus copy-back, views to `ViewMeta` |
| `torch._functionalize_replace` / `_commit_update` / `_mark_mutation_hidden_from_autograd` | 0 | 0 | reached only by `FunctionalTensor.replace_` etc. (functional_tensor.py:328-337), not on these graphs |
| `torch._freeze_functional_tensor` | 0 | 0 | export mode, `aten.dropout` only (functional_tensor.py:633) |

### 2.2 State queries (answer from what the machinery recorded)

| name | plain | mut |
|---|---|---|
| `torch._is_functional_tensor` | 60 | 68 (to_edge mut: 93, to_executorch mut: 4) |
| `torch._functionalize_has_data_mutation` | 4 | 4 |
| `torch._functionalize_has_metadata_mutation` | 4 | 4 |
| `torch._functionalize_was_storage_changed` | 9 | 9 |
| `torch._functionalize_was_inductor_storage_resized` | 4 | 4 |
| `torch._functionalize_are_all_mutations_hidden_from_autograd` | 4 | 4 |
| `torch._functionalize_are_all_mutations_under_no_grad_or_inference_mode` | 0 | 1 |
| `torch._functionalize_is_multi_output_view` | 1 | 1 |
| `torch._functionalize_was_shallow_copy_data` | 0 on 2.13, 4 on 2.14 | same |
| `torch._C._functionalization_reapply_views_tls` | 16 | 16 (exists on the shim) |
| `torch._C._dispatch_tls_is_dispatch_key_included/excluded(Functionalize)` | 44 / 28 | 52 / 36 (exist) |
| `torch._C._dispatch_has_kernel_for_dispatch_key(op, Functionalize)` | part of 1982 | (exists; answers for the shim's table) |

A query is only truthful if the machinery in 2.1 records what it reports. So the queries are
not a separate work item: they land with the machinery or not at all. Answering
`has_data_mutation -> False` from a stub would be the silent kind of answer AGENTS.md section
18 forbids, and for the "mut" module it would drop the buffer mutation from the graph signature.

### 2.3 What stays Python and is upstream's

`torch/_subclasses/functional_tensor.py` (`FunctionalTensor`, `FunctionalTensorMode`) is pure
Python and runs unchanged: 14 `FunctionalTensor.__new__`, 15 `FunctionalTensorMode.__enter__`,
16 `__torch_dispatch__` for the plain module. `FunctionalTensorMode` is an infra mode
(`_mode_key = FUNCTIONAL`), so it lives in the slot this round's W1 fix put on the logical
stack. Its core (functional_tensor.py:596-640) is

    with torch._C._ForceDispatchKeyGuard(include | Functionalize, exclude - Functionalize):
        old = torch._functionalize_enable_reapply_views(True)
        outs_unwrapped = func._op_dk(DispatchKey.Functionalize, *args_unwrapped, **kwargs_unwrapped)
        ...
        torch._disable_functionalization()

so the Python layer is a wrapper; the algorithm is the C++ `Functionalize` kernel reached
through `_op_dk`.

### 2.4 A hazard that exists today

The shim's `OpOverload._op_dk` (`bootstrap.py` `_get_operation_overload.op_dk`, "for every other
op the shim has one kernel per op and the key changes nothing") ignores the dispatch key for
every op without a Python kernel. Today that is unreachable for `Functionalize` because
`_to_functional_tensor` refuses first. The moment `_to_functional_tensor` exists in any form,
`_op_dk(Functionalize, ...)` would run the plain kernel and return an unwrapped result: a
silent non-functionalization. Step 0 below closes this before anything else lands.

## 3. Routes

### (i) A real functional wrapper as a `Repr` arm in Rust

`Repr::Functional(FunctionalWrapper)`, where the wrapper holds the current value (a shim
tensor), an `Arc` shared storage record (generation counter, the base value, and the flags the
2.2 queries read: data mutated, metadata mutated, storage changed, mutations hidden from
autograd, under no_grad/inference), and the list of view steps that produced it from its base.
The per-op `Functionalize` kernel is written once, generically, at the door: unwrap, then

* a non-mutating, non-view op: call, wrap each tensor output in a fresh storage record;
* a view op (the shim already knows these: `tensor.rs::note_view`): with `reapply_views`
  (always true under `FunctionalTensorMode`) re-apply the view to the value and share the
  base's storage record, recording the step;
* an in-place op: call its functional variant (schema `<name>_` to `<name>`; `out=` to the
  functional overload), replace the value, bump the generation, set the data-mutation flag;
  views of the same storage see the bump and replay on `_functionalize_sync`;
* mutation through a view: needs the inverse of each view step (`slice_scatter`,
  `select_scatter`, `diagonal_scatter`, `as_strided_scatter`): until written, refuse by name.

Truthfulness: highest. AGENTS.md section 11's property carries over: `tensor()` refuses a
non-`Dense` variant, so any kernel that reaches a wrapper without going through the
`Functionalize` kernel refuses loudly; a wildcard-free `match` makes every unhandled site a
compile error. Upstream's Python `FunctionalTensor`/`FunctionalTensorMode` run unchanged on
top. Cost: the largest Rust change of the three, roughly 1.5 to 2.5 kLOC across tensor.rs,
aten.rs (door), bootstrap.py (the `torch._functionalize_*` names); the view-inverse part
(step 4) is most of the risk and is only needed for mutation through views (LLM KV caches).

### (ii) A Python-level wrapper plus a thin metadata shim

`torch._to_functional_tensor` returns a Python `_make_wrapper_subclass` tensor whose
`__torch_dispatch__` runs the same algorithm as (i) in Python, with the storage record as a
Python object; the door reaches it through its existing subclass path
(`subclass_dispatch_target`). Cost: smaller and faster to iterate (no Rust rebuild per change),
maybe 0.8 to 1.5 kLOC Python.

Truthfulness: the algorithm is the same, so the answers can be the same. What is lost is the
structural guarantee: the wrapper is a `Dense` shim tensor underneath (a wrapper subclass has
some backing), so a kernel path that reads it without going through `__torch_dispatch__`
(a direct `.tensor()` in Rust, `capture.rs` replay, a `no_dispatch()` block) computes on the
backing silently instead of refusing. It also stacks two Python subclass layers
(`FunctionalTensor(elem)` around the wrapper) on the door's subclass path, which has never
been exercised two deep. "A thin C shim that tracks version/mutation metadata" is not thin:
`_op_dk(Functionalize)` *is* the machinery, so (ii) is (i)'s algorithm in another language,
not a smaller job.

### (iii) Bypass: give exir an already functional, already Core ATen program

torchnative's capture (`capture.rs` trace, `export/decompose.py` lowers to Core ATen with
upstream's own table, `export/refold.py` folds prims back) already produces functional Core
ATen for non-mutating graphs, and `capture.rs::note_mutation` sees mutations. The route builds
an `ExportedProgram` from it (fx graph, `ExportGraphSignature` with
`buffers_to_mutate`, `meta["val"]` fakes on every node) and calls
`executorch.exir.program._program._generate_edge_program` and `EdgeProgramManager(...)`
directly, skipping `run_decompositions`.

What it can and cannot do, from reading `_program.py`: `to_edge` itself cannot be used
(line 1433 always re-decomposes). `_generate_edge_program` is private and
`EdgeProgramManager.__init__` says "Should not be called directly by users"; both verify
(`EXIREdgeDialectVerifier`), so a wrong program would be refused rather than accepted.
Truthfulness: fine for what it produces, but it does not make `ep.run_decompositions()` or
`exir.to_edge(ep)` work on the shim, which is what #306 asks, and every upstream pass that
calls `run_decompositions` later (partitioners do: `_program.py:1138-1196`) hits the same wall.
It is also a new pipeline, which AGENTS.md section 7 puts in front of the owner; it may fit the
owner's 2026-11-07 direction (`.to(device.npu)` JIT with torchnative's capture as front end,
AGENTS.md section 20), but that is the owner's call. Cost: converter about 0.5 to 1 kLOC
Python, plus keeping in step with private exir APIs per executorch release.

## 4. Recommendation (agent proposal, not confirmed)

Route (i). It is the only one that makes the upstream entry points work and keeps the
"silent fallback is unrepresentable" property; (ii) is the same algorithm without that
property; (iii) is a useful second road for `.to(device.npu)` but does not close #306 and
needs an owner decision. PR-sized steps, each with a test that can fail:

0. **`_op_dk(DispatchKey.Functionalize, ...)` refuses by name** (any op without a live Python
   kernel). Test: `torch.ops.aten.add.Tensor._op_dk(DispatchKey.Functionalize, x, y)` raises
   naming `Functionalize`; red today (returns a tensor). Upstream comparison is not possible
   (upstream runs the Functionalize kernel), so the test asserts the refusal and is replaced
   by step 2's agreement test.
1. **The wrapper and its queries.** `Repr::Functional`, `_to_functional_tensor`,
   `_from_functional_tensor`, `_is_functional_tensor`, `_is_functional_tensor_base`,
   `_mirror_autograd_meta_to`, `_functionalize_sync` (a no-op while the generation is
   current), `_functionalize_enable_reapply_views`, `_disable_functionalization`, and the 2.2
   queries answering from the record. Test: a script run on both sides (the pattern of
   `tests/export/test_export.py::_STACK_SCRIPT`) printing round trips and every query on fresh
   and wrapped tensors; red today (`_to_functional_tensor` refuses). Nullification: answer
   `has_data_mutation` from a constant and the step 3 test must go red.
2. **Non-mutating ops and views.** The door's `Functionalize` kernel for non-mutating ops and
   forward view replay; mutation of anything refuses by name. Test: `run_decompositions()` of
   Linear+ReLU on the shim matches upstream node for node (targets and order: upstream gives
   `aten.permute.default, aten.addmm.default, aten.relu.default`) and in graph signature, and
   `ep2.module()(x)` agrees with eager within the derived tolerance. Nullification: return the
   unwrapped output from the kernel; `FunctionalTensorMode.wrap` must refuse and the test go red.
3. **Mutation of a non-view tensor.** Functional variant by schema, replace, generation bump,
   flags. Test: the "mut" module's `run_decompositions()` graph and signature
   (`buffers_to_mutate`) equal upstream's; then `exir.to_edge(ep).to_executorch()` for both
   modules, comparing the emitted instruction lists with upstream's.
4. **Mutation through views** (KV cache writes, `x[:, 1:].add_(1)`, `index_copy_` into a slice):
   inverse view steps through the `*_scatter` ops. Test: upstream comparison on those three
   patterns, then a Qwen3 decoder layer with a static cache through `run_decompositions`.

## 5. W6: parametrised JIT types

Measured on 2.13.0 (`jit_types.py`, `jit_up.json`) against the old shim (`jit_shim.json`):

| expression | upstream | shim |
|---|---|---|
| `torch.OptionalType.ofTensor()` | `Optional[Tensor]`, a singleton (`is` itself) | refuses (stub) |
| `torch.ListType.ofTensors()` / `ofInts()` | `List[Tensor]` / `List[int]` | refuses (stub) |
| `torch.ListType(torch.IntType.get())` | `List[int]` | **returns a meaningless stub instance** |
| `convolution` `bias` arg type | `Optional[Tensor]`, `== ofTensor()` True | `_SchemaType` `Tensor?` |
| `convolution` `stride` arg type (`SymInt[]` in the schema) | `List[int]`, `== ofInts()` True | `_SchemaType` `SymInt[]` |
| `add_.Tensor` `self` type (`Tensor(a!)`) | `Tensor`, `isinstance TensorType` True | `Tensor(a!)`, isinstance True |
| `index.Tensor` `indices` | `List[Optional[Tensor]]`, `isinstance ListType` True, element `Optional[Tensor]`, its element `Tensor` | `Tensor?[]`, `isinstance ListType` False, no `getElementType` |

exir needs (`verification/arg_validator.py:83,89`, `emit/_emitter.py:559-567`):
`OptionalType.ofTensor()`, `ListType.ofTensors()`, `ListType(elem)` as a constructor,
`isinstance(t, OptionalType)`, `isinstance(t, ListType)`, `getElementType()`, and `==` between
a schema argument's type and those objects.

Minimal truthful objects, all in `bootstrap.py` next to `_SchemaType`:

* `ListType(T)` / `OptionalType(T)` construct the interned `_SchemaType(f"{T}[]")` /
  `_SchemaType(f"{T}?")`; `ofTensor`, `ofTensors`, `ofInts`, and the other `of*` the stubs list
  are those constructions. Interning gives upstream's singleton identity for free.
* `_SchemaType.getElementType()` peels the **outermost** layer (`Tensor?[]` to `Tensor?`,
  `Tensor?` to `Tensor`) and refuses on a non-container type, as upstream has no such method there.
* `_ShimMeta.__instancecheck__` answers `ListType` / `OptionalType` from the outermost layer of
  `_decompose_type` (`Tensor?[]` is a `ListType`, not an `OptionalType`), the same table the
  existing `TensorType` check reads.
* `==` must compare **types**, as upstream's `arg.type` is: alias annotations dropped
  (`Tensor(a!)` is `Tensor`) and `SymInt` read as `int` (upstream's `type` vs `real_type`;
  `stride.type == ofInts()` is True upstream). Today `_SchemaType.__eq__` compares spellings, so
  this is a real change and needs its own upstream-table test over every argument of every
  schema in the table (the shim side and upstream side both enumerable).
* The stub constructor that silently returns an instance (`torch.ListType(x)`) must stop doing
  so in the same change; it is a check that cannot fail today.
* Not needed by exir and not proposed: `str()` in upstream's `List[int]` form (the shim prints
  the schema spelling `int[]`; nothing on the exir path was seen to read it, but this round did
  not trace `str` calls on types, so that is unverified), and a subtype lattice.

## 6. What this investigation could not find

* The c_call tracer sees builtins only; anything upstream does inside one C++ call (what the
  `Functionalize` kernel dispatches internally, how many `ViewMeta`s it records) is invisible
  to it. The step list in 4 is from reading `functional_tensor.py` and the C++ design as I know
  it, not from a trace of the kernel.
* The functional-variant lookup (step 3) for every mutating op in the shim's table was not
  enumerated; ops with no functional overload (upstream uses `auto_functionalize` for custom
  ops, functional_tensor.py:519) were not counted.
* Whether the shim's door can host a second `Repr` arm on the mode path without touching
  `dispatch_through_mode` was not checked.
* Under `FakeTensorMode` the shim gives a requires-grad result no `grad_fn` (measured:
  `fw * 2` has `grad_fn None` on the shim, `MulBackward0` upstream). `_mirror_autograd_meta_to`
  and the hidden-from-autograd flags depend on autograd meta, so this gap may surface in step 1.
* The shim's own `torch.export.export` of Linear+ReLU records `aten.t, aten.addmm, aten.relu`
  where upstream records `aten.linear, aten.relu` (measured, 2.13.0): the shim decomposes
  `linear` before export sees it. After `run_decompositions` upstream has
  `aten.permute, aten.addmm, aten.relu`, so step 2's node-for-node test will also catch `t`
  versus `permute` unless that difference is resolved first.
* Only Linear+ReLU and one buffer mutation were traced. LLM graphs (SDPA, rotary, KV cache)
  were not traced; their surface may add `_functionalize_replace`/`_commit_update`
  (`FunctionalTensor.replace_`), `_freeze_functional_tensor`, and the `*_scatter` inverses.
