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

## Step 2 (#333): the Functionalize kernel for non-mutating and view ops

Direction: leader decision 2026-11-15, agent proposal; not owner-confirmed.
Written with the code, before anything ran on a built shim (the round could
not build); every upstream fact below was measured on torch 2.13.0 with the
probes in `tests/bindings/test_functional_ops.py` and
`tests/export/test_functionalize_export.py`, which re-derive them on every run.

### What step 2 is

One generic kernel, `functional.rs::functionalize`, called by
`aten.rs::aten_dispatch_entry` after the CompositeImplicitAutograd Python
kernels and before the mode consult, for an op with a functional argument or
any op while Functionalize is TLS-active (included and not excluded); and
called directly by `OpOverload._op_dk(DispatchKey.Functionalize, ...)`,
whatever the TLS says. It is schema-driven (`bootstrap.py::_functionalize_plan`,
cached in Rust): an `(a!)` argument is a mutation (refused by name, step 3); a
return aliasing an argument is a view (a list return a multi-output view);
upstream's metadata queries are answered from the wrapper's own layout;
`lift`/`lift_fresh`/`lift_fresh_copy` wrap their plain argument; everything
else is pure. It unwraps every functional argument (inside lists too), runs
the inner op **back through the door** with Functionalize in the TLS exclude
set (`bootstrap.py::_functionalize_redispatch`, upstream's
`AutoDispatchSkipFunctionalize`), and wraps the outputs when an argument was
functional or there was no tensor argument (upstream's `should_wrap_outputs`):
a fresh `FunctionalStorage` for a pure op, the input's storage plus a recorded
`ViewStep` (op, non-tensor arguments, output index, `reapply_views`) for a
view. `_is_functional_tensor_base` is now "no view steps", replacing step 1's
`is_base`. `_shim_functionalize_counters()` counts what the kernel did, so a
test can tell the kernel's answer from any other path's (AGENTS.md 13.1).

The TLS the kernel reads is mirrored into Rust by property setters on
`bootstrap.py::_Tls` (`included`, `excluded`, `reapply_views`), so the door
reads one thread-local byte rather than calling Python per op.

### The dispatch order, measured

| call | who sees it, in order | argument kinds | Functionalize included |
|---|---|---|---|
| a pure op under `FunctionalTensorMode` (with Proxy and Fake below, as in AOT) | Functional, then Proxy, then Fake | `FunctionalTensor`, then the unwrapped fake tensors | no, then yes |
| a view op, same | Functional, then Proxy, then Fake | same | no, then yes |
| a factory op (`ones`), same | Functional, then Proxy, then Fake | none | no, then yes |
| an op on a C wrapper, no mode, a user mode active | the user mode only, with the inner op | the unwrapped plain tensors | no |
| a view on a C wrapper, `reapply_views` False | the user mode sees `<view>_copy` | plain | no |
| `run_decompositions()` of Linear+ReLU | Functional receives `linear` (decomposed by its CIA), `t`, `addmm`, `relu`, twice (metadata pass, then the trace); Proxy decomposes `t` to `transpose.int` to `permute` | | |

Upstream's Python key shows every mode Functionalize excluded, functional
tensor or not (measured on `ones([2])` with nothing functional anywhere); the
shim's mode path does not emulate that, so the tests compare the other flags.

### Measured corrections to step 1 and to the note below

* **AOT does not reach the kernel on the shim yet.** Step 1's claim that the
  metadata pass stops at permute's `_op_dk(Functionalize)` was reasoned, not
  run: `FunctionalTensor.__new__` (functional_tensor.py:198) calls
  `torch.Tensor._make_wrapper_subclass`, which is a surface stub here
  (`surface.json` lists it, nothing implements it), and `to_functional` builds
  a `FunctionalTensor` for every input before the first op. That is the wall
  `tests/export/test_functionalize_export.py` expects first. Behind it, read
  but not run: CompositeImplicitAutograd decomposition of aten ops
  (`_dispatch_has_kernel_for_dispatch_key(op, CompositeImplicitAutograd)` is
  False for aten on the shim, so `OpOverload.decompose` returns
  NotImplemented and `aten.t` would stay `t` where upstream's Proxy gives
  `permute`), and section 6's `t, addmm` versus `linear` export difference.
* **Through the door versus through `_op_dk`.** Autograd and
  `ADInplaceOrView` sit above Functionalize: through the door a view's
  wrapper is an autograd view of its input wrapper (`_base` is the input's
  root) and its unwrapped value is not a view; through `_op_dk`, which skips
  them, the wrapper is not an autograd view and the value is one (with
  `reapply_views` True). The kernel reproduces both (`Entry::Door` /
  `Entry::OpDk`; the door's `note_view` on the result, and a
  below-`ADInplaceOrView` flag on the inner call).
* **`reapply_views`** decides the inner op of a view: the view itself when
  True (what `FunctionalTensorMode` sets), its `*_copy` variant when False
  (the TLS default), and the wrapper's layout is the inner value's. The shim
  has no kernel for any `*_copy` op, so with `reapply_views` False a view on a
  wrapper refuses by name; `FunctionalTensorMode`'s path never needs them.
* **View classification is the schema's, with upstream's exceptions:**
  `detach` is a functional view but not an autograd one; `_unsafe_view` has no
  alias annotation and is a functional view (upstream's own kernel);
  `reshape` of a non-contiguous tensor is upstream's `clone` then
  `_unsafe_view` (not a base, not sharing the input's storage), and a view op
  the shim does not decompose that returns `self` returns the input wrapper.
* **A factory op's output is functional** under `_op_dk(Functionalize, ...)`
  even with the key not TLS-included, and with the key included through the
  door; with only plain tensor arguments it is plain.
* **`_functionalize_unsafe_set(dst, src)` swaps only the storage** (value,
  layout and base-ness stay; a later mutation of `dst` counts on `src`'s
  storage). Upstream also accepts a plain tensor on either side; the shim
  refuses that by name (a plain dense tensor's storage is a candle buffer).
  Its caller, `return_and_correct_aliasing`, passes `FunctionalTensor`s
  (wrapper tensors); corrected on 2026-11-16 below: two wrappers are now
  answered.
* **A wrapper's `untyped_storage()`** is its `FunctionalStorage`: shared by a
  base and its views (`StorageWeakRef` equal), sized as the first value's
  storage, on the wrapper's device, `data_ptr()` refused ("invalid python
  storage").
* **The metadata queries** (`sym_size.int`, `stride.default`, `dim.default`,
  `is_contiguous.default`, ...) dispatch nothing below Functionalize; the
  kernel answers them from the wrapper's layout.
* **`_enable_functionalization` is not on AOT's path** (it includes the key
  with `_ForceDispatchKeyGuard`), so it stays refused by name. (Superseded by
  section 5.6: `suspend_functionalization` restores the key with it inside
  AOT's path, and it is now upstream's TLS write.)

### Corrections from the first CI run (2026-11-16)

Agent proposal, not confirmed. Bundle run 43727860553 (Linux and macOS
identical) ran step 2 on a built shim for the first time. Every upstream fact
below was measured on torch 2.13.0 (spike-venv) on 2026-11-16; the shim
changes were checked with `cargo check` only and have not run.

* **The unwrapped value of a door-entered view is marked, not left bare.**
  Measured for `view`, `permute`, `t`, `unsqueeze`, `f[1:]`, `f[1]` and
  `split` with `reapply_views` True: `_from_functional_tensor(g)` shares
  storage with `_from_functional_tensor(f)`, `_is_view()` is False, `_base` is
  None, `_get_creation_meta` raises `meta != nullptr`, and a view made of it
  later has it as its `_base`. Through `_op_dk` the value is a view (its
  `_base` is the input's root). With `reapply_views` False the value is a
  fresh copy either way. The shim skipped the view hook for the inner call, so
  a value with a storage offset (`f[1:]`) fell to the layout rule and `_base`
  refused. The hook now records `ViewMeta { autograd: false }` there
  (`tensor.rs`), which answers upstream's four facts.
* **The `_copy` lookup was broken, not absent.** `_functionalize_plan` asked
  for overload `default` in a table that spells the unnamed overload `""`, so
  `view`, `t`, `detach` and every other `.default` view read as having no
  `_copy` variant, and the refusal said "has none". Upstream has
  `aten.view_copy.default`, `t_copy.default` and `detach_copy.default`. With
  the mapping, `reapply_views` False refuses naming the `_copy` op, which has
  no kernel in the shim (none of the view `_copy` ops has one).
* **A Python number as `other` of `aten.add.Tensor`.** Upstream accepts a
  number in a `Tensor` position only for the names in
  `should_allow_numbers_as_tensors` (`add`, `sub`, `subtract`, `mul`,
  `multiply`, `div`, `divide`, `true_divide`, `floor_divide`, their in-place
  and out forms, `to`, `_to_copy`, `copy`, `copy_`, `_conj`); `remainder`,
  `pow`, `eq` and the rest refuse ("Expected a value of type 'Tensor'"). So
  `remainder` and `pow` are not in the fix. For the named ops the
  `Tensor`-with-number call equals the `Scalar` sibling over 1512 cases (seven
  dtypes, six numbers, `alpha`, every `rounding_mode`, in-place identity and
  refusal text), and a `TorchDispatchMode` sees `aten.add.Tensor` with the
  Python number untouched. The kernel table runs the sibling
  (`aten.rs::number_as_tensor_sibling`), below the modes and the
  Functionalize kernel. Not covered: a number as `self` (upstream accepts
  `aten.add.Tensor(2, t)`), the `out=` forms, and `to`, `_to_copy`, `copy`,
  `copy_`, `_conj`.
* **`torch._sync` is Python upstream** (`torch._utils._functionalize_sync`,
  imported at `torch/__init__.py:50`; `hasattr(torch._C._VariableFunctions,
  "_sync")` is False). The stub `_VariableFunctions.pyi` lists it, the shim
  made a refusing member from that, and `torch/__init__.py:2212` copied the
  member over the Python function. The shim no longer has the member.
* **`_functionalize_unsafe_set` on AOT's path takes two wrapper tensors**, not
  meta or functional ones: `return_and_correct_aliasing` passes the
  `FunctionalTensor`s (`_make_wrapper_subclass` instances) a view op returns
  and receives. Measured: a wrapper's `untyped_storage()` has an identity of
  its own, `nbytes()` `itemsize * (storage_offset + 1 + sum((size - 1) *
  stride))` (0 with a zero-size dimension), device the wrapper's, and
  `data_ptr()` raises "Attempted to access the data pointer on an invalid
  python storage."; after `_functionalize_unsafe_set(dst, src)` `dst` reports
  `src`'s storage and `nbytes`, its layout unchanged; under
  `FunctionalTensorMode` `ft.view(-1)` and `ft[1:]` share `ft`'s storage and
  `ft + 1` does not. `Repr::Wrapper` now carries that identity,
  `untyped_storage()` answers it (the refusal #340 chose had no handle with a
  refusing `data_ptr` on a non-meta device; #333's `storage::functional` is
  one), and the swap is implemented for two wrappers. A dense or meta side,
  or two different kinds, still refuses by name. A wrapper's identity does not
  feed the door's view hook, so a `FunctionalTensor` view is still not an
  autograd view of its input on the shim (upstream: `ft.view(-1)._base is ft`).

Next walls for `run_decompositions()` of Linear+ReLU, read, not run, in
order: (1) `collect_metadata_analysis.py:303/333/462` key inputs and outputs
on `FunctionalTensor.untyped_storage()`, now answered; (2) `linear`: upstream's
`FunctionalTensorMode` decomposes it through `func.decompose` (its CIA
kernel), and the shim's `OpOverload.decompose` returns NotImplemented for aten
ops, so `linear` reaches `_op_dk` whole; (3) the shim's own export records
`aten.t, aten.addmm` where upstream records `aten.linear`, so the ops
`Functional` receives differ from the first one; (4) Proxy decomposes `t` to
`transpose.int` to `permute` upstream, through the same CIA question; (5) a
view's `FunctionalTensor` is not an autograd view on the shim (only matters
for an output that aliases). For `make_fx(dispatch_functionalize(fn))`: past
`torch._sync`, the next difference by reading is the log row `['Fake',
'aten.as_strided.default', ...]` upstream's `FakeTensorMode` emits for
`view`, which depends on the shim's fake view path.

### Walls (2) to (4) above, after #372 (2026-11-16)

Agent proposal, not confirmed; the direction (CIA decomposition before step 3)
is a leader decision 2026-11-16, agent proposal; not owner-confirmed. CI run
43739516306 showed the Linear+ReLU `run_decompositions()` test already `ok`,
because the shim's export never recorded `linear`. #372 adds the
CompositeImplicitAutograd frame (docs/design/DECOMPOSE.md): `F.linear` enters
the door as `aten.linear.default`, the shim's export records it as upstream's
does, and `FunctionalTensorMode` decomposes it through `OpOverload.decompose`
and `_op_dk(CompositeImplicitAutograd)`, which now runs the body. Wall (4) was
misread: `aten.t` is CompositeExplicitAutograd upstream, not CIA, and Proxy's
decomposition table turns it into `permute` on both sides. Written, compiled
(`cargo check`), not yet run.

### What step 3 needs

`_make_wrapper_subclass` (or this whole road stays unexercised on the shim;
it is a binding of its own and every wrapper subclass needs it); then, for
mutation: the functional variant of each mutating op by schema (`add_` to
`add`, `out=` to the functional overload; measured: upstream's inner op for
`add_` is `add.Tensor`, for `t_` is `t_copy`, for `copy_` is `copy.default`),
replacing `value`, bumping the storage generation and `mutation_counter`, the
hidden-from-autograd rule (FunctionalStorage's comment says why it is not a
plain counter equality), `_functionalize_replace`, `_functionalize_commit_update`,
`_functionalize_mark_mutation_hidden_from_autograd`; a metadata mutation
(`t_`, `resize_`) replaces the whole payload. Step 4 replays a lagging view's
`view_steps` from the storage's base, which step 4 must add to
`FunctionalStorage` (it is not declared yet because nothing reads it).

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
  **Wrong, corrected by wall W8 (section 5.1).** exir does read it.

### 5.1 W8 and the view-copy wall (CI run 43793269977, 2026-11-16)

Agent proposal, not confirmed. Two walls the probe met after W6, and neither is about
functionalization:

* **`to_edge` stopped on `aten.permute_copy.default`.** Not `reapply_views` False:
  executorch's `_generate_edge_program` (`exir/program/_program.py:799`) runs torch's
  `ReplaceViewOpsWithViewCopyOpsPass`, which re-runs every view node of the exported graph as
  its `_copy` variant (`get_view_copy_of_view_op`) under `FakeTensorMode`. On the shim the meta
  attempt raised `NotImplementedError` (`fake_tensor.py:3088`) and so did the real-tensor
  fallback (`run_fallback_kernel`), because the door had no `_copy` kernels. Upstream answers
  from the meta kernel (`has_meta` True; `run_fallback_kernel` never called, measured). Linear +
  ReLU reaches `permute_copy`; a small transformer block reaches `view_copy`, `permute_copy`,
  `expand_copy`, `select_copy.int`, `slice_copy.Tensor`, `split_with_sizes_copy`,
  `squeeze_copy.dims` and `unsqueeze_copy` (measured on 2.13.0). Change: `aten.rs::VIEW_COPY_OPS`
  and `view_copy`, the view routed as a direct call would be, then a fresh contiguous copy
  (dense `force_contiguous`, meta a fresh contiguous meta tensor, every other representation
  refused by name), for the nineteen pairs whose view the shim implements.
* **The XNNPACK stage's edge verifier reported float32 against float32 as a mismatch for
  `aten.linear`.** `exir/dialects/edge/spec/utils.py` decides which arguments are tensors by
  `str(arg.type) in ["Tensor", "Optional[Tensor]", "List[Tensor]"]`; the shim's `str` was the
  schema spelling `Tensor?`, so `bias` was missing from the op's dtype table and
  `FunctionDtypeConstraint.validate` returned False at its first check (an argument name the
  table does not hold). The dtypes themselves were never the problem: the shim's `torch.dtype`
  compares and hashes by value and its aliases are one object. Change: `_SchemaType.__str__`
  and `__repr__` return `annotation_str`, which equals upstream's `str` for all 19600 types of
  `_jit_get_all_schemas()` (measured), and every reader of the spelling (`_Argument`, the
  binder, `_functionalize_plan`, `aten.rs::reshape_to_schema`'s list-return test) reads
  `_spelling` or upstream's text.

Neither has run on a built shim.

### 5.2 After CI run 43802210944 (2026-11-16)

Agent proposal, not confirmed. That run built both changes above. `to_edge` gave upstream's
three edge operators and the edge dtype table for `linear` was upstream's. What it showed next:

* **A `split_copy`/`unbind_copy` element was a bare `TensorBase`.** The door's exit
  `promote` looks at the returned object only, not inside a list, so `view_copy` now promotes
  each element as every multi-output kernel does (`test_functional_ops`, `value_type`).
* **`unfold_copy` and `view_copy.dtype` on meta refuse**, because the shim has no meta kernel
  for `aten.unfold.default` or `aten.view.dtype`: the copy is its view followed by a copy. The
  test now derives that outcome from the view's own refusal on meta, measured in the same probe,
  and asserts that no op on ExecuTorch's measured path is among them.
* **`to_executorch` stopped with "Missing out variants: {'aten::relu',
  'aten::permute_copy'}".** `ToOutVarPass` finds an edge op's `.out` through
  `torch._C._jit_get_operation(name)[1]` and torchgen's parse of each overload's schema
  (`exir/dialects/edge/op/api.py:to_variant`). Upstream lists `['', 'out']` for both; the shim
  listed `['']`, because both `.out`s are `autogen:` entries of `native_functions.yaml`, which
  the schema scan does not read. `addmm.out` is a `- func:` line and was found. The two schemas
  are transcribed into `bootstrap.py::_GENERATED_ATEN_SCHEMA_TEXT` (upstream's
  `str(_schema)`). The shim has no kernel for either; calling one refuses by name in the door
  ("aten op not implemented in torch._C shim: aten.relu.out"), and nothing after the pass in
  `to_executorch` runs the graph. Tags are not transcribed: the shim answers `[]` for both,
  upstream `generated, out, pt2_compliant_tag` (and `view_copy` for `permute_copy.out`).
* **The XNNPACK stage stopped in its first pass, `ConvertToUpsampleBilinear2d`,** with the root
  exception not in the log: exir's pass manager re-raises as a bare `Exception` naming the pass.
  Before it looks at our graph, that pass builds its patterns
  (`backends/xnnpack/partition/graphs/bilinear_2d.py`) by running
  `torch.export.export(..., strict=True)` on an `F.interpolate(mode="bilinear")` module, then
  `to_edge`. Strict export is Dynamo's bytecode capture, and this shim refuses the frame hook
  Dynamo installs for `torch.compile` (`set_eval_frame`, docs/graph/COMPILE.md); upstream's
  strict export of that module gives one `aten.upsample_bilinear2d.vec` node (measured on
  2.13.0). Which Dynamo piece stops on the shim is NOT MEASURED. `exir_probe.py` now records the
  cause chain of every stage error and a report-only stage, `strict_export_interpolate`, that
  runs that capture alone, so the next run names the root. Not fixed: a larger wall than this
  section's, and not guessed at.

None of this has run on a built shim.

### 5.3 After CI log r382c (2026-11-16): `as_strided_`, and what the strict export needs next

Agent proposal, not confirmed. None of this has run on a built shim.

* **The root was named: `TensorBase.as_strided_`.** The cause chain ends in
  `torch/_dynamo/utils.py::clone_input` (via `builder.py::wrap_tensor` ->
  `cache_real_value_when_export` -> `_clone_input`), which restrides a fresh flat buffer in
  place before copying the input into it. So the strict export is past capture's entry: on
  2.13.0, `torch.export.export(strict=True)` with no dynamic shapes takes
  `use_new_tracer_experimental` (`torch/_export/config.py`, True) and
  `dynamo_graph_capture_for_export` -> `convert_frame.fullgraph_capture`, which interprets the
  function's bytecode in Python and never calls `set_eval_frame`. The paragraph above that
  blamed the frame hook was a reading, and the log contradicts it: Dynamo's builder ran.
  `as_strided_` is implemented now (docs/kernels/STRIDED.md §8), never run.
* **What the strict export reaches next, by reading only (not measured), most likely first:**
  1. Fake propagation of `aten.upsample_bilinear2d.default`. The shim's
     `torch._C._nn.upsample_bilinear2d` (`bootstrap.py`) computes the output size in Python and
     dispatches `.default`; under `FakeTensorMode` that op has no Rust meta arm, so it goes
     through upstream's Python decomposition `_upsample_linear` (`torch/_decomp/decompositions.py`)
     or its `activate_meta` registration, which uses `arange`, `clamp`, `reshape`, `.to(int64)`,
     `aten._unsafe_index` with `None` entries (decomposed to `aten.index.Tensor`), `sub`/`mul`/
     `add`, `suggest_memory_format` and `contiguous(memory_format=)`. The first of those without
     a meta arm in `aten.rs::meta_table` stops it, or, if `FakeTensorMode` takes its fallback,
     the dense CPU kernel runs on zeros. Size: one op family, from one to a few meta arms.
  2. A graph that differs from upstream's without stopping: upstream's strict export records
     one `aten.upsample_bilinear2d.vec` node (measured); the shim's binding records `.default`
     with a concrete size. `ConvertToUpsampleBilinear2d` builds its pattern from that graph and
     matches it against ours, so it can still complete; whether `to_edge` of the pattern graph
     accepts `.default` is not known.
  3. Dynamo's own reads of the wrapped input in `wrap_tensor` after `clone_input`
     (`_is_view()`, `is_contiguous(memory_format=...)`, the `torch._C._functorch` predicates,
     `torch._C._dynamo` guard objects). The non-legacy export path does not build guards
     (`functional_export.py` builds them only in `_dynamo_graph_capture_for_export`), so guards
     are the least likely.
  `tests/export/test_exir_shim.py` prints the strict stage's error and cause chain in the
  XNNPACK test's failure, so the next CI run names which one it is.

### 5.4 After CI run 43826381567: `aten._unsafe_index`, and packets with no `default`

Agent proposal, not confirmed. Compile-checked only; none of this has run on a built shim.

* **The wall, from the log.** `as_strided_` passed. The strict export's Dynamo builder then ran
  `F.interpolate(mode="bilinear")` on fake tensors and stopped on
  `NotImplementedError: torch._C shim has no meta kernel for aten._unsafe_index.default`.
  FakeTensorMode prefers a Python decomposition over the shim's meta arm for
  `aten.upsample_bilinear2d.default` (`fake_tensor.py`: an op in `decomposition_table` from
  `torch._decomp.decompositions` and not in the Python `meta_table`), as it does upstream, so
  it ran `_upsample_linear`, which calls the packet `aten._unsafe_index(input, [None, None,
  rows, cols])`. Upstream's packet matches the call to `_unsafe_index.Tensor` (its overloads
  are `Tensor` and `Tensor_hacked_twin`, no `default`; measured on 2.13.0), whose registered
  decomposition calls the packet `aten.index(x, indices)`, matched to `index.Tensor`, whose
  meta kernel answers. The shim's packet dispatched `<name>.default` for every packet, an
  overload upstream does not have, so no decomposition matched.
* **The fixes.** (1) `bootstrap.py::_packet_overload`: an aten packet with no `default`
  overload calls the one overload the call binds, dropping a `<base>_hacked_twin` whose base is
  listed. In `007f161` the binding was by arity only (positional count, keyword names, every
  argument without a default supplied), with none or several left keeping the old
  `<name>.default` key; argument kinds were added after the leader's review (section 5.5). Of
  the 336 implemented keys, none is a
  `.default` key of a name whose yaml entry has no `default` overload (counted from
  `aten.rs::IMPLEMENTED` and the vendored `native_functions.yaml`), so no call that worked
  changes key. (2) `aten._unsafe_index.Tensor` is implemented as `index.Tensor`'s kernel, dense
  and meta (one shared meta arm; `meta_stride_rule` leaves it `Unverified` like `index.Tensor`,
  which accepts the contiguous operands `_upsample_linear` passes). Upstream's eager kernel
  checks bounds too (`IndexError: index 7 is out of bounds for dimension 0 with size 4`,
  measured). Its schema is not in `overloads.json`: `Tensor?[] indices` is a type
  `_TypeChecker` cannot express (the `index_put_` note in `overloads.json`), so the entry would
  stop `import _C`; the yaml answers its schema. So `torch._unsafe_index` has no door here
  (upstream's binds `Tensor[]`, the hacked twin).
* **The rest of `_upsample_linear`, by reading and one upstream measurement.** A
  `TorchDispatchMode` under upstream's decomposition of `upsample_bilinear2d.default` and
  `.vec` logs `_to_copy.default`, `_unsafe_index.Tensor`, `add.Tensor`, `arange.default`,
  `clamp.default`, `mul.Tensor`, `sub.Tensor` and `view.default`; `contiguous(memory_format=)`
  dispatches nothing for an already contiguous result, and `prim.device` is answered by
  FakeTensorMode. Every one of the eight has a meta arm in `aten.rs::meta_table`
  (`_unsafe_index.Tensor` through `index.Tensor`'s). The index operands are `(8, 1)` and `(8,)`
  int64, contiguous, which the `Unverified` stride gate accepts. Not checked: whether each of
  those arms promotes `int64 - float32` and broadcasts `(1, 3, 8, 8)` against `(8, 1)` as
  upstream's do (`tests/ops/test_unsafe_index.py::test_fake_interpolate_answers_upstreams_shape`
  compares the result, not each step).
* **`upsample_bilinear2d` itself.** The shim has a dense kernel and a meta arm for
  `aten.upsample_bilinear2d.default` and none for `.vec`; its `torch._C._nn.upsample_bilinear2d`
  binding computes the output size in Python and dispatches `.default` (`bootstrap.py`), where
  upstream's F.interpolate records `.vec` (§5.3, measured there).
* **The predicted next wall (reading only, not measured).** The fake propagation should now
  complete. What the strict export records next differs from upstream: `.default` with a
  concrete size where upstream records one `aten.upsample_bilinear2d.vec` node. The two are
  different kinds of op upstream (measured on 2.13.0: `.vec` is CompositeImplicitAutograd and
  tagged `core`; `.default` has a CPU kernel, no CIA kernel, and is tagged only
  `pt2_compliant_tag`), so `to_edge`'s core-ATen decomposition is expected to keep one and
  decompose the other into the `_unsafe_index` pattern, and the shim's pattern graph and
  upstream's would differ node for node. The most likely next stop is therefore
  `ConvertToUpsampleBilinear2d` or the XNNPACK partitioner seeing a different graph (no match,
  or a different delegated op), rather than an exception; which of the two kinds is kept was
  not measured. If an exception comes first, the likely one is a meta arm among the eight
  above disagreeing on a promoted dtype or a broadcast stride.

<!-- DOCWATCH: op-implemented aten._unsafe_index.Tensor -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/bootstrap.py _packet_overload present -->
<!-- DOCWATCH: symbol-in-file tests/ops/test_unsafe_index.py test_fake_interpolate_answers_upstreams_shape present -->

### 5.5 Packet resolution: which overload `torch.ops.aten.<name>(...)` calls

Provenance: leader decision 2026-11-16 (not the owner's words), approving `007f161`'s change with
the condition that the binding also checks argument kinds as upstream's resolution does, that a
call several overloads accept refuses by name, and that the packets whose behaviour changes are
listed. Written and compile-checked (`python -m py_compile`), not run on a built shim. Upstream
measured on torch 2.13.0, 2026-11-16, with `torch._C._jit_resolve_packet` and a
`TorchDispatchMode`.

**Upstream's rule, measured.** A packet is TorchScript's `getOpWithStack`: the overloads are tried
in registration order and the first whose schema accepts the call runs. The checks for one
overload, in order: more arguments than parameters (`expected at most N argument(s) but received
M argument(s)`, keywords counted), more positional arguments than positional parameters (`takes N
positional argument(s) but M was/were given`), then each parameter in schema order (a missing one
without a default: `is missing value for argument 'x'`; a value of the wrong kind: `Expected a
value of type 'int' for argument 'dim0' but instead found type 'str'.` with `Position`, `Value`
and `Declaration` lines), then a keyword no parameter has (`Unknown keyword argument 'x' for
operator 'aten::empty'`). With no overload accepting, the call raises `RuntimeError`: the one
error alone for a packet with one overload (`aten.transpose` has only `.int` upstream; the
`Dimname` overloads are not TorchScript operators), otherwise `Overloaded torch operator invoked
from Python failed to match any schema:` followed by every overload's error. TorchScript's kinds
differ from `python_arg_parser`'s (which `torch.<op>` uses):

| value | TorchScript (packets) | `_TypeChecker` (`python_arg_parser`) |
|---|---|---|
| `True` for `int` | accepted (`transpose(t, True, 1)` computes) | refused |
| `np.int64(0)` for `int` | accepted | refused |
| a 0-dim tensor for `Scalar` (`number`) | refused (`fill_(t, tensor(1.))` takes `fill_.Tensor`, declared after `.Scalar`) | accepted |
| `np.float32(2)`, `np.int64(2)` for `Scalar` | refused | accepted (`numbers.Number`) |
| a Python number for `Tensor` | accepted for the packets `add_`, `sub_`, `mul_`, `div_`, `multiply`, `subtract`, `divide`, `true_divide`, `floor_divide_` (measured over every packet with a `.Tensor` and a `.Scalar` overload and no `default`), refused elsewhere | refused |
| `None` for `Tensor` | accepted (`rsub(t, None)` resolves `.Tensor`) | refused |
| a bare tensor for `Tensor?[]` | accepted (`index(t, tensor([0]))` computes) | refused |
| `1` for `bool`, `6` for `ScalarType`, `0` for `Layout`/`MemoryFormat` | accepted | refused |
| `0` for `Device` | refused | refused |
| a one-element integral tensor for `int` | accepted (read with `_local_scalar_dense`; a float one too) | accepted (integral or bool only) |

**The shim's rule.** `_packet_bind` reproduces the count, positional and keyword checks in
upstream's order and words, and checks each value with `_TypeChecker.predicate_for`, the
predicates `torch.<op>` resolution uses (one copy, not a second), plus three measured adjustments
that make the packet answer TorchScript's: `Scalar` takes only Python's own numbers (`bool`,
`int`, `float`, `complex`; not a tensor, not a NumPy scalar that is not a Python number), the
`_packet_numbers_as_tensors` packets take a Python number for `Tensor`, and `Tensor?[]` takes
`None` elements (`_TypeChecker._base_predicate` learned the `X?` element). Then:

* exactly one overload accepts: it is called (`aten.<name>.<overload>` through the one door);
* none accepts: `RuntimeError` with upstream's text for the shim's overloads (the `Cast error
  details` line is left out), and a last line saying the kinds are `_TypeChecker`'s;
* several accept, or an overload cannot be checked (a placeholder schema, or a kind
  `_TypeChecker` has no rule for: `t`, `Dimname`, ...): `NotImplementedError` naming the packet,
  the overloads and the reason.

**Limits, each a narrowing (the shim refuses where upstream computes), never a different
answer:** a call several overloads accept (`add_(t, 2)`: `.Tensor` and `.Scalar`;
`repeat_interleave(t, tensor([2]))`: `.self_Tensor` and `.self_int`), because upstream's choice is
its registration order and the shim's list is `native_functions.yaml`'s order (`bitwise_and`:
`Tensor_out` first here, `Tensor` first upstream); and the TorchScript-only acceptances in the
table above (`bool` for `int`, `None` for `Tensor`, a bare tensor for `Tensor?[]`, ints for `bool`
and the enums). Overloads upstream registers that the shim does not list (`select.t`,
`index.str`, `pow.int`, the generated `.out` variants the yaml lacks) are not candidates here, so
a call only they accept (`aten.pow(2, 3)`, `aten.select([1, 2], 0)`) raises the no-match
`RuntimeError` where upstream computes. A packet with a `default` overload was meant to be
untouched (it dispatches `<name>.default`); it was not, and section 5.6 is what that cost and
the rule that replaced it.

**What changes.** 49 packets have no `default` here and an implemented overload (computed from
`aten.rs::IMPLEMENTED`, `IMPLEMENTED_AWAITING_GOLDEN`, `native_functions.yaml` and upstream's
registry): 36 have none upstream either, and for 13 (`add`, `sub`, `mul`, `div`, `eq`, `ne`, `lt`,
`le`, `gt`, `ge`, `fmod`, `remainder`, `split`) upstream's `default` is a TorchScript op registered
after the native overloads (`aten::add(Scalar a, Scalar b) -> Scalar`), which the shim does not
list. Before `007f161` every call of the 49 dispatched `aten.<name>.default` and refused. The
calls that now compute, and which of them `007f161`'s arity rule already reached, are listed
in the pull request body (#384). `tests/bindings/test_packet_overload.py` computes that list from
the shim's own tables, prints it, and compares every packet's resolved overload with upstream's.

<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/bootstrap.py _packet_numbers_as_tensors present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_packet_overload.py test_packets_resolve_upstreams_overload present -->

### 5.6 CI run 43841297493: packets with a `default` read as packets without one

Agent proposal, not confirmed. Compile-checked only (`cargo check --tests`, Python `compile`);
nothing here has run on a built shim. Upstream measured on torch 2.13.0 (spike-venv),
2026-11-16.

* **The defect.** `_jit_get_operation` sent a packet down the no-default path when
  `"default"` was not in its overload list, but `native_functions.yaml` spells the unnamed
  overload `""` (`_ATEN_OVERLOADS_BY_NAME`). Every yaml packet with a default took the new
  path: `aten.im2col(...)`, `aten._unique2(...)`, `aten.alias(x)`, `aten.lift_fresh(x)` resolved
  `""` and dispatched `aten.<name>.` ("aten op not implemented in torch._C shim: aten.im2col.");
  `aten.upsample_nearest1d(x, [8], None)` bound `""` and `.vec` and refused as ambiguous; the
  decomposition census lost `_unsafe_view` and `detach` (`RAISED_INSIDE`), which call packets.
* **Upstream's rule for these packets, measured.** The overloads are tried in registration
  order. For a yaml packet the default is registered first, except for fifteen packets whose
  `overloads()` list other non-out overloads ahead of it (`range.step`, `sum.dim_IntList`,
  `max.other`, `where.self`, ... : `_PACKET_BEFORE_DEFAULT`); a default that is not in the
  yaml is a TorchScript op registered after every native one (`add`'s is sixteenth of
  sixteen). A marker kernel on `upsample_nearest1d.vec` showed `(x, [8], None)` (both bind)
  running `.default`, and `(x, None, [2.0])` and `(x, output_size=[8], scale_factors=None)`
  (only `.vec` binds) running `.vec`; `_jit_resolve_packet` answers `step` for `range(0, 5)`.
  All 217 implemented `aten.<name>.default` keys resolve to `default` upstream for a call
  built from the default schema's required arguments, positionally and by keyword.
* **The shim's rule** (`bootstrap.py::_packet_overload_with_default`). The yaml's `""` is a
  default. A packet whose only overload is its default dispatches `<name>.default` as before.
  Otherwise the walk follows upstream's order: an overload ahead of the default that binds is
  chosen; else the default unless the call is a definite mismatch for it; else the one later
  overload that binds. It never refuses. Wherever it cannot prove upstream's choice (an
  undecided kind, no overload or several later overloads binding) it answers `default`, the key
  every such call had before this branch, and the default kernel's own checks answer. One more
  fallback keeps every call that computed before: when upstream's choice has no kernel here and
  `default` has one, the call keeps `default` (`linspace(tensor(0.), tensor(1.), 5)` is
  `Tensor_Tensor` upstream and stays `linspace.default`; `upsample_nearest1d(x, None, [2.0])`
  is `.vec` upstream and reaches `.default`'s kernel, which refuses `None` as it did).
* **Known differences, each a call that keeps its old key:** the fallbacks above; `range(0, 5)`
  now asks for `range.step`, which upstream chooses and the shim does not implement (it
  refused naming `range.default` before; neither key is implemented).
* **Tests** (`tests/bindings/test_packet_overload.py`):
  `test_every_implemented_default_packet_resolves_to_default` (every implemented `.default`
  key, printed, against `_jit_resolve_packet`, through the packet's own `""`-spelled list),
  `test_the_ci_packets_dispatch_their_default` (the packets this run refused, called for real
  under a mode that stops at the first op) and
  `test_packets_with_a_default_choose_upstreams_overload` (the measured two-overload rows).
  Nullifications, reasoned and not run: reading only `"default"` again turns the first two red
  (`""` returned, `aten.im2col.` dispatched); dropping `_PACKET_BEFORE_DEFAULT` turns the
  third's `range|two` row red (both bind; `sum|dim` and `max|other` do not move, since
  `default` rejects those calls); dropping the `chosen` fallback turns
  `upsample_nearest1d|vec_keywords` red only if `.vec` gained a kernel, so it is held by the
  first test's `default` answers instead.
* **`_enable_functionalization`** (the XNNPACK wall of the same run). `ConvertToSDPAPass`
  builds its pattern by exporting an SDPA module and running `to_edge`
  (`executorch/backends/xnnpack/partition/graphs/sdpa.py`); inside that `run_decompositions`
  (Functionalize included by `_ForceDispatchKeyGuard`), `meta_utils.py` makes a fake tensor
  under `torch/_dispatch/python.py::suspend_functionalization`, which takes the key out with
  `_disable_functionalization` and puts it back with `_enable_functionalization`, refused here
  by name until now. Upstream's is a TLS write and nothing else (measured: it includes the key,
  sets `reapply_views` only when asked, raises its internal assert when the key is already
  included, and takes an int but not a str), the inverse of `_disable_functionalization`, which
  the shim already answered. It is answered the same way now; the kernel it turns on is the
  door's (#333), which refuses a mutation by name. What stops next was not measured.

<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/bootstrap.py _packet_overload_with_default present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/bootstrap.py _PACKET_BEFORE_DEFAULT present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_packet_overload.py test_every_implemented_default_packet_resolves_to_default present -->

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
