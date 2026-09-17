# `torch.export` on real architectures: eight walls, measured and closed

`docs/graph/EXPORT5.md` §10 measured `torch.export` at **0 of 40** real
`transformers` architectures and named three walls. This document is that
round's successor. It reports where the sweep stops **now**, op by op and
architecture by architecture, so the next round starts from a measurement
rather than re-deriving one.

Two things to read first, because they decide how everything below should be
taken:

* **The headline is still 0/40 under the bar that matters.** No architecture
  exports, replays, and agrees element-wise yet. What changed is *where the
  forty stop*: the twenty-six export-time failures used to be three walls deep
  and are now eight walls further in, on a completely different set of causes.
  **Six defects were fixed and six capabilities added** on the way there
  (itemised in §7). Counting those as "0 → 0" is as misleading as counting them
  as progress, so §1 reports where the forty stop as well as the headline.
* **Every claim here was re-measured, not inherited.** EXPORT5's own counts
  were reproduced first (§1.1) before anything was changed.

---

## 1. The numbers

### 1.1 EXPORT5's measurement, reproduced

`export_sweep.py --limit 40`, shim side, on `develop` before any change:

```
architectures swept   : 40
export                : 26
forward               : 10
construction          :  4

of the 26 export failures:
  14  AttributeError: '_SchemaType' object has no attribute 'annotation_str'
  11  NotImplementedError: TensorBase.set_: the storage has never been filled
   1  AssertionError: Could not find common device for aten.empty_strided.default
```

Identical to `docs/graph/EXPORT5.md` §10 in every figure. **EXPORT5 §10 is not
stale.**

### 1.2 After this round

```
architectures swept   : 40
export                : 26      <- unchanged in total
forward               : 10      <- untouched this round
construction          :  4      <- untouched this round
exported+replayed+agreed : 0/40 <- unchanged
```

and the twenty-six now stop here:

| n | wall | kind |
|---:|---|---|
| 10 | `torch.var_mean(...)` — overload resolution has no table entry | missing op |
| 5 | `torch._C._select_conv_backend` | missing surface |
| 5 | `no meta kernel for aten.as_strided.default` | the stride question (§6) |
| 3 | `aten op not implemented: aten.lift_fresh_copy.default` | missing op |
| 2 | `FakeTensorDeviceMismatchError` | unanalysed |
| 1 | `no meta kernel for prims.collapse_view.default` | missing meta kernel |

Not one of EXPORT5's three walls appears. Two are closed outright; the third
(`aten.as_strided` / the missing stride on `Repr::Meta`) is still open and is
now reached by five architectures instead of one — §6.

---

## 2. What was actually wrong: eight walls, in the order they appeared

Each wall was only visible once the one in front of it came down. That is why
EXPORT5 could name three and this round found eight: they were queued behind
`annotation_str`, which stops a trace within its first few operators.

| | wall | n | what it was |
|---|---|---:|---|
| 1 | `_SchemaType.annotation_str` missing | 14 | a schema surface that was never built |
| 2 | every argument reached a `TorchDispatchMode` as a **keyword** | 13 | collides with `OpOverload.decompose`'s bound `self` |
| 3 | `new_empty` / `new_zeros` had no meta kernel | 13 | `meta_embedding` is `weight.new_empty(...)` |
| 4 | `layout=torch.strided` refused | 10 | a request for the only layout the shim has |
| 5 | `pin_memory=False` refused | 10 | a request for nothing |
| 6 | `TensorType.get()` was identical to nothing | 10 | so no op was ever a tensor constructor |
| 7 | `Argument.default_value` was the schema's **source text** | 12 | `'None'`, the string |
| 8 | `set_` refused `meta_utils.py`'s metadata-only `set_` | 18 | EXPORT5's wall 2 |

**The `n` column is the count at the moment that wall was the blocker**, not a
partition of the twenty-six. They overlap heavily and they move: wall 8 was 11
architectures when EXPORT5 measured it and 18 by the time walls 1-7 were down,
because closing a wall in front of it let more architectures reach it. A column
that added up to 26 would be the wrong shape for a queue.

Walls 2, 6 and 7 are the interesting ones: all three are places where this shim
answered *conservatively* — a `False`, a string, a keyword — and the
conservative answer was not free. Each produced a failure several frames away
from its cause.

### 2.1 `annotation_str` (wall 1)

`torch/fx/operator_schemas.py:95` is

```python
return eval(ts_type.annotation_str, _type_eval_globals)
```

and `torch.export`'s normalisation reaches it for every argument of every op.
The mapping is not a reading: it was derived by parsing all 2584 `- func:`
entries of the vendored `native_functions.yaml` on **both sides** and pairing
argument by argument, which produced a total function of the spelling with
zero conflicts. `test_export6.py` re-derives the whole comparison on every run,
so a wrong entry is a failure rather than a drift.

The lossy rows are upstream's own: `ScalarType`, `Layout`, `MemoryFormat` and
`DeviceIndex` all annotate as plain `int`, because `_type_eval_globals` has no
name for any of them and a more faithful spelling would `eval` to `NameError`.

### 2.2 The keyword shape (wall 2)

`bootstrap.py` binds every argument by keyword whenever its fast path is not
taken (`dispatch(key, **bound)`), so `args` arrived as `()` and the whole call
was in `kwargs`. Eager dispatch does not care. `torch/_subclasses/fake_tensor.py:2970`
does:

```
r = func.decompose(*args, **kwargs)
TypeError: OpOverload.decompose() got multiple values for argument 'self'
```

`OpOverload.decompose` is `def decompose(self, *args, **kwargs)` and the first
argument of most aten schemas is *named* `self`. The call is now re-seated at
the mode boundary — schema-positional arguments in `args`, `kwarg_only` in
`kwargs` — and only at the mode boundary, so no mode-less number can change.

### 2.3 `default_value` was the source text (wall 7)

`has_default_value()` was `default_value is not None`, which forced
`default_value` to stay a string: 1112 of the file's arguments default to
`None`, and under that test every one of them would have answered "no default".
The three concepts are now separate — `default_source` (text),
`default_value` (value), `has_default_value()` (declared) — and all 9238
arguments are compared against upstream's answer.

The failure it produced:

```python
# torch/_subclasses/fake_impls.py:218
new_kwargs = {..., 'layout': 'None', 'device': 'None', 'pin_memory': 'None'}
r = func(*args, **new_kwargs)
RuntimeError: ... device type at start of device string: None
```

### 2.4 The TorchScript type singletons (wall 6)

`torch/_subclasses/fake_impls.py:159` decides "is this op a tensor constructor"
by **identity**:

```python
schema.returns[0].type is torch._C.TensorType.get()
```

Nothing this shim produced was ever that object. `_SchemaType`'s docstring
declined the correspondence on purpose, and was right to while it did not
exist. It exists now: `_SchemaType` is interned per spelling, `TensorType.get()`
returns the interned `Tensor`, and `_ShimMeta.__instancecheck__` reads the same
table so the five `isinstance(arg.type, torch.TensorType)` sites in the
vendored tree stop reading `False`.

The cost of the conservative answer was that `arange` fell through to the
generic fake path, which looks for a common device among arguments it does not
have.

### 2.5 `set_` on a meta tensor (wall 8, EXPORT5's wall 2)

`torch/_subclasses/meta_utils.py:2124` — the branch whose own comment says
"you're in crazy town" — builds a meta storage and `set_`s it onto a meta
tensor. No bytes exist on either side, so the `filled` check was asking a
question that cannot have a yes.

The narrowing is exactly the one EXPORT5 §10 named: **allow when both the
tensor and the storage are meta, refuse otherwise.** A dense receiver or a
dense storage still gets `docs/models/CKPT.md` §4's refusal, and
`test_export6.py` asserts that half beside this one — a test for the narrowing
alone would have passed against a shim that deleted the check.

Two things are refused by name rather than dropped: a non-zero
`storage_offset` (`Repr::Meta` has nowhere to put it) and a non-contiguous
stride (§6).

---

## 3. What this changed elsewhere, and why that is right

Three existing tests failed across the gate runs. All three were pinning a gap
this round closed, and all three were updated rather than worked around.

* `test_full_rejects_arguments_it_does_not_honour` passes the bare **string**
  `"strided"`, which is not a layout. `is_strided_layout` was tightened to
  require a real layout object (`_shim_name == "strided"`, or an upstream
  `torch.strided`), so the string is still refused and the test stands
  unmodified.
* `test_decompose_refuses_by_name_what_it_cannot_lower` used
  `aten.zeros_like.default` as wall 2's example, stopped on `pin_memory=False`.
  That is accepted now, so `zeros_like` lowers. The example moved to
  `aten.empty_like.default`, whose decomposition reaches `torch.empty_permuted`
  — an op with no entry in `overloads.json` at all. `zeros_like` is asserted
  **positively** as a case that now lowers, so the close is recorded rather
  than left as an absence.

* `test_ops_without_a_meta_kernel_name_themselves` is a census of which ops
  refuse on meta *by name*. `aten.squeeze.dims` was the last member of the
  `squeeze` family still on the refusing side and has moved to the answering
  side, together with `new_empty`, `new_zeros` and `prims.split_dim`, which were
  on neither. The test names ops rather than counting them, which is exactly why
  it caught this.

The note that used to stand in the decompose test argued the `pin_memory` gap was real
and should stay named. `torch.export` settled it the other way: it spells every
factory argument out, so refusing `pin_memory=False` refused a request for
nothing.

---

## 4. Nullification: 8 attempted, **0 uncaught**

Each landing was broken deliberately, rebuilt (`bootstrap.py` is
`include_str!`'d, so editing without rebuilding retests the old binary), re-run,
and reverted.

| | nullification | caught by |
|---|---|---|
| N1 | `annotation_str` answers the schema spelling | 2 tests |
| N2 | the mode boundary is not re-seated | 2 |
| N3 | `new_empty`/`new_zeros` lose their meta arm | 1 |
| N4 | `default_value` answers the source text | 2 |
| N5 | `layout=strided` / `pin_memory=False` refuse again | 1 |
| N6 | `TensorType.get()` is a stub instance again | 1 |
| N7 | meta+meta `set_` refuses again | 1 |
| N8 | `squeeze.dims` / `split_dim` lose their meta arms | 1 |

---

### 4.1 The nullification harness produced a false RED, and that is worth keeping

The harness restored each patched file with `shutil.copy` + `shutil.move`,
which gives the restored file the **backup's** mtime -- older than the build
output cargo had just produced. Cargo compares mtimes, saw nothing newer, and
skipped the rebuild. So after the last revert the *nullified* N8 binary was
still installed, and the next full gate reported `no meta kernel for
aten.squeeze.dims` on a tree whose source has one.

It failed loudly and was caught in minutes, which is the good direction. The
same mechanism in reverse would not have been: a revert that silently keeps a
*fix* installed while the source no longer has it would make every later run a
false green. Anything that edits a source file and rebuilds must touch it --
`shutil.copy2`, or an explicit `touch` -- and the eight verdicts in §4 are
unaffected, because each nullification was *written* with a fresh mtime and so
was genuinely built.

---

## 5. Left open, named

* **`torch.var_mean`** — 10 architectures. No `aten.var_mean.*` kernel exists;
  `aten.var.*` does. It returns a pair, so it is a real op addition with golden
  cases, not a table entry.
* **`torch._C._select_conv_backend`** — 5. A `_C` surface, not a kernel.
* **`aten.as_strided.default` on meta** — 5. §6. **Closed by
  `docs/graph/STRIDE.md`**, as is `prims.collapse_view` below; all six are
  architectures upstream itself does not export.
* **`aten.lift_fresh_copy.default`** — 3. No kernel at all, dense or meta. It is
  `clone` semantics, but adding it changes op coverage and wants golden cases.
* **`prims.collapse_view.default` on meta** — 1.
* **`FakeTensorDeviceMismatchError`** — 2, unanalysed.
  > **Analysed in `docs/graph/STRIDE.md` §6:** the shim's door consults
  > dispatch modes but not a tensor subclass's own `__torch_dispatch__`, so
  > fake mode's mode-less `view` implementation gets plain meta tensors back.
* **The `.Scalar`/`.Tensor` overload disagreement** — `docs/graph/EXPORT5.md` §9,
  untouched.

---

## 6. The stride question is unchanged, and is now the third-largest wall

> **Superseded by `docs/graph/STRIDE.md`.** `Repr::Meta` stores a stride,
> offset and storage size; `as_strided` and `collapse_view` have meta kernels;
> §2.5's two `set_` refusals are lifted. One premise below was already false
> when this was written: the meta `t`/`slice` arms existed and answered a
> contiguous stride, so the lie this section warns against was live
> (STRIDE.md §1).

EXPORT5 §10's third wall was `aten.t` / `aten.slice` on a meta tensor, because
`Repr::Meta` has no stride field and `docs/graph/EXPORT4.md` §6.5 rests an
invariant on there being no non-contiguous meta tensor.

It was attempted this round and **not closed.** What was learned:

* `aten.t` no longer appears in the sweep at all. The decompositions reach
  `prims.*` instead, and what surfaces now is `aten.as_strided.default` — the
  same question in its most direct form, since `as_strided` *is* a stride.
* The meta `set_` landed in §2.5 hits the same boundary from the other side and
  **refuses by name** when the requested stride is not contiguous, rather than
  accepting it and leaving a tensor reporting a layout it does not have. Axes of
  extent 0 or 1 are skipped, since their stride is unobservable.
* So the shape of the change is unchanged from EXPORT5's account: a stride field
  on `Repr::Meta`, `stride()` reading it, `EXPORT4` §6.5's invariant test
  rewritten, and `as_strided` / `t` / `slice` given meta kernels — **in one
  change**, because any subset leaves a meta tensor whose `stride()` lies.

That is a layout-model decision of the same class as the storage model, and it
is left for a round with room to verify it rather than landed at the end of this
one.

## 7. The gate

Two clean runs on the final tree, back to back, both identical:

```
GATE_EXIT=0    1647 ok    0 FAIL    DOCWATCH 1210/1210    VULKAN 23 ran / 0 skipped
GATE_EXIT=0    1647 ok    0 FAIL    DOCWATCH 1210/1210    VULKAN 23 ran / 0 skipped
```

against a `develop` baseline of 1636 ok / 0 FAIL / DOCWATCH 1198.

Two earlier runs are not counted, and why they are not is worth having:

* the first ran against the **nullified** N8 binary (§4.1) and failed
  `test_the_shape_only_meta_kernels_...`, correctly;
* the second failed `test_ops_without_a_meta_kernel_name_themselves` --
  `aten.squeeze.dims` answering on meta -- which is §3's third corrected test.

That same run also showed two failures that did **not** recur and pass when run
alone: `test_compute_plan_cleans_up_its_tempdir` (a leaked CoreML tempdir) and
`test_a_failed_cross_build_leaves_no_wheel_in_outdir`. They are recorded here as
suspected flakes under load rather than silently dropped; another agent was
running a model probe on the same machine at the time.

---

## 7.1 What this round was, separated

**Features added** -- surface or kernels the shim did not have:

1. `_SchemaType.annotation_str`, for all 2584 schemas.
2. `aten.new_empty.default` / `aten.new_zeros.default` meta kernels.
3. `aten.squeeze.dims` meta kernel.
4. `prims.split_dim.default` meta kernel.
5. `set_` on a meta tensor from a meta storage -- EXPORT5 §10's named narrowing.
6. The TorchScript type singletons: interning, `get()`, and `__instancecheck__`.

**Defects fixed** -- behaviour that was already there and wrong:

1. A `TorchDispatchMode` received schema-positional arguments as keywords.
2. A `TorchDispatchMode` received keywords equal to their own schema default.
3. `Argument.default_value` answered the schema's source text, not a value.
4. `has_default_value()` was `default_value is not None`, which made 1 and 3
   impossible to fix independently.
5. `layout=torch.strided` was refused by every factory.
6. `pin_memory=False` was refused by every factory.

**Tests added**: 11 in `rust/torch_c/pytests/test_export6.py`. Every one
compares against upstream torch's own answer in a second subprocess rather than
against a table written beside it; none asserts "export() returned".

**Tests corrected**: 3 in `test_shim.py`
(`test_parsed_schema_really_reads_the_schema`,
`test_decompose_refuses_by_name_what_it_cannot_lower`,
`test_ops_without_a_meta_kernel_name_themselves`) -- all three were pinning a
gap this round closed. §3.

**Docs**: this file; a superseded-note on `docs/graph/EXPORT5.md` §10.

**Removed**: nothing.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export6.py test_every_schema_type_answers_annotation_str_exactly_as_upstream_does present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export6.py test_a_dispatch_mode_receives_positional_arguments_positionally present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export6.py test_every_schema_default_is_the_python_value_upstream_gives_not_its_source_text present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export6.py test_the_torchscript_type_singletons_are_the_objects_a_schema_hands_out present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export6.py test_set_on_a_meta_tensor_with_a_meta_storage_is_metadata_and_is_allowed present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export6.py test_layout_strided_is_accepted_and_every_other_layout_is_still_refused present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export6.py test_pin_memory_false_is_accepted_and_pin_memory_true_is_still_refused present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _default_python_value present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs seat_positionally present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/storage.rs is_meta_storage present -->
<!-- DOCWATCH: op-implemented aten.new_empty.default -->
<!-- DOCWATCH: op-implemented aten.squeeze.dims -->
