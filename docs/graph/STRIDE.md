# A meta tensor's stride is stored, not derived

`docs/graph/EXPORT6.md` §6 left one change open and said why it had to be done
all at once: a stride field on `Repr::Meta`, `stride()` reading it,
`docs/graph/EXPORT4.md` §6.5's invariant test rewritten, and `as_strided` /
`t` / `slice` given meta kernels — because **any subset leaves a meta tensor
whose `stride()` lies**. This is that change, and what measuring it found.

Read these first, because they decide how the rest should be taken:

* **The headline is still 0 of 40 under the replay-and-agree bar** (§5).
  The `as_strided` wall (5 architectures) and the `collapse_view` wall (1) are
  gone. But all six are architectures **upstream itself cannot export** in
  this sweep: it refuses their `DynamicCache` output. Closing the stride
  question could not have moved the count, and it did not. The ten
  architectures upstream *can* export stop at `var_mean` (6) and
  `torch._C._select_conv_backend` (4), before and after.
* **The lie was already live when this round started** (§1). The premise that
  "any subset leaves a lying `stride()`" was about the *future*. It was
  already true: the meta `t`, `permute`, `slice`, `expand` and every pointwise
  arm answered a contiguous stride where upstream's is not.
* **Every layout claim below is compared with upstream torch in a separate
  process**, exactly (strides are integers; there is no tolerance). The
  comparison is `pytests/test_metastride.py`: 1,333 cases across 13 tests.

---

## 1. What was already wrong

A probe run on both sides before anything was changed
(`torch.empty(..., device="meta")` on each):

| expression | upstream | this shim |
|---|---|---|
| `m(3,4).t()` | `(1, 4)`, not contiguous | `(3, 1)`, contiguous |
| `m(3,4,5).permute(2,0,1)` | `(1, 20, 5)` | `(12, 4, 1)` |
| `m(3,4,5)[:, 1:3]` | `(20, 5, 1)`, offset 5 | `(10, 5, 1)`, offset 0 |
| `m(3,1).expand(3,4)` | `(1, 0)` | `(4, 1)` |
| `m(3,4).t() + 1` | `(1, 4)` | `(3, 1)` |
| `m(3,4).t().view(12)` | **refuses** (spans two subspaces) | `(1,)` — a view upstream says cannot exist |
| `m(3,4).as_strided((4,3),(1,4),2)` | `(1, 4)`, offset 2 | refuses: no meta kernel |

`docs/graph/EXPORT4.md` §6.5 answered `stride()` with the contiguous stride of
the shape, resting on "every meta tensor is contiguous", and pinned that
invariant with `test_a_meta_tensor_is_contiguous_so_its_stride_is_derivable`.
**The test asked a freshly constructed tensor, and only that.** The meta view
arms that falsified the invariant had existed for several rounds; no test
built a meta tensor through one of them and then asked its stride. That is
`CLAUDE.md` §5.4's shape: the check answered "is a new meta tensor
contiguous", and the claim was "is every meta tensor contiguous".

## 2. The layout model

`Repr::Meta` is now

```rust
Meta { shape, stride, storage_offset, storage_nbytes: Arc<AtomicUsize>, storage_id }
```

* **`stride` and `storage_offset` are stored as given**, and `stride()`,
  `storage_offset()` and `is_contiguous()` read them.
* **`storage_nbytes` is the storage's size, not the view's.**
  `x[1:].untyped_storage().nbytes()` is the base's upstream, and the prims
  `as_strided` meta (`torch/_prims/__init__.py:_as_strided_meta`) bounds-checks
  against it. A fresh tensor gets `computeStorageNbytes` of its layout, which
  is smaller than `numel * itemsize` for an overlapping layout.
* **The size is one cell shared** by the tensor, every view of it, and every
  storage handle `untyped_storage()` hands out. Upstream's `set_` *grows* a
  meta storage when the layout addresses past its end (measured: through the
  handle, the base and a view alike), and EXPORT6 pinned that growth as
  allowed. A plain `usize` per object could only refuse it or let the objects
  disagree about one number; the shared cell does what upstream does.
* **`set_` from a meta storage adopts any non-negative stride and any offset.**
  EXPORT6 §2.5 refused both by name *because there was nowhere to put them*.
  A negative stride is still refused: upstream accepts `(-1, 3)` and reads it
  back as `(6, 3)`, which is not a layout this shim can claim to reproduce.
* **`empty_strided` on meta builds the stride it is asked for.** It is the
  constructor behind every fake tensor. The dense half still refuses a
  non-contiguous stride by name — candle cannot hold one.

The arithmetic is in `rust/torch_c/src/layout.rs`, one function per upstream
rule, each naming the rule it ports: `computeStride` (`view`),
`inferExpandGeometry`, `inferUnsqueezeGeometry`, `computeStorageNbytes`,
`compute_elementwise_output_logical_to_physical_perm`,
`is_non_overlapping_and_dense`, `_collapse_view_helper`,
`compute_channels_last_contiguous_2d/3d`, `is_channels_last_strides_2d_s4/s5`
and `get_channels_last_strides_2d/3d`.

## 3. Every meta arm states its output layout, or refuses

A meta tensor that carries a stride makes every meta kernel a claim about
the stride of what it returns. So `aten.rs::meta_stride_rule` puts every op
in exactly one class, and `meta_dispatch` enforces the class before the arm
runs:

| class | what the arm does | checked by |
|---|---|---|
| `OwnLayout` (34 ops) | computes it: the views (sharing storage), the in-place initialisers and `copy_` (return the receiver), non-tensor results and the value-dependent refusals, `cat`, `contiguous`, `collapse_view`, CPU flash attention | `test_view_kernels_…`, `test_as_strided_…`, `test_cat_…`, `test_attention_…`; the non-tensor and refusing arms have no layout to check |
| `AlwaysContiguous` (26 ops) | `meta_result`, whatever the input | `test_always_contiguous_…`: every op × 5 input layouts incl. channels-last |
| `Elementwise` (56 ops) | built contiguous, re-laid by the elementwise rule over the operands broadcast to the output shape | `test_layout_following_…`: 64 op spellings × 15 layouts, plus broadcasting |
| `PreserveFormat` (3 ops) | as `Elementwise`, but a non-overlapping-and-dense input keeps its stride (`clone`, `_to_copy`, `prims.clone`) | same test |
| `Unverified` | **refuses by name if any meta input is not laid out contiguously** | `test_a_stride_unaware_…` (the refusal), `test_gated_…` (the contiguous answers) |

An explicit `memory_format=contiguous_format` on an elementwise or
preserve-format op is honoured (the old arms ignored it); a channels-last one
refuses by name.

### 3.1 What the measurement decided, that reading would have got wrong

Each of these was a first guess that the differential test rejected:

* **The pointwise kernels have no dense-input shortcut.** The Python function
  `compute_elementwise_output_strides` keeps a non-overlapping-and-dense
  input's stride. The meta kernels do not go through it — they go through
  `refs.empty_like`, which applies the permutation directly — and the two
  answer differently on an empty tensor: a transposed `(3, 0)` becomes
  `(1, 1)` under `relu`, `empty_like` and `zeros_like`, and stays `(1, 3)`
  under `clone` and `_to_copy`. Hence two classes, not one.
* **`contiguous()` returns `self`** when already contiguous — offset and storage
  included — and a fresh copy otherwise.
* **`cat` is not always contiguous.** It takes the inputs' common
  `suggest_memory_format()`, falling back to contiguous on any disagreement,
  with upstream's tie-breaks for `N111` and a unit channel axis. **A skipped
  1-D empty input still votes**, and makes the result contiguous.
* **`sort`, `index.Tensor` and `_weight_norm_interface` answer a
  non-contiguous layout for some non-contiguous inputs** (`sort` of a
  reversed-stride input keeps its stride; a gappy one does not). They are
  `Unverified`: they answer on contiguous inputs, where they were measured to
  agree, and refuse otherwise.
* **`convolution` is contiguous on meta even for a channels-last input**, and
  `native_layer_norm` is contiguous for every layout tried.

### 3.2 A wrong stride the gate could not see

The `Unverified` class rests on "contiguous in, contiguous out", and that is
not a law. Upstream's CPU flash-attention meta is

```python
attention = torch.empty_like(query)
logsumexp = torch.empty((B, T, H)).transpose(1, 2)
```

so `logsumexp` is `(15, 1, 3)` for a **contiguous** `(2, 3, 5, 4)` query. The
shim answered `(15, 5, 1)`, and a gate keyed on input contiguity lets that
through. It is now `OwnLayout` with upstream's two lines, and every op still
in `Unverified` is compared with upstream on contiguous inputs
(`test_gated_meta_kernels_agree_with_upstream_on_contiguous_inputs`). An op
added to the class later is not judged until it is added there.

## 4. Channels-last contiguity was already wrong on the dense side

`is_contiguous(memory_format=channels_last)` answered `False` for every
tensor, and `docs/graph/EXPORT5.md` §3 pinned that with a test of the premise
"no tensor in this build can be in that layout" — by trying
`.to(memory_format=channels_last)`. **A plain `permute(0, 3, 1, 2)` of an NHWC
tensor is channels-last** on both sides, with no memory-format request
anywhere, and the shim answered `False` for it. The premise was checked
through one door of several.

It is now read off the stride on every arm. Two more dense corrections came
with it: an empty permuted tensor is contiguous upstream (candle's rule has no
"fewer than two elements" clause), and the result is computed by upstream's
rule rather than candle's.

`.to(memory_format=channels_last)` still refuses by name: the dense side
cannot re-lay a tensor.

## 5. `torch.export`: where the forty stop now

> **Both walls named below are closed — `docs/graph/VARMEAN.md`.** `var_mean`
> went from 11 of the forty to 0 and `torch._C._select_conv_backend` from 5 to
> 0. The bar did not move: still **0 of 10**. Nine of the ten now stop at one
> new wall (a `return_types_native_layer_norm` that `proxy_tensor.py`'s
> `extract_val` cannot rebuild) and the tenth at §6's dispatcher question.

`export_sweep.py --limit 40`, shim side, before this round and after, with
upstream's own run beside it:

```
                                     before   after
architectures swept                      40      40
upstream exported+replayed+agreed        10      10
this shim, of those ten                   0       0     <- the bar
```

On the ten upstream exports, the shim stops at `var_mean` (6) and
`torch._C._select_conv_backend` (4), unchanged.

All forty, export-stage failures only:

| wall | before | after |
|---|---:|---:|
| `torch.var_mean` overload | 10 | 11 |
| `torch._C._select_conv_backend` | 5 | 5 |
| `no meta kernel for aten.as_strided.default` | 5 | **0** |
| `no meta kernel for aten.lift_fresh_copy.default` | 0 | 4 |
| `aten.lift_fresh_copy.default` (no kernel at all) | 3 | 3 |
| `FakeTensorDeviceMismatchError` | 2 | 3 |
| `no meta kernel for prims.collapse_view.default` | 1 | **0** |

The six that moved:

```
arcee, aria, aria_text, bitnet   as_strided     -> lift_fresh_copy on meta
bart                             as_strided     -> var_mean
audioflamingo3                   collapse_view  -> FakeTensorDeviceMismatchError
```

None of the six is exportable upstream (`DynamicCache` /
`EncoderDecoderCache` in the output), so none could have counted.

## 6. The next wall is a dispatcher question: subclass `__torch_dispatch__`

`docs/graph/EXPORT6.md` §5 listed `FakeTensorDeviceMismatchError` (2
architectures) as unanalysed. It is now analysed, and it is not a stride
question.

**The shim's single door consults dispatch *modes* but not a tensor
*subclass's* own `__torch_dispatch__`.** `FakeTensorMode` implements `view` by
calling `torch._refs._reshape_view_helper` on the `FakeTensor` *with the mode
popped* (`fake_impls.py:_view_meta`). Upstream, the `as_strided` /
`collapse_view` calls inside it re-enter fake mode through
`FakeTensor.__torch_dispatch__`. Here they reach the meta kernel directly and
come back as plain meta tensors, which `proxy_tensor.py:extract_val` then
records as tensors on `device="meta"`, and the next op finds `meta` and `cpu`
among its arguments. Measured on one line, both sides:

```python
with FakeTensorMode(): a = torch.empty(10, 8)
torch.ops.aten.as_strided.default(a, (8, 10), (1, 8), 0)
#   upstream:   FakeTensor, device cpu
#   this shim:  Tensor,     device meta
```

Before this round the same path raised earlier, at `as_strided`. The smallest
module that reaches it is an `nn.Linear` applied to a 3-D input (its `view`
to 2-D). It was **not** fixed here: it is a change to the door every op
passes through, not to the layout model, and it wants its own round and its
own cost measurement. The end-to-end test in this round's file stays 2-D for
that reason, and says so.

## 7. Nullification: 31 attempted, 0 uncaught

Each was written into the source, rebuilt, installed, run against the suites
named, and reverted by copy **and `touch`** (`docs/graph/EXPORT6.md` §4.1's
stale-mtime trap); the restored sources were compared byte for byte.

| | nullification | red |
|---|---|---:|
| N1 | meta `stride()` computes the contiguous stride from the shape | 9 tests, incl. EXPORT4's rewritten one |
| N2 | meta `storage_offset()` answers 0 | 6, incl. EXPORT4 |
| N3 | `as_strided` ignores the requested stride | 4 |
| N4 | `as_strided`'s default offset is 0, not the input's | 1 |
| N5 | every meta view is laid out contiguously | 9, incl. EXPORT4 |
| N6 | a meta view gets a fresh storage | 6 |
| N7 | `view` never refuses on layout | 1 |
| N8 | the `Unverified` gate lets everything through | 1 |
| N9 | elementwise outputs are not re-laid | 2 |
| N10 | the elementwise rule regains the dense-input shortcut | 1 |
| N11 | preserve-format copies lose it | 1 |
| N12 | channels-last contiguity is a constant `False` | 6, incl. EXPORT5's rewritten one |
| N13 | dense `is_contiguous` is candle's rule again | 1 |
| N14 | `cat` is always contiguous | 1 |
| N15 | `cat`'s vote skips the empty 1-D inputs | 1 |
| N16 | `set_` does not grow the storage | 1 |
| N17 | each storage handle copies the size instead of sharing it | 1 |
| N18 | meta `empty_strided` ignores the stride | 2 |
| N19 | `contiguous()` always copies | 1 |
| N20 | `collapse_view` never refuses | 1 |
| N21 | `memory_format=contiguous_format` is ignored | 1 |
| N22 | the meta storage size is `numel * itemsize` | 5 |
| N23 | `unsqueeze` gives the new axis stride 1 | 1 |
| N24 | `expand` gives every new axis stride 0 | 1 |
| N25 | `select` does not move the offset | 1 |
| N26 | `split` chunks all start at the base offset | 1 |
| N27 | `slice` does not multiply the stride by `step` | 3 |
| N28 | `set_` adopts a contiguous stride | 1 |
| N29 | attention's `logsumexp` is contiguous | 2 |
| N30 | the attention pair is not promoted | 2 |
| N31 | the layer-norm triple is not promoted | 1 |

N30 and N31 are caught only because the probe records the result's **type**
as well as its layout. It did not at first, and recording it is what exposed
§8's promotion defects.

The counts are from the runs as made: N1–N28 against the test file before
the attention, gated-class and receiver cases were added, N29–N31 after. N3's
first patch did not compile and was rerun with one that did. N12's run also
reddened `test_export5.py`'s meta-storage door test, but that was this
round's own rewording of the `resize_` refusal, present with or without N12,
and it is not counted.

Two things the nullifications showed about the *older* tests:
`test_export6.py`'s `meta_grows` case compares shape, dtype and device only,
so it stays green when `set_` does not grow the storage (N16) or adopts a
contiguous stride (N28); and nothing outside this round's file notices a meta
view losing its storage identity (N6).

## 8. Defects found on the way that are not about stride

* **`native_layer_norm`, CPU flash attention and `_weight_norm_interface`
  returned bare `TensorBase` objects inside their meta tuples** — the
  dispatcher's exit promotes a top-level tensor and does not look into a
  tuple. `F.scaled_dot_product_attention` on meta returned a `TensorBase`.
  All three now promote, as `max.dim` and `split` already did.
* **`torch.ops.aten.sub.Tensor(x, 1)` with a bare Python number** is refused
  here and accepted upstream. Recorded, not fixed; the probe uses 0-d tensors
  and the `Scalar` overloads instead.
* **The shim's `mm` shape refusal wording** differs from upstream's inside
  `torch.export` (`mat1 and mat2 shapes cannot be multiplied` against
  `a and b must have same reduction dim`). Recorded, not fixed.
* **CPU flash attention's `logsumexp` is `float32` for a `float64` query**
  here; upstream's meta uses `get_computation_dtype`, which keeps `float64`.
  Recorded, not fixed.

## 9. What this does not cover

* **Ops without a meta arm** are not judged — they refuse as before.
* **Ops with a meta arm are judged on the layouts the probe builds.** The
  classes are enforced for every op, but whether an `AlwaysContiguous` op is
  contiguous on a layout the probe never built is not measured.
* **`UntypedStorage.resize_` on a meta storage** still refuses by name. The
  shared size cell removes the reason it gave, but lifting it needs a
  measurement of what upstream does to the tensors when their storage shrinks.
* **The dense side cannot hold a caller-chosen stride**, so a layout-following
  op from a non-contiguous meta input onto a dense device refuses by name
  rather than answering contiguously.

## 10. This round, separated

**Features added** — representation or kernels the shim did not have:

1. `Repr::Meta` stores `stride`, `storage_offset` and a shared storage size.
2. `aten.as_strided.default` on meta.
3. `prims.collapse_view.default` on meta.
4. `set_` on meta adopts a non-contiguous stride and an offset, and grows the
   storage.
5. `empty_strided` on meta builds any non-negative stride.
6. `meta_stride_rule`: every meta arm in a layout class, and the gate that
   refuses a non-contiguous input for arms not shown to follow it.
7. The elementwise / preserve-format re-lay, `cat`'s memory-format vote, and
   explicit `memory_format=contiguous_format` on those ops.

**Defects fixed** — behaviour that was there and wrong:

1. The meta view arms (`t`, `transpose`, `permute`, `slice`, `select`,
   `expand`, `squeeze` ×3, `unsqueeze`, `split` ×2, `view`, `reshape`,
   `detach`, `alias`, `lift_fresh`, `prims.view_of`, `prims.split_dim`)
   answered contiguous strides, zero offsets and fresh storages.
2. The meta pointwise and preserve-format arms (57 ops) answered contiguous
   strides for non-contiguous inputs.
3. Meta `view` accepted layouts upstream refuses; meta `reshape` never
   distinguished a view from a copy.
4. Meta `contiguous()` always copied.
5. Meta `untyped_storage().nbytes()` was the view's size, not the storage's.
6. CPU flash attention's meta `logsumexp` was contiguous.
7. `is_contiguous(memory_format=channels_last)` was a constant `False`,
   including for dense channels-last permutations.
8. Dense `is_contiguous()` was `False` for an empty permuted tensor.
9. Three meta arms returned unpromoted `TensorBase` objects in tuples.

**Tests added**: 13 in `rust/torch_c/pytests/test_metastride.py` (1,333
cases compared with upstream) and 6 Rust unit tests in `layout.rs`.

**Tests rewritten**: 3 —
`test_export4.py::test_a_meta_tensor_reports_the_stride_it_stores_not_one_derived_from_its_shape`
(was `…_is_contiguous_so_its_stride_is_derivable`, §1),
`test_export5.py::test_channels_last_contiguity_is_read_off_the_stride_as_upstream_reads_it`
(was `…_is_false_as_a_fact_because_the_build_cannot_make_one`, §4), and
`test_export5.py::test_every_door_that_would_need_bytes_refuses_on_a_meta_storage`
(the `resize_` refusal's stated reason changed, §9). Both renamed tests now
compare with upstream in a subprocess.

**Docs**: this file; superseded notes on `docs/graph/EXPORT4.md` §4 and §6.5,
`docs/graph/EXPORT5.md` §3, `docs/graph/EXPORT6.md` §5–§6,
`docs/kernels/METAEMB.md` §6.2, `docs/kernels/METAFAM.md` §6.3 and
`docs/devices/META.md` §12.

**Removed**: nothing.

## 11. The gate

Two clean runs on the final tree, identical:

```
GATE_EXIT=0   1692 ok   0 FAIL   DOCWATCH 1258/1258   cargo test 40/40
golden 11478/11478 ops=304 failed=0 pending=0   self-test PASS
VULKAN 33 ran / 0 skipped
```

against a `develop` baseline of about 1679 ok / DOCWATCH 1238. The +13 is
this round's 13 tests and the +20 its 20 markers; `cargo test` went from 34
to 40 with `layout.rs`'s six.

Three runs are not counted, and why:

* the first failed `test_no_docs_reference_in_the_tree_dangles` (this file
  did not exist yet) and EXPORT6's `meta_grows` case, which is what turned
  `set_`'s by-name refusal past the storage's end into the shared, growing
  size of §2;
* one run between the two clean ones failed only
  `test_the_coreml_models_docs_npu_executed_ran_on_the_cpu`, with
  `('sigmoid', 'no compute operations in the plan at all')` — the CoreML
  compute-plan flake that fails on `develop` too and that another round is
  fixing. It stops the gate before golden and DOCWATCH, so the run is not
  evidence either way;
* the nullification campaign's 31 builds (§7), which are red by design.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs storage_nbytes present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs meta_view present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs meta_stride_rule present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs relay_elementwise present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/layout.rs elementwise_stride present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/layout.rs view_stride present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/layout.rs collapse_view present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/layout.rs strides_like_channels_last present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/storage.rs meta_len_cell present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metastride.py test_as_strided_on_meta_answers_upstreams_layout_and_refusals present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metastride.py test_view_kernels_on_meta_carry_the_real_stride_offset_and_storage present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metastride.py test_layout_following_meta_kernels_answer_upstreams_output_stride present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metastride.py test_always_contiguous_meta_kernels_are_contiguous_on_every_input_layout present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metastride.py test_gated_meta_kernels_agree_with_upstream_on_contiguous_inputs present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metastride.py test_a_stride_unaware_meta_kernel_refuses_a_non_contiguous_input present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metastride.py test_a_module_whose_trace_needs_real_meta_strides_exports_replays_and_agrees present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export4.py test_a_meta_tensor_reports_the_stride_it_stores_not_one_derived_from_its_shape present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export4.py test_a_meta_tensor_is_contiguous_so_its_stride_is_derivable absent -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export5.py test_channels_last_contiguity_is_read_off_the_stride_as_upstream_reads_it present -->
<!-- DOCWATCH: op-implemented aten.as_strided.default -->
