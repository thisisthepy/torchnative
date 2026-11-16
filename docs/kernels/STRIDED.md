# `as_strided` lands as a read-only view, and the barrier is what makes that honest

**The guard is real.** `aten.as_strided.default` is implemented, its values are
upstream's, and every write upstream's genuine view would have propagated, in
either direction, is a named refusal here rather than a wrong number.
Nullifying the barrier and changing nothing else puts back a **stale base**:
`x.as_strided((2,3),(3,1)).fill_(7.)` leaves `x` at `[0., 1., 2., …]` where
upstream leaves it `[7., 7., 7., 7., 7., 7., 6., …]`, with no exception
anywhere. §5 is the measurement.

`docs/kernels/TAIL3.md` §6 refused this op and `docs/kernels/TAIL4.md` §1 refused it again.
Both were right about the facts they checked and this round overturns neither
of them: candle 0.11.0 still cannot build the view, so the result is still a
materialised gather. What changed is that the third option, **refuse the
writes**, turned out to be reachable, and TAIL4's sizing of why it was not is
the one thing here that was wrong. §3.

`longformer` and `led` both move past this op. Neither forwards yet; both stop
one wall later, together, at `Tensor.new_zeros((tuple))`. §6.

---

## 1. What upstream's `as_strided` is, and what this one is

Upstream returns a tensor sharing the receiver's storage under a layout the
caller supplies outright. It is a view in **both** directions, measured on
2.13.0:

```text
x = torch.arange(6.).reshape(2, 3)
y = x.as_strided((2, 2), (3, 1))
y[0, 0] = 99.      ->  x[0, 0] is 99.      view  -> base
x[1, 1] = -7.      ->  y[1, 1] is -7.      base  -> view
```

candle 0.11.0 cannot produce that, and the reason is sharper than "no
constructor", it is that **both halves of the constructor are public and the
struct that joins them is not**:

```text
candle_core::Layout::new(shape, stride, start_offset)     pub
candle_core::Tensor::from_storage(storage, shape, ...)    pub, but takes an
                                                          OWNED Storage, makes a
                                                          fresh Arc, and its own
                                                          doc says "this uses
                                                          contiguous strides"
candle_core::Tensor_ { storage: Arc<RwLock<Storage>>,
                       layout: Layout, .. }               0 public fields
```

Nine sites inside `candle-core/src/tensor.rs` build a shallow view by writing
`Tensor_ { storage: self.storage.clone(), layout: <custom> }` directly:
`transpose`, `permute`, `narrow`, `squeeze`, `unsqueeze`, `reshape`,
`broadcast_as`, `detach`, and `slice_scatter`'s helper. None of them is
expressible from outside the crate. A candle patch exposing one
`from_storage_and_layout` would remove this whole document; `torchnative/rust/vendor/` already
carries `int8-candle-0.11.0-cpu.patch`, so the mechanism exists **, which is FALSE, corrected 2026-09-07.** The patch file is carried (now in `torchnative/rust/vendor/`, previously in `docs/`), but NOTHING APPLIES IT: `torchnative/rust/vendor/*.sh`, `torchnative/rust/vendor/*.py`, `Cargo.toml` and `build.rs` contain no patch step and no `[patch.crates-io]`. Carrying a diff is not a mechanism, and three documents used this sentence to argue that a candle fork would be cheap.

**Corrected again, 2026-11-16 (issue #380; agent edit, not a measurement).** The "nothing applies
it" correction above is stale. `scripts/vendor/vendor_candle.sh` materialises
`torchnative/rust/vendor/candle-core` from the published candle-core 0.11.0 plus this patch
(`--check` diffs the committed fork against that), and `torchnative/rust/torch_c/Cargo.toml`
reaches it through `[patch.crates-io]`. The patch now carries two shallow-view constructors:
`Tensor::narrow_with_step` (the stepped `slice.Tensor`) and, since #380,
`Tensor::view_with_layout(Layout)`, which joins the receiver's storage to any layout whose
addressed span lies inside the receiver's own and refuses anything else by name. The dense view
family (`view`, `_unsafe_view`, `reshape`, `view_as`, `reshape_as`, `unflatten`) uses it. **This
op does not, and that is open:** `as_strided` may address any part of the storage, including
elements its receiver does not, and `view_with_layout`'s bounds rule refuses those, so this
document's gather and barrier still stand. Making `as_strided` (and `unfold`) a real view needs a
bounds rule against the storage's own length, which the fork does not expose for all three
backends.

Nor is the view expressible as a composition of the ops candle does export.
`longformer`'s chunking asks for **overlapping** windows: `_chunk` halves the
second stride so consecutive chunks share half their elements, and no sequence
of `narrow`/`transpose`/`reshape` produces a stride smaller than the extent it
indexes.

So this implementation gathers, and the shim's `as_strided` result is a
**copy**. Everything below is about making that difference impossible to
mistake for the real thing.

<!-- DOCWATCH: op-implemented aten.as_strided.default -->
<!-- DOCWATCH: json-key torchnative/rust/torch_c/src/methods.json as_strided present -->
<!-- DOCWATCH: json-key torchnative/rust/torch_c/src/overloads.json as_strided present -->

---

## 2. The barrier, and why an address is a sound key here

A gather is silently wrong for a writer in **both** directions at once, so the
result and the base are both barred from in-place writes for as long as the
result is alive.

```text
storage.rs   STRIDED_BARRIER : storage address -> live barrier count
             StridedBarrier { keys, keep_alive: [base.clone(), view.clone()] }
             Drop -> remove the keys, THEN release keep_alive
tensor.rs    PyTensorBase.strided : Option<Arc<StridedBarrier>>
             write_into() -> refuse if crate::storage::write_is_barred(&dest)
aten.rs      as_strided_default() -> the only caller of
             bar_writes_as_strided_view()
```

**The bar is read at the single write door.** `tensor::write_into` is the only
thing in the crate that writes through a tensor's layout, and `aten.rs::write_back`
is its only caller, counted, not read (`aten.rs` 1 occurrence of `.write_into(`,
`tensor.rs` 0). That is `docs/kernels/TAIL4.md` §1.1's finding, re-verified here and now
load-bearing rather than a precondition for a sizing.
`test_the_write_door_is_still_single_which_is_what_makes_the_barrier_total`
fails the day either number moves.

**The key is the storage address, which is the identity that survives
aliasing.** `capture.rs::storage_key` already uses it for `STORAGE_VERSIONS`.
`docs/kernels/TAIL4.md` §1.2 rejected it for this purpose and gave the right reason: an
address is reused after the allocation it named is freed, so a permanent entry
starts refusing writes to unrelated tensors allocated later at the same address.

What answers that objection is one field. `StridedBarrier` holds a **clone of
both candle tensors**, so neither storage can be freed while its key is
registered, so neither key can be reused while it means anything. `Drop` removes
the keys and only then releases the clones, the entry and the address
reservation end in the same statement, which is what makes reuse unobservable
rather than merely unlikely.

Holding the base alive for the lifetime of the view is **not** a leak relative
to upstream. Upstream's `as_strided` result holds the base's storage alive by
aliasing it. This is the same lifetime, reached a different way. And the barrier
is not permanent: drop the result and the base is writable again, which
`test_the_barrier_lifts_when_the_view_dies_which_is_what_keeps_it_from_poisoning`
asserts, because a barrier that only ever grows is a leak with a refusal on top.

<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/storage.rs StridedBarrier present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/aten.rs as_strided_default present -->
<!-- DOCWATCH: symbol-in-file tests/ops/test_strided.py test_the_write_door_is_still_single_which_is_what_makes_the_barrier_total present -->

---

## 3. Where `docs/kernels/TAIL4.md` §1.3's sizing was wrong, which is the finding

TAIL4 sized the smallest sufficient change as *"one `bool`/`Option<…>` field on
`PyTensorBase` that survives aliasing, plus propagation through the view-producing
ops"*. **That would not have been enough**, and the reason is a direction rather
than a detail:

> **Propagation runs forward from the mark. The exposure runs backward from it.**

```python
x = torch.arange(12.)
v = x.view(3, 4)              # an alias of the base, taken FIRST
y = x.as_strided((2, 3), (3, 1))
v.fill_(7.0)                  # upstream: y is all 7s. A gather: y sees nothing.
```

`v` existed before `as_strided` was ever called. No field attached to `y`, and
no propagation forward from `y` through the view ops, can reach it. A key on the
storage reaches it, because `v`, `x` and `y`'s base are one buffer, measured,
and `test_the_barrier_reaches_an_alias_of_the_base_that_PREDATES_the_call` pins
it.

This is not a corner case dressed up as one. **`longformer`'s `_chunk` calls
`as_strided` on a `hidden_states` that is itself the result of a `.view(…)` one
line earlier**, so the base is already an alias by the time the op is reached.

The other escape TAIL4 named, `z = y.view(-1); z.fill_(0)`, which defeats a
`TensorId`-keyed set because candle mints a fresh `TensorId` for every shallow
view, is closed by the same key, for the same reason. Verified in candle's
source: all nine shallow-view sites write `id: TensorId::new()`.

So the corrected sizing, for the record TAIL4 §1.3 was keeping: **a storage-keyed
registry with a keep-alive, plus one `Arc` field on `PyTensorBase` to own the
handle.** The field is still needed, but for the barrier's *lifetime*, not for
its *identity*, and getting that backwards is what made the earlier estimate
both smaller and insufficient.

---

## 4. The narrowing, stated as a narrowing

Upstream is a two-way view; this is a read-only one. Everything upstream would
have propagated is refused:

| | upstream | here |
|---|---|---|
| values, including overlapping and zero strides | n/a | **identical** |
| the five refusals (length, negative stride, negative size, negative offset, out of bounds) | n/a | **identical messages** |
| write through the view, base read after | propagates | `RuntimeError`, names `as_strided` |
| write to the base, view read after | propagates | `RuntimeError`, names `as_strided` |
| write through an alias of the view | propagates | `RuntimeError` |
| write through an alias of the base taken **before** the call | propagates | `RuntimeError` |
| receiver is non-contiguous | reads the **storage** | `NotImplementedError`, §4.1 |

Both write directions are registered in `tests/golden/cases.py` as
`expect="c_error"`, one case each so that closing one cannot hide the other, and
`compare.py` prints them every run. That is a **stronger** register than the two
ops already in that file with the same shape of divergence:
`aten.slice.Tensor` (step > 1) and `aten.view.dtype` are `expect="diverge"`,
they alias upstream, materialise here, and a write through them is *silently*
lost. `as_strided` refuses instead. The bar `docs/kernels/COMPLEX2.md` set was that a
wrong answer be unrepresentable rather than unlikely, and a refusal is how that
is reached when the aliasing itself cannot be.

### 4.1 The contiguity requirement is upstream's semantics, not tidiness

Upstream's `as_strided` addresses the **storage** and ignores the receiver's own
layout entirely:

```text
x.reshape(3, 4).t().as_strided((2, 2), (1, 1))   ->  [[0., 1.], [1., 2.]]
                                                     storage order
gathering the receiver's LOGICAL order would give  [[0., 4.], [4., 8.]]
```

Same shape, same dtype, different numbers, the failure no shape or dtype
comparison can see. This kernel gathers out of the receiver's elements, which
are in storage order only when the receiver is contiguous, so a non-contiguous
receiver is refused by name. `.contiguous()` first is **not** an equivalent
rewrite and the message says so.

### 4.2 The one window this does not close

An alias of the *result* that outlives the result:

```python
y = x.as_strided(size, stride)
z = y.view(-1)      # aliases y's copy
del y               # the last barrier handle drops; both keys are released
z.fill_(0)          # accepted -- writes the copy, and x never sees it
```

While `y` is alive, `z` is refused (§4's table, row four). It is the `del` that
opens it, and closing it would mean an entry whose life is tied to the *copy's*
storage rather than to the result wrapper, which needs a `Weak` handle to
`Arc<RwLock<Storage>>` that candle does not expose, or a per-storage id it does
not have. The two lifetimes actually available are permanent (a monotonically
growing poison set, fail-loud but eventually useless) and wrapper-scoped (this),
and there is no third against candle 0.11.0's public surface. It is written down
here rather than left to be discovered, and the candle patch of §1 removes it
along with everything else in this document.

<!-- DOCWATCH: symbol-in-file tests/ops/test_strided.py test_the_barrier_reaches_an_alias_of_the_base_that_PREDATES_the_call present -->
<!-- DOCWATCH: symbol-in-file tests/ops/test_strided.py test_a_noncontiguous_receiver_is_refused_because_upstream_reads_storage_order present -->

---

## 5. Nullifying the barrier, and what it puts back

`docs/kernels/COMPLEX2.md` did not argue that `tensor()`'s non-`Dense` refusal mattered;
it deleted the refusal and measured that 6 of 10 sampled ops computed silently.
The same experiment here. One edit, `if false &&` in front of
`write_is_barred(&dest)` in `tensor::write_into`: a rebuild, and nothing else:

```text
                                    upstream            barrier off
x.as_strided((2,3),(3,1)).fill_(7.)
  the view                          [[7,7,7],[7,7,7]]   [[7,7,7],[7,7,7]]   agree
  the BASE x afterwards             [7,7,7,7,7,7,6,...] [0,1,2,3,...,11]    STALE

v = x.view(3,4); y = x.as_strided(...); v.fill_(7.)
  the VIEW y afterwards             [[7,7,7],[7,7,7]]   [[0,1,2],[3,4,5]]   STALE
```

**What is corrupted, by name, is the base**, and in the caller that wanted this
op, the base is `hidden_states`, the tensor `_chunk` is handed at
`modeling_longformer.py:719` and `modeling_led.py`'s copy of it. The shim
answers the right shape and the right dtype with values from before a write that
upstream applied, and raises nothing.

The build with the barrier off is not merely wrong, it is **caught**, which is
the other half of the check (`AGENTS.md` §17.5, a new public path that does not
break when nullified is a path nobody uses):

```text
golden       11334/11336, 2 failed   both as_strided c_error cases
test_strided 4 FAIL of 12
```

One note for whoever runs that experiment next: `compare.py` reports a failing
`c_error` case as *"gap appears CLOSED: promote this case to expect=match"*.
Here that advice is wrong and following it would be the silent failure, the
gap is not closed when both sides succeed, it is closed when candle can build
the view.

<!-- DOCWATCH: symbol-in-file tests/ops/test_strided.py test_the_barrier_is_read_at_that_door_and_nullifying_it_is_what_STRIDED_md_measured present -->

---

## 6. `longformer` and `led`

Both were blocked on `aten.as_strided.default` in `docs/architectures/ARCH200.md` and both were
left there deliberately by `docs/kernels/TAIL4.md` §8. Re-run with
`tests/_support/arch_sweep.py --one`, shim side:

| architecture | before | after |
|---|---|---|
| `longformer` | `aten.as_strided.default` | past `_chunk` and the chunked matmul; stops at `Tensor.new_zeros((tuple))` |
| `led` | `aten.as_strided.default` | the same wall, in `modeling_led.py:440` |

**Neither forwards, and neither is a partial result.** They stop at the same
line of the same borrowed function,
`diagonal_chunked_attention_scores.new_zeros((batch, chunks + 1, w, 2w + 1))`,
which is an **argument form**, not a missing op: `new_zeros` is implemented and
its schema takes `SymInt[]`, and this call passes a single tuple.
`docs/kernels/TAIL4.md` §8.2 recorded `fastspeech2_conformer` arriving at exactly this
shape one wall earlier (`torch.zeros((tuple), dtype=, device=)`), so it is a
third caller for one fix rather than two new gaps. It is data for the next
round in `docs/architectures/ARCH200.md`'s sense.

Worth saying plainly, because "moved to a new wall" has been used to dress up a
non-result before: what was verified here is that the chunked attention path
**runs**: `_chunk`'s overlapping `as_strided`, the `einsum` over it, and the
padding after it, and that the barrier did not fire anywhere inside either
model. A barrier that refused a write these models actually make would have
turned a gap into a regression, and it does not.

---

## 7. What is verified, and how

Split the way `AGENTS.md` §17.3 asks, because "one op" and "one test file" are
not the same unit of work:

* **Feature added (1):** `aten.as_strided.default`, plus `torch.as_strided` and
  `Tensor.as_strided` table rows.
* **Safety mechanism added (1):** `storage.rs::StridedBarrier` and its reader at
  the write door. It is not an op and does not appear in any op count.
* **Tests added:** 12 in `tests/ops/test_strided.py`, 29 golden cases.
* **Tests inverted (2), not deleted:**
  `test_tail4.py::test_the_as_strided_refusal_was_INVERTED_by_the_strided_round`
  and `…::test_the_as_strided_reach_allowlist_entry_was_removed_when_the_gap_closed`.
  `docs/kernels/FFT.md`'s round is the model and the previous bodies' own instructions
  said to do it this way; each keeps the reason the answer used to be no.
* **Documents corrected (1):** `docs/kernels/TAIL4.md` §1, marked superseded in place
  with the specific thing it got wrong (§3), not rewritten to have been right.
* **Deletions (1):** the `reach_allow.json` entry, as its own text instructed.

Numbers, all re-measured on this worktree:

<!-- DOCWATCH: count golden_cases_total ge 11336 -->
<!-- DOCWATCH: count golden_cases_passed ge 11336 -->
<!-- DOCWATCH: count golden_ops_covered ge 299 -->
<!-- DOCWATCH: count golden_pending eq 0 -->

* Golden: **11336/11336**, 299 ops, 0 pending builders (11307/298 at this
  worktree's base). The 29 new cases are four dtypes across five
  (size, stride, offset) shapes chosen where a plausible wrong gather differs,
  overlapping, zero-stride, wider-than-extent, explicit offset, plus the two
  degenerate ranks, the five upstream refusals, and the two write directions.
* `device.rs::MPS_HOST_READBACK_OPS` is **unchanged, and that is a result**:
  the index list is built on the host out of the *shape*, and no element of any
  tensor is read back. `test_none_of_this_rounds_kernels_reads_a_tensor_back_to_the_host`
  in `test_tail4.py` is the precedent for asserting this directly.
* `capture.rs` and `tape.rs` are untouched. The name does not end in `_`, so
  `is_mutating` does not claim it; it is out-of-place, so it is not in
  `_EXPECTED_MUTABLE`.

---

## 8. `as_strided_`: the receiver restrided over its own storage (issues #306, #380)

Status: **implemented, never run.** The kernel, the fork constructor, the Rust unit test, the
suite and the golden cases are written and compile-checked only (`cargo check --tests`,
`python -m py_compile`); nothing here has been built or run. Agent proposal, not
owner-confirmed; the constructor added to the fork is in the category the leader approved for
#380 (one more method in the same patch, like `narrow_with_step` and `view_with_layout`).

### 8.1 Why it is needed

CI log r382c (exir stage `xnnpack`): executorch's XNNPACK pass `ConvertToUpsampleBilinear2d`
runs a strict `torch.export.export` of an `F.interpolate(mode="bilinear")` module. Dynamo's
builder copies every input it wraps (`builder.py::cache_real_value_when_export` ->
`_clone_input` -> `torch/_dynamo/utils.py::clone_input`), and `clone_input` does:

```python
needed_size = sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride()))
result = torch.empty(needed_size + 32, dtype=dtype or x.dtype, device=x.device)
cache_line_offset = ((x.data_ptr() - result.data_ptr()) % 32) // x.element_size()
result.as_strided_(x.size(), x.stride(), cache_line_offset)
result.copy_(x.clone())          # a RuntimeError here falls back to torch.clone(x)
```

The shim had no `TensorBase.as_strided_` and raised `NotImplementedError` by name.

### 8.2 Upstream, measured on 2.13.0

* `x.as_strided_(size, stride, off)` returns `x` itself, with its storage, dtype and
  `requires_grad`; shape, stride and storage offset become the given ones.
* The bound is the **storage**, not what the receiver addressed:
  `arange(12.)[2:4].as_strided_((10,), (1,), 2)` succeeds and reads `[2., ..., 11.]`.
* The default offset is the receiver's own (`arange(12.)[3:].as_strided_((2, 2), (2, 1))`
  keeps offset 3).
* Checks, in order: `mismatch in length of strides and shape`; `as_strided: Negative strides
  are not supported at the moment, got strides: [...]`; `numel: integer multiplication
  overflow` (a negative size); `Tensor: invalid storage offset -1` (also for an empty layout);
  `setStorage: sizes [...], strides [...], storage offset N, and itemsize K requiring a storage
  size of B are out of bounds for storage of size S` (bytes; the same words on meta).
* An empty layout is accepted at any offset (`(0, 3), (3, 1), 100` on 12 elements).
* Views made before the call are separate tensors over the same buffer and keep their layouts:
  writes go both ways through the storage.
* View-ness does not change: a root stays a root (`_is_view()` False, `_base` None) at any
  offset or extent and is its later views' base; a view keeps its `_base`.
* The version counter moves 0 -> 1. A leaf that requires grad is refused outside `no_grad`
  (`a leaf Variable that requires grad is being used in an in-place operation.`); a non-leaf
  gets `AsStridedBackward1`. `clone_input` runs under `no_grad` on a fresh buffer.
* In `clone_input`, an expanded input's `copy_` raises the internal-overlap RuntimeError and
  the function returns `torch.clone(x)` instead (stride `(3, 1)` for a `(4, 3)` expand).

### 8.3 The mechanism

`aten.rs::as_strided_inplace` checks the arguments in upstream's order and words, resolves the
default offset, and calls `tensor.rs::PyTensorBase::restride_in_place`, which swaps a new candle
tensor into the wrapper, `t_`'s mechanism: the Python object, its tag, `requires_grad`, view
metadata and any `as_strided` barrier handle stay as they were. No second tensor exists, so no
barrier is taken (§2's barrier is about a gathered copy, which this is not). The new candle
tensor shares the storage `Arc`, so a write through it reaches every alias and the reverse, by
the single write door (`write_into`, which writes through the destination's layout into the
shared buffer). That was verified by reading, not run.

* **CPU:** upstream's rule exactly. The storage's byte length is read (`cpu_storage_nbytes`),
  the refusal is upstream's `setStorage` text, and the layout is built by the fork's new
  `Tensor::view_with_layout_in_storage`, which bounds by the storage's element count instead of
  by what the receiver addresses (`view_with_layout`'s rule).
* **CUDA and Metal:** the storage length is not readable here. A layout inside what the
  receiver already addresses is accepted (every element of it is in the storage); any other is
  refused by name rather than accepted unchecked.
* **Meta:** the same kernel; the bound is the meta storage's `storage_nbytes`, with upstream's
  message. `meta_stride_rule` lists the op as `OwnLayout`.
* **Quantized, Vulkan, Complex, Functional, Wrapper:** refused by name, matched exhaustively.
* **View-ness:** a tensor no view op made whose layout proves nothing (`layout_says_view`) is
  marked "not an autograd view" before the restride, so `_is_view()` stays False and `_base`
  None afterwards, as upstream. Without the mark, `clone_input`'s buffer (offset up to 7,
  footprint smaller than its storage) would answer `_is_view()` True and `_base` would refuse.
* **The one narrowing:** an empty layout past the end of the storage. Upstream accepts it;
  candle reads a contiguous tensor as `storage[offset .. offset + numel]`, which panics past
  the end even for no elements, so the shim refuses it by name (up to the storage's end is
  accepted). `clone_input` never asks for it (its offset is below 32 / itemsize inside a
  buffer 32 elements longer than needed).
* **Not reproduced, as `t_` does not:** the leaf-requires-grad refusal and `AsStridedBackward1`.

### 8.4 The fork change (one method)

`torchnative/rust/vendor/candle-core/src/tensor.rs` gains `Tensor::view_with_layout_in_storage`
beside `view_with_layout`, carried in `int8-candle-0.11.0-cpu.patch` (the `src/tensor.rs` section
regenerated from the published crate; `sh scripts/vendor/vendor_candle.sh --check` reported the
fork byte for byte equal to crate plus patch, run with the cached crates and no download). It
refuses a CUDA or Metal storage and the four packed dtypes by name, because their buffer length
is not their element count. Rust unit test:
`layout.rs::tests::view_with_layout_in_storage_is_bounded_by_the_storage_not_the_receiver`.

<!-- DOCWATCH: op-implemented aten.as_strided_.default -->
<!-- DOCWATCH: json-key torchnative/rust/torch_c/src/methods.json as_strided_ present -->
<!-- DOCWATCH: json-key torchnative/rust/torch_c/src/overloads.json as_strided_ present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/vendor/candle-core/src/tensor.rs view_with_layout_in_storage present -->
<!-- DOCWATCH: symbol-in-file tests/ops/test_asstrided_inplace.py test_clone_input_answers_upstreams_layout_and_values present -->

### 8.5 Registered

`_aten_implemented()` (`aten.as_strided_.default`), the dispatch arm, the meta arm and
`OwnLayout`, `methods.json` (`Tensor.as_strided_`) and `overloads.json` (`torch.as_strided_`)
with upstream's schema text, a golden builder (`cases.py::as_strided_inplace_cases`: 45 layout
cases over five dtypes, six refusals, one `c_error` case for the narrowing), and
`test_shim.py::_EXPECTED_MUTABLE`. The op is not tagged `core` upstream (`inplace`,
`inplace_view`, `pt2_compliant_tag`), so the core count does not move.

### 8.6 Tests (never run) and what turns each red

`tests/ops/test_asstrided_inplace.py`, shim against upstream in separate subprocesses, field
by field: 70 layout rows (identity, shape, stride, offset, contiguity, `_is_view()`, `_base`,
dtype, shared storage and its size, values), 37 refusals (type and whole message), 8 write
rows in both directions, 4 view-ness rows, 16 meta rows, `clone_input` itself on nine inputs
(contiguous, transposed, sliced with an offset, expanded, int64 transposed, a leaf requiring
grad, zero-size, 0-d, NCHW; the offset is compared as "inside one cache line" because its
value comes from two allocation addresses), the `torch.`, keyword, `torch.ops` and
`torch.Size` spellings, and the narrowing as a named refusal. The suite's docstring lists the
nullification that turns each test red.

What this cannot see: a CUDA or Metal receiver (no device run covers it), and whether the
strict export that called `clone_input` now goes further (`tests/export/test_exir_shim.py`
prints that report-only stage with its cause chain; `docs/design/FUNCTIONALIZATION.md` §5.3
has the reading of what comes next).
