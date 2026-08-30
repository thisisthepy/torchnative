# VIEWS.md — the four kernel gaps docs/GROUPED_MM.md §6.4 recorded

Binding the seven `TensorBase` members Mixtral needed (docs/GROUPED_MM.md §6.4) closed every
*name*. Doing so exposed four things that are not names: they are kernels that resolve and then
refuse, or capabilities the storage model does not have. §6.4 recorded them precisely rather than
writing them, because that change could not touch `aten.rs`. This document is what happened when
someone could.

Ordered smallest first, which is also the order they were done in.

| | gap | verdict |
|---|---|---|
| §1 | `aten.ge.Tensor` has no kernel | **kernel, one arm** — closed |
| §2 | `index_put_` refuses a bool mask | **kernel** — closed |
| §3 | `index_put_` refuses non-1-D operands | **kernel** — closed |
| §4 | `select.int`/`slice.Tensor` return copies, not views | **see §4** |

Baseline, before any of it (this worktree, `e50084f`):

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh    223 ok
$PY tools/golden/compare.py                  2971/2971, ops=121
$PY tools/golden/compare.py --self-test      13 comparators x 11 fault modes, 0 problems
$PY rust/torch_c/pytests/verify_schemas.py   4231/4231
```

---

## 1. `aten.ge.Tensor` — the sixth comparison, and the only one without a Tensor overload

`le.Tensor`, `lt.Tensor`, `gt.Tensor`, `eq.Tensor` and `ne.Tensor` all had a kernel.
`ge` had only `.Scalar`. §6.4 put both of `ge`'s schema strings into `methods.json` for symmetry
with the other five, which made `x >= tensor` **resolve** — and then refuse inside
`_aten_dispatch`, by name, on a key that was not in `_aten_implemented()`.

That is the good failure mode, and it is also exactly one arm of work.

### The change

`rust/torch_c/src/aten.rs`, two lines: `"aten.ge.Tensor"` in the implemented list, and

```rust
"aten.ge.Tensor" => compare_tensor(py, args, kwargs, "aten.ge.Tensor", Cmp::Ge),
```

There is no new arithmetic. `compare_tensor` already reads both operands into one exact common
representation (`f64` if either side is floating, `i64` otherwise) and `Cmp::Ge` was already a
variant of the enum, used by `ge.Scalar`. The five siblings are the same call with a different
`Cmp`.

### What the cases pin, and what they could not have pinned

`ge_tensor_cases` mirrors `gt_tensor_cases` deliberately, because the two kernels differ in one
enum variant and a wrong variant is the realistic defect. Three cases separate them:

* **`x >= x` is all `True`** where `gt.Tensor`'s matching case is all `False`. This is the
  assertion that distinguishes `Cmp::Ge` from `Cmp::Gt` and nothing else does.
* **NaN on either side is `False`**, including `nan >= nan` — `>=` is the reflexive comparison
  everywhere except on NaN, so a kernel that "helpfully" made it reflexive would pass every other
  case here.
* **the causal-mask idiom**, `arange(S)[:,None] >= arange(S)[None]`, which is `le.Tensor`'s
  measured case written the other way round.

Plus the eight dtypes × two broadcast scenarios the whole comparison family shares, and
`bool` compared as 0/1.

**Six of the cases go through the member**, not through `_aten_dispatch`: `x >= y`, `x.ge(y)` and
`x.__ge__(y)` on three dtypes, plus a 0-d right-hand side (which still picks `ge.Tensor`, since
the overloads are told apart by the argument's *type* and not its rank) and a NaN pair. That
distinction is §6.4's own lesson: the kernel-level cases for `clamp_`/`div_`/`masked_fill_` passed
for weeks while the members raised `NotImplementedError`. Here it runs the other way — the member
bound and the kernel refused — and the member cases are what fail if either half regresses.

### Sabotage

Deliberately broken, rebuilt, and counted rather than assumed:

| injected fault | golden cases failed | smoke tests failed |
|---|---:|---:|
| `Cmp::Ge` → `Cmp::Gt` on the new arm | **30** of 3002 | 1 (`test_the_mixtral_member_names_reach_the_kernels_that_were_already_there`) |
| the arm deleted entirely (back to refusing by name) | **31** of 3002 | — |

The two numbers differ by one, and the one is the `x >= x` equality-boundary case: it is the only
case in the suite whose *answer* changes between "computed with the wrong comparison" and "refused".
Every other case fails both ways, for different reasons — value mismatch in the first, one side
raising where the other computed in the second.

Restored from a `cp` backup afterwards and confirmed with `git diff --stat`.

### Counts after §1

```
run.sh                 223 ok       (unchanged — the new assertions are inside an existing test)
compare.py             3002/3002    ops=122   (+31 cases, +1 op)
compare.py --self-test 13 x 11, 0 problems    (unchanged — no new comparator)
verify_schemas.py      4233/4233    (+2: one more `_schema` text/is_mutable row and one more
                                     `OpOverload.tags` row, both derived from the implemented set)
```

One smoke-test constant moved and it is a real consequence, not a fixture edit:
`test_core_ops_and_op_tags_agree`'s `tag_core_count` went **77 → 78**. It counts *implemented* ops
that upstream tags `core`, and `ge.Tensor` is core upstream exactly as the `le`/`lt`/`gt` siblings
already counted there are. The test asserting a number rather than a range is what made that
visible in one line.

---

## 2-3. `aten.index_put_.default` — a bool mask, and operands above rank 1

These are two entries in §6.4's list and **one cause**. The kernel did not implement `index_put_`;
it built a `scatter.src` call and let that op do the work:

```rust
// before
let scatter_args = PyTuple::new(py, [receiver, 0i64, index, values])?;
let result = scatter_src(py, &scatter_args, None)?;
```

`scatter.src` requires an int32/int64 index, and index/src/self all of the same rank. So:

* a **bool mask** hit its dtype check — `Expected dtype int32 or int64 for index, got bool` — which
  §6.4 recorded as a `c_error` golden case rather than leaving uncased;
* **anything but rank 1** hit an explicit guard put in front of it
  (`only a 1-D self/index/values is implemented in torch._C shim`), which is why `x[t] = 5`
  refused: the number lifts to a 0-d tensor, and 0 is not 1.

Borrowing another op's arithmetic was the right call for the one shape Mixtral needed
(`inv_perm[perm] = arange(...)`, int64 and 1-D on every side). It does not generalise, because
`index_put_` and `scatter` do not mean the same thing.

### A mask is a different operation, not a cast

The tempting fix is to convert the mask to an integer index. That is wrong, and wrong with a
plausible shape, which is the failure mode this op invites: `x[tensor([True,False,True,False])]`
selects positions **0 and 2**, whereas the integer index `[1,0,1,0]` selects positions
**1, 0, 1, 0**. Same tensor, four elements either way, different answer.

The correct lowering is mask -> coordinates, and it is upstream's own move
(`at::native::expandTensors`). It was **already implemented here** as `mask_to_indices`, for
`index.Tensor`, along with the shape check and upstream's two-part error message. So `index_put_`
calls it rather than growing a second mask reader that could disagree with the first.

Two consequences that the tests pin because they are easy to get wrong:

* **A `k`-dimensional mask consumes `k` axes** and contributes one axis of length `count` to the
  result. The same mask values give different answers depending on the mask's rank —
  `x(2,3)[mask(2,3)] = [1,2,3]` writes three elements and `x(2,3)[mask(2,)] = [1,2,3]` writes a
  whole row. Both are cased, with the same numbers, so a kernel that reads the contents and
  ignores the rank fails one of them.
* **`uint8` is a mask too.** Upstream treats it as a deprecated spelling of `bool` and warns; both
  spellings are cased.

### The general kernel

`index_put_` now does its own address arithmetic. One index group at axis `a` consuming `m` axes,
and the indexing result shape is `dims[..a] ++ index_shape ++ dims[a+m..]` — always the spliced
form, because with **one** group `index.Tensor`'s fronting-versus-splicing rule (which needs two
separated groups to matter) cannot apply. `values` broadcasts onto that shape right-aligned, and
the walk is row-major over it.

Everything it does was measured against torch 2.13.0 first and reproduced second, including the
refusals:

| | upstream | here |
|---|---|---|
| bool / `uint8` mask | selects true positions | same, via `mask_to_indices` |
| `self` of any rank, 1-D index | indexing result `dims[..a] ++ S ++ dims[a+1..]` | same |
| index of any shape | spliced in whole | same |
| `[None, t]` (`x[:, t] = v`) | index sits at axis 1 | same |
| `values` broadcast | right-aligned onto the result | same |
| `values` that does not fit | `shape mismatch: value tensor of shape [3] cannot be broadcast to indexing result of shape [2, 2]` | verbatim |
| dtype mismatch | `Index put requires the source and destination dtypes match, got Float for the destination and Long for the source.` | verbatim, no promotion |
| negative index | wraps | wraps (`scatter` had no rule for this and refused) |
| out-of-range index | `index 9 is out of bounds for dimension 0 with size 5` | verbatim |
| mask shape mismatch | `The shape of the mask [3] at index 0 does not match the shape of the indexed tensor [4] at index 0` | verbatim |
| float index | `tensors used as indices must be long, int, byte or bool tensors` | verbatim |
| empty index / all-false mask | writes nothing, returns `self` | same |
| repeated position | last write wins | same |
| two index tensors | computes | **refused by name** — not measured, not guessed |
| `accumulate=True` | computes | **refused by name** — unchanged |

The two remaining refusals are deliberate and unchanged: refusing where upstream computes is a
recorded gap, and computing where upstream refuses is the silent divergence this repository does
not ship.

### One thing moved that is not in `aten.rs`: `_lift`'s dtype

`bootstrap.py`'s `__setitem__` lifts a bare Python number to a 0-d tensor before dispatching.
It inferred the dtype **from the Python type** — `int` -> `int64`, `float` -> `float32`.
Upstream does not. Measured with a `TorchDispatchMode` logger:

```
float32 x;  x[t] = 5     ->  lift_fresh(float32()) ...  index_put_
int64   x;  x[t] = 5.0   ->  lift_fresh(int64())   ...  index_put_
float32 x;  x[t] = True  ->  lift_fresh(float32()) ...  index_put_
bool    x;  x[t] = 2     ->  lift_fresh(bool())    ...  index_put_
```

**It is always the receiver's dtype.** The old rule survived because `index_put_` only accepted a
1-D receiver, so the one call that reached it was int64 on both sides — in the agreeing half. Both
halves are reachable now, and `index_put_` requires the dtypes to match exactly, so the old rule
would turn a write upstream performs into a *refusal*.

### Cases

`aten.index_put_.default` goes from 11 cases to 45, and 15 of those go through the **member**
(`x[mask] = v`, `x[idx] = 5`, `x[:, idx] = v`) rather than through `_aten_dispatch`. Every fault
below fails one door-level case and one member-level case, which is what that pairing is for.

Two case shapes are worth naming separately:

* **the ones that read the original binding.** `index_put_` returns `self`, so a case that reads
  the return value passes just as well against a kernel that built a fresh tensor and handed it
  back. Two cases throw the return value away and read the name that was passed in. That is the
  only shape that can fail when a write lands in a copy — the failure mode this whole document is
  about.
* **the ones that admit they cannot fail.** `x[:] = 3.0` on an int64 receiver and `x[:] = 2` on a
  bool receiver go through `fill_.Tensor`, and `fill_` takes its dtype from the receiver on both
  sides — so **neither of them discriminates between the two lift rules.** That is written into
  their note rather than implied away. The case that does discriminate is `x[idx] = 2` on a bool
  receiver, where the old rule refuses; it was added after sabotage F showed the `fill_` case
  staying green, which is what the sabotage pass is for.

### Sabotage

Each fault was injected into the built artefact, rebuilt, and counted:

| injected fault | golden failed | which cases | smoke |
|---|---:|---|---:|
| a mask always consumes exactly one axis | *aborted* | panicked at the 2-D-mask case (index out of bounds) | — |
| mask offsets drop the axis stride | 2 / 3037 | the 2-D mask, at the door and through the member | 1 |
| `values` broadcast left-aligned instead of right | 2 / 3037 | the `(2,)` broadcast and the 1-D-mask-over-2-D row | 1 |
| negative indices not wrapped | 2 / 3037 | the negative-index pair | 1 |
| leading un-indexed axes dropped from the destination stride | 2 / 3037 | the `[None, index]` pair | 1 |
| `_lift` back to inferring from the Python type | 3 / 3037 | the three number-on-the-right cases | 2 |
| the old rank-1 restriction reinstated | **18** / 3037 | everything §3 opened | 2 |

The first row is a real result and is reported as it happened: the fault made the offsets and the
result shape disagree and the kernel **panicked** rather than failing gracefully. That is a caught
fault, but an uncountable one, so the remaining faults were chosen to stay in bounds. Reaching that
panic from a *legitimate* input is not possible: `mask_to_indices` validates the mask's shape
against the receiver before any offset is built, integer indices are bounds-checked as they are
read, and the value offset is bounded by the broadcast check — so every write is inside the flat
buffer by construction.

Two of the seven faults are ones a reasonable implementer would actually write (the left-aligned
broadcast and the dropped leading axes), and both were caught by exactly one door case and one
member case, which is the resolution the case set was built for rather than a lucky wide net.

### Counts after §2-§3

```
run.sh                 225 ok       (+2: the two new tests below)
compare.py             3037/3037    ops=122   (+35 cases over §1)
compare.py --self-test 13 x 11, 0 problems    (unchanged)
verify_schemas.py      4233/4233    (unchanged -- no new key)
```

The two new smoke tests are `test_index_put_takes_a_mask_a_matrix_and_a_number`, which pins the
shapes that used to be *refused* so a regression to the `scatter` delegation fails by name and not
only by value, and `test_index_put_writes_into_the_receiver_and_not_into_a_copy`, which is §4's
question asked of the op §2-§3 just rewrote.

---

## 4. `select.int` and `slice.Tensor` — the views question

**Verdict: a storage-model redesign, not a kernel change. Not implemented, and §6.4's refusal
stands.** The reason §6.4 gave for it is wrong, though, and the correct reason changes what the
redesign has to be — so the refusal is kept and its justification is replaced.

### What §6.4 said, and what is actually true

§6.4, and the docstrings that carried its wording, said:

> `aten.select.int` and `aten.slice.Tensor` return copies, not views — a candle tensor is a value.

**That is measurable, and it is false.** candle's `Tensor` is `Arc<Tensor_>` and `Tensor_` holds
`storage: Arc<RwLock<Storage>>`; `narrow` and `squeeze` clone that `Arc` and rebuild only the
`Layout` (`candle-core-0.11.0/src/tensor.rs:902` and `:880`). So `select.int` returns a tensor that
**aliases its input's buffer**.

Measured rather than read off the source. candle's `same_storage` is `pub(crate)`, but `slice_set`
consults it and bails with a distinguishable message, which makes it an oracle from outside the
crate. A temporary `_probe.same_storage` key was added to `aten.rs`, built, measured, and removed:

```
x vs select.int(x, 0, 1)         SHARED
x vs slice.Tensor(x, 0, 1, 3, 1) SHARED
x vs slice.Tensor(x, 0, 5, 2)    independent buffers   (step > 1 goes through index_select)
x vs alias(x)                    SHARED
x vs detach(x)                   SHARED
x vs unsqueeze(x, 0)             SHARED
x vs view(x, [5])                SHARED
x vs clone(x)                    independent buffers
x vs a freshly built tensor      independent buffers
```

And the write, on the same build:

```
v = select.int(x, 0, 1);  copy_(v, 3.0)
x      -> [0,0,0,0,0]      unchanged
v      -> 3.0              changed
v vs x -> no longer shared
```

So the sequence does not fail because the view is a copy. It fails because **`copy_` replaced the
view wrapper's tensor with a fresh buffer instead of writing into the one it was pointing at.**
The copy is not in `select.int`; it is in `PyTensorBase::replace_with`.

This distinction is the whole reason to write §4 down. "Teach candle about views" is not the work
— candle already does views. The work is a write path, and that is somewhere else entirely.

### Why that makes it a redesign

**`replace_with` is the write primitive for every in-place op — 26 call sites.** It is defined as
"swap the wrapper's tensor", and its doc comment already says the consequence out loud: an alias
taken before the call does not see the write. `fill_`, `zero_`, `copy_`, `add_`, `relu_`,
`clamp_`, `div_`, `masked_fill_`, `uniform_`, `normal_`, `index_put_` and `set_`/`.data` all go
through it. Four things follow, and each of them is why the kernel-sized version of this fix would
be worse than the refusal:

1. **A partial fix is inconsistent in a way the refusal is not.** Making only `copy_` write
   through gives `x[0] = v` while `x[0].fill_(3)` still silently drops. Today the rule is uniform
   ("no in-place op is visible through an alias"), stated in one place, and the one path that
   would otherwise be silent refuses by name.

2. **Making writes observable creates a new divergence in the other direction.** `detach(x)`
   already shares storage with `x` — measured above — and that is harmless *only because nothing
   writes into storage*. The moment one op writes through, the ops that still swap become
   hazardous the other way: `y = detach(x); x.fill_(0)` moves `x` onto a fresh buffer and leaves
   `y` on the old one, where upstream has `y` see the fill. The divergence does not get closed by
   a partial write-through; it gets relocated and doubled.

3. **candle's public write surface does not cover the cases that would be needed.**
   `storage()`, `storage_mut()` and `same_storage()` are all `pub(crate)`. From outside the crate
   there are exactly two write-through paths:

   * `Tensor::slice_set(&self, src, dim, offset)`, which requires **both sides contiguous**,
     equal rank, matching shape off `dim` — and **refuses when the two share storage**, which is
     precisely what `x[0:2] = x[1:3]` is. A `select.int` along dim 0 is contiguous; along dim 1 it
     is not, so `x[:, 0] = v` is outside it. Building on `slice_set` alone would make some
     subscript writes work and leave the rest silently not working, which is the failure this
     refusal exists to avoid.
   * the `InplaceOp1/2/3` custom-op traits, which *do* receive the `&Layout` and so could write
     a strided view correctly. This is the real path, and it means the redesign is possible rather
     than blocked — but a custom op is written per dtype against `CpuStorage`, so it is a new
     component, not an edit.

4. **Most in-place kernels here cannot write through without being rewritten.** The dominant shape
   in `aten.rs` is `read_flat` -> compute into a `Vec` -> `write_flat`, which produces a new buffer
   by construction. That includes the `index_put_` §2-§3 just wrote. Write-through means
   re-expressing each of them as a masked write into an existing buffer.

There is also a modelling gap, though it is the smallest of the five: `PyTensorBase` has no notion
of "this wrapper is a view of that one". The *data* half is already there — the offset and strides
live in candle's `Layout` — so this is not the obstacle it would be against a value-typed tensor.
It matters for `.data`, `set_`, and anything that would need upstream's `_base`.

### The shape of the redesign, for whoever does it

Stated so that §4 is a decision to be taken rather than a wall:

1. Give `PyTensorBase` a **write-through mutation primitive** beside `replace_with` — something
   like `write_into(&self, region: &Layout, src: &Tensor)`, implemented as an `InplaceOp2` so it
   gets the layout and can handle a non-contiguous destination.
2. **Move all 26 in-place sites onto it**, in one change rather than incrementally. The
   inconsistency in point 1 above is a property of a partial migration, so a partial migration is
   the one thing not to do.
3. Then, and only then, delete `__setitem__`'s basic-index refusal branch.

`test_setitem_refuses_the_basic_index_write_rather_than_dropping_it` is the signal for step 3: it
asserts the **probe** — that the write through `select.int` does not reach the base — rather than
the refusal alone. When step 2 lands, that test goes red on the assertion and points at the branch
to delete. It is deliberately built that way and it is still green, which is the honest report of
where this stands.

### What did change here

Nothing in `aten.rs`. Three pieces of prose that gave the wrong reason were corrected, because a
wrong reason in a live docstring is what sends the next person to build the wrong fix:

* `bootstrap.py`'s `__setitem__` docstring, which said select and slice "return a copy, because a
  candle tensor is a value";
* the `NotImplementedError` message that branch raises, which said "this shim's select and slice
  return copies rather than views" — it now says the narrowing aliases correctly and the
  write-through is what is missing;
* `test_setitem_refuses_the_basic_index_write_rather_than_dropping_it`'s docstring and its
  assertion message.

The assertions themselves are unchanged, so the test still fails exactly when mutable views arrive.

---

## 5. Where this landed

|  | before | after |
|---|---|---|
| `run.sh` | 223 ok | **225 ok** |
| `compare.py` | 2971 / 2971, ops=121 | **3037 / 3037**, ops=**122** |
| `compare.py --self-test` | 13 comparators × 11 modes, 0 problems | unchanged |
| `verify_schemas.py` | 4231 / 4231 | **4233 / 4233** |
| `aten.ge.Tensor` | resolves, refuses | kernel |
| `index_put_` bool mask | refuses (`c_error` case) | kernel, cased against upstream |
| `index_put_` rank > 1 | refuses | kernel, cased against upstream |
| mutable views | refuses by name | **still refuses, with the correct reason** |

Golden cases by key: `aten.ge.Tensor` 31 (new), `aten.index_put_.default` 11 -> 45,
`aten.fill_.Tensor` 13 -> 15. Of the new cases, 21 go through a tensor **member** rather than
through `_aten_dispatch`.

Sabotage totals, all measured by injecting the fault, rebuilding, and counting — never by reading
a green run as proof:

| fault | golden failed |
|---|---:|
| `ge.Tensor` computed with `Cmp::Gt` | 30 |
| `ge.Tensor` arm deleted | 31 |
| mask offsets drop the axis stride | 2 |
| `values` broadcast left-aligned | 2 |
| negative indices not wrapped | 2 |
| leading un-indexed axes dropped | 2 |
| `_lift` back to the Python type | 3 |
| the old rank-1 restriction reinstated | 18 |
| a mask forced to consume one axis | harness panicked (uncountable, but caught) |

One case was found by that pass to be incapable of failing (`x[:] = 3.0` through `fill_`, which
takes its dtype from the receiver on both sides regardless of the lift rule) and a discriminating
one was added beside it rather than the note being left to imply a guarantee it did not give.
