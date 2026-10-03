# The tenth was not the representation question it was recorded as

`docs/graph/LIFTFRESH.md` §6 left `torch.export` at **9 of 10** under the
replay-and-agree bar and named the tenth precisely: `canine`, stopping at
`aten.zeros_like.default` on a non-contiguous meta tensor, in
`docs/graph/STRIDE.md` §3's territory -- *a dense tensor here cannot be built
with a caller-chosen stride*.  It said that was a **representation** question
rather than an operator, and declined to attempt it for that reason rather than
leaving it as "not gotten to".  That was the right thing to do with what was
measured.

**Re-measured at the head of this round, `canine` does not stop there.**  It
stops one wall earlier, and the earlier wall is not a representation question
at all.

Read these four things first:

* **The count moved: 9 of 10 -> 10 of 10.**  That is upstream's ceiling: the
  other thirty of forty are refused by upstream itself in this sweep, so ten is
  full marks.  §4.  The ten also agree with upstream's *own* replay, 10/10
  within 1e-05.
* **The wall was `aten.constant_pad_nd.default`, not `zeros_like`** -- and it
  was `docs/graph/EXPORT6.md` §5's recurring shape: an operator **already in
  `_aten_implemented()` that merely lacked a meta kernel**.  The brief asked to
  check for that shape first; this is the sixth time it has been the answer.
  §1.
* **`STRIDE.md` §3's dense-stride limit is still real and still unmoved**, and
  nothing in this round argues with it.  It simply was not what `canine` was
  standing behind.  §3 says what would have had to change had it been, and why
  that estimate is now the *only* thing keeping it on the board -- no
  architecture in this sweep reaches it any more.
* **The layout is asserted as a difference between two branches**, not as a
  stride written beside the kernel, because `constant_pad_nd` has two rules and
  a single-rule kernel gets one of them right.  §2, and §5's N2/N3.

---

## 1. What `canine` actually stopped at

The sweep, re-run before anything was changed:

```
NotImplementedError: torch._C shim: the meta kernel for
  aten.constant_pad_nd.default has not been shown to produce upstream's output
  stride, and one of its inputs is not laid out contiguously
  (shape [1, 64, 8], stride [512, 1, 64])
```

That message is `MetaStrideRule::Unverified`'s gate, which `STRIDE.md` §3 put
in front of every meta arm whose layout claim had not been measured.  The gate
fired, so the arm behind it was never reached -- and **there was no arm behind
it.**  `meta_table` had no `constant_pad_nd` entry at all, so the op would have
refused on a *contiguous* meta input too, with a different message.  The gate
was reporting the wrong reason for the right refusal.

So there are two separate facts and only the first is `canine`'s:

| | |
|---|---|
| the op has no meta kernel | **`canine`'s wall.**  Closing it is shape-and-stride inference, no representation change |
| a dense tensor cannot carry a caller-chosen stride | real, `STRIDE.md` §3, and **not on `canine`'s path** |

`zeros_like` on a non-contiguous meta tensor is the second, and it is the
refusal `LIFTFRESH.md` §6 recorded.  `relay_elementwise` still raises it and
that arm is untouched.  What changed is only which wall comes first: with
`constant_pad_nd` closed, `canine` reaches the end without ever presenting a
non-contiguous meta tensor to a dense-device `zeros_like`.  §3.

## 2. Upstream's layout rule, measured

`torch/_meta_registrations.py::_constant_pad_nd_meta`, confirmed against a
separate upstream subprocess over **182 layout x pad cases, 180 answered**
(`pytests/test_canine.py::test_constant_pad_nd_meta_layout_agrees_with_upstream`).
Strides are integers, so the comparison is exact and there is no tolerance
here to widen.

There are two branches and they answer **differently on the same input**:

```
(3, 4) stride (1, 3)   pad [ 1,  1]  ->  (3, 6) stride (6, 1)    a fresh buffer
(3, 4) stride (1, 3)   pad [-1, -1]  ->  (3, 2) stride (1, 3)    the input's layout
```

* **every entry `<= 0`**: the input is `narrow`ed and `clone`d, so the output
  carries the **preserve_format** stride of the *narrowed* shape over the
  *input's* stride.  A narrow changes an extent and never a stride, which is
  why one call to `layout::preserve_format_stride` covers it.
* **otherwise**: `empty(new_shape, memory_format=suggest_memory_format(input))`
  -- contiguous for every layout except a channels-last-*strided* one, which
  stays channels-last, at rank 4 and rank 5 alike.

### 2.1 Three things the measurement decided that reading would have got wrong

* **The branch test is `p <= 0`, not `p < 0`.**  The meta registration and the
  `_refs` decomposition genuinely differ here -- `_refs` says `all(p < 0)`, and
  comments at length about *not* early-exiting on an all-zero pad, which is
  exactly the case the meta registration does early-exit on.  The meta
  registration is what the dispatcher reaches.  So `constant_pad_nd(t, [0, 0])`
  of a transposed `t` **keeps `(1, 3)`** rather than contiguating, and a rule
  copied from `_refs` is wrong on that row and on no other.  It is in the
  matrix for that reason.
* **Upstream's two channels-last ambiguity fallbacks both apply**, and they
  pull in opposite directions, so a "is any stride out of order" test cannot
  stand in for `suggest_memory_format`.  An `N111` whose batch stride equals
  its channel stride (`(2,1,1,1)` stride `(1,1,1,1)`) comes out **contiguous**;
  a unit-channel `(2,3,1,1)` stride `(3,1,3,3)` comes out **channels-last**.
  `layout::strides_like_channels_last` already had both, from `cat`'s round.
* **An overlapping input is not a special case.**  An expanded `(3,4)` stride
  `(1,0)` answers contiguous under a positive pad and contiguous under a
  negative one -- the latter because `preserve_format_stride` falls back to the
  elementwise permutation for a layout that is not non-overlapping-and-dense.
  No extra arm; the existing helper is already upstream's rule.

## 3. What `STRIDE.md` §3's dense limit would still have to become

This round did not need it, and that is a change in its status worth recording
rather than a reason to drop it: **after this round no architecture in the
forty reaches it.**  It is now a limit with no measured caller, which is a
weaker case for paying its price than `LIFTFRESH.md` §6 implied.

The refusal is in `relay_elementwise`: a meta input whose stride is not
contiguous, an op whose `device=` puts the result on a dense device.  Upstream
lays that dense result out following the input; this build lays every dense
tensor out contiguously, because a dense tensor here **is** a
`candle_core::Tensor` and that is the representation question.

Candle's `Tensor` does carry a `Layout` -- shape, stride and a start offset --
so it can *describe* a non-contiguous tensor; that is how `transpose` and
`narrow` work there.  What it has no constructor for is **allocating a fresh
tensor into a caller-chosen stride**: `Tensor::zeros`/`ones`/`from_vec` all
build a contiguous `Layout` over a fresh storage, and the only way to reach
another stride is to take a view of something that already exists.  So the
change is not "teach the dense side about strides".  It is one of:

1. an `empty_strided`-shaped constructor in candle (a vendored patch, on top of
   the `int8-candle` patch already carried), plus every kernel here learning
   that its input may be non-contiguous where today `.contiguous()` is called
   defensively; or
2. allocating the contiguous buffer and returning a **view** of it with the
   requested stride -- which works only when the requested stride is
   non-overlapping and dense, i.e. exactly the layouts a permutation of a
   contiguous buffer can reach.  That covers the `zeros_like`-of-a-permuted-
   input case and does **not** cover an expanded or gappy stride, so it would
   have to refuse by name for the rest and the refusal would not go away.

Option 2 is bounded and cheap and would close the recorded `zeros_like` case;
option 1 is the one that makes the limit disappear.  **Neither is attempted
here**, because with `canine` at 10 of 10 there is no measured caller to judge
either against, and `STRIDE.md` §3's own argument -- that a partial stride model
leaves a tensor whose `stride()` lies -- applies to option 2 unless every dense
kernel is audited for the assumption.  It stays on the board, named, with no
caller.

## 4. The numbers, under the bar

`export_sweep.py --limit 40`, both sides.  Load average was **1.93** at the
start of the shim sweep and **1.69** at the start of the upstream one
(`uptime`); the other worktree's round (`tn-anesubgraph`) was running
throughout, which is recorded because `docs/graph/VARMEAN.md`'s bisect was
ruined by not recording it.  These are pass/fail verdicts rather than timings,
so load bears on how long they took and not on what they say.

```
                                     before   after
architectures swept                      40      40
upstream exported+replayed+agreed        10      10
this shim, of those ten                   9      10     <- THE BAR
```

**The `after` column is a full forty-architecture sweep on both sides, run
here.  The `before` column is not**, and saying so is the point of this
paragraph: it is `LIFTFRESH.md` §5's forty-architecture measurement, plus a
re-run of `canine` *alone* at the head of this round, which failed at
`constant_pad_nd` (`0/1 exported+replayed+agreed`).  The other thirty-nine
were not re-swept before the change.  What that leaves unverified is whether
anything else had drifted since `LIFTFRESH`; the `after` sweep answers that
for the ten that matter -- all ten agree -- so the only claim resting on the
un-re-run `before` column is the *number* 9, and the direction of the move is
measured on both ends for `canine` itself.

**9 of 10 -> 10 of 10**, which is upstream's ceiling.  Not "export() returned":
each of the ten returned an `ExportedProgram`, replayed it, and agreed
element-wise with its own eager module -- and separately with *upstream's*
replay:

```
cross-side: shim replay vs upstream replay, 10 comparable
            agreeing within 1e-05: 10/10
            worst:  aimv2_vision_model 1.0e-06   altclip 1.0e-06   blip 1.0e-06
                    canine 1.0e-06   albert 1.0e-06   beit 1.0e-06
                    bert 1.0e-06     big_bird 1.0e-06
                    bert-generation 0.0   camembert 0.0
```

| architecture | before | after |
|---|---|---|
| the nine of `LIFTFRESH` §5 | agrees | **agrees** |
| `canine` | `aten.constant_pad_nd.default` on a non-contiguous meta tensor | **agrees** |

### 4.1 The reproducer

`canine`'s shape of expression with no dependence on `transformers` -- a
transpose, a padding pass, a cropping pass -- clears the full bar rather than
the export stage:

```
CANINEEXPORT: exported+replayed+agreed, worst relative 0.000e+00;
              4 distinct call targets including constant_pad_nd
```

The tolerance is derived, `docs/numerics/AGREE.md` §2's method -- the p90 of
upstream's own float32-vs-float64 relative error on these very outputs, floored
at 8 ulp.  There is no constant in the test for anyone to widen.  The
call-target assertion is `docs/graph/COMPILE.md`'s: an `ExportedProgram` holding
no operators would replay and agree trivially, so `constant_pad_nd` is required
to be *in* the graph.

## 5. Nullification: 4 attempted, 0 uncaught

Each written into the source, `touch`ed (`EXPORT6` §4.1's stale-mtime trap),
rebuilt, installed, run, reverted.

| | nullification | red |
|---|---|---|
| N1 | the whole arm absent (the round's starting state) | 4 tests |
| N2 | the narrow branch answers `contiguous` instead of `preserve_format` | 2 |
| N3 | the branch test is `_refs`' `p < 0` instead of the meta rule's `p <= 0` | 2 |
| N4 | the channels-last branch removed | 2 |

**N2 is the one worth keeping.**  It is the implementation a reader would reach
for first -- the op fills a fresh buffer, so the fresh buffer is contiguous --
and it is right for every positive pad, which is every pad a reader is likely
to try.  It fails only on the non-positive branch, and only because the layout
is asserted as a *difference between the branches* rather than as a stride
beside the kernel.

**The export test stayed green under N2, N3 and N4, and that is reported rather
than smoothed over.**  It is a bar test, not a layout test: a wrong meta stride
that is still a legal stride does not change the replayed numbers for this
module.  The layout tests are what catch those three, which is why the round
has both kinds and not only the headline one.

## 6. Left open, named

* **`STRIDE.md` §3's dense stride limit** -- §3.  Still real, now with **no
  architecture in the forty reaching it**, and with the two possible shapes of
  fix costed.
* **A non-contiguous `FakeTensor` reaches `TensorBase._base`** -- unchanged
  from `LIFTFRESH.md` §6.
* **`UntypedStorage._expired`** -- refuses; `LIFTFRESH.md` §2's reason.
* **`aten.detach_.default`** -- upstream emits it into the exported graph for a
  tensor literal and this shim does not; unchanged.
* **The per-op `Meta` dispatch-key predicate of `VARMEAN` §4.1** -- still real,
  still unlanded, still not touched.
* **The thirty upstream itself refuses.**  Ten is this sweep's ceiling, not
  forty, and nothing here makes the other thirty this shim's deficit.  Moving
  any of them means first making *upstream* export them.

## 7. This round, separated

**Features added** -- capability the shim did not have:

1. `aten.constant_pad_nd.default` on **meta**: shape, dtype and both of
   upstream's layout branches (§1, §2).  The dense kernel is untouched.

**Defects fixed**: none.  The `Unverified` gate was reporting the wrong reason
for `constant_pad_nd` -- it named the input's layout when the op had no meta arm
at any layout -- but the refusal itself was correct and the gate is behaving as
`STRIDE.md` §3 designed it to.  It is not counted as a defect.

**Tests added**: 4, all in `rust/torch_c/pytests/test_canine.py`, each comparing
against upstream torch's own answer in a second subprocess.

**Tests corrected**: none.

**Docs corrected**: 1.  `docs/graph/LIFTFRESH.md` §6 names `canine`'s wall as
`aten.zeros_like.default`; re-measured it is `aten.constant_pad_nd.default`.
The correction is made *here*, in §1, rather than by editing that file, which
is `LIFTFRESH.md`'s own convention for a superseded reading -- and the
distinction matters, because the sentence that was wrong is the one that said
the tenth needed a representation change.

**Docs**: this file.

**Removed**: nothing.

**Golden cases**: none added.  `aten.constant_pad_nd.default` is already in
`_aten_implemented()` with golden cases from `docs/architectures/ARCH20.md` §2;
meta support is a property of an op already on the list and the golden harness
compares values, which a meta tensor has none of (`docs/devices/META.md` §7).
Op coverage is unchanged, which is why the gate's `ops=` count does not move.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs strides_like_channels_last present -->
<!-- DOCWATCH: op-implemented aten.constant_pad_nd.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_canine.py test_constant_pad_nd_meta_layout_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_canine.py test_constant_pad_nd_meta_negative_pad_preserves_where_positive_pad_contiguates present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_canine.py test_constant_pad_nd_meta_keeps_a_channels_last_input_channels_last present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_canine.py test_padding_a_noncontiguous_tensor_exports_replays_and_agrees present -->
