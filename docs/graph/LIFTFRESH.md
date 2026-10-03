# One operator was the brief; three walls stood in a line behind it

`docs/graph/STRUCTSEQ.md` §5 left `torch.export` at **7 of 10** under the
replay-and-agree bar and named this round's whole task: `aten.lift_fresh_copy
.default`, the one operator `big_bird`, `blip` and `canine` all stopped at.

That was right, and it was not sufficient. The operator was closed in the first
hour and **the count did not move.** What the sweep then showed was that
`lift_fresh_copy` was not one wall but the first of four in a single code path
-- fake mode's *constant propagation* -- each one unreachable until the one in
front of it fell:

```
aten.lift_fresh_copy.default                 no kernel at all, dense or meta
  -> UntypedStorage._weak_ref                inherited `raise NotImplementedError`
     -> torch.Storage                        the alias was never set
        -> torch._C._SchemaInfo.is_mutable   a generated placeholder
           -> Tensor(a!)[] alias parsing     a parser defect the oracle found
```

Read these four things first:

* **The count moved: 7 of 10 -> 9 of 10**, and the nine agree with upstream's
  own replay cross-side, 9/9 within 1e-05. §5.
* **Ten is still upstream's ceiling.** The other thirty of forty are refused by
  upstream itself. Nobody should read a denominator of 40 into this.
* **The tenth is `canine`, and it is not in this chain.** It stops at
  `docs/graph/STRIDE.md` §3's structural limitation -- a dense tensor here
  cannot be built with a caller-chosen stride -- reached through
  `aten.zeros_like.default` on a non-contiguous meta tensor. That is a
  representation question, not an operator, and it is untouched. §6.
* **The last two walls were not predicted; they were found by the sweep and by
  an exhaustive oracle.** §4 is a defect this round's own test caught in
  *existing* code, and it had nothing to do with `lift_fresh_copy`.

---

## 1. What `lift_fresh_copy` actually has to guarantee

Measured on both sides in separate subprocesses before anything was written,
because the brief asked what it *guarantees* rather than what it computes.

`aten::lift_fresh_copy(Tensor self) -> Tensor`. It exists so that a tensor
built from a Python literal stops aliasing whatever it was built over:
functionalisation rewrites `aten.lift_fresh.default` into it, and the rewrite
is the point.

Two properties separate it from the ops it looks like, and each has a test:

| | aliases its input | output layout |
|---|---|---|
| `aten.lift_fresh.default` | **yes** -- identity | its input's |
| `aten.clone.default` | no | **its input's**, preserved |
| `aten.contiguous.default` | yes, when already contiguous | contiguous |
| `aten.lift_fresh_copy.default` | **no** | **contiguous** |

The layout row is the load-bearing one and it was measured, not assumed. On a
transposed `(4, 3)` input of stride `(1, 4)`:

```
lift_fresh_copy  ->  stride (3, 1)     a fresh contiguous buffer
clone            ->  stride (1, 4)     the input's layout, preserved
```

So it is `contiguous`'s layout with `clone`'s unconditional allocation, and it
is neither of those kernels reused -- `contiguous_blocked` hands back the input's
own storage when the input is already contiguous, which would make it an alias.
`test_lift_fresh_copy_is_contiguous_where_clone_preserves` asserts the
*difference* between it and `clone` rather than just its own stride, so a
kernel that spelled one as the other cannot pass.

### 1.1 Meta or dense: **both**, and for two different reasons

The brief asked to check this shape first, because four rounds this week were
caught by an operator already in `_aten_implemented()` that merely lacked a meta
kernel. **This was not that shape.** `lift_fresh_copy` had no kernel at all:

```
dense:  NotImplementedError: aten op not implemented in torch._C shim
meta:   NotImplementedError: torch._C shim has no meta kernel for ...
```

Both are needed, and not merely because tracing touches both. Fake mode's
`_dispatch_impl` runs the **real** kernel on the real constant and wraps the
result with `from_real_tensor(make_constant=True)`, so the *dense* kernel is on
the tracing path too; and the `ExportedProgram` replays the operator on the real
constant afterwards. The meta kernel is `MetaStrideRule::AlwaysContiguous`, for
§1's reason, not the `PreserveFormat` that `clone` uses.

### 1.2 `lift_fresh` was deliberately left alone, and that is a measurement

It is already implemented as the identity it is upstream, and **it never
appears in an exported graph**: functionalisation has rewritten it before the
graph is built. Measured on both sides and asserted in
`test_export_emits_lift_fresh_copy_and_not_lift_fresh`, so the choice is a fact
in the suite rather than a preference in a commit message. Changing or adding
to it would have unblocked nothing.

---

## 2. Wall two: `_weak_ref`, and what the caller actually wanted

With the operator in, all three architectures moved together to:

```
fake_tensor.py:388   weak_st = StorageWeakRef(fake_tensor.constant._typed_storage())
reductions.py:32     self.cdata = storage._weak_ref()
storage.py:177       raise NotImplementedError
```

`storage.py:177` is upstream's own `_StorageBase` abstract stub -- byte-identical
in the vendored tree and in the installed wheel. Upstream overrides it on the C
`UntypedStorage`; this shim did not.

**What that caller needs is an identity, not a weak reference.**
`StorageWeakRef.__hash__` and `__eq__` read `cdata` and nothing else, and
`fake_tensor.py` **never calls `_expired`** -- it tracks liveness with Python
`weakref.ref` on the *tensors* and uses `StorageWeakRef` purely as a dict key.
So `_weak_ref` answers `_cdata`, which is already this shim's storage identity
and already has upstream's relation: shared between a storage and its views,
distinct across storages.

`_free_weak_ref` is a no-op, because nothing was retained -- and it has to exist
and not raise, since `StorageWeakRef.__del__` calls it and an exception there is
only *ignored*.

**`_expired` is left refusing, deliberately.** Answering `False` is the cheap
way to make it return something and it is a claim that is wrong exactly when it
matters. `test_expired_still_refuses_rather_than_guessing` holds the refusal in
place so that answering it later requires a liveness mechanism rather than a
constant.

### 2.1 `torch.Storage` was simply not set

`StorageWeakRef.__init__` reads `torch.Storage._free_weak_ref`. Upstream's
`_initExtension` sets that alias and it is `FloatStorage` there, not
`UntypedStorage` -- measured on 2.13.0, where `torch.Storage is
torch.FloatStorage` is True and `is torch.UntypedStorage` is False. It is set
from the same class here, in the same place, for the same reason.

---

## 3. Wall three: `_SchemaInfo`, a derivation that turned out to have a table

Next in the same function:

```
fake_tensor.py:3300   if any_constant and schema_info.is_mutable():
```

`invalidate_written_to_constants` asks whether an op *writes* to an argument, so
that a write to a traced constant invalidates every alias of it.
`torch._C._SchemaInfo` was a generated placeholder and every call raised.

It is a derivation, not new information: an argument is mutated exactly when it
carries a write alias annotation, `Tensor(a!)`, and this shim's
`arguments[i].alias_info.is_write` already agreed with upstream's. So it is
written in Python over the parsed schema rather than in Rust over the text.

**The oracle is upstream over all 2584 `- func:` entries** of the vendored
`native_functions.yaml` -- the same file the shim parses -- for `is_mutable()`,
`is_mutable(name)` for every argument, and `has_argument`, including a name that
is in no schema so that a constant `True` cannot pass. 987 mutable, 1597 not, so
the sweep contains both answers and "agrees with upstream" is not satisfiable by
a constant.

### 3.1 The derivation is wrong for exactly 14 pairs, and the test found them

Not predicted -- the 2584-schema comparison failed on `batch_norm`:

```
aten::batch_norm  running_mean  annotation says False   upstream says True
```

Upstream's `SchemaInfo` hardcodes that the batch-norm family writes
`running_mean`/`running_var`. It cannot be spelled in a static alias annotation
because the mutation is **conditional on the `training` argument**, so upstream
answers the conservative `True`. Exactly 14 (op, argument) pairs over 7 names,
and the table is copied as measured.

**The key includes the overload, and that is not tidiness.** Upstream's table
contains `native_batch_norm.out` but **not** `cudnn_batch_norm.out`, whose
`running_mean`/`running_var` it reports as *not* mutable even though
`cudnn_batch_norm`'s are. Keying on the base name made this shim answer `True`
for `cudnn_batch_norm.out` and the comparison caught it on the next run. There
is no rule behind that asymmetry to infer; it is a hand-maintained list.

---

## 4. Wall four: a parser defect in existing code, found by the oracle

The same comparison then failed on an op with nothing to do with any of this:

```
aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, ...)
    upstream:   self.alias_info.is_write == True
    this shim:  self.alias_info is None
```

`_parse_argument` required the type spelling to **end** with `)`:

```python
if "(" in spelling and spelling.endswith(")"):
```

True for `Tensor(a!)`, false for `Tensor(a!)[]`. So **a mutable list of tensors
carried no `alias_info` at all**, and every consumer -- not just `_SchemaInfo`,
but `FunctionSchema.is_mutable` and the aliasing predicates beside it -- read it
as non-mutating. The annotation is the first parenthesised group and a list or
optional suffix can follow it; it is read that way now.

This is a **defect fixed in code that predates this round**, and it is listed
separately in §7 for that reason. It was not reachable from the brief; it was
reachable from choosing an oracle that could disagree.

---

## 5. The numbers, under the bar

`export_sweep.py --limit 40`, both sides, run solitary. Load average was 1.3
and 2.1 at the starts of the two sweeps that bracket the change (checked with
`uptime`); the other worktree's round was idle throughout. These are pass/fail
verdicts rather than timings, so load bears on how long they took and not on
what they say -- but it is recorded because `docs/graph/VARMEAN.md`'s bisect
was ruined by not recording it.

```
                                     before   after
architectures swept                      40      40
upstream exported+replayed+agreed        10      10
this shim, of those ten                   7       9     <- THE BAR
```

**7 of 10 -> 9 of 10.** Not "export() returned": each of the nine returned an
`ExportedProgram`, replayed it, and agreed element-wise with its own eager
module -- and separately with *upstream's* replay:

```
cross-side: shim replay vs upstream replay, 9 comparable
            agreeing within 1e-05: 9/9
            worst:  aimv2_vision_model 1.0e-06   altclip 1.0e-06   blip 1.0e-06
                    albert 1.0e-06   beit 1.0e-06   bert 1.0e-06
                    big_bird 1.0e-06
                    bert-generation 0.0   camembert 0.0
```

Where the ten stop:

| architecture | before | after |
|---|---|---|
| the seven of `STRUCTSEQ` §5 | agrees | **agrees** |
| `big_bird`, `blip` | `aten.lift_fresh_copy.default` | **agrees** |
| `canine` | `aten.lift_fresh_copy.default` | `aten.zeros_like.default` on a non-contiguous meta tensor (§6) |

All forty, export-stage failures only, by first wall:

| wall | `STRUCTSEQ` §5 | after |
|---|---:|---:|
| `aten.lift_fresh_copy.default` | 9 | **0** |
| `zeros_like` on a non-contiguous meta tensor | 0 | 1 |

### 5.1 The four-line reproducer

A module with two Python literals in its `forward` is the export test, and it
clears the full bar rather than the export stage:

```
LFCEXPORT: exported+replayed+agreed, worst relative 0.000e+00;
           3 distinct call targets including lift_fresh_copy
```

The tolerance is derived, `docs/numerics/AGREE.md` §2's method -- the p90 of
upstream's own float32-vs-float64 relative error on these very outputs, floored
at 8 ulp. There is no constant in the test for anyone to widen. The
call-target assertion is `docs/graph/COMPILE.md`'s: an `ExportedProgram` holding
no operators would replay and agree trivially, so `lift_fresh_copy` is required
to be *in* the graph.

---

## 5.2 The gate

Two full runs. The first attempt is reported too, because it failed and the
failure was real.

```
attempt 1   GATE_EXIT=1   suites 91/91   ok=1753   FAIL=1
            FAIL test_reach_every_declared_name_reaches_a_kernel_and_every_kernel_a_name
            -- `aten.lift_fresh_copy.default` has a kernel and golden cases and
               no Python spelling reaches it.  §7's allowlist entry is the fix.
            The golden self-test and DOCWATCH never ran: `run.sh` puts them
            after the suite ledger, so a suite failure hides both.

run A       GATE_EXIT=0   suites 91/91   ok=1754   FAIL=0   SKIP=0
run B       GATE_EXIT=0   suites 91/91   ok=1754   FAIL=0   SKIP=0

            both runs, identically:
            golden self-test  PASS -- 26 comparators x 11 fault modes,
                              0 problems, 0 comparators never exercised
            golden            11627 cases, 0 failed, 0 pending, ops=308
            DOCWATCH          PASS -- 1313/1313
            VULKAN            42 ran (42 ok, 0 failed) / 0 skipped
```

Against the `develop` baseline of 1745 ok / 0 FAIL / 90 suites / DOCWATCH 1304
/ golden 11606 cases, ops=307, every delta accounts for itself:

| | baseline | here | why |
|---|---:|---:|---|
| ok | 1745 | 1754 | **+9** = this round's 9 new tests |
| FAIL | 0 | 0 | |
| suites | 90 | 91 | **+1** = `test_liftfresh.py` |
| DOCWATCH | 1304 | 1313 | **+9** = this round's 9 markers |
| golden ops | 307 | 308 | **+1** = `aten.lift_fresh_copy.default` |
| golden cases | 11606 | 11627 | **+21** = its 21 cases, 0 failed |

The golden rows are the load-bearing ones for §4, which changed the *parser*
that every schema goes through: 11627/11627 pass and 0 fail, so no kernel's
numbers moved.

---

## 6. Left open, named

* **`canine`** -- `aten.zeros_like.default` of a non-contiguous meta tensor onto
  a dense device. `docs/graph/STRIDE.md` §3 already analysed it: a dense tensor
  here cannot be built with a caller-chosen stride. It is a representation
  change, not an operator, and it is the whole of the distance from 9 to 10.

  **Superseded -- `docs/graph/CANINE.md` §1.** Re-measured at the head of the
  next round, `canine` stops one wall *earlier*, at
  `aten.constant_pad_nd.default` on a non-contiguous meta tensor: an operator
  already in `_aten_implemented()` that merely lacked a meta kernel, with no
  representation question in it. Closing that took the count to **10 of 10**.
  `STRIDE.md` §3's dense stride limit is still real and still unmoved; it was
  simply not what `canine` was standing behind, and no architecture in the
  forty reaches it now.
* **A non-contiguous `FakeTensor` reaches `TensorBase._base`**, which this shim
  does not implement -- for **every** copy-shaped operator, `clone` and
  `contiguous` identically to `lift_fresh_copy`. Not this operator's gap, and
  held down as "all three behave alike" by
  `test_a_noncontiguous_fake_tensor_is_not_this_operators_gap` rather than
  omitted, so it fails if `lift_fresh_copy` ever becomes the odd one out.
  `torch.export` does not reach it: a Python literal's constant is contiguous.
* **`UntypedStorage._expired`** -- refuses; §2's reason.
* **`aten.detach_.default`** -- upstream emits it into the exported graph for a
  tensor literal and this shim does not. Asserted to be *exactly* that one
  target of difference, so any other drift fails, but not otherwise analysed.
* **The per-op `Meta` dispatch-key predicate of `VARMEAN` §4.1** -- still real,
  still unlanded, still not touched.

---

## 7. This round, separated

**Features added** -- capability the shim did not have:

1. `aten.lift_fresh_copy.default`, dense and meta (§1).
2. `UntypedStorage._weak_ref` / `_free_weak_ref`, and the `torch.Storage`
   alias (§2).
3. `torch._C._SchemaInfo` -- `is_mutable`, `has_argument` (§3).

**Defects fixed** -- wrong answers in code that predates this round:

1. `_parse_argument` dropped the alias annotation of a list-typed argument, so
   `Tensor(a!)[]` read as non-mutating everywhere `alias_info` is consulted
   (§4).

**Tests added**: 9, all in `rust/torch_c/pytests/test_liftfresh.py`, each
comparing against upstream torch's own answer in a second subprocess.

**Tests corrected**: none.

**Docs corrected**: none. `STRUCTSEQ` §5 is not wrong that `lift_fresh_copy`
was where the three stopped; it is this file's starting point.

**Docs**: this file.

**Removed**: nothing.

**Golden cases**: `aten.lift_fresh_copy.default` gets a builder, 21 cases over
7 dtypes and 3 shapes, so the op does not land in `_aten_implemented()` without
a comparison.

**Reachability allowlist**: one entry, `shape2_kernel_without_spelling`. The
gate caught this and it is not a waiver -- upstream exposes no
`torch.lift_fresh_copy` and no `Tensor.lift_fresh_copy` (measured), exactly as
for `aten.alias.default` already in that file. The op is reached from a
traced/exported graph and nowhere else, so inventing a spelling would be the
door upstream does not have that `docs/bindings/SPELLINGS.md` refuses. The
allowlist is matched **exactly in both directions**, so if a spelling ever does
reach it the entry itself fails.

### 7.1 Nullification: 8 attempted, 0 uncaught

Each written into the source, rebuilt (`touch`ed first -- `EXPORT6` §4.1's
stale-mtime trap), installed, run, and reverted.

| | nullification | red |
|---|---|---|
| N1 | the whole repair absent (the round's starting state) | 5 tests |
| N2 | dense `lift_fresh_copy` spelled as `clone` | 2 |
| N3 | dense `lift_fresh_copy` returns its input, as `lift_fresh` does | 2 |
| N4 | meta rule `PreserveFormat` instead of `AlwaysContiguous` | 1 |
| N5 | `_weak_ref` answers a constant | 1 |
| N6 | `_SchemaInfo` ignores the batch-norm table | 1 |
| N7 | `is_mutable()` always `True` | 1 |
| N8 | `has_argument` always `True` | 1 |
| N9 | the list-alias parser fix reverted | 1 |

**N2 is the one worth keeping.** It is the implementation a reader would reach
for first -- `lift_fresh_copy` is "a copy", `clone` is "a copy" -- and it gets
every value right. It fails only on the layout, and only because the layout is
asserted as a *difference from `clone`* rather than as a stride written down
beside the kernel.

N1 reddening 5 rather than 9 is not under-reach: the `_weak_ref` and
`_SchemaInfo` tests did not exist yet at that point, and each was observed red
on its own before its implementation was written -- §2 and §3 are separate TDD
cycles, not one.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs lift_fresh_copy_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/storage.rs _weak_ref present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/storage.rs _free_weak_ref present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _TRAINING_DEPENDENT_WRITES present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_liftfresh.py test_lift_fresh_copy_is_contiguous_where_clone_preserves present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_liftfresh.py test_a_module_with_a_tensor_literal_exports_replays_and_agrees present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_liftfresh.py test_storage_weak_ref_is_an_identity_the_constant_map_can_key_on present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_liftfresh.py test_schema_info_answers_mutability_for_every_schema_upstream_does present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_liftfresh.py test_expired_still_refuses_rather_than_guessing present -->
