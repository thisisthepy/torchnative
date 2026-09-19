# `_dispatch_is_alias_key` and `_dispatch_has_backend_fallback`: the chain terminates, and it terminates at 1440

`docs/graph/BKEYSET.md` §3 mapped the remaining chain, said it was exactly
these two names plus one the sweep never consults, and projected **3301**
resolutions for the end of it. This round is that end.

    Features added    2   (torch._C._dispatch_is_alias_key,
                           torch._C._dispatch_has_backend_fallback)
    Defects fixed     0
    Tests added       9   (rust/torch_c/pytests/test_bfallback.py)
    Docs corrected    1   (BKEYSET.md §3's projection and §5's forward pointer)
    Removed           0

**The number is 1440, not 3301, and that is the result rather than a
shortfall.** 3301 was measured in BKEYSET with *upstream's* fallback set
patched in. That set is the 37 dispatch keys upstream's C++ build registers
catch-every-operator fallback kernels at. This shim has registered none, so
answering from upstream's set would have handed back the dispatch key itself
for **1861** more `(op, key)` pairs on the promise that a fallback will catch
the call. §2 is the evidence for "none", and it is derived from this shim's own
registry rather than reasoned about.

**What did move is what the other 3453 do.** They stop dying on a name this
shim has not implemented and start raising upstream's own `could not find
kernel`. After this round **no `resolve_key` result on the aten surface dies on
a gap in this shim.** That is what BKEYSET meant by the chain terminating,
reached honestly instead of by copying, and it is the first round in this
family whose headline number moved at all.

---

## 1. `_dispatch_is_alias_key` — the cheap one, and it adds no table

Upstream answers `True` for six of its 145 keys and never raises:

```
Autograd  CompositeExplicitAutograd  CompositeExplicitAutogradNonFunctional
CompositeImplicitAutograd  CompositeImplicitAutogradNestedTensor
FuncTorchBatchedDecomposition
```

The only caller in the vendored tree is
`OperatorBase.has_kernel_for_any_dispatch_key` (`torch/_ops.py:113`), which
skips alias keys when asking whether an op has a kernel in a keyset — an alias
key in `py_kernels` says how an op is put together, not which backend runs it.

**The implementation is membership in `_ALIAS_EXPANSION`, not a new six-name
list.** `docs/graph/ALIASINC.md` §1.1 measured that exactly six of upstream's
keys expand beyond themselves under `_dispatch_is_included_in_alias`, and those
six are these six. Adding a second list beside the first would give this file
two places to rot independently, so the predicate reads the one that already
exists and is already held to upstream over all 15129 askable pairs.

That identity is **re-derived from a live upstream every run** rather than
assumed: `test_the_alias_keys_are_exactly_the_keys_that_expand_beyond_themselves`
computes both sets in a separate subprocess and reddens if they come apart.

### 1.1 The probe for "expands beyond itself" has to compare enum *values*

A name-inequality probe reports **twelve** names as expanding, not six —
`Meta`, `AutogradMeta`, `SparseMeta`, `QuantizedMeta`, `SparseCsrMeta`,
`NestedTensorMeta` and five `EndOf*` sentinels. That is not a second finding:
it is `docs/graph/ALIASINC.md` §3 arriving from the other direction. Upstream
spells six keys twice — `EndOfDenseBackends` **is** `Meta` — so
`is_included_in_alias(Meta, EndOfDenseBackends)` is `True` by value while the
names differ, and a name-based probe reads that as an expansion.

It was written the naive way first and the test failed with all twelve named.
Worth recording because the failure looked like "upstream's two surfaces
disagree" and was actually the probe.

### 1.2 What it buys

`resolve_key(op, k)` over every aten overload against `Meta`, `CPU` and
`AutogradCPU` — 4893 results — goes **1346 → 1440**: 87 onto
`CompositeImplicitAutograd` and 7 onto `Autograd`. That is the first movement
in this chain after two consecutive rounds of zero, and it is 94.

---

## 2. `_dispatch_has_backend_fallback` — the judgement, and the evidence for it

`resolve_key` (`torch/_ops.py:257`) consults this last: if it says `True`,
`resolve_key` hands back the dispatch key *itself*, commented "the dispatch key
will implicitly route to backend fallback". Upstream can promise that; this
shim cannot.

### 2.1 What upstream has: 37 keys, all of them C++

Measured in a separate subprocess over all 145 keys, zero raises, 37 `True`:
`ADInplaceOrView`, four `Autocast*`, twelve `Autograd*`, `BackendSelect`,
`Conjugate`, `Negative`, `ZeroTensor`, `Functionalize`, `Python`,
`PythonDispatcher`, `PythonTLSSnapshot`, `PreDispatch`, five `FuncTorch*`,
`Meta`, `MPS`, and three `StartOf*`/`EndOf*` second names for keys already in
the list.

### 2.2 What this shim has: none, derived rather than asserted

**The only door a registration can arrive through here is
`_C._dispatch_library(...)`**, whose `fallback` method appends to
`_shim_registrations`. After a full `import torch` on this shim that list holds
**1895 registrations — 131 `define`, 1764 `impl`, and zero `fallback`.**

So the honest answer is not a choice between upstream's 37 and some smaller
number. It is that this shim has registered no backend fallback at all, read
off its own registry. `_SHIM_BACKEND_FALLBACKS` is the set the predicate
answers from, and it is empty.

### 2.3 A recorded fallback is deliberately *not* an effective one

`_install_library`'s docstring already records the largest thing this file
papers over: registrations that arrive through `_dispatch_library` are recorded
and **dropped**, because `_aten_dispatch` is the only thing that answers a call
and knows nothing about them. So a `fallback` that did arrive must **not** make
this predicate answer `True` — that would claim a kernel that can never run,
which is the same failure as copying upstream's 37, just smaller.

The two sets are therefore kept apart, and the asymmetry is asserted rather
than left to this paragraph:
`test_a_python_registered_fallback_is_recorded_but_is_not_effective` pushes a
fallback through the real door and pins that `_shim_registrations` grows by one
while `_shim_backend_fallbacks()` stays empty and the predicate still answers
`False` for that key. The day a round wires Python fallbacks through to
`_aten_dispatch`, that test reddens and forces both sides to move together.

`_C._shim_backend_fallbacks()` exists for the same reason `_shim_registrations`
and `_C._shim_unknown_tags()` do: the size of the set is the size of what
`resolve_key`'s last branch can honestly promise, and it is countable instead
of implicit.

### 2.4 What the honesty costs, and what it buys

Measured on this tree, four ways, 4893 results each:

| answer | resolved | the rest |
|---|---:|---|
| before this round | 1346 | 2719 × `GAP:_dispatch_is_alias_key`, 828 × `GAP:_dispatch_has_backend_fallback` |
| + `_dispatch_is_alias_key` | **1440** | 3453 × `GAP:_dispatch_has_backend_fallback` |
| + this, answered honestly | **1440** | 3453 × `NOKERNEL` — upstream's own refusal |
| + upstream's 37-key set | 3301 | 1592 × `NOKERNEL` |

BKEYSET §3 projected the last row for this step. **The third row is the real
one.** The 1861 difference is the size of the claim declined: 1585 results that
would have been handed `AutogradCPU` and 276 handed `Meta`, each on the promise
of a fallback that does not exist here. `_DISPATCH_REGISTRATIONS` refuses the
same shape at a different scale — answering from upstream's file there "would
claim 1500 kernels this shim does not have".

**And the chain does terminate.** In the honest row every one of the 4893
either resolves or raises `could not find kernel`; none dies on an
unimplemented name. `_dispatch_autogradother_backends` is still an
`_Unimplemented` and is still never blamed, because `resolve_key` only reads it
when `k == AutogradOther` and these three keys never are —
`test_no_resolve_key_result_on_the_whole_aten_surface_dies_on_a_gap_any_more`
would name it if that changed.

---

## 3. The capability gaps this answer names: 32

Every key where upstream answers `True` and this shim answers `False` is a
backend fallback kernel upstream's build has and this one does not. Upstream
has 37; five cannot be spelled in the vendored enum, and three of those five
(`EndOfDenseBackends`, `StartOf`/`EndOfAutogradFunctionalityBackends`) are
second names for keys that can. That leaves **32 real, nameable gaps**:

```
ADInplaceOrView                   Functionalize
AutocastCPU                       MPS
AutocastCUDA                      Meta
AutocastXPU                       Negative
AutogradCPU                       PreDispatch
AutogradCUDA                      Python
AutogradFunctionality             PythonDispatcher
AutogradHPU                       PythonTLSSnapshot
AutogradLazy                      ZeroTensor
AutogradMPS                       BackendSelect
AutogradMTIA                      Conjugate
AutogradMeta                      FuncTorchBatched
AutogradOther                     FuncTorchDynamicLayerBackMode
AutogradPrivateUse1               FuncTorchDynamicLayerFrontMode
AutogradXLA                       FuncTorchGradWrapper
AutogradXPU                       FuncTorchVmapMode
```

The two that cannot be reached here and are not second names are `AutocastMPS`
and `AutogradMAIA`.

This list is **not written into the implementation.** It is re-derived from
both live sides on every run by
`test_the_capability_gaps_this_honest_answer_names`, so the count moves the day
either side does — which is the point of printing it beside the 1440.

---

## 4. Nullification: five deliberate breaks, and two tests that do not notice

Every break was built, the shim reinstalled, and the suite run.

| break | `import torch` | which tests reddened |
|---|---|---|
| `_dispatch_is_alias_key` always `False` | **survives** | agreement, naming all six |
| drop `Autograd` from the six | survives | agreement — **one key**, named |
| copy upstream's 37-key fallback set | survives | the empty-registry test, the gap list, the declined-claim test |
| `_dispatch_has_backend_fallback` always `True` | survives | five of nine, including the stub-only-keys guard |
| make a recorded Python fallback effective | survives | the asymmetry test, naming `CPU` |

`import torch` survives all five, so `docs/graph/METAKEY.md` §4's warning — a
wrong default in this machinery stopping the tree from loading — does not reach
either of these names. Measured, not inherited.

The second row is the point, and it is the same one ALIASINC §5 and BKEYSET §4
each made about their own one-key break: a single wrong key out of six is
invisible everywhere except the cross-key agreement test, which catches it and
names it.

### 4.1 One test was toothless and was given teeth rather than counted

`test_the_honest_fallback_answer_resolves_nothing_extra_and_that_is_the_result`
stayed **green** under all five breaks in the first pass. Its assertions were a
control and a one-directional "nothing stopped resolving", and both blanket-
`True` breaks *add* resolutions, so neither tripped it.

Widening it into "the count must stay 1440" would have been a test against
progress (`docs/graph/METAKEY.md` §5). What it asserts instead is that growth
must be **paid for**: if more results resolve than in the control, the shim's
fallback registry must be non-empty. A real fallback landing later raises the
count and keeps the test green; a copied table raises it with an empty registry
and reddens. Re-run under the always-`True` break it now fails with *"3453 more
results resolve because of this name while the shim's fallback registry is
empty — those are claimed kernels"*.

### 4.2 Two tests still have no teeth against a wrong answer, and are counted as such

Stated rather than buried, in BKEYSET §4.1's shape.

* `test_no_resolve_key_result_on_the_whole_aten_surface_dies_on_a_gap_any_more`
  stayed green under all five breaks. It guards *that both names answer at
  all*, which is the §2.4 result and is the thing this round claims — and it
  was RED before the implementation, blaming both names 3547 times. But it
  cannot tell a correct answer from a constant.
* `test_the_alias_keys_are_exactly_the_keys_that_expand_beyond_themselves` is
  an **upstream-side** identity. No break to this shim can redden it, by
  construction; it exists to redden on a vendor bump that invalidates §1's
  licence to reuse `_ALIAS_EXPANSION`.

Seven of nine are sensitive to the shim's answers. The two that are not are
named here rather than counted as coverage.

---

## 5. What this round did not do, and why

* **Did not implement `_dispatch_autogradother_backends.`** It is the third
  name on BKEYSET §3's chain and it is not blamed by any of the 4893 results,
  because `resolve_key` only reads it when `k == AutogradOther`. Implementing
  it would move no number in this sweep; a round that wants it should pick a
  probe key set that reaches it.
* **Did not register any backend fallback.** §3's 32 are named, not closed.
  Closing one means giving `_aten_dispatch` a catch-every-operator path at that
  key, which is a kernel question and not a predicate question.
* **Did not wire `_dispatch_library(...).fallback(...)` through to the
  dispatcher.** §2.3 is why, and the asymmetry test is the marker left for the
  round that does.
* **Did not assert the resolved count stays at 1440.** Same reason BKEYSET
  declined to pin 1346: it is a test against progress. The count is printed
  beside the guarantee, and §4.1's replacement is what has teeth without it.

---

## 6. The gate

Run twice from the worktree root, `vendor/install_shim.sh` re-run after every
source change and once more after §4's five breaks were reverted and the
restored `bootstrap.py` was confirmed byte-identical to the pre-break copy.

Baseline on `develop`, measured **in a worktree**: suites 99/99, ok 1802,
FAIL 0, SKIP 24, DOCWATCH 1373/1373. The main repository's baseline is ok 1801 /
SKIP 25, and the one-test difference is `test_toolguard_wheel_staging` refusing
to fake a wheel over a real cross-build artefact that only the main tree has —
`CLAUDE.md` §2 records it, and the two are not compared directly.

```
run 1   GATE_EXIT=0   suites 100/100   ok=1811   FAIL=0   SKIP=24
run 2   GATE_EXIT=0   suites 100/100   ok=1811   FAIL=0   SKIP=24

        DOCWATCH   PASS -- 1383/1383
```

That is **+1 suite, +9 tests, +10 markers**. **The `ok` count rose by exactly
nine, the number of tests written** — checked deliberately, because
`docs/graph/ALIASINC.md` §6.1 records a run in this same family where the suite
had no `if __name__ == "__main__":` block, the runner imported the file, eight
test functions were defined and none called, and the gate counted a suite that
ran zero tests as passing with `ok` sitting exactly at baseline. The unmoved
count was the only tell.

`bootstrap.py` changed, so the artefact is this branch's rather than
`develop`'s.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_the_alias_key_predicate_agrees_with_upstream_on_every_shared_key present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_the_alias_keys_are_exactly_the_keys_that_expand_beyond_themselves present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_this_shim_has_registered_no_backend_fallback_and_the_predicate_says_so present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_a_python_registered_fallback_is_recorded_but_is_not_effective present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_the_capability_gaps_this_honest_answer_names present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_the_honest_fallback_answer_resolves_nothing_extra_and_that_is_the_result present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_no_resolve_key_result_on_the_whole_aten_surface_dies_on_a_gap_any_more present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_upstreams_fallback_set_would_resolve_1861_more_by_claiming_kernels present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bfallback.py test_the_enum_divergence_that_bounds_what_this_agreement_can_mean present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _SHIM_BACKEND_FALLBACKS present -->
