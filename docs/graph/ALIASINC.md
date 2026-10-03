# `_dispatch_is_included_in_alias`: implemented, agreeing with upstream, and moving nothing yet

`docs/graph/METAKEY.md` §5 named this gap and declined it: *"Did not implement
`_dispatch_is_included_in_alias`, which is what `resolve_key` actually dies on
here (§2.2). It is a different gap and it is the one that would have to close
first for the Meta answer to matter."* This round closed it.

    Features added    1   (torch._C._dispatch_is_included_in_alias)
    Defects fixed     0
    Tests added       8   (tests/bindings/test_aliasinc.py)
    Docs corrected    1   (METAKEY.md §5's forward pointer)
    Removed           0

**The predicate now agrees with upstream on all 15129 askable (key, alias)
pairs, and zero `resolve_key` outcomes changed.** Both halves of that sentence
are measured, and the second is the honest answer to "what does closing this
unblock": nothing downstream, yet. What it bought is that the wall now names
the real blocker instead of this one.

---

## 1. What upstream actually answers

`torch._C._dispatch_is_included_in_alias(key, alias)` over the full cross
product of upstream's 145 `DispatchKey`s — 21025 pairs, in a separate
subprocess, no set inferred from a name.

| | count |
|---|---:|
| pairs answered | **20881** |
| of those, `True` | **471** |
| pairs that **raise** | **144** |

Every one of the 144 is `alias == Undefined`, and the message is upstream's own
internal assert, not a Python error:

```
RuntimeError: t != DispatchKey::Undefined INTERNAL ASSERT FAILED
              at "c10/core/DispatchKeySet.cpp":9
```

**The two edge rules are asymmetric**, and that is a real behaviour rather than
an accident: `key == Undefined` returns `False` and does *not* raise, because
upstream tests `k != Undefined` first and short-circuits. So
`(Undefined, Undefined)` is `False` while `(CPU, Undefined)` raises.

### 1.1 Only six of the 145 keys expand beyond themselves

This is the finding that decides the shape, and it is not guessable from the
names.

| alias | keys it includes |
|---|---:|
| `CompositeImplicitAutograd` | 122 |
| `CompositeExplicitAutograd` | 82 |
| `CompositeExplicitAutogradNonFunctional` | 57 |
| `Autograd` | 27 |
| `CompositeImplicitAutogradNestedTensor` | 26 |
| `FuncTorchBatchedDecomposition` | 7 |

Every other one of the 145 includes **only itself**. `ADInplaceOrView` reads
like an alias key and is not one — it expands to `{ADInplaceOrView}`. The
handful of aliases that look like they have two members (`Meta` → `{Meta,
EndOfDenseBackends}`) are one key with two names; see §3.

So upstream's entire behaviour is

```
k != Undefined and (k == alias or k in EXPANSION[alias])
```

with six entries in `EXPANSION`. Checked rather than asserted: that rule
reproduces **20869 of the 20881** answers.

---

## 2. Table or rule, and how staleness becomes visible

**Both — a rule with a six-entry table inside it**, and the reason a table is
defensible here is not the same reason METAKEY rejected one for the Meta
predicate. There the answer depended on *which ops this shim implements*, which
is why a derivation was wanted and a list would have encoded a moving target.
Here the sets are fixed by upstream's `DispatchKeySet` definitions and have
nothing to do with what this shim implements, so there is no derivation to
prefer: the alias sets *are* the specification.

That does not make a list safe. METAKEY records that this project spent a week
removing rotted lists, and `_ALIAS_EXPANSION` would rot the moment upstream
adds a dispatch key. **What separates a list with a test from a bare list is
that the test has to fail for the right reason.** Three do, and they fail at
three different distances:

| if upstream... | which test reddens | and how it fails |
|---|---|---|
| changes any key's membership | `..._agrees_with_upstream_over_the_full_key_by_alias_cross_product` | names the disagreeing `(key, alias)` pairs |
| adds or promotes an **alias** key | `test_exactly_six_aliases_expand_beyond_themselves` | prints the new list of expanding aliases |
| changes the **enum** the comparison is valid over | `test_the_enum_divergence_that_bounds_what_agreement_can_mean` | prints the new partition |

The third is the one that is easy to leave out and is the reason the other two
can be trusted. The cross-product test can only compare names *both* enums
have. Without the partition pinned, a vendor bump that dropped forty keys from
the stub would make that test compare less and still report green — the
false-green shape this repo keeps finding. Here the partition itself is the
assertion, so coverage shrinking is a failure rather than a smaller number
nobody reads.

Every upstream number above is recomputed in a separate subprocess on every
run. There is no expected-value table beside the implementation.

### 2.1 The vendored enum and upstream's runtime enum are not the same enum

Measured while establishing the above, and worth stating because it bounds
every other claim here:

```
torch/_C/__init__.pyi   declares 140 DispatchKeys
surface.json            mirrors all 140 exactly (0 added, 0 dropped)
upstream's pybind enum  has 145
                        -> 123 shared | 17 stub-only | 22 runtime-only
```

**It is upstream's own stub that is behind its own C++ enum, not this shim's
copy of it.** The 17 stub-only names are ones upstream's dispatcher dropped and
its `.pyi` still declares — `Vulkan`, `MKLDNN`, `MkldnnCPU`, `Metal`, `IDEEP`,
`OpenCL`, `OpenGL`, `FPGA`, `Named`, `Tracer`, `Batched`, `BatchedNestedTensor`,
`VmapMode`, `Autocast`, `CustomRNGKeyId`, `TESTING_ONLY_GenericMode`,
`TESTING_ONLY_GenericWrapper`. The 22 runtime-only are the `StartOf*`/`EndOf*`
sentinels, the `MAIA` backends and `Quantized`.

Consequences, both of which are asserted:

* **"Agreement with upstream" is only definable on the 123.** The cross-product
  test compares 123 × 123 = **15129** pairs, not 21025.
* **The 17 stub-only keys answer `False` for every alias.** Upstream cannot be
  asked about them, so any other answer would be this shim inventing a
  membership. `False` is the direction that never claims a kernel exists.
  `test_the_enum_divergence_that_bounds_what_agreement_can_mean` asserts it, so
  a later round cannot quietly guess `Vulkan` into the dense backends.

---

## 3. The name rule's only error, and why it cannot be reached here

The rule in §1 compares names. Upstream's enum exposes six pairs of names that
are *the same number* — `EndOfDenseBackends` **is** `Meta`,
`EndOfSparseBackends` **is** `SparseMeta` — and upstream answers `True` for both
orders of each. A name-equality rule answers `False`. That is 12 pairs, and it
is the rule's entire disagreement with upstream.

**None of the six offending pairs can be formed in the vendored enum**, because
the `EndOf*` half of every one of them is in §2.1's runtime-only 22.
`test_the_six_value_alias_pairs_the_name_rule_cannot_see_are_all_absent_here`
asserts both halves of that: that the error is still exactly 12 pairs, and that
each pair still has a member the vendored enum lacks. If a vendor bump ever
introduces `EndOfDenseBackends`, that test reddens and says to compare enum
values instead of names.

---

## 4. What closing it unblocks: nothing downstream, measured

`resolve_key(op, key)` for every aten overload against `Meta`, `CPU` and
`AutogradCPU` — **4893 results**, swept before and after.

```
                                    before          after
died on _dispatch_is_included_in_alias   3547            0
resolved (branch 1, via py_kernels)      1346         1346
died on _dispatch_get_backend_keyset_from_autograd
                                            0         3547
```

**Not one of the 4893 resolves that did not resolve before.** The 3547 that
used to die on this predicate now die one name later, on
`_dispatch_get_backend_keyset_from_autograd`, which is the next
`_Unimplemented` on that path and sits above `resolve_key`'s branches 2.3–2.5.

And the export that motivated the whole chain is unchanged:
`test_the_export_that_motivated_this_still_makes_zero_alias_queries` prints
`exported=True alias_queries=0`. METAKEY §2.2 measured that the shim makes zero
`Meta` queries during that export because nothing here enters `_get_dispatch`;
the same is true of this predicate, for the same reason. That count is
**printed, not asserted** — asserting it would be a test against progress,
exactly as METAKEY §5 argued.

So the honest result is that **this round moved no downstream number.** The
justification for the change is agreement with upstream over 15129 pairs and a
named next blocker, not an unblocked feature. The next round that wants
`resolve_key` to actually work on this shim should start at
`_dispatch_get_backend_keyset_from_autograd`.

> **Done, and it moved no downstream number either: `docs/graph/BKEYSET.md`.**
> The 3547 do not resolve when that name answers; they split, 2719 onto
> `_dispatch_is_alias_key` and 828 onto `_dispatch_has_backend_fallback`. What
> that round added instead is the map this section could not draw: **the
> remaining chain is exactly three names and it terminates.** With all of them
> answered from a live upstream, 3301 of the 4893 resolve and the other 1592
> raise upstream's own `could not find kernel` — none dies on an unimplemented
> name. `_dispatch_is_alias_key` is the first one that moves the resolved count
> (1346 → 1440) and is six names wide; `_dispatch_has_backend_fallback` is
> where the mass is (1440 → 3301) and is the one that must **not** be copied
> from upstream, because its `True` set is upstream's registered fallbacks.
>
> **Both are now done, and the terminal number is 1440, not 3301:**
> `docs/graph/BFALLBACK.md`. This shim's own registry holds zero backend
> fallbacks, so the honest predicate answers `False` everywhere; 3301 was
> measured with upstream's 37-key set patched in and was an upper bound. The
> chain does terminate — the other 3453 raise upstream's own `could not find
> kernel` rather than dying on a gap.

---

## 5. What a wrong default breaks — the guess was wrong, and the build said so

The brief's warning was METAKEY §4: a blanket `True` on the *Meta* predicate did
not merely disagree with upstream, it stopped `import torch`, because
`torch/library.py:493` consults it before allowing a meta registration. This
predicate sits in the same machinery, so the same question was asked here — by
building it, not by reasoning about it.

**It does not repeat, and the measurement contradicted what had already been
written down.** An earlier draft of this implementation's docstring asserted
that a blanket `True` would make `resolve_key` hand back
`CompositeExplicitAutogradNonFunctional`, and that a blanket `False` would make
it fall through to `could not find kernel`. Both were plausible from reading
`torch/_ops.py:213`. Both are false. Four deliberate breaks were built and run:

| break | `import torch` | cross product | §4 sweep |
|---|---|---|---|
| blanket `True` | **survives** | 14737 / 15129 wrong | **0 of 4893 differ** |
| blanket `False` | **survives** | 514 / 15129 wrong | **0 of 4893 differ** |
| `Undefined` alias returns `False` instead of raising | survives | 122 / 15129 wrong | 0 differ |
| one key dropped from `Autograd`'s 27 | survives | **1** / 15129 wrong | 0 differ |

No answer to this predicate is distinguishable at `resolve_key` on this shim
today. Branch 1 (`py_kernels`) answers 1346 of the 4893, and the other 3547 hit
§4's next gap first. Branches 2.1 and 2.2 *are* consulted before that — but
nothing in this tree registers a `py_kernel` at
`CompositeExplicitAutograd[NonFunctional]`, so they never fire whatever this
predicate says.

That is why the round's bar had to be agreement with upstream rather than any
observable behaviour: **the observable behaviour cannot tell a correct
implementation from a constant.** The last row is the point — a single wrong key
out of 27 is invisible everywhere except the cross-product test, which catches
it and names it.

---

## 6. The gate

Run twice from the worktree root, `scripts/vendor/install_shim.sh` re-run after each
source change and once more after §5's four breaks were reverted.

```
run 1   GATE_EXIT=0   suites 96/96   ok=1786   FAIL=0
run 2   GATE_EXIT=0   suites 96/96   ok=1786   FAIL=0

        DOCWATCH   PASS -- 1332/1332
        golden     11627 cases passed, 0 failed, pending=0
```

Against the baseline on `develop` (95/95, 1778 ok, DOCWATCH 1324/1324) that is
**+1 suite, +8 tests, +8 markers**. `bootstrap.py` changed, so the artefact is
this branch's rather than `develop`'s; golden's numbers are unmoved, which is
§4's result stated a second way.

### 6.1 A third run was thrown away, and it is the reason to run the gate at all

The first gate run of this round reported

```
GATE_EXIT=0   suites 96/96   ok=1778   FAIL=0   DOCWATCH PASS -- 1332/1332
```

Green, the suite count already up by one, the DOCWATCH markers already
passing — and **`ok` exactly equal to `develop`'s baseline**, because
`test_aliasinc.py` had no `if __name__ == "__main__":` block. The runner
executed the file, the file defined eight functions and called none of them,
and the ledger counted a suite that ran zero tests as a passing suite.

It is the shape this repo keeps writing down: a check that cannot fail. Nothing
in the gate catches it, because every signal a suite emits on success is also
what a suite emits when it does nothing. **The one number that gave it away was
`ok` not moving** — the count the round was supposed to add was the count that
stayed still.

That run is discarded and the two above replace it. Worth stating in the same
breath as §4's "nothing downstream moved": an unmoved number is a legitimate
result *and* the first thing to check when a new suite looks green, and telling
those two apart is the whole job.

<!-- DOCWATCH: symbol-in-file tests/bindings/test_aliasinc.py test_the_shim_agrees_with_upstream_over_the_full_key_by_alias_cross_product present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_aliasinc.py test_an_undefined_alias_raises_while_an_undefined_key_answers_false present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_aliasinc.py test_exactly_six_aliases_expand_beyond_themselves present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_aliasinc.py test_the_six_value_alias_pairs_the_name_rule_cannot_see_are_all_absent_here present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_aliasinc.py test_the_enum_divergence_that_bounds_what_agreement_can_mean present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_aliasinc.py test_no_resolve_key_result_on_the_whole_aten_surface_still_blames_this_predicate present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_aliasinc.py test_closing_the_gap_lets_resolve_key_answer_where_it_used_to_raise present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/bootstrap.py _ALIAS_EXPANSION present -->
