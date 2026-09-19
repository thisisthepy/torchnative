# `_dispatch_get_backend_keyset_from_autograd`: implemented, agreeing with upstream, and the map of what is left

`docs/graph/ALIASINC.md` §4 named this as the next blocker and measured it
exactly: after that round, **3547 `resolve_key` resolutions that used to die on
`_dispatch_is_included_in_alias` died on this name instead, and zero newly
resolved.** This round is that name.

    Features added    1   (torch._C._dispatch_get_backend_keyset_from_autograd)
    Defects fixed     0
    Tests added       9   (rust/torch_c/pytests/test_bkeyset.py)
    Docs corrected    1   (ALIASINC.md §4's forward pointer)
    Removed           0

**Answering it also unblocks nothing, and the same 3547 is the reason.** They do
not resolve; they *split*, 2719 onto `_dispatch_is_alias_key` and 828 onto
`_dispatch_has_backend_fallback`. That is the second consecutive round whose
honest headline is a number that did not move, and it is reported first rather
than last.

**What is new, and is worth more than the function, is that the chain
terminates.** It is exactly three more names, and with all of them answered from
a live upstream *no* `resolve_key` result dies on an unimplemented name: 3301
resolve and 1592 raise upstream's own `could not find kernel`. §3 is that map.
It is the thing that says this line of work ends.

---

## 1. What upstream actually answers

`torch._C._dispatch_get_backend_keyset_from_autograd(k)` for all 145 upstream
`DispatchKey`s, in a separate subprocess, membership read key by key.

| | count |
|---|---:|
| keys that answer | **145** — there are no edge cases; `Undefined` answers too |
| of those, **empty** | **129** |
| of those, non-empty | **16** |
| keys that raise | **0** |

`Autograd` itself is one of the 129. So is every backend key, every autocast
key and `ADInplaceOrView`.

### 1.1 The four `Autograd<Backend>` keys that answer empty

This is the finding that decides the shape, and it is the one a derivation gets
wrong.

The obvious rule — strip `Autograd`, return that backend — holds for fifteen
keys and is **wrong for four**:

```
AutogradHIP            -> {}        not {HIP}
AutogradVE             -> {}        not {VE}
AutogradMTIA           -> {}        not {MTIA}
AutogradFunctionality  -> {}
```

All four exist in upstream's enum. Upstream's `getBackendKeySetFromAutograd` is
a `switch` with explicit cases and those are not among them. A derived
implementation would have invented three backend memberships, in the direction
that claims a kernel exists.
`test_four_autograd_keys_answer_empty_so_the_name_rule_is_wrong` asserts the
exception against a live upstream, so if upstream ever adds those cases the
test reddens — and at that point the derivation becomes correct and the table
should go.

The other twelve non-empty answers are `AutogradCPU`/`CUDA`/`HPU`/`IPU`/`Lazy`/
`MPS`/`Meta`/`PrivateUse1..3`/`XLA`/`XPU` → `{Dense, <backend>}`, plus
`AutogradNestedTensor` → 16 `NestedTensor*` backends and `AutogradOther` → 47
`Quantized*`/`Sparse*`/`SparseCsr*` backends.

### 1.2 The second upstream surface that corroborates the largest entry

`AutogradOther`'s 57 members are also reachable as the
`_dispatch_autogradother_backends` **value**, which `resolve_key` consults
fourteen lines later. Upstream builds both from the same `c10` set, and
`test_the_autogradother_keyset_is_exactly_the_dispatch_autogradother_backends_value`
asserts the identity on a live upstream. The biggest entry in the table is
therefore corroborated by a second upstream surface rather than by one probe.

---

## 2. Two ways the measurement itself was wrong before it was right

Both were caught by the tests rather than by reading, and both are now
assertions, because either one silently changes every number above.

### 2.1 `DispatchKeySet.has()` answers `True` seven times on an *empty* keyset

Upstream's empty keyset reports `has()` `True` for all six alias keys and for
`Undefined`. It is a property of the bit-set representation, not a membership.

Read naively, every one of the 145 answers looked like it contained seven
members, and the first agreement run reported **123 of 123 keys disagreeing**
with the shim — whose `has()` does not do this. The fix is not to teach the
shim the same behaviour; it is to put those seven outside the comparison on
both sides, which leaves 116 comparable members per key.
`test_the_membership_probe_is_only_valid_away_from_the_alias_keys` pins that
the spurious set is **exactly** those seven, so the exclusion is a measured
bound rather than a habit — and if upstream fixes `has()`, the test reddens and
the probe can widen instead of silently continuing to skip them.

It also asserts the shim's empty keyset claims **no** members, so a later round
cannot "fix" the disagreement by copying upstream's bug.

### 2.2 `repr(DispatchKeySet)` prints names that are not in upstream's enum

The first table for this round was read out of `repr()`, which produced
members named `Vulkan`, `Metal`, `MkldnnCPU`, `FPGA` and `CustomRNGKeyId`.
`docs/graph/ALIASINC.md` §2.1 had already measured `Vulkan` as **stub-only** —
a name upstream's `.pyi` declares and its runtime enum does not. Both cannot be
true. `repr` is C++'s `toString`, which prints legacy names for bit positions;
`__members__` is the enum.

The contradiction is the only reason it was caught, and it is worth stating
because the wrong table **produced identical `resolve_key` numbers** — all five
staged sweeps in §3 were first run against it and every count matched to the
unit. Nothing downstream could have told the two apart. Every number in this
document is built from `__members__` and `has()`; nothing is parsed from a
repr.

### 2.3 The partition the agreement is valid over

The guard `docs/graph/ALIASINC.md` §2.1 installed applies here unchanged and is
re-asserted, because it bounds this comparison too:

```
upstream's pybind enum  145 keys
vendored DispatchKey    140 keys
                        -> 123 shared | 17 stub-only | 22 runtime-only
```

Consequences, both asserted by
`test_the_enum_divergence_that_bounds_what_this_agreement_can_mean`:

* **Two of the 16 non-empty keys are unreachable here** — `AutogradMAIA` and
  `EndOfAutogradFunctionalityBackends` (which is `AutogradMeta`'s value under a
  second name). So `_AUTOGRAD_BACKEND_KEYSET` has **14** entries, and a vendor
  bump that introduces either is a RED test rather than two answers quietly
  going missing.
* **The 17 stub-only keys answer the empty set.** Upstream cannot be asked
  about them, and empty is the direction that never claims a backend.

---

## 3. The map: three more names, and then it stops

The question ALIASINC's §4 could not answer for its own successor. Each stage
patches one more name in from a live upstream and re-sweeps
`resolve_key(op, k)` over every aten overload against `Meta`, `CPU` and
`AutogradCPU` — 4893 results per stage.
`test_the_chain_terminates_and_no_fourth_name_appears` is this table.

| stage | what is answered | resolved | dies on a gap |
|---|---|---:|---|
| −1 | (control: this name put back to raising) | 1346 | 3547 × `_dispatch_get_backend_keyset_from_autograd` |
| 0 | **this round** | **1346** | 2719 × `_dispatch_is_alias_key`, 828 × `_dispatch_has_backend_fallback` |
| 1 | + upstream's own table for this name | 1346 | 2719 + 828, identical |
| 2 | + `_dispatch_is_alias_key` | **1440** | 3453 × `_dispatch_has_backend_fallback` |
| 3 | + `_dispatch_autogradother_backends` | 1440 | 3453, unchanged |
| 4 | + `_dispatch_has_backend_fallback` | **3301** | **none** |

Read across, that is the whole remaining shape:

* **This round moves 1346 → 1346.** Zero.
* **`_dispatch_is_alias_key` is the one that first moves the number**, and only
  by 94 (87 `CompositeImplicitAutograd`, 7 `Autograd`). It is a six-name
  predicate — `test_the_membership_probe_...` already measures which six — so it
  is the cheapest remaining name by a wide margin and it is the one to do next.
* **`_dispatch_autogradother_backends` changes nothing** in this sweep. It is
  only consulted when `k == AutogradOther`, which these three keys never are.
  It is on the chain but not on this sweep's path.
* **`_dispatch_has_backend_fallback` is where the mass is**: 1440 → 3301. It is
  also the one that cannot simply be copied — upstream's `True` set is the
  fallbacks *upstream's build registered*, and answering from it here would
  claim 37 fallbacks this shim does not have. That is a decision for the round
  that does it, not a table lookup, and §4 of `docs/graph/METAKEY.md` is the
  precedent for taking it seriously.
* **At stage 4 no result dies on an unimplemented name.** 3301 resolve and 1592
  raise upstream's own `could not find kernel` — an honest refusal that names
  itself, not a gap. Stage 4 is measured with upstream's answers patched in, so
  it is a claim about the **shape of the remaining work**, not a claim that the
  shim does it.

Stage 0 and stage 1 agreeing to the unit is a second, weaker check on the
table: the shim's own 14 entries and upstream's live 16 produce the same 4893
outcomes.

---

## 4. Nullification: four deliberate breaks, and three tests that do not notice

Every break was built, the extension rebuilt, and the suite run.

| break | `import torch` on the shim | which tests reddened |
|---|---|---|
| always return the empty set | **survives** | agreement (14 of 123 keys) |
| the name rule of §1.1 (`Autograd<X>` → `{Dense, X}`) | survives | agreement (6 of 123), and the four-empty-keys test, naming `AutogradFunctionality` |
| drop `NestedTensorMPS` from `AutogradNestedTensor` | survives | agreement — **1 of 123 keys** |
| always return the full keyset | survives | agreement (123 of 123), the four-empty-keys test, the enum-partition test, and the `has()` probe test |

The third row is the point, and it is the same point ALIASINC §5 made about its
own one-key break: a single wrong backend out of 19 is invisible everywhere
except the cross-product agreement test, which catches it and names it.

**`import torch` survives all four**, so METAKEY §4's warning — that a wrong
default in this machinery stops the tree loading — does not extend to this
function either. That is measured, not inherited: every probe in the suite
asserts `is_shim` on the side it ran, and all nine kept doing so under all four
breaks.

### 4.1 Three of the nine tests are insensitive to the table's contents

Stated rather than buried. Under all four breaks,
`test_no_resolve_key_result_on_the_whole_aten_surface_still_blames_this_name`,
`test_the_staged_sweep_shows_this_name_alone_unblocks_nothing` and
`test_the_chain_terminates_and_no_fourth_name_appears` stayed **green**.

They are not worthless — they guard *that the name answers at all* and *where
the population lands*, which is what §3 is — but they cannot tell a correct
table from a constant. That is §3's own result restated: at `resolve_key`, no
answer to this function is distinguishable today. It is exactly why the bar for
this round had to be element-wise agreement with upstream (14268 membership
questions) and not any observable behaviour.

The two that are sensitive have a control apiece. The agreement test asserts
the shim raised for **zero** keys before comparing, so a missing implementation
is a failure rather than an empty comparison. The sweep test runs stage −1 —
the name patched back to what an `_Unimplemented` raises — and asserts that
control blames the name more than a thousand times, so "nothing blames it"
cannot pass by the sweep having stopped reaching it.

That control was added because it was needed: the first green run of the sweep
test used stage 0 as its own control and was therefore comparing the
implementation with itself.

---

## 5. What this round did not do, and why

* **Did not implement `_dispatch_is_alias_key`.** It is the next name, it is the
  first one that moves the resolved count, and it is six names wide — but it is
  a different function and this round's bar is agreement on this one. §3 says
  what it buys: 94 of 4893.
* **Did not implement `_dispatch_has_backend_fallback`.** It is where the mass
  is (1440 → 3301) and it is the one that must not be copied from upstream: its
  `True` set is upstream's registered fallbacks, and 37 of them are not this
  shim's. Copying it would be the claim-a-kernel direction that
  `docs/graph/METAKEY.md` §3 rejected for the Meta predicate.
* **Did not assert that the resolved count stays at 1346.** That would be a
  test against progress (`docs/graph/METAKEY.md` §5). The guarantee is
  one-directional — no result that resolved before may stop resolving — and the
  count is printed beside it.

---

## 6. The gate

Run twice from the worktree root, `vendor/install_shim.sh` re-run after each
source change and once more after §4's four breaks were reverted and the
restored `bootstrap.py` was confirmed byte-identical to the pre-break copy.

```
run 1   GATE_EXIT=0   suites 98/98   ok=1808   FAIL=0
run 2   GATE_EXIT=0   suites 98/98   ok=1808   FAIL=0

        DOCWATCH   PASS -- 1359/1359
        golden     11627 cases passed, 0 failed, pending=0, ops=308
        VULKAN     ran=47 ok=47 failed=0 skipped=0 (Apple M1)
```

Against the baseline on `develop` (97/97, 1799 ok, DOCWATCH 1349/1349) that is
**+1 suite, +9 tests, +10 markers**. `bootstrap.py` changed, so the artefact is
this branch's rather than `develop`'s; golden's numbers are unmoved, which is
§3 stated a second way.

**The `ok` count rose by exactly nine, the number of tests written.** That is
checked deliberately: `docs/graph/ALIASINC.md` §6.1 records a run in this same
family where the suite had no `if __name__ == "__main__":` block, so the runner
imported the file, defined eight test functions, called none, and the gate
counted a suite that ran zero tests as passing — with `ok` sitting exactly at
baseline. The unmoved count was the only tell.

Both runs were taken while another round's gate was running in a different
worktree (load average ~3.4–4.1). The stage lock guards duplicate runs inside
one worktree, not across worktrees. Neither run showed a red in
`test_coremlops` or `test_anedecode`, so nothing here needed the `PLAN`-line
adjudication CLAUDE.md §2 describes — but the contention is stated because the
two CoreML suites are the ones it would have shown up in. Nothing in this
round's own numbers is load-sensitive: agreement against a live upstream and
`resolve_key` outcome counts mean the same thing under contention.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_the_shim_agrees_with_upstream_on_the_backend_keyset_for_every_shared_key present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_only_sixteen_keys_answer_a_nonempty_backend_keyset present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_four_autograd_keys_answer_empty_so_the_name_rule_is_wrong present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_the_autogradother_keyset_is_exactly_the_dispatch_autogradother_backends_value present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_the_enum_divergence_that_bounds_what_this_agreement_can_mean present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_the_membership_probe_is_only_valid_away_from_the_alias_keys present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_no_resolve_key_result_on_the_whole_aten_surface_still_blames_this_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_the_staged_sweep_shows_this_name_alone_unblocks_nothing present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bkeyset.py test_the_chain_terminates_and_no_fourth_name_appears present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _AUTOGRAD_BACKEND_KEYSET present -->
