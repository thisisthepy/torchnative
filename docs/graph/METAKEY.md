# The per-op `Meta` dispatch-key predicate: measured, and left alone

`docs/graph/VARMEAN.md` §4.1 named this gap and set it aside; `docs/graph/STRUCTSEQ.md`
found the real cause of the same wall and left it untouched again. Both were
right, and neither measured the gap on its own terms. This round did, changed
nothing, and this document is the count.

    Features added    0
    Defects fixed     0
    Tests added       5   (rust/torch_c/pytests/test_metakey.py)
    Docs corrected    1   (VARMEAN.md §4.1's forward pointer)
    Removed           0

**The conclusion is that the constant `False` should stay.** Not because the
disagreement is small — it is 1375 overloads on the raw predicate — but because
three separate measurements say every available repair is worse than the gap,
and the one caller that asks the question is not reached here. Each of those is
below, with the number that makes it.

---

## 1. The full comparison

`torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta")` for every aten
overload, both sides, in separate subprocesses
(`test_upstream_answers_false_for_a_measured_minority_so_true_is_not_the_target`,
`test_the_meta_predicate_answers_false_for_every_name_and_never_raises`).

| | upstream `True` | upstream `False` | upstream **raises** |
|---|---:|---:|---:|
| shim `True` | 0 | 0 | 0 |
| shim `False` | **1375** | **408** | **300** |

2083 overloads. VARMEAN.md §4.1's 408 reproduces exactly.

**The third column is new and nobody had named it.** For 300 names —
`aten::ldexp`, `aten::acos.int`, `aten::_unsafe_index.Tensor_hacked_twin`, the
TorchScript residue `docs/design/REGISTRATIONS.md` §3.2 enumerated — upstream does
not answer at all, it raises `RuntimeError: operator ... does not exist`. The
shim answers `False`. REGISTRATIONS.md §3.4 already argued that this
never-validating behaviour is the paired half of `_dispatch_has_kernel = True`
and that upstream's own `activate_meta()` dies without it. The predicate is
therefore not two-valued on the upstream side, and "agree with upstream" cannot
be stated as a boolean.

### 1.1 The number a caller can see is 148, not 1375

`OpOverload.has_kernel_for_dispatch_key` (`torch/_ops.py:891`) is

```python
return super().has_kernel_for_dispatch_key(k) or \
    torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), k)
```

and `super()`'s half is `k in self.py_kernels`. This tree runs upstream's
`_meta_registrations.activate_meta()`, which does
`op_overload.py_impl(DispatchKey.Meta)(fn)` — so **1227 of upstream's 1375
`True` overloads already answer `True` here**, through the Python half, with the
C half saying `False`
(`test_most_of_upstreams_meta_registrations_already_answer_true_here_through_py_kernels`).

The residual is **148**, of which **93** are `.out` variants or in-place
spellings (`aten::_adaptive_avg_pool2d.out`, `aten::add.Scalar_out`,
`aten::bernoulli_.Tensor`, `aten::as_strided_scatter.out`). The 1375 headline
describes the predicate; the 148 describes the disagreement.

---

## 2. Which direction is harmful, and to whom

### 2.1 The one caller

Instrumenting the predicate and exporting a four-line `LayerNorm` module on
upstream: **nine** `Meta` queries, and every one of them arrives from one place
(`test_the_only_caller_that_asks_about_the_meta_key_is_resolve_key_and_it_asks_about_prims`):

```
torch/_ops.py:1009  OpOverload._get_dispatch
  -> torch/_ops.py:213   resolve_key
    -> torch/_ops.py:894  OpOverload.has_kernel_for_dispatch_key
```

Exporting a `BertModel` instead gives 17, same single site. **Eight of the
nine, and fifteen of the seventeen, are `prims::` names** — `prims::mul`,
`prims::broadcast_in_dim`, `prims::rsqrt`. The aten names asked about are
`aten::as_strided` and `aten::set_.source_Storage_storage_offset`, and neither
is in §1.1's 148.

That reframes the gap before any repair is considered: **a registry of this
shim's aten meta arms would not have answered the question being asked.**

Other Meta-key call sites exist in the vendored tree and none is live for aten:
`torch/_library/utils.py:551` (`has_fake_kernel`) is gated behind
`not library_utils.is_builtin(func)` in `fake_tensor.py:3002`, and aten is
builtin; `torch/_custom_op/impl.py` and `torch/_library/fake_impl.py` ask about
a custom op's own qualname; `torch/_inductor/utils.py:480` asks about
`torchvision::nms`.

### 2.2 What the wrong answer costs, in each direction

**`False` where upstream says `True`** — a caller takes a more conservative
path. Here it is worse than conservative: `resolve_key` falls past its branch 1
into branch 2.1, which calls `torch._C._dispatch_is_included_in_alias`, which
this shim does not implement. Measured:

```
                                  upstream            this shim
resolve_key(aten.as_strided.default, Meta)
                                  -> Meta             -> NotImplementedError:
                                                         _dispatch_is_included_in_alias
resolve_key(aten.native_layer_norm.default, Meta)
                                  -> Meta             -> Meta      (via py_kernels)
```

So the harm is real but **latent**: the same probe run on this shim makes
**zero** `Meta` queries while exporting either module, because nothing here
enters `_get_dispatch`. Both exports succeed anyway. `ep.run_decompositions()`,
the next thing that would reach it, stops earlier on an unrelated gap
(`_Unimplemented.set_autograd_compiler`).

**`True` where upstream says `False`** — a caller attempts something that then
fails. This direction is currently unreachable, because the shim never answers
`True`. §3 is what happens when it can.

---

## 3. Deriving the answer rather than storing it, and why it does not work here

The brief for this round proposed deriving the predicate from what the shim
implements — `_aten_implemented()` and the `meta_dispatch` arms — rather than
from a list that must be maintained. The derivation is easy (the arms answer the
question about themselves: only `meta_table`'s fallthrough says "has no meta
kernel for"), it cannot rot, and **it is wrong.**

`test_deriving_the_answer_from_this_shims_meta_arms_would_disagree_in_the_harmful_direction`:
of the **114** aten overloads with a real meta arm here, **28** are ones
upstream answers `False` for.

```
aten.clone.default   aten.detach.default   aten.alias.default    aten.expand.default
aten.t.default       aten.transpose.int    aten.permute.default  aten.reshape.default
aten.slice.Tensor    aten.select.int       aten.squeeze.{default,dim,dims}
aten.unsqueeze.default   aten.split.Tensor  aten.split_with_sizes.default
aten._to_copy.default    aten.copy_.default  aten.lift_fresh{,_copy}.default
aten.constant_pad_nd.default   aten.clip.default   aten.matmul.default
aten.where.{default,ScalarOther,ScalarSelf}   aten.var_mean.{default,dim}
```

Upstream's reason is visible in its own answers:

```
aten::clone      Meta=False  CompositeExplicitAutograd=True   CompositeImplicitAutograd=False
aten::t          Meta=False  CompositeExplicitAutograd=True   CompositeImplicitAutograd=False
aten::matmul     Meta=False  CompositeExplicitAutograd=False  CompositeImplicitAutograd=True
aten::view       Meta=True   CompositeExplicitAutograd=False  CompositeImplicitAutograd=False
```

These ops *do* work on a meta tensor upstream — through an **alias** key, which
`resolve_key`'s branches 2.1–2.3 exist to find. There is no registration at the
`Meta` key and upstream's honest answer is `False`.

**The two predicates are not asking the same question.** `meta_dispatch`
answers "can this be computed without storage". Upstream answers "is there a
registration at the `Meta` key". Deriving one from the other gets 28 of 114
wrong — 25% — and all 28 in §2.2's harmful direction.

### 3.1 A sound derivation does exist, and it buys 40

`native_functions.yaml` is vendored and already line-scanned by
`bootstrap.py::_scan_dispatch_registrations`. The rule "explicit `Meta:` in the
`dispatch:` block, or `structured: True`, or `structured_delegate:`" was
measured against upstream over the 1563 entries that line up:

```
rule True,  upstream True     658
rule False, upstream False    326
rule False, upstream True     579     <- incomplete
rule True,  upstream False      0     <- and never wrong in the harmful direction
```

Sound, incomplete, and derived from a file rather than a list. Crossed with
§1.1's residual it would take the caller-visible disagreement from **148 to
108** — the 579 false negatives are mostly ops `activate_meta()` already covers
through `py_kernels`.

**It was not implemented.** 40 overloads, none of them asked about by the one
caller that asks, at a call site this shim does not reach, against
`docs/design/REGISTRATIONS.md` §3.4's measured warning about touching this
predicate alone. That is a change with no measurable beneficiary, and this
document is worth more than it is. The rule is written down here so the next
round that finds a caller does not re-derive it.

---

## 4. What a blanket `True` actually does — it does not merely disagree

The bar for this round was "do not repair this by answering `True` broadly",
and §1's 408 was the reason. The deliberate break that proves
`test_the_meta_predicate_answers_false_for_every_name_and_never_raises` is a
real test found a stronger one: with
`_dispatch_has_kernel_for_dispatch_key = lambda *a, **k: True`, the shim does
not import at all.

```
torch/_library/fake_impl.py:87  lib.impl(qualname, meta_kernel, "Meta")
torch/library.py:496            RuntimeError: We should not register a meta kernel
                                directly to the operator 'debug_mode_ops::annotate',
                                because it has a CompositeImplicitAutograd kernel in core.
```

`torch/utils/_debug_mode/_mode.py:116` registers a `custom_op` at import time,
and `torch/library.py:493` checks this predicate before allowing a meta
registration. A universal `True` makes every op look composite, so no meta
kernel may be registered, so `import torch` raises. It is not a quality
regression; the tree stops loading.

---

## 5. What this round did not do, and why

* **Did not change the predicate.** No measured caller, and both candidate
  repairs (§3, §3.1) are either wrong or buy 40 overloads nobody asks about.
* **Did not implement `_dispatch_is_included_in_alias`**, which is what
  `resolve_key` actually dies on here (§2.2). It is a different gap and it is
  the one that would have to close first for the Meta answer to matter.
* **Did not assert that the shim keeps making zero `Meta` queries.** A test
  that reddens when the shim starts reaching `_get_dispatch` would be a test
  against progress. The count is printed, not asserted.

## 6. The gate

Run twice from the worktree root, `vendor/install_shim.sh` re-run after each
source change and once more after the deliberate breaks of §4 were reverted.
Both runs identical:

```
run 1   GATE_EXIT=0   suites 95/95   ok=1774   FAIL=0   VULKAN 44 ran / 0 skipped
run 2   GATE_EXIT=0   suites 95/95   ok=1774   FAIL=0   VULKAN 44 ran / 0 skipped

        DOCWATCH   PASS -- 1324/1324
        golden     11627/11627 cases passed, 0 failed, ops=308, pending=0
```

Against the baseline on `develop` (94/94, 1769 ok, DOCWATCH 1319/1319) that is
**+1 suite, +5 tests, +5 markers and nothing else**. No Rust source changed, so
the artefact is `develop`'s: golden's numbers are unmoved because there was
nothing for them to move for.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metakey.py test_the_meta_predicate_answers_false_for_every_name_and_never_raises present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metakey.py test_upstream_answers_false_for_a_measured_minority_so_true_is_not_the_target present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metakey.py test_deriving_the_answer_from_this_shims_meta_arms_would_disagree_in_the_harmful_direction present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metakey.py test_most_of_upstreams_meta_registrations_already_answer_true_here_through_py_kernels present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metakey.py test_the_only_caller_that_asks_about_the_meta_key_is_resolve_key_and_it_asks_about_prims present -->
