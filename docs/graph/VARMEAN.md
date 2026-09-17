# `var_mean` and `_select_conv_backend`: both walls closed, and the count did not move

`docs/graph/STRIDE.md` §5 measured where `torch.export` stops and narrowed the
question sharply: of the forty `transformers` architectures the sweep covers,
only **ten** are ones upstream torch can export at all, and those ten stopped
at exactly two places — `torch.var_mean` (6) and
`torch._C._select_conv_backend` (4).

This round closed both. Read these three things first, because they decide how
the rest should be taken:

* **The headline is still 0 of 10 under the replay-and-agree bar**, and the bar
  is not "export() returned" — it is that the exported program replays and its
  outputs agree element-wise with the unexported module. §3 reports it.
* **Both named walls are at zero, measured.** `var_mean` went from 11 of the
  forty to 0 and `_select_conv_backend` from 5 to 0. Nine of the ten now stop
  at **one** new wall, the same one for all nine, and it is not a missing
  operator (§4) — it is a result *type*. Two explanations for it were tested
  and rejected; §4.1 records them, because both look right. The tenth reaches
  `FakeTensorDeviceMismatchError`, which `STRIDE.md` §6 analysed and
  deferred.
* **`_select_conv_backend` was not the wall it looked like**, and this was
  measured before it was written. §2.3: with the query stubbed out, three of
  its four architectures fall straight through to §4's wall and the fourth to
  `STRIDE.md` §6's. Closing it could not have moved the count and did not.
  It is closed anyway, honestly and in upstream's own vocabulary, because the
  next round should not have to re-derive it.

---

## 1. `var_mean` — a real operator, returning a pair

There was no `aten.var_mean.*` kernel of any kind, and no `var_mean` entry in
`rust/torch_c/src/overloads.json`, so `torch.var_mean(...)` refused at overload
resolution before reaching a dispatcher. `aten.var.*` and `aten.std.*` existed
and still do.

**The pattern named in this round's brief — an operator already implemented but
lacking a *meta* kernel — was checked for first and is not what this was.**
`aten.var_mean` was absent on both sides of that line. It was worth checking
anyway: the same check applied to §4's wall found a real instance of the
pattern — `aten.native_layer_norm.default` **has** a meta arm here and the
surface answers that it has none — which turned out not to be the cause of
that wall either (§4.1), but is a measured gap now on the record.

### 1.1 What landed

Three overloads, all three reaching one implementation:

```
aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
aten::var_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
aten::var_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
```

plus a meta arm for each, and an entry in `meta_stride_rule`'s
`AlwaysContiguous` class.

`var_reduce` was split so the pair can share its arithmetic rather than copy
it: `var_reduce_values` returns `(out, means, out_dims)` and both `var_reduce`
(one tensor, optionally rooted for `std`) and `var_mean_reduce` (the pair) are
thin exits over it. That sharing is the point and not a tidiness: the five
clamped-denominator rows `var` records — `max(0, n - correction)`, `inf` where
`m2 > 0` and `nan` where `m2 == 0` — apply to the pair because it is the same
code, and a fresh implementation would have had to remember them.

**The mean is a result, not a by-product.** It comes out of the same `f64`
accumulator the variance was computed from and is narrowed once. Recomputing it
through `mean.dim` would narrow twice, which upstream's fused kernel does not
do, and a comparator that checked only the variance would accept a kernel that
returned the wrong tensor as its second element entirely — which is one wrong
variable away here, since the mean is already sitting in scope.

### 1.2 The dtype and correction behaviour, measured

* **`correction=None` means 1, not 0.** `var`'s trap, identically. At n=2 the
  two conventions differ by a factor of two (`var_mean([1., 3.])` is `(2., 2.)`
  and `correction=0` gives `(1., 2.)`), which is where it is pinned.
* **`correction` is a `Scalar`, so a fractional one is legal** and is in the
  probe.
* **Both halves keep the input's dtype**, including `float16` and `bfloat16`;
  `complex64` gives a real variance and a complex mean (recorded, not
  implemented — this shim's `var` family does not take complex either).
* **The refusal wording is not `var`'s.** Upstream's CPU kernel says
  `var_mean only support floating point and complex dtypes`, where `var` says
  `std and var only support floating point and complex dtypes`. Transcribed
  rather than shared.

### 1.3 Two divergences, pinned rather than left unstated

**The overload the Python binding resolves to.** Upstream's `PythonArgParser`
sends *every* spelling of `torch.var_mean` to `aten.var_mean.correction`,
translating the deprecated `unbiased` forms on the way. This shim's resolver
takes the first schema in `overloads.json` that binds, so `torch.var_mean(x)`
reaches `.default` and `torch.var_mean(x, dim=0)` reaches `.dim`.
**`torch.var` has had exactly this disagreement since it landed**
(`docs/graph/EXPORT5.md` §9) — measured on both sides, not assumed — and
`var_mean` inherits the table's shape rather than inventing a second
convention. The values agree, so no value test can see it; it is pinned by
`test_var_mean_resolves_the_overload_var_resolves_and_that_is_not_upstreams`
so that a silent change in either direction reddens something.

**The meta refusal's wording.** Upstream disagrees with itself: its CPU kernel
refuses an integral input with `var_mean only support...` and its *meta* kernel
refuses it with `mean(): could not infer output dtype...`, because its meta arm
is the `_refs` decomposition and the refusal surfaces from the `mean` inside
it. `docs/devices/META.md` §7.1 settles which one this shim follows — the dense
kernel's, called rather than restated — and `_weight_norm_interface` already
carries four divergences of this shape.

---

## 2. `_select_conv_backend` — a query, not a computation

### 2.1 What it is

Its only consumer in the vendored tree is
`torch/_subclasses/fake_impls.py:1811`. It asks which kernel a convolution
would dispatch to, hands that to `_conv_determine_backend_memory_format`, and
uses the answer for exactly one thing: a `t.to(memory_format=...)` on the
result. **Nothing convolves.** So the honest answer is a statement about this
shim's own convolution, not a reimplementation of ATen's selection.

### 2.2 The answer is upstream's own

`_ConvBackend.Overrideable` is upstream's name for "a backend outside this
enumeration handles this", and upstream returns exactly that whenever it cannot
see a device it knows. `torch/_meta_registrations.py:2793` says so in a
comment; it is measured here rather than taken on that comment's word —
upstream answers `Overrideable` for a meta-tensor convolution in all six shapes
the probe builds (2d, 3d, 1d, depthwise, transposed, dilated), for both the
`bias=` and the `bias_sizes=` spelling `fake_impls.py` uses.

This shim has **one** convolution path and it is none of the twenty-two
upstream enumerates — no cudnn, no mkldnn, no nnpack, no xnnpack, no Winograd.
Naming any of them would be a claim about which kernel runs.

`_ConvBackend`'s twenty-two names and values cannot come from the vendored
tree: `torch/_C/__init__.pyi` declares `class ConvBackend(Enum): ...` with
**zero members**, which is why the generated `torch._C.ConvBackend` was an
empty enum. They are transcribed, exactly as `overloads.json` is, and
`test_convbackend.py` re-derives them from a live upstream rather than
reviewing them. `MpsTranspose,` carries a trailing comma in upstream's own
definition and is transcribed as found; the value gap at 9 is upstream's too.

Two things the measurement decided that reading would have got wrong:

* **The first draft refused every native backend by name**, on the strength of
  a *CPU* measurement where `Slow2d` answers `channels_last` for a
  channels-last input. Re-measured on the device this is actually asked about,
  that refusal was a **divergence and not a narrowing**: on a meta tensor
  upstream answers `torch.contiguous_format` for `Slow2d`, `Empty` and
  `Overrideable` alike, contiguous input and channels-last input alike. The
  constant is now upstream's own answer everywhere this shim is asked, and the
  one place it differs — a *dense* channels-last input with a native backend —
  is recorded by a test that also shows it is unreachable through this shim's
  own query.
* **`torch._C.ConvBackend` must NOT be pointed at the real class.**
  `torch/__init__.py:1091` walks every public name in `dir(_C)` and rewrites
  `__obj.__module__` to `"torch"`. Aliasing the two names — which the `.pyi`
  annotation invites — silently moved the class's `__module__` off `torch._C`,
  where upstream's is. Upstream has no runtime `ConvBackend` at all. Caught by
  the differential test, not by reading.

### 2.3 It could not have moved the count, and that was measured first

Before writing anything, both queries were stubbed out in a probe — a token
backend and "no memory-format request", which is what a shim whose convolution
is always contiguous would answer — and the four architectures behind them were
re-run:

```
altclip, beit, blip     -> §4's wall (return_types_native_layer_norm)
aimv2_vision_model      -> FakeTensorDeviceMismatchError (STRIDE.md §6)
```

So the wall was one architecture deep. It is closed anyway, because the answer
is short, honest and now measured; but it is reported as **capability added,
not as progress**, and §3's count is the reason to insist on the distinction.

---

## 3. The numbers, under the bar

`export_sweep.py --limit 40`, both sides, on an otherwise idle machine
(load average 1.8–2.5 across the runs).

```
                                     before   after
architectures swept                      40      40
upstream exported+replayed+agreed        10      10
this shim, of those ten                   0       0     <- THE BAR
```

**0 of 10 before and 0 of 10 after.** Not one architecture exports, replays and
agrees. What changed is where they stop.

All forty, export-stage failures only, by first wall:

| wall | before | after |
|---|---:|---:|
| `torch.var_mean` — no table entry | 11 | **0** |
| `torch._C._select_conv_backend` | 5 | **0** |
| `return_types_native_layer_norm.__new__()` (§4) | 0 | 14 |
| `aten.lift_fresh_copy.default` | 7 | 7 |
| `FakeTensorDeviceMismatchError` | 3 | 5 |

The `before` column is `STRIDE.md` §5's, reproduced: `var_mean` was confirmed
as the live wall by direct probe before any change, and the intermediate sweep
taken after `var_mean` landed and before the conv query did shows the 11
moving to §4's wall as one block and the 5 still standing.

Where the **ten** stop, individually:

| architecture | before | after |
|---|---|---|
| `albert`, `bert`, `bert-generation`, `big_bird`, `camembert`, `canine` | `var_mean` | §4 |
| `altclip`, `beit`, `blip` | `_select_conv_backend` | §4 |
| `aimv2_vision_model` | `_select_conv_backend` | `FakeTensorDeviceMismatchError` |

Nine of the ten now stop at one wall, and it is the same wall.

---

## 4. The next wall, measured — and one hypothesis checked and rejected

All nine stop here, and a four-line module reproduces it:

```python
class M(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.ln = torch.nn.LayerNorm(6)
    def forward(self, x): return self.ln(x)

torch.export.export(M().eval(), (torch.randn(4, 6),))
```
```
  File "torch/fx/experimental/proxy_tensor.py", line 714, in extract_val
    return val.__class__([extract_val(x) for x in val])
TypeError: return_types_native_layer_norm.__new__() missing 2 required
           positional arguments: 'out1' and 'out2'
```

`extract_val` rebuilds a sequence result by handing its class **one iterable**.
Upstream's `torch.return_types.native_layer_norm` is a structseq and accepts
that. The object arriving here is not one: it is the `typing.NamedTuple` that
`torch/_prims_common/wrappers.py:281`'s `_out_wrapper` builds, and a NamedTuple
rebuilds positionally.

The divergence is one line, both sides:

```python
with FakeTensorMode(): torch.ops.aten.native_layer_norm.default(x, [6], w, b, 1e-5)
#   upstream:   <class 'tuple'>
#   this shim:  return_types_native_layer_norm
```

### 4.1 What was checked and is NOT the cause

Two plausible explanations were tested and **both are wrong**. They are
recorded because each looks right and would have been published as the cause:

* **"the shim reaches `_refs` where upstream does not."** It does not: the
  route is identical on both sides —
  `fake_tensor.py:3052` → `fake_impls.py:205` →
  `_refs/__init__.py:3513 native_layer_norm_fake` → the `_out_wrapper`'d ref —
  and `torch._refs.native_layer_norm(...)` called directly returns
  `return_types_native_layer_norm` on **both** sides. `_out_wrapper`'s exit is
  `return out if is_tensor else return_type(*out)`, unconditionally.
  So is `native_layer_norm_fake(fake_mode, func, ...)` called by hand inside
  `FakeTensorMode`: NamedTuple on both. Only the **full dispatch** differs.
* **"`_dispatch_has_kernel_for_dispatch_key(name, 'Meta')` is a constant
  `False` here and `True` upstream, so the meta kernel is skipped."** The
  measurement is real — `bootstrap.py` answers `lambda *a, **k: False`, and
  its own comment says the answer is conservative — but it is **not the
  mechanism for this**: the branch that consults a meta kernel
  (`fake_tensor.py:2946`) goes through `cpp_meta_supports_symint`, which is a
  fixed `ordered_set` and identical on both sides, and `OpOverload.decompose`
  asks for `CompositeImplicitAutograd`, which both sides answer `False` for
  `native_layer_norm`.

  The blanket repair would have been wrong anyway, and that is worth keeping:
  of the **2083** aten overloads on upstream, **408 answer `False`** for
  `Meta`. The predicate is per-op, not per-namespace, so answering it honestly
  needs a registry of which ops this shim has a meta arm for. That is a real
  gap, and it is now measured; it is simply not this wall.

### 4.2 What is left to localise

Upstream's full fake dispatch produces a plain `tuple` for
`aten.native_layer_norm.default` while reaching, by every probe made here, the
same `_refs` function that returns a NamedTuple. The remaining candidates are
`_cached_dispatch_impl`'s cache (the traced run that showed the identical route
had monkey-patched the ref, which forces a miss and may not be the path a cold
upstream run takes) and a branch above `fake_tensor.py:3049` that returns
before the `op_implementations_checks` loop. **That is the next round's first
task**, and it is one line of measurement away — not a redesign.

Stated plainly so nobody re-derives it: **the shortest road from 0/10 is now
this one result type, shared by nine of the ten. It is not a missing
operator.** The tenth, `aimv2_vision_model`, stops at
`FakeTensorDeviceMismatchError`, which `docs/graph/STRIDE.md` §6 analysed and
deferred; it is still deferred.

---

## 5. The gate

Five full runs. The last two are clean and identical:

```
run 4   GATE_EXIT=0   suites 89/89   ok=1729   FAIL=0   VULKAN 33 ran / 0 skipped
run 5   GATE_EXIT=0   suites 89/89   ok=1729   FAIL=0   VULKAN 33 ran / 0 skipped

        golden self-test  PASS -- 26 comparators x 11 fault modes,
                          0 problems, 0 comparators never exercised
        DOCWATCH          PASS -- 1281/1281
```

The golden *corpus* is not part of `run.sh` (it runs the self-test only), so it
was run against the same staged artefact the gate hands its own stages
(`TORCH_C_ARTEFACT=$stage/_C.abi3.so`):

```
golden   11606/11606 cases passed, 0 failed, ops=307, pending=0
```

`_var_mean_pair_check` catches 9 of the 11 injected fault modes -- the same
profile as `_triple_result_check` and `_weight_norm_pair_check`, the two
comparators nearest it in shape.

The three earlier runs are not counted, and why is worth having:

* **run 1** found **four** failures, every one a *consequence* of the new op
  correctly reported by a test doing its job -- the MPS and CUDA readback
  lists, the transcribed-schema count, and `vit`'s refold count. All four are
  fixed in §6 by naming what moved rather than re-baselining a number.
* **runs 2 and 3** were identical at `suites 89/89, ok=1726, FAIL=1`, the one
  failure being the gloo pin nullification -- **not this round's**, and closed
  in §5.2 by conditioning rather than weakening.

The `develop` baseline is 1717 ok / 0 FAIL / 87 suites / DOCWATCH 1268 /
golden ops=304, and every delta accounts for itself:

| | baseline | here | why |
|---|---:|---:|---|
| ok | 1717 | 1729 | **+12** = this round's 12 new tests |
| FAIL | 0 | 0 | |
| suites | 87 | 89 | **+2** = `test_varmean.py` and `test_convbackend.py`; the ledger discovers `test_*.py` by glob and accounted for all 89 |
| DOCWATCH | 1268 | 1281 | **+13** = this round's 13 markers (the two `test_gloopin.py` tests carry none) |
| golden ops | 304 | 307 | **+3** = `var_mean`'s three overloads |
| golden cases | 11478 | 11606 | **+128** = their case builders, reusing `var`'s tables |

### 5.1 Nullification: 14 attempted, **0 uncaught**

Each was written into the source, rebuilt (`bootstrap.py` is `include_str!`'d,
so editing without rebuilding retests the old binary), installed, run, and
reverted by `copy2` **and an explicit `utime`** — `EXPORT6` §4.1's stale-mtime
trap, which makes cargo skip a rebuild and leave the wrong binary installed.

| | nullification | red |
|---|---|---|
| N1 | `correction` defaults to 0, not 1 | 2 tests |
| N2 | the pair is returned `(mean, var)` | 3 |
| N3 | the meta arm ignores `keepdim` | 2 |
| N4 | the meta pair is not promoted | 2 |
| N5 | `var_mean` is `Elementwise`, not `AlwaysContiguous` | 1 |
| N6 | the dense kernel skips its dtype refusal | 3 |
| N7 | `_select_conv_backend` answers `Slow2d` | 2 |
| N8 | the memory format is `channels_last` | 2 |
| N9 | one enum value is wrong | 1 |
| N10 | the `overloads.json` entry is removed | 3 |
| N11 | the interface detector returns what it parsed instead of failing | 1 |
| N12 | the precondition answers "cannot run" unconditionally | 1 |
| N13 | the precondition counts loopback as a local address | 1 |
| N14 | the precondition accepts an address not on this machine | 2 |

N2 is the one worth keeping: swapping the pair reddens the **export** test as
well as the value tests, because the exported program's replay then disagrees
with eager. A `var`-only comparator would have passed it.

---

### 5.2 The gloo nullification, closed by conditioning rather than weakening

Runs 1-3 carried one failure that was **not this round's**, and runs 4-5 do
not, because it is now a *conditioned* assertion:
`test_gloopin.py::test_removing_the_pin_restores_the_dependency_on_host_name_resolution`.
It is the *nullification* of the gloo loopback pin -- it asserts that removing
the pin makes the child either fail on name resolution or bind a routable
address, because if removing it changes nothing then the three tests above it
prove nothing.

The mechanism, measured:

```
hostname            irackui-Macmini.local
getaddrinfo         218.38.137.27
actual interfaces   127.0.0.1, 10.9.8.15, 10.9.8.33
```

`218.38.137.27` is not on this machine -- an ISP wildcard is answering the
`.local` lookup. gloo will not treat a non-local address as local, says so
(`[W ProcessGroupGloo.cpp:555] Unable to resolve hostname to a (local)
address. Using the loopback address as fallback.`) and falls back to loopback
on its own. Both arms then bind loopback, and the comparison distinguishes
nothing.

**The assertion was not weakened. It is byte-identical to `develop`** -- the
diff for this file is `179 insertions(+), 0 deletions(-)`. What was added is a
*precondition that is measured and reported*:

* `_interface_addresses()` reads this host's real interface addresses;
* `_hostname_resolution()` resolves this host's own name;
* `_nullification_can_run()` answers whether any resolved address is on one of
  those interfaces and is not loopback.

If it cannot run, the test **says so in the gate output**, naming the resolved
address and the interfaces it is not among, and does not assert. If it can, it
asserts exactly as before.

Three things keep this from being a skip:

1. **A broken detector fails rather than answering.** `_interface_addresses`
   raises unless `127.0.0.1` is among what it parsed -- a fact about every
   host, not about this one. A detector that quietly returned an empty set
   would put the gate permanently in the cannot-run branch: an assertion
   switched off with nothing saying so, which is CLAUDE.md §5.5's shape and
   the direction this project has been bitten in before.
2. **Both branches of the decision are exercised here**, on a host that only
   ever takes one. `_nullification_can_run` is a pure function of two measured
   lists, and `test_the_precondition_decides_both_ways_on_synthetic_inputs`
   drives five cases through it -- including a real local address, which must
   make the nullification run.
3. **Four nullifications, 0 uncaught** (§5.1's N11-N14). N14 is the one worth
   keeping: widening the precondition by a single term, so that it accepts a
   resolved address whether or not it is on this machine, makes the original
   assertion fire again. The condition is therefore exactly as narrow as this
   host requires and no narrower.

Two tests added: `test_the_interface_detector_fails_rather_than_reporting_no_addresses`
and `test_the_precondition_decides_both_ways_on_synthetic_inputs`.

## 6. This round, separated

**Features added** — kernels or surface the shim did not have:

1. `aten.var_mean.default` / `.dim` / `.correction`, dense.
2. Meta arms for all three, in the `AlwaysContiguous` layout class.
3. A `var_mean` entry in `overloads.json`, so `torch.var_mean(...)` resolves.
4. `torch._C._ConvBackend`, upstream's twenty-two names and values.
5. `torch._C._select_conv_backend`, answering `Overrideable`.
6. `torch._C._conv_determine_backend_memory_format`, answering
   `torch.contiguous_format`.
7. `var_mean`'s three overloads added to `device.rs::MPS_HOST_READBACK_OPS`
   (87 → 90) — a *refusal* is the capability here: they read an `mps` tensor
   back to the host, as their `var` siblings do, so they must be refused at
   the door rather than computing on the CPU under an `mps` label. The gate
   derived this itself: `test_the_mps_readback_list_is_what_the_kernels_actually_do`
   scans each dispatch target for a readback and named all three.

**Defects fixed**: none. Nothing that was here was wrong; these were absences.
The four gate failures this round produced were all *consequences* correctly
reported by tests doing their job, not pre-existing defects — see below.

**Tests added**: 12 — 7 in `rust/torch_c/pytests/test_varmean.py`, 3 in
`rust/torch_c/pytests/test_convbackend.py`, and 2 in
`rust/torch_c/pytests/test_gloopin.py` (§5.2: the interface detector's own
check, and the precondition driven both ways). Every one compares against upstream
torch's own answer in a second subprocess; the numeric tolerances are imported
from `tools/golden/dtypes.py` rather than restated, so widening one is not
possible without widening the golden harness's. Plus 3 golden case builders
(`aten.var_mean.*`), which reuse `var`'s own tables with a `(var, mean)`
comparator — so the five clamped-denominator rows and the `correction=None`
default apply to the pair by construction.

**Tests conditioned**: 1 — `test_gloopin.py`'s pin nullification, §5.2. Not
corrected and not weakened: the assertion is byte-identical to `develop`
(`179 insertions(+), 0 deletions(-)`); what changed is that it now measures
and reports whether the comparison is possible on this host instead of failing
blind to it.

**Tests corrected**: 2 in `test_shim.py`, both consequences of the new op and
both updated by **naming** what moved rather than re-baselining a number:

* `test_schema_text_survives_the_round_trip_through_the_transcribed_tables`
  368 → 371 distinct `(qualname, overload)` pairs. **+3, one per overload**,
  and `overloads.json`-only because upstream has no `Tensor.var_mean`; the
  comment records that +6 would mean a method spelling had been invented and
  fewer than +3 that an overload had been dropped.
* `test_the_refold_recovers_vits_regression_and_claims_nothing_more`
  `vit` goes 11 → 12 outside NNAPI and 10 → 11 after the refold, because
  `_refs.native_layer_norm` now terminates at `aten.var_mean.dim` instead of
  refusing. The test now asserts **that op by name** and that the refold still
  removes exactly one — so a different op moving outside would still fail it.

**Docs corrected**: 3 live counts in `README.md` (the `mps` host-readback set,
87 → 90, in all three places that state it) and a forward note on
`docs/graph/STRIDE.md` §5, whose two named walls this round closed. The
historical records of that number in `REPEAT.md`, `MPSATTN.md`,
`REMEASURE2.md`, `GAPS.md` and the release notes are round history and are
left alone.

**Docs**: this file.

**Removed**: nothing.

**Measured and deliberately NOT written**: a port of ATen's `select_conv_backend`
CPU rules (§2.2 — `Overrideable` is upstream's own answer and the port would be
a claim about kernels not in this build), and the per-op `Meta` dispatch-key
predicate of §4.1 — which is a real gap (408 of upstream's 2083 aten overloads
answer `False` there, so the blanket answer is wrong) but is **not** the cause
of §4's wall, and landing it on that mistaken ground would have been the
round's worst move.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs var_mean_reduce present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs var_reduce_values present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs var_mean_correction present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_conv_backend_query present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_varmean.py test_var_mean_agrees_with_upstream_element_wise_on_both_halves present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_varmean.py test_var_mean_correction_defaults_to_one_not_zero present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_varmean.py test_var_mean_on_meta_answers_upstreams_shape_dtype_and_stride present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_varmean.py test_a_module_that_uses_var_mean_exports_replays_and_agrees present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_convbackend.py test_the_conv_backend_enum_is_upstreams_names_and_values present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_convbackend.py test_select_conv_backend_answers_overrideable_exactly_where_upstream_does present -->
<!-- DOCWATCH: op-implemented aten.var_mean.default -->
<!-- DOCWATCH: op-implemented aten.var_mean.dim -->
<!-- DOCWATCH: op-implemented aten.var_mean.correction -->
