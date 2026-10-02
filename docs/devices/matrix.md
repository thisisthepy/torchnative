# The (dtype × device) matrix — re-measured, and what the first table got wrong

Round date 2026-09-16. Branch `work/dtmtests`, on develop `53fff42`. Host Apple M1
(arm64 macOS, Metal), CPython 3.13, upstream torch 2.13.0 as the oracle,
`candle-core` 0.11.0 (this repository's fork). **No CUDA, no NPU and no Android
device on this machine** — those columns are not in this table at all, because a
column of "untested" reads like a column of results.

A previous round produced this matrix and fixed four defects with it. It added
**no tests**, so none of its fixes could be merged: three of them guard real
behaviour on a real device and nothing would have noticed if they came back.
This round ports that work, gives each surviving change a test that goes red
when the thing it guards is gutted, and **re-measures every cell** rather than
carrying the verdicts across.

> **Answer first — the 2026-09-16 round.** §7 re-measures all of it.
>
> | | |
> |---|---|
> | Cells measured | **4864** — 304 operators × 8 dtypes × 2 devices |
> | AGREES / REFUSES / BREAKS / REACHES / n/a | **2487 / 1671 / 624 / 50 / 32** |
> | Refusals that name themselves | **1201 of 1671.** The other **470** hand back a candle symbol — see §4.3 |
> | Cells whose verdict differs from the first table | **730 of 4864 (15.0%)** |
> | Cells the first table recorded as `AGREES` at `float64_mps` | **17.** Every one is measured here to refuse or break. §2 |
> | Operators the first table gave a single identical verdict across **all 16 columns** | **16** — the signature of a cell that was never placed on the device §2 |
> | `float64` on `mps`, this round | **0 AGREES, by construction.** It refuses by name on all 22 roads onto the device §3.1 |
> | Silent CPU fallback found | **`aten.view.dtype`** returns a **cpu** tensor from an `mps` input, on 4 cells. §4.1 |
> | Values that disagree with upstream and are not RNG | **4 cells**, named in §4.2 |

> **Answer first — the 2026-09-19 re-measurement.** Full detail in §7.
>
> | | |
> |---|---|
> | Cells measured | **4928** — 308 operators × 8 dtypes × 2 devices |
> | AGREES / REACHES / REFUSES / BREAKS / n/a | **2515 / 52 / 1801 / 528 / 32** |
> | Refusals that name themselves | **1327 of 1801.** The other **474** hand back a candle symbol — unmoved, §4.3 |
> | Cells where the 2026-09-16 table was **wrong about reality** | **2 of 4864**, both `aten.bernoulli_.float`, both RNG. §7.1 |
> | Cells the 2026-09-16 table **did not publish at all** | **64** — 4 operators landed after it. §7.1 |
> | Cells the 2026-09-16 table graded too **pessimistically** | **108** — a named refusal classified BREAKS. §7.2 |
> | Cells moved REFUSES → AGREES this round | **4** — `aten.abs.default` on `mps`, all four dtypes Metal allows. §7.3 |
> | Metal dispatch counter | **still none.** This is the ceiling on every `mps` cell's placement claim. §7.5 |
> | `aten.view.dtype`'s silent CPU fallback | **closed 2026-09-19**, by refusal. The `mps`/`cuda` cells are REFUSES and name the reason; the derivation that missed it for two rounds now reaches across files. §4.1 |


> **Answer first — the 2026-09-20 round.** Full detail in §7.7–§7.9.
>
> | | |
> |---|---|
> | Cells measured | **4928** — 308 operators × 8 dtypes × 2 devices |
> | AGREES / REACHES / REFUSES / BREAKS / n/a | **2626 / 53 / 1689 / 528 / 32** |
> | Cells moved REFUSES → AGREES | **109, across 32 operators.** The whole in-place family on Metal. §7.7 |
> | Cells moved REFUSES → REACHES | **3** — `uniform_`, which has no oracle because it draws. §7.7 |
> | The gate that moved them | **one**: `write_into` was a host-side strided scatter over a `&mut CpuStorage`. §7.7 |
> | Change to `vendor/candle-core` | **none.** The device path is candle's published `slice_set` → `copy2d` → Metal blit. §7.7 |
> | In-place operators that now reach on `mps` | **31 of 43**, measured one by one. The other 12 are two named gates, not twelve. §7.8 |
> | Silently wrong answers found by re-measuring | **1, and it is older than this round** — `clamp`/`clamp_min` drop NaN on Metal. Fixed. §7.9 |
> | Metal dispatch counter | **still none.** Unchanged ceiling on every `mps` placement claim in this document. §7.5 |


---

## 1. Three grades, four verdicts

The repository's standard is *builds / reaches / **agrees***, and this document
says which one every cell establishes. A dtype that produces numbers on a
device and disagrees with upstream is worse than one that refuses, because it
is silent — so the table grades at *agrees* wherever an oracle exists, and says
so explicitly where one does not.

| verdict | what it claims |
|---|---|
| **AGREES** | both sides computed and every element matches within `tools/golden/dtypes.py`'s tolerance **for the result's dtype**. The only verdict that is a claim about numbers. |
| **REACHES** | the shim computed and the claim stops there — upstream refused (no oracle), or both computed and the values differ. A REACHES is **never** recorded as an AGREES. |
| **REFUSES** | the shim raised a refusal. Counted in two halves: refusals that name the dtype/device/reason, and refusals that hand back a candle symbol. CLAUDE.md §6 makes only the first kind acceptable. |
| **BREAKS** | anything else — a panic, a hard crash, or a cell this harness could not build. A BREAKS is as much a statement about the harness as about the shim. |
| **n/a** | not a verdict. The operator takes neither a tensor nor a `device=`, so it has no `mps` cell at all. |

`rust/torch_c/pytests/dtype_device_matrix.py` is the sweep that produces the
table below, and it is a script rather than a test for the same reason
`agree_sweep.py` is: it makes numbers a document quotes, so the document can be
re-measured instead of re-asserted.

---

## 2. What the first table got wrong, and how

The first harness captured each operator's operands through a proxy and then
placed the **tensor** arguments on the target device. An operator with no
tensor arguments — every factory: `ones`, `full`, `eye`, `arange`, `linspace`,
`scalar_tensor`, `empty`, `hann_window`, `kaiser_window` — has nothing to
place, so it **ran on the CPU in all sixteen columns**, and its CPU answer was
recorded under `mps`.

That is not a subtle error. It is the failure this whole namespace exists to
prevent, in the reporting layer instead of the compute layer: *a correct answer
computed somewhere other than where the label says*. Sixteen operators carry a
single identical verdict across all sixteen columns as a result, which is the
signature — a row that cannot tell `float64` from `bool`, or `cpu` from `mps`,
is a row where neither axis was ever applied.

Measured, on this build:

```
aten.ones.default, dtype=float64, device=mps   ->  RuntimeError: unsupported const-set f64
the same cell as the first harness ran it      ->  a float64 tensor on the CPU
```

Seventeen operators were recorded `AGREES` at `float64_mps`. Metal has no
`double`; not one of them can produce that tensor. This round measures 14 of
them as REFUSES and 3 as BREAKS.

This harness reads each operator's **schema** and injects `device=` and
`dtype=` whenever the operator accepts them. For a factory the dtype kwarg is
the only way the dtype column can be expressed at all; for a tensor operator
the dtype is expressed by the operands, and adding an output-dtype kwarg on top
would ask a different question. Where a cell genuinely cannot be placed on the
device, it is recorded `n/a` — 32 cells, 4 operators — and never given a
verdict.

**Not every difference is a correction.** Of the 730 differing cells, the
largest single block is the `int8_cpu` column (256 cells): the candle fork that
gave the CPU `DType::I8` landed in `52ca23e`, *after* the commit the first table
was measured on. That column is new capability, not a fixed error. The next
largest blocks are this round's own fixes (`REFUSES -> AGREES`, 191 cells —
mostly `mps` operands that could not be built before §3.2) and the factory
correction above (`AGREES -> BREAKS`, 168, and `AGREES -> REFUSES`, 54).

---

## 3. The three changes, and the one that was declined

### 3.1 `float64` on Metal refuses by name — including the factories

Metal has no `double`. `metal_dtype_gate` (`rust/torch_c/src/device.rs`) already
refused it with upstream's own sentence on every road through
`PyTensorBase::new`, which is the one constructor every dense tensor passes
through. **The factories did not reach it**: they call candle first, and candle
answered in its own vocabulary —

```
aten.ones.default:         candle: unsupported const-set f64
aten.arange.default:       candle: Metal contiguous to_dtype I64 F64 not implemented
```

Eight roads spoke that way: `ones`, `full`, `scalar_tensor`, `arange`,
`ones_like`, `full_like`, `new_ones`, `new_full`. A caller cannot act on either
sentence — both name an internal symbol for a fact about Metal's API — and it is
the shape `test_intmps.py` already rejected for the integer dtypes.

`storage_for` (`rust/torch_c/src/aten.rs`) pairs `PyDtype::storage` with the
**existing** gate and is called from the nine factory sites. This is not a
second guard: it is the same guard asked one step earlier on the paths that
would otherwise never reach it, so nullifying `metal_dtype_gate` takes both out
together. Two guards that shadow each other is the defect CLAUDE.md §5.5
records, and it is what the declined change below would have created.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs metal_dtype_gate present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs storage_for present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dtmdev.py test_float64_refuses_by_name_on_every_road_onto_metal present -->

### 3.2 `_tensor_from_flat` builds on the host and moves last

It built on the target device and cast there, so it inherited every gap in
candle's Metal `to_dtype` table: **ten of the eleven storable dtypes** raised
`Metal contiguous to_dtype F64 <X> not implemented`, and only `bool` (which
leaves by another path) survived. Since `tools/golden/build.py` builds every
operand through this function, an `mps` operand of any arithmetic dtype could
not be constructed at all.

Building on the CPU, casting there, and moving the already-narrow result fixes
it and is also the cheaper order — a float16 operand now moves a quarter of the
bytes. Ten dtypes land carrying upstream's values; `float64` still refuses by
name and `int8` refuses naming `I8` (this repository's candle fork is CPU-only,
docs/numerics/INT8.md §1.2).

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dtmdev.py test_tensor_from_flat_lands_upstreams_values_on_the_device present -->

### 3.3 `tolist` reads on the host

`flat_objects` read the tensor where it lay, so `tolist()` on an `mps` tensor
raised for float32, float16, bfloat16, int32 and int16 — while int64, uint8,
uint32 and bool happened to work, because candle implements *those* casts on
Metal. Whether values could be read off the device depended on candle's kernel
table rather than on anything this shim decided.

**This is a host readback and is allowed to be one.** `tolist` is a host read by
definition and has exactly one caller. What must not acquire a quiet host hop is
`read_flat`, which is what twenty-odd *kernels* use: docs/devices/MPS.md §2
measured thirteen operators returning correct values the GPU never computed once
that gate was removed. The two functions stay separate, and
`test_fixing_tolist_did_not_open_the_host_readback_hole` fails behaviourally —
not by a source scan, which docs/devices/MPSATTN.md §3.1 records how to defeat —
if a kernel is ever routed through the new path.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dtmdev.py test_tolist_on_a_device_tensor_is_upstreams_values present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dtmdev.py test_fixing_tolist_did_not_open_the_host_readback_hole present -->

### 3.4 Declined: a `float64` guard in `visit_for_device`

The ported branch also added a check in `visit_for_device` (`aten.rs`) refusing
`float64` on a Metal device at **dispatch** time. It is not ported, and the
reason is worth recording because the next round will otherwise re-derive it.

It is unreachable. `metal_dtype_gate` sits on the constructor every dense tensor
passes through, so the object that guard inspects — an existing `float64` tensor
on Metal — cannot be built. Twenty-two roads onto the device were probed and not
one produces it. A test for that guard could not be written: its precondition is
unconstructible, so gutting it leaves every test green, which is the
"verification that cannot fail" shape of CLAUDE.md §5.5. Worse, as a second
guard behind the first, the two would shadow each other and neither could be
nullified alone.

### 3.5 Nullification

Each change was broken on purpose and the suite re-run (§6 of CLAUDE.md's rule;
an unnoticed nullification is worth more than a feature):

| nullification | result |
|---|---|
| `metal_dtype_gate` returns `Ok(())` | **RED** — 22 of 22 roads stop refusing; 2 tests fail |
| `_tensor_from_flat` builds on the device again | **RED** — the values test fails, and 2 tests that depend on device operands |
| `flat_objects` reads the tensor where it lies | **RED** — the tolist test alone, with the other 4 still green |
| all three restored | 5 of 5 green |

---

## 4. What the matrix found

### 4.1 `aten.view.dtype` answers on the CPU under an `mps` label — closed 2026-09-19

The one silent fallback in this sweep, and it was a real one:

```
input.device=mps:0  ->  aten.view.dtype  ->  result.device=cpu     (shim, before)
input.device=mps:0  ->  Tensor.view      ->  result.device=mps:0   (upstream)
```

Measured on all four dtype pairs tried (`float32->int32`, `int64->float64`,
`int32->float32`, `bool->uint8`). The result was correct and the device was
wrong, which is the quiet form of the failure docs/graph/NPU2.md is about — and
because the *output* was a cpu tensor, everything downstream of it silently left
the device too. It was recorded BREAKS in the table below.

**It is closed now, by refusal rather than by a kernel.** The verdict on the
`mps` and `cuda` columns is REFUSES, naming the op, the device and the reason.

#### Why it stayed open, and what that says about the derivation

`device.rs::MPS_HOST_READBACK_OPS` is exactly the gate for this shape and it did
not fire. The kernel is two lines:

```rust
let bytes = crate::tensor::to_le_bytes(OP, input.tensor()?)?;
let wrapped = crate::tensor::from_le_bytes(OP, &bytes, &dims, want)?;
```

`to_le_bytes` is a `to_vec1` per dtype — a host readback — and `from_le_bytes`
opens with `let device = candle_core::Device::Cpu;`. **Both live in `tensor.rs`,
and the derivation scanned `aten.rs` only.** The six helper names it followed
were all defined in the same file as the kernels, so a readback one module away
was invisible to the per-op derivation *and* to the classification test that was
built to cover what the per-op scan misses. That is the cross-file form of the
defeat docs/devices/MPSATTN.md §3.1 records, and it had to be fixed before the
operator could be seen at all.

`test_shim.py::_cross_file_readback_helpers` now derives, from every `*.rs` in
`src/` other than `aten.rs`, the set of functions whose bodies hold a readback
marker, and the per-op derivation matches kernels against them **by qualified
path** (`crate::tensor::to_le_bytes(`), not by bare name — `to_le_bytes` is also
an inherent method on every Rust integer, and a bare-name match would have
marked a dozen clean kernels.
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py _cross_file_readback_helpers present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py _reaches_cross_file_readback present -->

#### Why a refusal and not a device kernel

`view.dtype` reinterprets bytes. candle 0.11.0 exposes no bit-reinterpretation
of a `Tensor` on any backend, and its Metal storage is not reachable at a level
where a buffer could be re-tagged; a device-resident version is a Metal shader
plus a candle-fork API, not a kernel edit. And one of the four pairs cannot be
done on Metal at any effort: `int64 -> float64` asks for a double. §3.1 closed
twenty-two roads onto the device for `float64`; **this operator was the
twenty-third, and it was open.**

#### What the evidence is, per device

| device | evidence | measured here |
|---|---|---|
| `mps` | structural + the artefact's own table. Refused before the kernel runs; the name is re-derived from `aten.rs` + `tensor.rs` every gate run | yes — four dtype pairs, `test_viewdtype.py` |
| `cuda` | the same list (`CUDA_HOST_READBACK_OPS` is an alias), so the one-line addition closes it, and `_cuda_counters()` reports the refusal | **no — there is no CUDA device on this machine.** Structural only |
| `vulkan` | never silent: that backend is an allowlist and `aten.view.dtype` is not on it | asserted from the allowlist, not from a run |
| `cpu` | unaffected, and pinned at grade *agrees* against a subprocess oracle | yes — four pairs, exact equality |

There is still **no Metal dispatch counter** in this build (§7.5), so the `mps`
row above is the ceiling this document keeps naming, not a counter reading.
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_viewdtype.py test_view_dtype_refuses_on_mps_rather_than_answering_from_the_host present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_viewdtype.py test_no_aten_kernel_reaches_a_cross_file_readback_unrefused present -->
<!-- DOCWATCH: op-implemented aten.view.dtype -->

### 4.2 Four cells whose values disagree with upstream

44 cells compute a value that is not upstream's. **40 of them are RNG**
(`bernoulli_`, `normal_`, `uniform_`, `multinomial`, `randint`) where two
independent RNG implementations cannot be compared by value at all —
`tools/golden/cases.py` says so itself and carries `value_check` hooks for
exactly this. The remaining four are candidates worth a round of their own:

| cell | upstream | shim |
|---|---|---|
| `aten.acos.default` `int8_cpu` | `3.14159274` | `nan` |
| `aten.bitwise_xor.Scalar` `bool_cpu` | `11.0` | `1.0` |
| `aten.col2im.default` `bool_cpu` | `1.0` | `2.0` |
| `aten.view.dtype` `int8_cpu` | `-122.0` | `127.0` |

### 4.3a The 474 symbol refusals are **five causes, and one of them is 60%**

Scoped 2026-09-19, and the answer changes what the number means. The 474 were
read as "the largest piece of work this matrix names"; clustered by the message
they actually carry, they are not 474 kernels and they are not one problem
either.

**Provenance, stated because the number is quoted.** Derived from
`/tmp/matrix_pub.json` — the 2026-09-19 round's own sweep output on develop
`b4f89f3`, re-read rather than re-run, so this costs the machine nothing and
adds no measurement of its own. The total reproduces exactly: **474**. It is
the state *before* this round's `view.dtype` change, which moves 4 cells out of
BREAKS and into a named refusal and does not touch any of the five below.

| # | cause | cells | ops | stage | where |
|---|---|---|---|---|---|
| **A** | **`_tensor_from_flat` cannot build the operand** — `candle: unsupported dtype I8 for op to_dtype` | **284** | 284 | `operands` | `int8_mps`, all of it |
| **B** | Metal has no cast between the two dtypes — `Metal contiguous to_dtype A B not implemented` | **137** | 45 | `op` | `mps` |
| **C** | candle's matmul lacks the dtype — `unsupported dtype .. for op matmul`, `mlx matmul doesn't support ..` | **27** | 6 | `op` | 19 `cpu`, 8 `mps` |
| **D** | candle cannot set a const of the dtype — `unsupported const-set ..` | **25** | 9 | `op` | `mps` |
| **F** | one cell: `aten.sign.default` `bool_mps`, `Metal contiguous unary usign U8 not implemented` | **1** | 1 | `op` | `mps` |

**A is not an operator refusal at all.** Its stage is `operands`: the sweep
could not *construct* the int8 tensor on Metal, so the cell never reached the
kernel. It is one gate — the candle fork's `DType::I8` is CPU-only
(docs/numerics/INT8.md §1.2) — wearing 284 different operator names, exactly
the shape the in-place family turned out to have (33 operators behind one
`write_back`). **60% of the headline number is one thing, and it says nothing
about the 284 operators it is filed under.** Until int8 lands on Metal the
honest verdict for those cells is a named refusal or `n/a`, not a candle symbol
attributed to the op.

**B is mostly `float64` again.** Of the 137, **117 name `F64` on one side of the
cast** (`F16->F64` 32, `BF16->F64` 32, `F32->F64` 31, `F64->*` 16, `U8->F64` 3,
`I64->F64` 2). §3.1 refuses `float64` on Metal by name on all twenty-two roads
*onto* the device — these are casts reached **inside** a kernel, after the
operands were already placed, which is a road `metal_dtype_gate` does not stand
on. The remaining 20 are `I64->I32` (10) and `I64->I8` (10), the same
CPU-only-integer story as A. So B is two causes, not 45 operators.

**Correction, 2026-09-22 (§7.18): the larger of B's two is neither a gate
placement question nor 117 cells.** This paragraph read "117 name `F64` on one
side of the cast" as 117 callers asking for `float64`, and concluded the gate
was standing in the wrong place. Re-clustered by **the dtype the caller
actually asked for** rather than by the text of the message, the 117 are
`float32`/`float16`/`bfloat16` 108, `int64`/`bool` 8, **`float64` 1**. In 116
of 117 nobody asked for `float64`: the `F64` is this crate's own, introduced
by `read_flat` widening on its way to `to_vec1::<f64>`, and it appears in the
message because candle names the dtypes of the cast it could not find. Moving
a gate would not have touched a single one of them. §3.1's twenty-two roads
are the right number of roads, and they are all still needed.

**C is the family §4.3 originally named** — `mm`, `bmm`, `matmul`, `addmm`,
`baddbmm`, `convolution` — and it is the smallest real one: **6 operators.**
**D is the factories**, 9 of them. **F is one cell.**

    Work actually named here: 1 int8-on-Metal gate (A, and half of B's tail),
    0 float64-cast gate placements (B -- retracted 2026-09-22; see the
      correction above and Section 7.18), 6 matmul ops (C), 9 factories (D),
    1 cell (F).

**C and D were re-derived on 2026-09-20 and neither line above survived it.**
See §4.3b. The counts reproduce exactly — 27/6 and 25/9, both stage `op` — but
"6 matmul ops" and "9 factories" are both wrong about what the work *is*, in
the same way cause A was: the clustering is by message text, and a single
underlying door wearing several operator names reads as several doors.

**What this clustering cannot see.** It groups by the *text* of the message, so
two different causes that happen to raise the same candle sentence are merged,
and one cause whose wording differs between call sites is split. It also
inherits every limit §5 states about the sweep — one shape per operator, and a
cell that refuses for the first reason it meets hides any second reason behind
it. Cause A hides the most: 284 operators have never been asked the question at
all on that cell.

**Asked, 2026-09-20 (§7.17): cause A is closed and it was worth 137, not 284.**
Once the operand builds, 0 of the 284 still fail at stage `operands` and
**137 reach AGREES**. The gap is not shaders: 99 of the 284 do not have
`int8_cpu` at AGREES either, so they were never in the prize — which is the
error this very paragraph warns about, committed by the paragraph above it.

### 4.3b C and D re-derived: 52 cells, and neither is what it was filed as

Scoped 2026-09-20 from the same `/tmp/matrix_pub.json` §4.3a used, re-read
rather than re-run. **The arithmetic reproduces exactly**: C is 27 cells over
6 operator names, D is 25 over 9, and every one of the 52 is stage `op` — so,
unlike cause A, these really did reach a kernel. What does not survive is the
description.

#### D is one helper that nine operators never adopted

Every one of the 25 cells says `unsupported const-set f64` — and **the cell's
own dtype is `float32`, `float16` or `bfloat16` in all 25.** The shim was
asking Metal to materialise a *double* it had no reason to want. That is not
"candle cannot const-set the dtype"; candle const-sets all three of those
dtypes perfectly well.

The door was already open in this repository. `aten.rs::host_const` builds the
constant on the host and moves it, landed for `mul.Scalar` inside a rotary
embedding (docs/devices/MPSFWD.md §2), and its docstring already carries the
whole argument. Nine operators simply still called `Tensor::full(v, shape,
device)`. `host_full` is `host_const` with a shape, and adopting it closes all
25 cells — **no kernel, and nothing in `vendor/candle-core` touched.**

The trap in the obvious cheaper fix is recorded because it is invisible on
`float32`, which is the dtype anyone would test it on: narrowing the `f64` to
`f32` and letting the device const-set *that* rounds twice for a `float16`
destination. `f16(f32(0.031265258789971995))` is `0.03125` where
`f16(0.031265258789971995)` is `0.031280517578125`.

The same scan found **three more call sites of the identical shape** that the
sweep never reached, because those cells refuse earlier for another reason:
`amax`'s NaN seed, `nan_shaped_like`, and `round.decimals`' scale constant.
They are fixed too, and a derivation in `test_constset.py` now refuses the
*shape* of the defect in `aten.rs`, so a tenth is caught by arithmetic rather
than by the next sweep.

#### C is four different things, and only 15 cells are a missing kernel

| what | cells | answer |
|---|---:|---|
| `bool`, any device | 6 | **upstream refuses it too** |
| signed integer, `cpu` | 15 | a real kernel — exact |
| `int64`, `mps` | 5 | refusal, by name (**re-measured 2026-09-22: 15**, `int8`/`int32`/`int64` — see §4.3c) |
| `convolution` `bfloat16_cpu` | 1 | **not done**; see below |

**The bool six were never a gap.** Measured against torch 2.13.0: `torch.mm`,
`torch.matmul` and `torch.addmm` on `bool` raise `"addmm_impl_cpu_" not
implemented for 'Bool'`, and `bmm`/`baddbmm` name themselves. `addmm` in this
build already answered upstream's sentence; the other four answered `mlx
matmul doesn't support U8`. The verdict does not change — it was REFUSES and
stays REFUSES — and what changes is that the sentence is now upstream's, which
is exactly the work §4.3 names.

**The upcast route is a fudge here, and this was checked rather than assumed.**
`gemm_accumulate_in` already widens `float8_e4m3fn` to `f32`, multiplies and
narrows, and that is honest only because upstream's own answer was *measured*
bit-identical to it over 700 cases. Integers are the other case:

```text
torch.mm(int8[[100, 100]], int8[[100], [100]])  ==  32
```

not `20000` (widening) and not `127` (saturating) — `20000 mod 256 == 32`.
Upstream wraps in the storage width. **The cell §4.3a filed this under is a
2x3x2 of small values where all three behaviours agree**, which is the `clamp`
shape of mistake §1 warns about, so `test_gemmint.py` carries inputs that
overflow every width *and* a test that fails if they ever stop overflowing.

So the kernel accumulates in `i64` with wrapping arithmetic and truncates to
the storage width at the end, which is the same number: reduction mod `2**n`
is a ring homomorphism and `2**8`, `2**16` and `2**32` all divide `2**64`, so
a sum of products — built from `+` and `*` and nothing else — has the same
image either way, including when the `i64` evaluation itself overflows. That
is `test_intmps.py`'s argument, reused for the one operator built from nothing
but ring operations.

**The `mps` five stay refused — and a re-run on this tree makes it ten.**
(§4.3c: the `int8`/`mps` five sat behind cause A's `operands` stage when this
was scoped, and became visible only once int8 reached Metal.) The kernel is a host computation, and running
it for an `mps` operand would return a value the GPU did not compute under an
`mps` label — the failure docs/graph/NPU2.md records. The refusal now names
the operator, the dtype, the device, the reason and both roads out, and
deliberately **does not quote candle's sentence**: this document clusters by
message text, so carrying `mlx matmul` would keep counting the cell as the
candle-symbol refusal it had just stopped being.

**Not done: `aten.convolution.default` `bfloat16_cpu`, 1 cell.** Giving it
`gemm_accumulate_in`'s widening is tempting and was not done, because the
bit-identity argument does not carry over: widening a *cast* is elementwise
and cannot move a value, whereas widening before im2col changes the
**summation order** relative to upstream's kernel, and a reassociated float
sum is a different number. That claim needs its own measurement against
upstream, which this round did not make, so the cell is left refusing rather
than answered on an argument nobody checked.

#### Evidence, and its ceiling

The 52 cells are graded **agrees** where they answer — element-wise against
upstream in a separate subprocess, exactly (integers) or bit for bit
(constants), with no tolerance anywhere in either file to widen. Eight mutants
were built **through `vendor/install_shim.sh`**, so each reached the vendored
tree and not only the stage, and each went RED in the test written for it:
double-rounding the constant, filling on the device, saturating instead of
wrapping, the `f64` upcast, removing the device gate, removing the bool gate,
clamping the narrow, and a genuine host readback.

**Withdrawn, 2026-09-21: there is a Metal dispatch counter in this build.**
This paragraph said there was not, citing §7.5 — but §7.11 built one, and the
round that wrote this was scoped against a tree that predated it. The `mps`
half of cause D is no longer resting on a structural derivation: it is
bracketed by `_C._metal_counters()` in §4.3c, where the host-twin experiment
CLAUDE.md §2 records as impossible on Metal has now been run on Metal.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs host_full present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs exact_int_matmul present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs gemm_multiply present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs gemm_broadcast_multiply present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs reject_bool_gemm present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs reject_device_int_gemm present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs exact_int_gemm_dtype present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_constset.py test_every_float_factory_in_cause_d_answers_on_mps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_constset.py test_the_constant_is_rounded_once_and_not_twice present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_constset.py test_no_float_constant_is_still_materialised_on_the_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_constset.py test_the_cpu_answers_are_unchanged present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_gemmint.py test_upstream_integer_matmul_wraps_in_the_storage_width present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_gemmint.py test_integer_matmul_agrees_with_upstream_including_at_overflow present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_gemmint.py test_the_overflow_cases_really_do_overflow present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_gemmint.py test_bool_matmul_refuses_in_upstreams_own_words present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_gemmint.py test_the_mps_integer_refusal_names_the_op_the_dtype_and_the_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_gemmint.py test_the_mps_integer_refusal_is_not_served_by_a_readback present -->

### 4.3c §4.3b re-run rather than re-read — the work landed, three of its numbers did not

§4.3a and §4.3b were both scoped **by re-reading `/tmp/matrix_pub.json`**, and
they say so. This section is the first time those cells were **re-measured**,
and the provenance matters more than usual: `882c45a`, the commit that
implemented §4.3b, is **not a descendant of `7457147`** (the int8 /
`candle-metal-kernels` round). Its sole parent is `c210f7c`; it reached this
branch through a later merge. So §4.3b's numbers are **pre-int8 and
pre-counter**, and every one of them was owed a re-run.

**Method.** `dtype_device_matrix.py --worker` over the operators of causes C
and D, run twice against two builds of this tree: `HEAD`, and a counterfactual
build with `882c45a`'s `aten.rs` replaced by `7457147`'s — which is exactly
"this tree minus the change under test", since the merge applied the WIP on
top of that file. Same harness, same machine, same process shape, 37 minutes
apart. Neither run re-reads a cached sweep.

#### What moved: 39 AGREES and 25 relabelled, and the grades are not interchangeable

| | cells | grade |
|---|---:|---|
| signed integer `mm`/`bmm`/`matmul`/`addmm`/`baddbmm` on `cpu` | **15** | REFUSES → **AGREES** |
| cause D `full` and `scalar_tensor` on `mps` | 6 | REFUSES → **AGREES** |
| cause D `constant_pad_nd`, `fill_`, `full_like`, `new_full`, `round.decimals`, `round_.decimals` on `mps` | 18 | REFUSES → **AGREES** |
| `bool` gemm, both devices | **10** | REFUSES → REFUSES, sentence now upstream's |
| `int8`/`int32`/`int64` gemm on `mps` | **15** | REFUSES → REFUSES, sentence now ours |
| `aten.convolution.default` `bfloat16_cpu` | 1 | REFUSES, unchanged — declared not done, and it is |

**39 cells reach AGREES and 25 are relabelled refusals.** They are different
claims and this table does not add them up into one.

**Corrected 2026-09-22, by re-running the sweep a third time rather than
re-reading this section.** The paragraph above previously read "37 cells" in
its heading and "21 cells reach AGREES and 16 are relabelled" in its body,
while the table beside it summed to 39 and 16. **Neither prose figure was
derivable from the table it introduced**, which is the §5.3 shape: a number
that looks measured because a measured table is next to it. Two of the table's
own rows were also short, and both for the same reason — a row was named after
the dtypes the round had gone looking for rather than after the dtypes the
sweep returned:

* `bool` gemm is **10** cells, not 6: five operators × two devices, and every
  one of the ten comes back in upstream's own words (`"addmm_impl_cpu_" not
  implemented for 'Bool'`, `"bmm" not implemented for 'Bool'`, `"baddbmm" not
  implemented for 'Bool'`). 6 cannot be reconstructed from any run of this
  tree; the population it was drawn from was the cached `/tmp/matrix_pub.json`
  that no longer exists.
* the `mps` integer gemm refusal is **15**, not 10 and not 5. §4.3b said five
  (`int64`), §4.3c raised it to ten (`int64` + `int8`) — and both missed
  `int32`, whose five cells refuse with exactly the same sentence of ours.
  **The count was corrected once and was still wrong**, because the correction
  chased the dtype that had just been vendored instead of asking the sweep
  which dtypes refused.

Measured figures for the whole 224-cell sweep over these fourteen operators,
from `dtype_device_matrix.py --tally`: **AGREES 150, REFUSES 61 (59 named, 2 by
candle symbol), BREAKS 13, disagrees 0.** All thirteen BREAKS are
`where.ScalarSelf`, the harness limitation named below.

#### Two cause-D operators still carry a candle symbol on `int32`/`mps`

The two unnamed refusals in that tally are `aten.full.default` and
`aten.scalar_tensor.default` at `int32_mps`, both quoting
`candle: Metal contiguous to_dtype I64 I32 not implemented`. They are the same
defect as the `eye`/`linspace` cells named under "Still open" below — an
integer constant built at `i64` and converted on the device — sitting inside
cause D's own two headline operators, which `host_full` closed for the three
float dtypes and not for this one. §4.3c did not name them. **Not fixed here:
it is operator work, not verification, and it belongs with the `eye`/`linspace`
round.**

**Updated 2026-10-02 (§7.18).** `scalar_tensor` at `int32_mps` now agrees: its
integer constant is narrowed on the host. `full` at `int32_mps` still breaks,
but on `Metal copy_strided I32 not implemented` — the broadcast that fills the
shape, not the narrowing.

#### Three numbers in §4.3b do not survive the re-run

**C's `mps` half is 10, not 5** — and a third run makes it 15; see the correction above.
*(superseded)* **C's `mps` half is 10, not 5.** §4.3b names five `int64`/`mps` cells. On this
tree the same change relabels **ten**: the five `int8`/`mps` cells were behind
cause A's `operands` stage when §4.3b was scoped — the sweep could not build an
int8 Metal operand at all, so those cells never reached the matmul to be
counted in C. `7457147` moved them, and they landed in the same refusal. The
implementation was already right; the count was written before it could be.

**D's 25 is not reproducible, and the honest figures are 24 and 27.** On this
tree the message `unsupported const-set f64` appears on **24** sweep-visible
cells — eight operators × three dtypes — and all 24 now agree. The ninth
cause-D operator, `where.ScalarSelf`, cannot be expressed by this harness at
all (§5: it casts every tensor operand to the cell's dtype, including the
`bool` condition, so upstream rejects the cell), which is why it is absent
from the 24 and why `test_constset.py` measures it directly. Nine operators
× three dtypes = **27**, which is what that file covers and what closed.
**25 was a pre-int8 figure and no run of this tree produces it.** The claim
§4.3b was making — one helper, nine operators, no kernel, nothing in
`vendor/candle-core` touched — is confirmed; only its arithmetic is not.

**`round.decimals` was never outside the sweep.** §4.3b lists three call sites
"the sweep never reached": `amax`'s NaN seed, `nan_shaped_like`, and
`round.decimals`' scale. The sweep reaches the third one directly — it is
measured above moving REFUSES → AGREES. Two are genuinely outside it, both for
the same reason: the sweep builds one shape per operator and that shape has no
NaN, so the seed is dead code for it.

#### The two call sites that really were outside the sweep, run

Neither had ever been executed; §4.3b's evidence for them was a derivation over
the source text. Run now, in `test_the_nan_seed_call_sites_answer_where_they_are_reachable`:

* on the `cpu`, `max.default`, `max.dim` and `min.dim` **agree with upstream**
  on a NaN input, so `host_full` did not disturb the seed.
* on `mps`, `max.dim` and `min.dim` **refuse by name** — they are in the
  host-readback family, so `nan_shaped_like` is not reachable there.
* `amin` is not implemented by this build at all, which the test asserts
  rather than assumes.

#### A silently wrong answer found on the way, older than this branch

**`aten.amax.default` on `mps` drops NaN.** Over `[1.25, nan, 0.5, 3.75]` it
returns `3.75` where upstream returns `nan`; the `cpu` path returns `nan`
correctly. The cause is `aten.rs::amax_keepdim_anywhere`, which sends a Metal
tensor to candle's `max_keepdim` — the NaN-skipping fold that the docstring
above `nan_along_dim` calls "the third repair of one predicate and ... meant
to be the last". **It is the fourth.**

It is **not** this round's change and **not** cause D: the function is
byte-identical in `7457147`'s `aten.rs`, so it predates both this branch and
the int8 round. It is left unfixed deliberately — it is a different operator
family from the one this round was sent to verify, and the same helper is also
called from the softmax row-max, so the blast radius wants its own round.
`test_the_nan_seed_on_mps_is_either_refused_or_a_recorded_defect` **pins** it:
it asserts the wrong answer is still the wrong answer, and goes RED the moment
somebody repairs the operator, which forces the agreement assertion back in.
This is the `clamp` shape of §7.9 again, found the same way — by giving a cell
an input its published shape never had.

#### The bracket, and the host-twin experiment run on Metal

`_C._metal_counters()` exists in this build (§7.11). **Bracket**: counters read
immediately before and immediately after one `_aten_dispatch`, in-process,
after an unmeasured dispatch has warmed the device.

    aten.full.default, [2,3], on mps      compute  uploads  bytes  downloads
      float32                                   1        1      4          0
      float16                                   1        1      2          0
      bfloat16                                  1        1      2          0
    aten.scalar_tensor.default, on mps          0        1    4/2/2        0

`compute_encoders > 0` and `host_downloads == 0`, as asked. The load-bearing
number is **`host_upload_bytes`**: it is one element of the *storage* dtype,
not `numel × itemsize` (a host-side fill of the whole block) and not 8 (an
unconverted `f64`). `scalar_tensor`'s `compute_encoders` is legitimately 0 —
a one-element result has nothing to broadcast — and the test excludes it by
name rather than asserting a bug into existence.

**CLAUDE.md §2 says this experiment cannot be run on Metal. It can now, and it
was.** A mutant replacing `host_full`'s device path with a host-side fill of
the whole block, built through `vendor/install_shim.sh` so it reached the
vendored tree:

| mutant | values | counters |
|---|---|---|
| M-B: whole block filled on the host, then uploaded | **all green** — `test_every_float_factory_in_cause_d_answers_on_mps` and the rounding witness both pass | **RED**: 24 bytes uploaded where 4 were expected, 0 compute shaders where ≥ 1 were expected |

That is the host twin, on Metal, invisible to every value in the file and
caught only by the counter — the same result CLAUDE.md §2 records three times
on Vulkan and explicitly declines to claim cross-backend. It may now be
claimed.

#### Nullification

| mutant | reached the vendored tree | result |
|---|---|---|
| M-A: narrow the constant through `f32` before the device sees it | yes | **RED** in `test_the_constant_is_rounded_once_and_not_twice` — `float16` gave `0.03125`, upstream `0.031280517578125`. **Every other test in the file stayed green**, which is the point: the trap is invisible on `float32` and invisible to a value of `1.5` |
| M-B: fill the whole block on the host and upload it | yes | **RED** only in the counter bracket and the structural test; all six value tests green |
| the whole of `882c45a` removed (`7457147`'s `aten.rs`, rebuilt) | yes | **7 of 11 RED** — the pre-implementation state, which is stronger evidence than a mutant and is what a TDD red looks like after the fact |

**Which tests are toothless, stated plainly.** Four of the eleven stayed green
against the pre-implementation build: `test_the_cpu_answers_are_unchanged`
(a regression guard — green by design),
`test_upstream_integer_matmul_wraps_in_the_storage_width` (measures upstream,
not this build), `test_the_overflow_cases_really_do_overflow` (guards another
test's inputs), and `test_the_mps_integer_refusal_is_not_served_by_a_readback`
— which is the only uncomfortable one, because the pre-change build also
refused, just in candle's words. It could not distinguish the two builds. It
has been given the counter bracket so that it is at least an instrument
against a future readback rather than an assertion that something raised.

**Re-nullified 2026-09-22 on a freshly rebuilt tree, and the count was stale.**
"Eleven" was the suite count of `882c45a`; this section then added three more
tests (the counter bracket and the two NaN-seed tests) and did not re-run its
own nullification, so four of the fourteen had never been broken on purpose.
All three mutants below were built through `vendor/install_shim.sh`, so each
reached the vendored `_C.abi3.so` rather than a stage:

| mutant | result |
|---|---|
| M-A, re-run: `host_const` narrows through `f32` before the steps | **exactly one RED** — `test_the_constant_is_rounded_once_and_not_twice`, on `float16`/`mps`, `0.03125` against upstream's `0.031280517578125`. The other seven tests in the file stayed green, which is the claim: `float32` cannot see this |
| M-B, re-run: `host_full` fills the whole block on the host and uploads it | **two RED** — the counter bracket (24 bytes where 4 were expected, 0 compute encoders where ≥ 1) and `test_host_full_exists_and_converts_before_it_broadcasts`. **All six value tests green.** The host twin, on Metal, seen only by the counter |
| M-C, new: the exact integer matmul **saturates** instead of wrapping (`clamp` before the narrowing cast, `int8` and `int32`) | **RED in `test_integer_matmul_agrees_with_upstream_including_at_overflow` on all 20 case keys**, `int8` and `int32` alike |
| the overflow inputs defanged to ±2 in `_random_case` | **RED in `test_the_overflow_cases_really_do_overflow`**, naming all three dtypes |

M-C and the defang together close the question §4.3a's single 2×3×2 cell could
not: the inputs really do overflow, and a saturating kernel — one of the three
behaviours that cell cannot distinguish — is caught in every case. `int64` has
no saturating mutant to write, because its arm returns the accumulator
unnarrowed.

#### The `mps` integer refusal is not one instantiation line away

Asked because the vendored `candle-metal-kernels` crate now exists and could
in principle have made the refusal premature. It has not:
`kernels/mlx_gemm.rs` knows `GemmDType::{F32, F16, BF16}` and nothing else,
`metal_src/mlx_gemm.metal` carries `instantiate_gemm_transpose_helper` lines
for `f32`, `f16` and `bf16` only, and its `accum_type` is `typedef float`.
An integer GEMM there is not a new instantiation line; it is a new accumulator
type, because a `float` accumulator cannot hold the exact `int32`/`int64`
products the CPU kernel is graded against. §7.13 and §7.15 reached the same
place for `DType::I8` by a different road. **The refusal stands, on measured
grounds rather than inherited ones.**

#### The gate had never been run on this work, and it was red in six suites

§4.3b and §4.3c were both written by rounds that ended before a gate. Run for
the first time on 2026-09-22, the tree came back **suites 109/109, ok 1862,
FAIL 21, SKIP 24** — not the FAIL 0 the branch was being reported at. Five
suites were red **because of the work in this section**, and each failure was a
neighbour the implementing commit did not come back to:

| suite | what it said | why |
|---|---|---|
| `test_dtypedev` | `now computes and did not (4): int8\|matmul, int16\|matmul, int32\|matmul, int64\|matmul` | `_CPU_REACHES` and DTYPEDEV.md §3.1 freeze the exact set of cells that compute. `exact_int_matmul` widened it and neither was updated. Fixed: four rows go 18 → 19, and **`int16` is in that gain**, which §4.3b's fifteen does not contain — its sweep has no `int16` column |
| `test_mpsinplace` | `aten.fill_.Scalar now works on mps. That is good news: make this an agreement test` | the pin §7.7 left behind did exactly what it was written to do. Promoted: `test_fill_on_mps_agrees_with_upstream` plus a second test holding the `int32` cell that still refuses |
| `test_intmps` | `int16 on mps refused matmul without offering the int64 road` | **the message was right and the test was wrong.** `int64` matmul on `mps` refuses too, so recommending it would dead-end. The invariant "every refusal offers the `int64` road" was true only while the refusing operators were elementwise; it is now per-operator, and `matmul`'s device-keeping road is a float cast |
| `test_shim` | `these functions in aten.rs read device bytes to the host and are neither a refused kernel, a known readback helper, nor an exemption with a reason: ['exact_int_matmul']` | the scan is a real instrument and it fired correctly. `exact_int_matmul` refuses a non-host operand at its first statement, so it is an exemption with a reason, and it is written down as one rather than filtered out |

**And twenty golden cases, which the suites cannot see.** With the five suites
green the gate was still red, in the one marker CLAUDE.md §2 says `ge` is
useless for: `golden_cases_failed eq 0`, reading **20**. Every one of the twenty
said the same thing — *"gap appears CLOSED: both sides now succeed, promote this
case to expect=match and diff real values"*. They are `mm`, `bmm`, `matmul`,
`addmm` and `baddbmm` at `int64`, `int32` and `int16`, recorded in
`tools/golden/cases.py` as a candle gap since before this branch, plus the three
`baddbmm` `alpha=0` cases and the two `alpha=1.9`/`alpha=1` truncation cases that
were pinned as *unverifiable* precisely because the shim never reached the
multiply. `exact_int_matmul` reaches it. All twenty are promoted to
`expect="match"` — so their **values** are now diffed against upstream, which is
a stronger claim than the refusal they replaced — and `_MM_C_ERROR_DTYPES`
shrinks to `["uint8"]`, the one width the signed kernel does not cover. Golden:
**11627/11627, 0 failed, ops=308, pending=0.** The `alpha=1.9` case is the one
worth naming: it now *measures* the truncation its own comment said could not be
verified, and it is bit-for-bit identical to `alpha=1`.

The sixth was `test_ane_subgraph_components`, which died at import on an
absolute `sys.path.insert` naming a worktree that no longer exists — unrelated
to this work, and the reason it was invisible is worth keeping: **a suite that
dies at import contributes 0 ok and 0 FAIL**, so it costs the ledger nothing
and reads as present.

#### Sixteen of the twenty-one failures were one character of path

The rest were not this branch at all. Fifteen suites decide "did this probe get
the shim or upstream?" with

    "torchnative" in (torch.__file__ or "")

and the workspace move that put the caches inside the repository made
`/Volumes/macMini/caches` a symlink to
`/Volumes/macMini/thisisthepy/torchnative/.caches`. **Upstream torch now lives
under a directory literally named `torchnative`**, so that predicate is a
constant `True` and every one of those suites failed with "the upstream-side
probe got the shim". A sixteenth, `test_shim`'s HF-quantiser provenance check,
asks the same question of traceback frames and answered the same way — it
reported that transformers' own refusal came from torchnative.

This is the CLAUDE.md §5.5 shape inverted: not a check that cannot fail, but a
check that cannot pass, and both are the same defect — the instrument stopped
depending on the thing it claims to measure. Replaced with
`hasattr(torch._C, "_aten_implemented")`, which is what actually distinguishes
the two builds, and with a path-component test that excludes `site-packages`
for the traceback case.

**Found twice, independently, within the same hour.** The coordinating session
diagnosed it from the other side — it had made the workspace move — and landed
the same replacement on `develop`, at the same nineteen call sites and with the
same predicate. That is worth recording for a reason beyond bookkeeping: two
rounds converged on `_aten_implemented` because it is the only thing in this
directory that is a fact about **which torch was imported** rather than about
where a file happens to sit. **A substring of a path is not that fact**, and
this is the second time a workspace move has invalidated one: the first was the
absolute `sys.path.insert` in `test_ane_subgraph_components` above. The fix in
this worktree is redundant with develop's and should be dropped in favour of it
at merge — kept here only so the gate numbers below were measured on a tree
where the instrument worked.

#### Still open, and named

**`eye` and `linspace` build an `f64` on the device — cause D's defect wearing
cause B's message.** Measured, unchanged by this round: `linspace` on `mps`
refuses with `Metal contiguous to_dtype F64 F32 not implemented`, and
`aten.rs` shows why — `Tensor::from_vec(values, n, &device)` puts a `Vec<f64>`
straight onto Metal and converts afterwards. That is exactly the shape
`host_full` exists to remove, one call shape further out, and
`test_no_float_constant_is_still_materialised_on_the_device` does not catch it
because it only inspects `Tensor::full`. Eight measured cells
(`eye` and `linspace`, `float32`/`float16`/`bfloat16`/`int32` on `mps`) are
behind it. **Not done here** — it is new operator work, not verification, and
it is proposed rather than taken.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs amax_keepdim_anywhere present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_constset.py test_the_mps_fill_is_bracketed_by_the_metal_counters present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_constset.py test_the_nan_seed_call_sites_answer_where_they_are_reachable present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_constset.py test_the_nan_seed_on_mps_is_either_refused_or_a_recorded_defect present -->
<!-- DOCWATCH: op-implemented aten.amax.default -- the operator the mps NaN defect above is pinned on -->
<!-- DOCWATCH: op-not-implemented aten.amin.default -- asserted by the NaN table rather than assumed -->

### 4.3 470 refusals still hand back a candle symbol

1201 of 1671 refusals name themselves. The other 470 quote an internal candle
symbol — `mlx matmul doesn't support I64`, `Error while loading function:
badd_i16` — concentrated in `mm`/`bmm`/`matmul`/`addmm`/`baddbmm` and the
factories on `mps`. Each is a true refusal; none of them tells the reader what
to do. This is the same gap `test_intmps.py` closed for `int16`/`int32` and
`storage_for` closes for `float64`, and it is the largest piece of work this
matrix names.

---

## 5. What this table structurally cannot see

* **One shape per operator** — the first case `tools/golden/cases.py` builds. A
  defect that needs broadcasting, a non-contiguous input or an empty tensor is
  invisible here.
* **The oracle is asked on the `cpu`** even for `mps` cells, following
  `rust/torch_c/pytests/test_dtypedev.py`: the question is whether the shim's
  number is upstream's number, not whether upstream would produce it on that
  device.
* **Casting every tensor operand to the cell's dtype casts index operands too**,
  so rows like `index_select`, `gather`, `scatter` and `masked_fill` report on
  an input upstream itself rejects. Those are BREAKS, and they are a limit of
  the sweep, not a finding about the shim — the largest BREAKS causes are
  upstream's own refusals (`result type Float can't be cast to the desired
  output type`, 101 cells; `where expected condition to be a boolean tensor`,
  48).
* **No `cuda`, `vulkan` or `npu` column.** This machine has none of them.
* A cell is one measurement, not a proof. `docs/numerics/DTYPEDEV.md` remains
  the document for the dtype axis, and its frozen `mps` column in
  `test_dtypedev.py` is what actually holds those cells down in the gate.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/dtype_device_matrix.py result_dtype_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/dtype_device_matrix.py tally present -->

---

## 6. The table

**Re-measured 2026-09-20** by `rust/torch_c/pytests/dtype_device_matrix.py
--drive` against the artefact built from this tree — 308 operators, 4928 cells.
The 2026-09-19 table it replaces is compared against cell by cell in §7.7; the
2026-09-16 one before that, in §7.1.
Reproduce with:

```sh
PYTHONPATH=<stage>:rust/torch_c/pytests python3 \
    rust/torch_c/pytests/dtype_device_matrix.py --drive \
    --json /tmp/matrix.json --markdown /tmp/matrix.md
```

| Operation | float32_cpu | float16_cpu | bfloat16_cpu | float64_cpu | int64_cpu | int32_cpu | int8_cpu | bool_cpu | float32_mps | float16_mps | bfloat16_mps | float64_mps | int64_mps | int32_mps | int8_mps | bool_mps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aten._grouped_mm.default | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten._is_all_true.default | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten._local_scalar_dense.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten._log_softmax.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten._safe_softmax.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten._scaled_dot_product_flash_attention_for_cpu.default | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| aten._softmax.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten._to_copy.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten._unique2.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten._unsafe_view.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten._weight_norm_interface.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.abs.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.abs_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.acos.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REACHES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.adaptive_avg_pool1d.default | AGREES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.adaptive_avg_pool2d.default | AGREES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.add.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.add.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.add_.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.add_.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.addmm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.alias.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.all.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.all.dim | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.all.dims | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.allclose.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.amax.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.any.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.any.dim | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.arange.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.arange.start | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.arange.start_step | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.argmax.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.argsort.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.argsort.stable | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.as_strided.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.avg_pool2d.default | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.baddbmm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bernoulli_.float | REACHES | REACHES | AGREES | REACHES | REACHES | REACHES | REACHES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bitwise_and.Scalar | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bitwise_and.Tensor | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bitwise_not.default | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bitwise_or.Scalar | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bitwise_or.Tensor | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bitwise_xor.Scalar | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bitwise_xor.Tensor | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bmm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.broadcast_tensors.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| aten.bucketize.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.bucketize.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.cat.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| aten.ceil.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.ceil_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.chunk.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.clamp.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.clamp_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | BREAKS | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.clamp_min.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.clamp_min_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.clip.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.clone.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.col2im.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.constant_pad_nd.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.convolution.default | AGREES | BREAKS | REFUSES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.copy_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.cos.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.cos_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.cumprod.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.cumsum.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.detach.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.diag.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.diff.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.div.Scalar_mode | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.div.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.div.Tensor_mode | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.div_.Scalar | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | REFUSES |
| aten.div_.Tensor | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.embedding.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.empty.memory_format | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | BREAKS | AGREES |
| aten.empty_like.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.empty_strided.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | BREAKS | AGREES |
| aten.eq.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.eq.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.equal.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.erf.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.erf_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.erfinv.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.exp.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.exp_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.expand.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.expm1.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.expm1_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.eye.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.eye.m | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.fill_.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.fill_.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.flip.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.floor.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.floor_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.floor_divide.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.floor_divide.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.fmod.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.fmod.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.full.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.full_like.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.gather.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.ge.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.ge.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.gelu.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.glu.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.greater.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.greater.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.gt.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.gt.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.hann_window.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS |
| aten.hann_window.periodic | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS |
| aten.hardtanh.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REACHES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REACHES |
| aten.histc.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.i0.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.im2col.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.index.Tensor | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.index_add.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.index_add_.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.index_copy.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.index_copy_.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.index_put_.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.index_select.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.is_floating_point.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.isin.Tensor_Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.kaiser_window.beta | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | BREAKS |
| aten.kaiser_window.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | BREAKS |
| aten.kaiser_window.periodic | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | BREAKS |
| aten.le.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.le.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.leaky_relu.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.lift_fresh.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.lift_fresh_copy.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.linalg_qr.default | AGREES | REFUSES | REFUSES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.linalg_vector_norm.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.linspace.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.log.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.log2.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.log2_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.log_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.logical_and.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.logical_not.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.logsumexp.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.lstm.input | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.lt.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.lt.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.masked_fill.Scalar | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.masked_fill_.Scalar | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.masked_scatter.default | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.masked_select.default | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.matmul.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.max.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.max.dim | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.max.other | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.max_pool1d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.max_pool2d.default | AGREES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.maximum.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.mean.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.mean.dim | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.min.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.min.dim | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.min.other | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.mm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.mul.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.mul.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.mul_.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.mul_.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.multinomial.default | AGREES | REACHES | REACHES | REACHES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.multiply.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.multiply.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.native_batch_norm.default | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS |
| aten.native_dropout.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.native_group_norm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.native_layer_norm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.ne.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.ne.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.neg.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.neg_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.new_empty.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.new_full.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.new_ones.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.new_zeros.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.nll_loss_forward.default | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.nonzero.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.norm.ScalarOpt_dim | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.normal_.default | REACHES | REACHES | REACHES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.one_hot.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.ones.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | BREAKS | BREAKS | AGREES |
| aten.ones_like.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.permute.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.pow.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.pow.Tensor_Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.pow.Tensor_Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.prod.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.prod.dim_int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.randint.default | REACHES | REACHES | REACHES | REACHES | REACHES | REACHES | REACHES | BREAKS | REACHES | REACHES | REACHES | REFUSES | REACHES | REFUSES | REFUSES | BREAKS |
| aten.randint.low | REACHES | REACHES | REACHES | REACHES | REACHES | REACHES | REACHES | BREAKS | REACHES | REACHES | REACHES | REFUSES | REACHES | REFUSES | REFUSES | BREAKS |
| aten.randperm.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES |
| aten.reciprocal.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.reciprocal_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.reflection_pad1d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.reflection_pad2d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.reflection_pad3d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.relu.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.relu_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.remainder.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.remainder.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.repeat.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.repeat_interleave.Tensor | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.replication_pad1d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.replication_pad2d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.replication_pad3d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.reshape.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.rms_norm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.roll.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.round.decimals | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.round.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.round_.decimals | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.round_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.rsqrt.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.rsqrt_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.rsub.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.scalar_tensor.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.scatter.src | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.scatter.value | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.scatter_.src | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.scatter_.value | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.scatter_reduce.two | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.select.int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.sigmoid.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.sigmoid_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.sign.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
| aten.silu.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.sin.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.sin_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.sinc.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.slice.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.softplus.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.sort.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.split.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.split_with_sizes.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.sqrt.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.sqrt_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.squeeze.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.squeeze.dim | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.squeeze.dims | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.stack.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| aten.std.correction | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.std.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.std.dim | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.stft.center | AGREES | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.stft.default | AGREES | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.sub.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.sub.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.sub_.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.sub_.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.sum.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.sum.dim_IntList | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.t.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.t_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.tanh.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.tanh_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.topk.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.transpose.int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.tril.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.triu.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.unbind.int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.unflatten.int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.unfold.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.uniform_.default | REACHES | REACHES | REACHES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REACHES | REACHES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.unsqueeze.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.upsample_bicubic2d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.upsample_bilinear2d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.upsample_linear1d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.upsample_nearest1d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.upsample_nearest2d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var.correction | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var.dim | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var_mean.correction | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var_mean.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var_mean.dim | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.view.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.view.dtype | AGREES | BREAKS | BREAKS | AGREES | AGREES | AGREES | REACHES | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.view_as.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.where.Scalar | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.where.ScalarOther | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.where.ScalarSelf | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | REFUSES |
| aten.where.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.where.self | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.zero_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.zeros_like.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| prims.broadcast_in_dim.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| prims.clone.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| prims.cos.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| prims.erf.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| prims.neg.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| prims.reciprocal.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| prims.rsqrt.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| prims.sin.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| prims.split_dim.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| prims.sqrt.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| prims.tanh.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| prims.transpose.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| prims.view_of.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |


---

## 7. Two re-measurements: 2026-09-19 and 2026-09-20

Both on branch `work/cmlsilence`, same host and same oracle (upstream torch
2.13.0). **§7.1–§7.6 are the 2026-09-19 round** (on develop `11fc109`);
**§7.7–§7.10 are the 2026-09-20 round** (on develop `b4f89f3`), which lifted
the gate §7.2 identified. They are kept in one section, and in order, because
the second is legible only against the first: §7.2 is the finding and §7.7 is
what was done about it.

Every cell in §6 was re-derived by the sweep rather than carried, in each
round, and in each round the sweep was run again *after* the change so that
the table published above is the table this tree produces. §6 is therefore the
2026-09-20 measurement; the before/after counts in §7.7 are the two halves of
that round.

### 7.1 What the 2026-09-16 table got right, and the two cells it did not

Of the **4864** cells that table published, **4862 still read exactly what it
says.** That is the headline, and it is worth stating plainly because the
round that produced it opened by finding its own predecessor wrong in 730
cells: a matrix built by reading each operator's *schema* and injecting
`device=`/`dtype=` is reproducible, and a matrix built by placing tensor
arguments through a proxy was not.

The two that differ are both `aten.bernoulli_.float`, at `float32_cpu` and
`bool_cpu`, and both moved `AGREES -> REACHES` because the shim's RNG drew
different bits from upstream's. They are not a regression and not a defect:
§4.2 already says the 40 RNG cells cannot be compared by value at all, and
`tools/golden/cases.py` carries `value_check` hooks for exactly this family.
**What they are is a reason not to grade an RNG cell AGREES**, which this
table still does for them whenever the draws happen to coincide — a cell that
flips between two verdicts across identical runs is measuring the seed.

**None of the 4864 is wrong in the dangerous direction.** The check that
matters here is a cell the document calls AGREES where reality refuses, or
where the answer comes from a device other than the label — the failure §2 was
written about. There is none. The two disagreements are AGREES -> REACHES,
which is the *safe* direction: the document over-promised only to the extent
of two RNG draws, and the sweep's own placement check
(`answered on %s under a %s label`) fired on no new operator.

**64 cells were never published.** Four operators reached
`_aten_implemented()` after that table was measured — `aten.var_mean.default`,
`aten.var_mean.dim`, `aten.var_mean.correction` and
`aten.lift_fresh_copy.default` — so the table had 304 rows for 308 operators.
A missing row reads like an operator nobody thought to measure, which is the
same failure mode §5 warns about for a missing column. They are in the table
above.

### 7.2 108 cells were published as BREAKS and are named refusals

The largest single correction, and it runs in the **pessimistic** direction:
nothing became AGREES.

`write_back` refuses an in-place write to a non-CPU tensor, and its sentence is

```
aten.zero_.default: writing through a view is implemented for the CPU backend
only in torch._C shim; this tensor is on mps:0
```

The sweep's `classify_refusal` looked for `"not implemented"`, `"unsupported"`
and four more fragments. This sentence says *"is implemented for"*, so it
matched none of them and every one of these cells fell through to BREAKS —
which §1 defines as "a panic, a hard crash, or a cell this harness could not
build", and "never read as a refusal". **112 cells across 33 in-place
operators** were published that way (108 of them inside the 4864 the previous
table covered; the rest are in the four new rows).

The fix is one fragment added to `_REFUSALish`, and it is a defect in the
harness, not in the shim. What it uncovers is worth naming, because BREAKS was
hiding it: **the entire in-place family refuses on Metal, for one reason.**
`add_`, `mul_`, `div_`, `sub_`, `clamp_`, `copy_`, `fill_`, `masked_fill_`,
`relu_`, `sigmoid_`, `tanh_`, `zero_` and twenty more all stop at the same
gate. That is one fix, not thirty-three, and it is now legible as one.

### 7.3 `abs` on Metal — four cells from REFUSES to AGREES

```python
import torch
x = torch.randn(1024, 1024, device="mps")
x.abs()                      # NotImplementedError, before this round
```

Three lines, the second and third of which anyone writes. `aten.abs.default`
read `REFUSES` in all eight `mps` columns, and the refusal was
`mps_host_readback_gate`'s: a kernel that moves its operand's bytes to the
host computes on the CPU under an `mps` label, and this repository refuses
that rather than doing it silently (docs/devices/MPS.md §2).

**The gate was right and the kernel was half wrong.** `abs`'s floating path
was already one candle op (`Tensor::abs`) and never touched the host; its
integral path went out through `to_vec1::<i64>()` and a scalar `wrapping_abs`
loop. The refusal list is *op* granular, so the integral half was refusing the
floating half's three dtypes as well.

**The readback was removed, not the name.** Taking a name off
`MPS_HOST_READBACK_OPS` while leaving the `to_vec1` in place is the defeat
docs/devices/MPSATTN.md §3.1 records, and
`test_shim.py::test_the_mps_readback_list_is_what_the_kernels_actually_do`
re-derives the whole list from `aten.rs` on every gate run specifically to
catch it. `integral_abs_on_device` (`aten.rs`) is `maximum(x, 0 - x)`:

* the negation is a **binary** subtraction from zero, not `Tensor::neg`,
  because candle's `unary_op!` macro fills every integer arm with `todo!()` —
  `neg` on an `i64` tensor panics rather than raising. `bin_op!` is the
  opposite and has real arms for every integer width.
* **the wrap is unchanged.** `0 - INT_MIN` wraps to `INT_MIN` in the storage
  width and `maximum(INT_MIN, INT_MIN)` is `INT_MIN`, which is what upstream
  returns. All three signed widths are pinned against upstream in
  `test_absmps.py`.
* **unsigned storages are the identity and must be.** `0u8 - 5` wraps to
  `251`; `maximum` would pick it and corrupt every nonzero element.

Measured after the change: `float32_mps`, `float16_mps`, `bfloat16_mps` and
`int64_mps` all **AGREES**. The other four `mps` dtypes stay refused for
reasons that are not `abs`'s and are not lifted here — `float64` because Metal
has no double (§3.1), `int32`/`int8` because this build refuses them on Metal
by name (`_shim_mps_unsupported_int_dtypes`, and the candle fork's CPU-only
`DType::I8`, docs/numerics/INT8.md §1.2), and `bool` because upstream refuses
it too.

### 7.4 `abs_` stayed REFUSES, and its refusal stopped lying

> **Superseded on 2026-09-20 by §7.7.** `abs_` now AGREES on all four `mps`
> dtypes Metal allows: the write-back gate this section calls "a separate gate
> this round does not lift" was lifted. The paragraph is kept as the record of
> what was true on 2026-09-19, and `test_absmps.py`'s refusal test became an
> agreement test — which its own docstring had said to do if this happened.

`abs_`'s kernel lost the same readback, so it left the list too. It still does
not run on `mps`: it hits §7.2's gate instead. **The cell is published as
REFUSES and nothing better** — but the sentence it gives changed from one that
was no longer true ("this kernel reads the tensor back to host memory") to the
one that is. A refusal naming the wrong reason sends its reader to the wrong
fix. `test_absmps.py` pins both halves: the write-back sentence must be there,
and the readback sentence must not.

### 7.5 What this round could not prove, and the one thing that would fix it

> **Closed on 2026-09-20 by §7.11.** `_C._metal_counters()` exists, the fork
> carries the counters, and §7.12 records the substitution experiment this
> section says could not be run. The paragraph below is left as it was written
> because §7.11 is only readable against it; read it as history, not as the
> current state.

**There was no Metal dispatch counter in this build.** `device.rs` has
`_cuda_counters()` and `_vulkan_counters()`; Metal had neither, and candle's
`MetalDevice` exposes no countable kernel launch — `capture()` writes a
`.gputrace` and nothing smaller. So the placement evidence for §7.3's four
cells is (a) the kernel performs no host readback *by construction*, re-derived
from source by the gate on every run, and (b) the artefact's own refusal table.
That is the standard `softmax_on_device` was landed at (MPSATTN.md §3.1), and
it is **weaker than a counter**, which is why this section says so instead of
grading around it.

A counter would have to go in `vendor/candle-core`, at the Metal
backend's host-readback path. That is a change to a fork whose documented
contract is "two inputs and nothing else" (docs/numerics/INT8.md §1.2) and
whose patch `vendor/vendor_candle.sh --check` verifies byte for byte in the
gate, so it is not a change to make inside a round about something else. It is
**the** piece of infrastructure that would raise every `mps` cell in this
document above the present ceiling, and it is not built here.

### 7.6 Nullification

Each change was broken on purpose, rebuilt, and the suite re-run.

| nullification | result |
|---|---|
| `integral_abs_on_device` returns its input unchanged | **RED** — 2 of 4 |
| the unsigned identity arm removed (`uint8` goes through `maximum`) | **RED** — 2 of 4 |
| `maximum(x, 0)` instead of `maximum(x, 0 - x)` (relu, not abs) | **RED** — 3 of 4, including the wrap test |
| `aten.abs.default`/`abs_` put back on `MPS_HOST_READBACK_OPS` | **RED** — 3 of 4 |
| all restored | **4 of 4 green** |

The third row is the one that matters for
`test_abs_wraps_at_the_signed_minimum_exactly_as_upstream_does`: the first two
mutants leave it green, because `abs(INT_MIN) == INT_MIN` is also what an
identity returns. A test that no mutant can redden is worthless, and that one
needed a fourth mutant to show it is not.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs integral_abs_on_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_absmps.py test_abs_agrees_with_upstream_on_both_devices present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_absmps.py test_abs_wraps_at_the_signed_minimum_exactly_as_upstream_does present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_absmps.py test_abs_on_mps_does_not_come_back_through_the_readback_gate present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_absmps.py test_abs_inplace_agrees_with_upstream_on_cpu_and_on_mps present -->
<!-- DOCWATCH: op-implemented aten.abs.default -->


---

### 7.7 The in-place family on Metal — one gate, 109 cells, no fork change

§7.2's finding was that **112 cells across 33 in-place operators stop at one
sentence**, and that it was one gate rather than thirty-three gaps. This
section lifts it. Re-measuring the whole matrix before and after, on the same
machine, with the same sweep:

| | before | after |
|---|---|---|
| AGREES | 2516 | **2626** |
| REACHES | 51 | 53 |
| REFUSES | 1801 | **1689** |
| BREAKS | 528 | 528 |
| refusals that name themselves | 1327 | 1215 |

Cell by cell, **exactly 112 cells left REFUSES**: 109 to AGREES across 32
operators, 3 to REACHES (`uniform_`, which draws and therefore has no oracle —
REACHES is the correct verdict and AGREES would be measuring the seed, §7.1).
One further cell flips `aten.bernoulli_.float` `REACHES → AGREES`, which is the
same RNG flapping §7.1 named and not a result. **Nothing moved in the
dangerous direction**: no cell went AGREES → anything, and no new
disagreement appeared outside the three RNG cells.

**What the gate actually was, named before it was fixed.** Not a missing
kernel and not a missing aliasing analysis. `tensor.rs::write_into` is the
single write door, and it was implemented as a *host-side* strided scatter:
`flat_storage` pulled the computed replacement out through `to_vec1`, and
`WriteThrough` implements candle's `InplaceOp1::cpu_fwd` only, walking the
destination's layout over a `&mut CpuStorage` **slice**. There is no slice
behind a Metal buffer. The refusal was honest; what it named was the door, not
the operator — which is why one sentence sat under thirty-three names.

**What lifts it.** For a **contiguous** receiver, the storage positions the
view addresses are one unbroken run of `numel` elements starting at
`layout.start_offset()`. candle already publishes a door for exactly that run:
`Tensor::slice_set` → `BackendStorage::copy2d` → on Metal, with
`src_s == d2 == dst_s`, a **blit on the device's own command queue**
(`metal_backend/mod.rs`). `tensor.rs::write_on_device` is that call plus two
guards, and every line of it is on the pinned crate's public API.

**No change to `vendor/candle-core`, and that was checked rather than assumed
before any code was written.** The fork's contract is "two inputs and nothing
else" (docs/numerics/INT8.md §1.2) and `vendor/vendor_candle.sh --check`
verifies its patch byte for byte in the gate. Two candle-side designs *were*
considered and both rejected for needing the fork: an `InplaceOp1::metal_fwd`
that replaces the storage wholesale (`MetalStorage`'s element count is a
private field, so the "does this view cover the whole buffer" test cannot be
written from outside), and a generic device scatter-by-layout (candle exposes
no rank-n scatter; `copy2d` takes a single stride pair). `slice_set` needs
neither.

**What is NOT lifted.** A *non-contiguous* receiver — `x[:, 0:2]`, a
transpose, an `expand` — still refuses, because writing a broken run needs the
scatter candle does not expose. **It must not fall through to the host path**:
serving a device receiver through `flat_storage` would compute the write on
the CPU under an `mps` label, which is what docs/devices/MPS.md §2 exists to
refuse. So the refusal stays, and it changed sentence: it now names the
*layout*, says what does work, and tells the reader to `.contiguous()`.
Naming the backend is what sent 112 cells to the wrong diagnosis.

**Placement evidence, and it is the same ceiling as §7.5.** There is still no
Metal dispatch counter, so this round cannot prove "the blit ran on the GPU"
the way a `_vulkan_counters()` round can. The available standard is
structural: `write_on_device` contains no readback marker and does not call
`flat_storage`, and `write_into` dispatches to it *before* `flat_storage` is
reached — both re-derived from `tensor.rs` on every gate run by
`test_mpsinplace.py::test_the_device_write_door_performs_no_host_readback`.
That is weaker than a counter and is stated as weaker. It matters concretely:
three rounds on 2026-09-19 planted a host-computed twin, found every value
correct and their agreement tests green, and only dispatch counters caught the
fallback. On Metal that instrument does not exist.

### 7.8 31 of 43, and the twelve that did not move are two gates

Every in-place operator `_aten_implemented()` reports was dispatched on an
`mps` tensor one at a time, rather than inferring coverage from the matrix
(whose cell is one shape per op, §5).

| | count | why |
|---|---|---|
| reach on `mps` after this round | **31** | the write door serves them |
| refused by the **host-readback** gate | **4** | `expm1_`, `log2_`, `index_add_`, `index_put_` — a real and different gate, correctly still closed |
| blocked by **f64 on Metal** | **8** | `fill_.Scalar`, `fill_.Tensor`, `masked_fill_.Scalar`, `round_.decimals`, `bernoulli_.float`, `normal_.default`, `scatter_.value`, `scatter_.src` |

The third row is **one gate under eight names**, the same shape §7.2 found:
these kernels build a constant or an intermediate as `f64` (`Tensor::full(
value.as_f64(), ...)`, a `to_dtype` through `F64`), and candle's Metal backend
has no `F64` at all — `unsupported const-set f64`, `Metal contiguous to_dtype
F64 F32 not implemented`. The cells that *do* move for these operators are
exactly the dtypes whose constant is not an `f64` (`bool`, `int64`), which is
what makes the diagnosis legible rather than guessed. **It is not fixed here**
— it is a change to `aten.rs`'s constant construction with its own agreement
surface, and folding it into a round about the write door would make both
harder to judge. `test_mpsinplace.py::
test_fill_on_mps_is_blocked_by_a_different_gate_than_this_one` pins it so it
cannot be quietly reclassified.

**Closed 2026-09-22, for the float dtypes.** §4.3b's `host_full` is exactly
"the change to `aten.rs`'s constant construction" this paragraph declined to
make, and it removed the `unsupported const-set f64` gate for `float32`,
`float16` and `bfloat16`. The pin did its job — it went RED when `fill_` on
`mps` started answering — but the commit that closed the gate did not come back
to it, so it sat red. It has been promoted as it instructed:
`test_fill_on_mps_agrees_with_upstream` grades the three float cells
element-wise against upstream, and
`test_fill_on_mps_is_refused_by_name_for_the_integer_dtypes` keeps the `int32`
cell, which still refuses, from vanishing with the old test. The eight names in
the third row above are **not** all closed: the gate is closed where the
constant is a float the destination dtype can hold, and `to_dtype through F64`
(`eye`, `linspace`, `full`/`scalar_tensor` at `int32`) is untouched — see
§4.3c.

**Updated 2026-10-02 (§7.18).** Of the eight names, `fill_.Tensor`,
`bernoulli_.float` and `normal_.default` now agree on the three float dtypes,
and `scalar_tensor` at `int32` agrees; `full` at `int32` still breaks, on a
different candle symbol. `scatter_.value` and `scatter_.src` stay refused by
the host-readback gate (§7.16), which was always in front of them.

### 7.9 A silently wrong answer, older than this round, found by re-measuring

**`clamp` and `clamp_min` have dropped NaN on Metal since `mps` landed.**
Upstream propagates it — `torch.tensor([nan]).clamp_min_(0.)` is `nan` — and
so does candle on the CPU. Its Metal `maximum`/`minimum` are MSL `max`/`min`,
which return the *non-NaN* operand, so `clamp_values` returned `0.0`.

Two things are worth keeping apart.

* **This document graded those cells AGREES in every published table, and was
  not wrong to within what it can see.** Each cell is `tools/golden/cases.py`'s
  first case for the operator, and that case has no NaN in it. §5's "one shape
  per op" is this blind spot, and this is the first time it has been
  demonstrated rather than warned about. The sweep caught the divergence only
  because the *in-place* case builder happens to carry a NaN and the
  out-of-place one does not — `clamp_min_` landed as `REACHES` with
  `disagrees`, three cells, while its identical out-of-place sibling sat at
  AGREES.
* **Lifting §7.7's gate would have converted a refusal into a wrong answer.**
  `clamp_`/`clamp_min_` were refused on Metal before this round, so the
  divergence was unreachable through them. That is the one direction CLAUDE.md
  §4 does not permit, and it is the argument for re-measuring the whole matrix
  after a change rather than testing the change.

The fix is in `clamp_values` and is device-resident: `where(x != x, x,
clamped)` — an elementwise compare and a select, both candle kernels on
whatever device the tensor is on, so nothing returns to the host and the op
stays off `MPS_HOST_READBACK_OPS`. It is applied only off the CPU, because the
CPU is already right and `clamp` is on `mamba`'s per-step hot path; the
condition is `!is_cpu()` rather than `is_metal()` because CUDA's kernels have
the same `fmax`/`fmin` shape and there is no CUDA on this machine to rule it
out on.

### 7.10 Nullification

Each new guarantee was broken on purpose, **rebuilt through
`vendor/install_shim.sh` so the mutant reached the vendored tree and not only
the stage**, and the two suites re-run. A mutant that only reached the stage
would let a subprocess probe read the unmutated `.so` and report a real test
as toothless.

| nullification | `test_mpsinplace.py` | `test_absmps.py` |
|---|---|---|
| `write_on_device` returns `Ok(())` without writing | **RED — 6 of 8** | **RED — 1 of 4** |
| the `.copy()` that breaks source/destination storage sharing removed | **RED — 1** (`..._self_overlapping_copy_...`) | green |
| the contiguity refusal removed, so a strided receiver reaches `slice_set` | **RED — 1** (`..._strided_receiver_...`) | green |
| `flat_storage` reached before the device receiver is dispatched | **RED — 1** (`..._no_host_readback`) | green |
| the NaN restore in `clamp_values` disabled | **RED — 1** (`..._clamp_propagates_nan_...`) | green |
| `ne` written as `eq` in the NaN restore (select inverted) | **RED — 1** (same) | green |
| all restored | **8 of 8 green** | **4 of 4 green** |

Rows two through six each redden **exactly one** test, which is the property
worth having: each guarantee has a test that is about it and not about
something else. Row one is the blanket mutant and reddens six, including the
NaN test — `clamp_min_` is an in-place op, so a write door that claims success
without writing makes its receiver keep the unclamped values.

The two that stay green under row one are the right two:
`..._no_host_readback` is a claim about *source* and a mutant that removes the
write does not add a readback, and `..._fill_...` (now
`test_fill_on_mps_agrees_with_upstream`) is about a blocker upstream of the
write door that row one cannot reach.

**One test in this file has no mutant of its own, and that is reported rather
than papered over.**
`test_a_view_taken_before_the_write_sees_it_on_mps` — the write-through
property, the thing docs/kernels/VIEWS.md §6 changed the family for — goes red
only under the first row, alongside five others. No plausible mutant of
`write_on_device` produces *correct values* while hiding them from an alias,
because `slice_set` writes into the shared buffer by construction; making one
would mean replacing the mechanism, not breaking it. So that test
characterises the mechanism rather than discriminating a fault in it. It is
kept: it is the test that would fire if a future round served devices by
rebinding the wrapper, which is the obvious wrong way to do this.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs write_on_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs clamp_values absent -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs clamp_values present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_in_place_ops_agree_with_upstream_on_mps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_a_view_taken_before_the_write_sees_it_on_mps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_writing_into_an_offset_slice_on_mps_touches_only_that_slice present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_a_self_overlapping_copy_on_mps_is_not_half_overwritten present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_a_strided_receiver_on_mps_refuses_and_names_its_layout present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_the_device_write_door_performs_no_host_readback present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_clamp_propagates_nan_on_mps_exactly_as_upstream_does present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_fill_on_mps_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_fill_on_mps_is_refused_by_name_for_the_integer_dtypes present -->
<!-- DOCWATCH: op-implemented aten.zero_.default -->
<!-- DOCWATCH: op-implemented aten.add_.Tensor -->
<!-- DOCWATCH: op-implemented aten.copy_.default -->
<!-- DOCWATCH: op-implemented aten.clamp_min_.default -->

---

### 7.11 The Metal dispatch counter — §7.5's missing instrument, built

§7.5 said a Metal counter "is **the** piece of infrastructure that would raise
every `mps` cell in this document above the present ceiling, and it is not
built here." It is built now. `_C._metal_counters()` returns six numbers and a
`built` flag, in the shape `_cuda_counters()` and `_vulkan_counters()` already
use:

| key | door it is counted at | what it means |
|---|---|---|
| `compute_encoders` | `MetalDevice::command_encoder()` | kernel launches |
| `blit_encoders` | `MetalDevice::blit_command_encoder()` | device copies |
| `host_uploads`, `host_upload_bytes` | `MetalDevice::new_buffer_with_data()` | host → device |
| `host_downloads`, `host_download_bytes` | `MetalStorage::to_cpu()` | device → host |

**The counters are in `vendor/candle-core`, and they had to be.** §7.5 named
that as the reason not to build it inside a round about something else; this
round was given the fork. They cannot live on this side of the FFI boundary:
a counter in `aten.rs` counts what this crate *intended*, and a host-computed
twin intends exactly what the real kernel intends. Only candle can say whether
a compute encoder was opened. Every edit went into
`vendor/int8-candle-0.11.0-cpu.patch` and the tree was regenerated from it, so
`sh vendor/vendor_candle.sh --check` still passes byte for byte.

**What `compute_encoders` is, stated narrowly enough that it cannot be
over-quoted.** It is a successful `command_encoder()` call, which candle hands
straight to a `candle_metal_kernels::call_*` that encodes at least one
`dispatch_thread*`. So it is a **lower bound on GPU dispatches** and an exact
count of candle's GPU op invocations — *not* a count of `dispatch_threads`,
which happen in `candle-metal-kernels`, a crate this vendoring does not cover.
Quoting it as "N kernels ran" would be the same kind of inflation-by-citation
CLAUDE.md §2 records for the "four rounds"/"five rounds" count.

Measured, one `aten.abs.default` on an `mps` `float32` `[2, 3]`:

    build the operand   uploads 1 (+24 bytes), compute 0
    abs                 compute 1, downloads 0
    .cpu()              blits   1, downloads 1 (+24 bytes)

**Which `mps` claims this upgrades.** §7.3's four `abs` cells
(`float32`/`float16`/`bfloat16`/`int64`) move from structural evidence — "no
host readback by construction", re-derived from source — to **counted**
evidence: `test_metalcount.py` asserts `compute_encoders >= 1` and
`host_downloads == 0` across the dispatch itself, on all four dtypes, beside
an agreement check against a subprocess oracle.

**Which it does not reach.** Every other `mps` cell in §6's table is still at
the old standard, because the counter is an instrument and not a sweep: no
test in this round asserts a counter delta for `softmax_on_device`, for the
§7.7 in-place family, or for any of the 43 operators §7.8 measured. Those
remain where §7.5 put them until someone writes the assertion. The instrument
now exists; the work of pointing it at each cell does not.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs metal_counters present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-core/src/metal_backend/mod.rs COMPUTE_ENCODERS present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-core/src/metal_backend/mod.rs note_host_download present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-core/src/metal_backend/device.rs note_compute_encoder present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalcount.py test_abs_on_mps_opens_a_metal_kernel_and_reads_nothing_back present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalcount.py test_a_cpu_op_moves_no_metal_counter present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalcount.py test_a_readback_costs_exactly_the_tensors_bytes present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalcount.py test_metal_counters_answers_with_all_six_names present -->
<!-- DOCWATCH: op-implemented aten.abs.default -->

### 7.12 The experiment that could not be run on Metal, run

CLAUDE.md §2 records three rounds that replaced a device kernel with a
host-computed twin — `7dff9f0`, `35e002f`, `22d9158` — and notes that all three
are Vulkan, because Vulkan was the only backend with a counter. This is the
Metal one.

**The subject.** `integral_abs_on_device` (the integral path of
`aten.abs.default`, `maximum(x, 0 - x)`), the kernel §7.3 moved onto Metal and
whose four cells §7.11 upgrades. **Two twins were planted**, both computing
`wrapping_abs` on the host from `to_vec1::<i64>()` and rebuilding the tensor on
the same device, and both returning values identical to upstream's:

| twin | where the readback lives | `test_shim.py` classification scan | `test_absmps.py` (4 tests) | `test_metalcount.py` |
|---|---|---|---|---|
| M1 | inside `integral_abs_on_device`, in `aten.rs` | **RED** — named the function | 4 of 4 green | **RED** |
| M2 | one file deeper, `tensor.rs::twin_abs_i64` | **green — 480 ok, 0 FAIL** | 4 of 4 green | **RED** |

M2 is the result that matters, and it is the one the Vulkan rounds predicted.
The entire existing gate — every value test, the wrap-at-`INT_MIN` test, the
readback-gate table test, and the source scan whose whole purpose is to catch
this — stayed green while `abs` on `int64`/`mps` was computed on the CPU. The
counter said:

    compute_encoders 0   blit_encoders 1   host_downloads 1 (+48 bytes)

**M1 is a finding in the other direction and is reported as one.** The scan is
not as blind as §7.5 implied: it catches a readback added directly to a
function in `aten.rs`, by name, which is exactly what it claims to do. Its
blind spot is narrower than "one call deeper" — it is *one file over*, because
`_aten_rs_functions()` parses `aten.rs` and only `aten.rs`. That is worth
knowing precisely rather than approximately.

**No kernel already claiming device residency was found to be falling back.**
The four `abs` cells were tested with the counter before any mutant was
planted and all four opened a compute encoder and downloaded nothing. The
substitution above is a mutant this round introduced and removed, not a defect
it discovered.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs integral_abs_on_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs twin_abs_i64 absent -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_every_host_readback_in_aten_is_classified present -->

### 7.13 `DType::I8` on Metal — declined, with the reason measured

> **Superseded by §7.17 (2026-09-20), which does not correct it.** The second
> crate was approved and vendored; everything this section says about a
> `candle-core`-only change remains true and is why the second crate was
> needed. The one number below that did **not** survive is "284": the ceiling
> was 185, because 99 of the 284 do not agree on the CPU either.

284 matrix cells fail at stage `operands`: `_tensor_from_flat` cannot build an
`int8` tensor on `mps` at all, because the fork's `DType::I8` is CPU-only
(docs/numerics/INT8.md §1.2) and `storage_from_cpu_storage` refuses
`CpuStorage::I8`. Lifting that refusal was considered this round and **is not
done**, for a reason that is a fact about a crate rather than a preference:

`candle_metal_kernels::DType` has exactly six variants — `BF16 F16 F32 I64 U32
U8` — and the MSL sources instantiate every kernel for those six and no others
(`binary.metal`'s `init_binary` macro). `I8` is not among them. So the reach
of a change confined to `candle-core` is precisely this: the *buffer* could be
built, and every kernel would then refuse for want of a shader symbol. That
converts 284 `operands` failures into 284 kernel-stage refusals and moves **no
cell to AGREES** — a tensor that exists and can do nothing, which is a worse
answer than the refusal it replaces, not a better one.

**Taken, 2026-09-20.** See §7.17 for what the second crate cost (51 lines,
zero new shader bodies) and what it moved (137 cells).

This is the same wall `int16`/`int32` hit, and `device.rs`'s
`_shim_mps_unsupported_int_dtypes` is the decision already taken for it
(docs/numerics/DTYPEDEV.md §4.2). Making `int8` reach a Metal kernel means
adding shader instantiations to `candle-metal-kernels`, a **second** crate to
vendor. This round's approval was to patch this fork, so that decision is left
where it belongs: with the user, stated rather than taken.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs shim_mps_unsupported_int_dtypes present -->

### 7.14 The counter pointed at the rest of the table — and twelve cells are computed on the host

§7.11 built `_C._metal_counters()` and said, in its own words, that it had not
pointed it anywhere except `abs`:

> no test in this round asserts a counter delta for `softmax_on_device`, for
> the §7.7 in-place family, or for any of the 43 operators §7.8 measured.

This round points it. **The finding first, because it is a claim this
repository published and it is wrong.**

#### The finding

**Twelve `mps` cells across eight operators are published `AGREES` in §6's
table and are computed on the CPU.** The operand is downloaded to host memory,
a scalar loop runs there, and the answer is uploaded back to the device it came
from — so every value is correct, every agreement test is green, and
`.device` still reads `mps`. That is precisely the failure §7.12 had to plant
deliberately in order to observe. It was already here.

| operator | cells | measured delta on `mps` | how it reaches the host |
|---|---|---|---|
| `aten.sort.default` | `int64`, `bool` | `compute 0–2, downloads 1 (48 B)` | `order_along` |
| `aten.argsort.default` | `int64`, `bool` | `compute 0–1, downloads 1 (48 B)` | `argsort_core` |
| `aten.argsort.stable` | `int64`, `bool` | `compute 0–1, downloads 1 (48 B)` | `argsort_core` |
| `aten.topk.default` | `int64` | `compute 0, downloads 1 (48 B)` | `order_along` |
| `aten.floor_divide.default` | `int64` | `compute 0, downloads 2 (64 B)` | `floor_divide_impl` |
| `aten.floor_divide.Scalar` | `int64`, `bool` | `compute 0–1, downloads 1 (56 B)` | `floor_divide_impl` |
| `aten.scatter_.src` | `int64` | `compute 0, blit 4, downloads 3 (80 B)` | `scatter_inplace` → `scatter_src` → `read_flat` |
| `aten.scatter_.value` | `int64` | `compute 0, blit 3, downloads 2 (96 B)` | `scatter_inplace` → `scatter_value` → `read_flat` |

The last two rows are §7.7's and §7.8's. §7.8 classified `scatter_.value` and
`scatter_.src` as "blocked by `f64` on Metal", which is true of their **float**
columns and is why the diagnosis read as complete; their `int64` column is not
blocked, reaches, and computes on the host. So **two of the 109 cells §7.7
moved `REFUSES → AGREES` were moved onto the host, not onto the device.** The
other 107 were not: they are counted below and they are clean.

**Why the gate that exists to refuse this did not refuse it.**
`MPS_HOST_READBACK_OPS` refuses `aten.sort.default`'s *kind* of kernel by
name, and none of these eight is on it, because nothing put them there.
`test_shim.py`'s derivation re-derives the list from `aten.rs` on every gate
run — it is the check §7.3 leans on and §7.12 praised — and it follows
**exactly six helper names** plus a derived set of *cross-file* helpers. Each
of these eight kernels reaches `read_flat` through a seventh in-file helper
(`order_along`, `argsort_core`, `floor_divide_impl`) or through another
operator's kernel function (`scatter_inplace` calls `scatter_src`). §7.12
described the scan's blind spot as "**one file over**". That was too narrow and
this corrects it: the blind spot is **one un-named hop**, in the same file.
Twelve production cells sit in it.

**It was not fixed in the round that found it, and that was a decision rather
than an omission** — a capability decision, which CLAUDE.md §5.7 leaves with
the user, and an audit is not the round to take it in. `test_metalplace.py`
pinned all three halves instead — the download happens, the op is not refused,
the scan does not derive it — so that fixing any one of them would turn the
suite red and force this section and §6's table to move with it.

**§7.16 is that fix**, taken on the user's decision, and all three pins moved
together as designed. The eight are refused by name, two more turned up in the
same blind spot once the derivation was deepened, and the assertions in
`test_metalplace.py` now run in the opposite direction. The table above and
the counter deltas in it are kept as the **record of the measurement that
found them**, not as a description of the current build: `sort` on
`int64`/`mps` raises now and moves no counter at all.

#### The claims that are now counted, and the bracket they were counted in

The bracket is the same for every number below, and it is stated because
counters from different brackets read like regressions side by side (CLAUDE.md
§2): operands are built **before** the first snapshot, the counters are read
immediately before and immediately after the single `_aten_dispatch` under
test, and results are read back **after** the second snapshot. Nothing else
runs between the two reads.

| claim | cells counted | result |
|---|---|---|
| `softmax_on_device` (MPSATTN.md §3.1) | 6 — `_softmax` and `_safe_softmax` × `float32`/`float16`/`bfloat16` | `compute_encoders` **5**, `host_downloads` **0** |
| §7.7's in-place family | 52 — the 14 operators × their dtypes | `host_downloads` **0**, `host_upload_bytes` ≤ 8, `compute_encoders ≥ 1` for the 12 that compute |
| §7.8's 43 in-place operators | every (operator, dtype) pair that reaches | `host_downloads` **0** for all but `scatter_.src`/`scatter_.value` |
| §7.3's four `abs` cells (control) | 4 | already counted by §7.11; unchanged |

Each of these is asserted beside element-wise agreement against an oracle
computed in a **separate subprocess** at `tools/golden/dtypes.py`'s derived
tolerance, except the 43-operator sweep, which is **placement only and says
so** — its values are graded by `test_mpsinplace.py` and by §6's table, and
duplicating an oracle for 43 operators would have made the file about
agreement instead of about where the work happened (CLAUDE.md §4).

#### What the counter cannot reach, stated rather than approximated

* `compute_encoders` is a **lower bound on GPU dispatches** — a successful
  `command_encoder()`, handed to a `candle_metal_kernels::call_*` that encodes
  at least one `dispatch_thread*`. It is not a `dispatch_threads` count; those
  live in `candle-metal-kernels`, which this vendoring does not cover.
* `blit_encoders` moves for a device-to-device copy **and** for a readback,
  because a readback blits first. The two are not separated, so
  `blit_encoders > 0` is never used here as evidence of device residency.
  `host_downloads == 0` is.
* **199 `mps` `AGREES` cells move no counter at all** and correctly so: they
  are views, layout changes and dtype-identity cases (`view`, `permute`,
  `slice`, `t_`, `clone`, `_to_copy` within a dtype, `round`/`ceil`/`floor` on
  an integer). For these `compute_encoders ≥ 1` is unavailable and demanding it
  would be widening the claim to make it countable. What *is* available is that
  `host_uploads` and `host_downloads` are both zero, which a host-computed twin
  cannot achieve — it has to move the bytes both ways.
* **Two operators are correctly blit-only** and are asserted as such rather
  than excused: `zero_` (a device `const_set` plus a copy) and `copy_` (the copy
  alone) have `compute_encoders == 0` by construction. The assertion for them
  is `blit_encoders ≥ 1` with both byte counters at zero.
* **`uniform_` downloads 24 bytes and is exempt, not clean.** It is one of the
  two names in `MPS_READBACK_BUT_ALLOWED`: its readback is of a *constant* it
  built itself (`narrow_roundtrip_f32`), not of an input. The counter cannot
  tell those apart — it counts bytes leaving the device, and the reason they
  are leaving is not in the count. The same is true of
  `_local_scalar_dense` (`.item()`), whose readback is what the caller asked
  for. **Both are the class of claim this instrument cannot settle**, and both
  remain settled by reading the kernel.
* **A cell passing is not an operator being right** (§7.9's `clamp`). The
  counter says where the work happened, never whether the shape exercised the
  operator. Of the cells counted above, the ones whose single shape cannot
  distinguish a correct kernel from a wrong one are named in
  `test_metalplace.py`'s docstrings rather than counted as proof.

#### Nullification

Three mutants, each built through `vendor/install_shim.sh` so it reached the
**vendored** tree, and each removed afterwards
(`sh vendor/vendor_candle.sh --check` passes byte for byte).

Each was built and measured **alone**, so the attribution below is measured
rather than inferred.

| mutant | what it broke | `test_metalplace` | `test_metalcount` |
|---|---|---|---|
| M-A: `note_compute_encoder` gutted in the fork | the compute counter | softmax RED, in-place family RED | `abs` RED |
| M-B: `note_host_download` gutted in the fork | the download counter | §7.14 pin RED, 43-operator sweep RED | readback calibration RED |
| M-C: a host-computed softmax twin in `tensor.rs`, one file over from `aten.rs` | `softmax_on_device`'s placement | softmax RED (`compute 0, blit 1, uploads 1 (24 B), downloads 1 (24 B)`) | green — it is about `abs` |

M-C also left **`test_mpsattn.py` — the suite whose entire subject is
`softmax_on_device` — green, 0 FAIL**, while softmax on `mps` computed on the
CPU.

**One honest weakness, found by running M-A and M-B separately.** The
`host_downloads == 0` half of the softmax and in-place tests **survives M-B**:
a download counter that can never move satisfies `== 0` trivially. What stops
that is `test_metalcount.py::test_a_readback_costs_exactly_the_tensors_bytes`,
which asserts a `.cpu()` costs exactly the tensor's bytes and does go RED under
M-B. So the zero in this file is only meaningful because that calibration runs
in the same gate — it is not self-supporting, and it is written down here
rather than left to be discovered by the next round.

M-C is the result worth keeping. It repeats §7.12's M2 on a different kernel
and gets the same answer: the suite built for that kernel cannot tell that the
kernel stopped running on the device. One test elsewhere did fail under M-C —
`test_shim.py`'s central-difference tape check — but on **values**, because the
twin accumulated in `f32` rather than the op's accumulation dtype. That is the
twin being imperfect, not the gate detecting a fallback, and it is recorded
that way rather than counted as a catch.

**Which of the new tests survive which mutant.**
`test_the_readback_derivation_scan_does_not_reach_these_kernels` survives all
three, correctly — it is a claim about source, and no counter mutant can touch
it. Every other test in the file dies to at least one: softmax and the in-place
family to M-A (and softmax also to M-C), the §7.14 pin and the 43-operator
sweep to M-B.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalplace.py test_softmax_on_mps_opens_metal_kernels_and_reads_nothing_back present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalplace.py test_the_in_place_family_on_mps_computes_on_the_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalplace.py test_every_in_place_operator_that_reaches_on_mps_reads_nothing_back present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalplace.py test_eight_operators_answer_an_mps_dispatch_from_the_host present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metalplace.py test_the_readback_derivation_scan_reaches_these_kernels present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs softmax_on_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs twin_softmax absent -->
<!-- DOCWATCH: symbol-in-file vendor/candle-core/src/metal_backend/mod.rs note_host_download present -->

### 7.15 `candle_metal_kernels::DType` — the instantiation is macro-driven, and that is the shape of the `I8` decision

§7.13 declined `DType::I8` on Metal and left the second-crate decision with the
user. The one fact that decision needs, read rather than implemented:

**Adding `I8` to `candle-metal-kernels` is adding instantiation lines to
existing macros. It is not writing a Metal shader per operator.** Every kernel
family in `src/metal_src/*.metal` is a C++ template instantiated through an
`init_kernel` macro, and the six-dtype list is a hand-written list of macro
*invocations* — one line per dtype — inside a per-family macro:

```c
#define init_binary(bop)                             \
    init_binary_k(bop, bop, f32,  float,    float)   \
    init_binary_k(bop, bop, f16,  half,     half)    \
    init_binary_k(bop, bop, bf16, bfloat,   bfloat)  \
    init_binary_k(bop, bop, u8,   uint8_t,  uint8_t) \
    init_binary_k(bop, bop, u32,  uint32_t, uint32_t)\
    init_binary_k(bop, bop, i64,  int64_t,  int64_t)
```

One added line there lights up **all six arithmetic binaries and all six
comparison binaries at once**, each in its nine layout variants
(`_strided`, `_lstrided`, `_scalar`, `_cs`, `_sc`, …), because `init_binary_k`
fans those out. `cast.metal` is the same shape: `init_cast_all` is a six-line
macro and one more line plus one more top-level `init_cast_all(i8, int8_t);`
produces every cast pair to and from `i8`.

The families that are **not** macro-fanned over dtype are hand-enumerated at
the top level, one line per (index dtype, value dtype) pair:
`INDEX_OP` ×16, `INDEX_ADD_OP` ×18, `GATHER_OP` ×16, `SCATTER_OP` ×10,
`SCATTER_ADD_OP` ×10, `WHERE_OP` ×18, `ARGSORT` ×6. Adding `i8` as a *value*
type there is one line per index dtype per family, not a new shader.

So the count is roughly **30 added lines and zero new shader bodies** for the
elementwise, cast, indexing, ternary and sort families. Two things are outside
that:

* `init_unary_float` covers `f32`/`f16`/`bf16` only, and the integer unary
  instantiations are a hand-written three (`copy` for `u8`, `u32`, `i64`).
  **candle has no integer `sin`/`exp`/`sqrt` on Metal and adding `I8` does not
  change that** — nor should it, since upstream refuses those on integers too.
* `quantized.metal`, `mlx_gemm.metal`, `gemv.metal` and
  `scaled_dot_product_attention.metal` are float-only and untouched by this.

The Rust half is the ordinary enum widening: `DType` gains a variant,
`size_in_bytes` gains an arm, and every kernel-name suffix table gains `"i8"`.

**Nothing was implemented and nothing was vendored for this section.** It is a
reading of `candle-metal-kernels-0.11.0` as published, recorded so that §7.13's
open decision is made against a measured shape rather than a guessed one.

### 7.16 The twelve cells are withdrawn, and the derivation that missed them now follows calls

§7.14 found twelve published `AGREES` cells computed on the host and left the
capability decision with the user. The decision was taken: **refuse them.**

#### Why refusal, and not a third category

The values were never wrong. What was false is the **placement** — and on a
device whose whole point is avoiding host round trips, an invisible sync point
inside a `generate` loop is a performance cliff nobody can see. The obvious
alternative was an "allowed but host-assisted" grade, and it was rejected for a
reason that is about the instrument rather than about taste:
`MPS_HOST_READBACK_OPS` already answers exactly this question for every other
operator that reaches the host, and a third category would create a state
`_C._metal_counters()` cannot distinguish from a real defect. That counter is
the only instrument that has ever caught a host-computed twin (§7.12, and three
Vulkan rounds before it). `MPS_READBACK_BUT_ALLOWED` is deliberately two names
for the same reason.

#### What it costs, named rather than glossed

**All twelve withdrawn cells are integral dtypes.** The float columns of these
operators were *already* not running on `mps`, and that is measured rather than
assumed: `sort` on `float32`/`mps` raises

    aten.sort.default: candle: Metal contiguous to_dtype F32 F64 not implemented

from inside candle, because `read_flat` widens to `f64` and Metal has no
double. So the refusal takes the integral columns and **re-labels** the float
ones; it does not remove a working float path.

| operator | withdrawn (`AGREES` → `REFUSES`) | re-labelled |
|---|---|---|
| `aten.sort.default` | `int64`, `bool` | 6 already-`REFUSES` cells |
| `aten.argsort.default` | `int64`, `bool` | 6 |
| `aten.argsort.stable` | `int64`, `bool` | 6 |
| `aten.topk.default` | `int64` | 6, plus `bool` `REACHES` → `REFUSES` |
| `aten.floor_divide.default` | `int64` | 7 |
| `aten.floor_divide.Scalar` | `int64`, `bool` | 6 |
| `aten.scatter_.src` | `int64` | 3 `BREAKS` + `bool` `BREAKS` + 3 `REFUSES` |
| `aten.scatter_.value` | `int64` | the same shape |
| `aten.linalg_vector_norm.default` | — | 3 `BREAKS` → `REFUSES`, 5 already `REFUSES` |
| `aten.norm.ScalarOpt_dim` | — | the same shape |

**27 cells in §6's table change; 12 of them are a capability withdrawal and
15 are a re-labelling.** Counting them together would have made the
withdrawal look twice its size, and counting only the withdrawal would have
hidden that the document now says `REFUSES` where it said `BREAKS`.

**Every refusal names what still works.** `device.rs::MPS_HOST_READBACK_NOTES`
carries, per operator, the dtypes whose **CPU** column agrees with upstream,
and the gate appends it:

    aten.sort.default: not implemented for the mps device. This kernel reads
    the tensor back to host memory and computes there, [...] Move the tensor
    with .cpu() to ask for the CPU on purpose. On the CPU this op agrees with
    upstream element-wise for float32, float16, bfloat16, float64, int64,
    int32, int8, bool; no dtype of it computes on mps, so .cpu() is the whole
    answer rather than a dtype change.

The note does **not** say "the float column works on mps", because it does
not. Writing the comforting sentence would have been §7.9's `clamp` in prose —
a claim published because it sounded right, never re-run. Instead every dtype
in every note is re-measured against an oracle in a **separate subprocess** by
`test_mpsrefuse.py::test_every_dtype_a_refusal_note_advertises_agrees_on_the_cpu`,
at `tools/golden/dtypes.py`'s derived tolerance.

#### The half that matters: the derivation, not the list

Adding names to a list fixes twelve cells once. What stops the class recurring
is the derivation, and its blind spot had by then been mis-stated twice:

| said | where | correct? |
|---|---|---|
| "one call deeper" | §7.5 | no |
| "one *file* over" | §7.12 | no — §7.12's M2 happened to be one file over |
| "an un-named hop **in the same file**" | §7.14 | yes, and twelve production cells were in it |

`test_shim.py`'s derivation no longer matches names at all. It builds the call
graph of **every `src/*.rs`**, seeds it with the functions whose own bodies
hold a readback marker (`.to_vec[0-3]`, `.to_scalar`, `.to_cpu(`), and closes
it transitively by reverse BFS. A dispatched operator is derived if any path
from its kernel reaches a readback, and the derivation returns the **witness
path**, so a failure prints how the kernel gets to the host rather than only
that it does.

    sort_default, topk_default        -> order_along        -> read_flat
    argsort_default, argsort_stable   -> argsort_core       -> order_along -> read_flat
    floor_divide_{default,scalar}     -> floor_divide_impl  -> read_flat
    scatter_inplace                   -> scatter_src        -> read_flat
    linalg_vector_norm_default,
    norm_scalaropt_dim                -> norm_pow_walk      -> read_flat

**It found ten, not eight.** `aten.linalg_vector_norm.default` and
`aten.norm.ScalarOpt_dim` share `norm_pow_walk`, whose accumulate-in-`acc_t`
reduction opens with an unconditional `read_flat` (§7.14 never looked at them
because no counter was pointed there). They were **not** on the list of eight
this round was given, which is the only interesting thing about them: the
derivation produced two names nobody had handed it. Their cost is zero
capability — every `mps` cell of both was already `REFUSES` or `BREAKS`.

**The independence is proved mechanically, not by reading.** A derivation that
works because somebody added the answer to a list is the same defect wearing a
fix, so `test_mpsrefuse.py::test_the_derivation_finds_the_ten_without_being_told_their_names`
stubs `_C._shim_mps_host_readback_ops()` to the empty list, re-runs the
derivation, and asserts the result is unchanged and still contains all ten,
each through a path of **more than one hop** ending at a function that really
holds a marker.

**The false-positive direction is guarded at the call site.** A previous round
found that matching `to_le_bytes` by bare name flags a dozen clean kernels,
because it is an inherent method on every Rust integer. Calls are matched as
`name(` **not preceded by `.` or `:`** for a function defined in the same
module, and as `(crate::)?module::name(` across modules; a method call is
neither. `test_a_readback_behind_two_un_named_hops_is_still_derived` asserts
both directions on **synthetic** sources rather than on the real tree, so it
cannot be satisfied by an accident of what `aten.rs` contains today.

`_MPS_READBACK_EXEMPT` names are **barriers** in the graph rather than nodes:
`scalar_arg` reads a zero-dim tensor *argument*, `scale_by_alpha` and
`narrow_roundtrip_f32` read constants they built themselves. A kernel calling
one has moved no dispatched tensor, and deriving through it would have put
clean ops on the list.

`test_cuda.py` stopped keeping its own copy of the predicate and imports
`_ops_that_reach_the_host`. Its docstring already recorded that a copy had
diverged once (`aten.view.dtype`, 2026-09-19); it would have diverged a second
time here, by ten names.

#### Nullification

Both mutants were built through `vendor/install_shim.sh` so they reached the
**vendored** tree, run alone, and removed afterwards.

| mutant | what it breaks | result |
|---|---|---|
| N-1: `_callees` returns `set()` — the transitive hop blinded, one-hop matching only | the derivation | `test_shim.py` derivation RED, `test_metalplace` RED, `test_mpsrefuse` RED — all ten stop being derived, i.e. straight back to §7.14 |
| N-2: `aten.sort.default` removed from `MPS_HOST_READBACK_OPS`, readback left in place | the refusal, not the readback — the MPSATTN.md §3.1 defeat | `test_shim.py` derivation RED (`missing: ['aten.sort.default']`), `test_mpsrefuse` RED, `test_metalplace` RED |
| N-3: a refusal note widened to advertise a dtype that does not work | the note, the only part of this a user reads | RED — but see below |

**N-3 is the one worth reading, because its first form survived.** Adding
`int8` to `scatter_.src`'s note left the whole suite green. So did adding
`float32` — and §6 grades `scatter_.src`'s `float32_cpu` cell **BREAKS**. The
subprocess oracle was not lying: at the one shape that test uses, `[2, 3]`
with an `int64` index, the shim and upstream really do agree. §6's BREAKS
comes from a different probe. **One shape cannot speak for a column**, which
is §7.9's `clamp` in miniature and was rediscovered here rather than
remembered.

The fix is not a bigger oracle. It is that every dtype a note advertises is
now also required to be graded `AGREES` in §6's CPU column for that operator,
so the note has the whole matrix sweep behind it instead of one tensor. With
that assertion in place both forms of N-3 go RED, and it was confirmed by
running them separately — `linalg_vector_norm` at `int64` (which upstream
itself refuses) and `scatter_.src` at `float32` (which §6 grades BREAKS).

**Which tests survive which mutant, including the ones that survive both.**
`test_a_readback_behind_two_un_named_hops_is_still_derived` dies to N-1 and
survives N-2, correctly: it is a claim about the derivation, and N-2 does not
touch it. `test_the_matrix_grades_every_mps_cell_of_the_ten_as_refuses` reads
the document and the build's note table, and no counter, so it survives
**both** N-1 and N-2 — it is a consistency check between written things and
is labelled as one rather than counted as proof of placement. It is the only
test that catches N-3. `test_no_newly_refused_op_answers_any_mps_dtype`
survives N-1 (the list is still right) and dies to N-2.

**One weakness carried forward rather than rediscovered.** §7.14 recorded that
`host_downloads == 0` survives gutting the download counter, because a counter
that never moves satisfies `== 0`. `test_eight_operators_answer_an_mps_dispatch_from_the_host`
now asserts exactly that zero, so it inherits the weakness: its `== 0` half is
meaningful only because `test_metalcount.py::test_a_readback_costs_exactly_the_tensors_bytes`
runs in the same gate. Its other two halves — the refusal fires, and it is the
readback gate's wording — do not depend on any counter.

**A cell passing is not an operator being right** (§7.9's `clamp`). None of the
above says these operators are correct on the CPU at every shape; it says the
dtypes each note advertises agree with upstream at the shape measured, and the
shape is `[2, 3]`.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs MPS_HOST_READBACK_NOTES present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs host_readback_note present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py _ops_that_reach_the_host present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py _host_reaching_functions present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py _callees present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsrefuse.py test_the_ten_newly_refused_ops_are_refused_and_the_message_says_what_works present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsrefuse.py test_no_newly_refused_op_answers_any_mps_dtype present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsrefuse.py test_every_dtype_a_refusal_note_advertises_agrees_on_the_cpu present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsrefuse.py test_the_derivation_finds_the_ten_without_being_told_their_names present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsrefuse.py test_a_readback_behind_two_un_named_hops_is_still_derived present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsrefuse.py test_the_matrix_grades_every_mps_cell_of_the_ten_as_refuses present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs norm_pow_walk present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs order_along present -->
<!-- DOCWATCH: op-implemented aten.sort.default -->
<!-- DOCWATCH: op-implemented aten.topk.default -->

### 7.17 `DType::I8` on Metal — vendored, and 137 of the 284 cells now agree

§7.13 declined this and left the decision with the user; §7.15 read the
instantiation so the decision could be made against a measured shape. This
section is the round that made it. **§7.13 is not corrected below — it was
right about what a `candle-core`-only change would buy.** What changed is that
the second crate was approved, so the other half could be built.

#### The survey was re-checked first, because it is why this was approved

§7.15 was read-only and this round depended on it, so it was verified against
`candle-metal-kernels-0.11.0` as published — the crate whose sha256
`242e83c6…3d3e` is the `checksum` `rust/torch_c/Cargo.lock` already carried,
i.e. cargo's own pin and not one chosen here.

| §7.15 said | measured |
|---|---|
| `candle_metal_kernels::DType` has six variants | **holds** — `BF16 F16 F32 I64 U32 U8`, `src/lib.rs` |
| `init_binary` / `init_boolean_binary` are per-dtype macro invocation lists | **holds** |
| `init_cast_all` is one macro plus one top-level call | **holds** |
| `INDEX_OP` ×16, `INDEX_ADD_OP` ×18, `GATHER_OP` ×16, `SCATTER_OP` ×10, `SCATTER_ADD_OP` ×10, `WHERE_OP` ×18, `ARGSORT` ×6 | **holds, all seven counts exactly** |
| `init_unary_float` is float-only; integer unary is `copy` only | **holds** — so no `int8` `sin`/`exp`/`sqrt`, as upstream also refuses |
| `quantized`/`mlx_gemm`/`gemv`/`sdpa` are float-only | **holds** |
| "roughly 30 added lines and zero new shader bodies" | **zero new shader bodies holds.** The line count was low: **43 MSL lines** (40 instantiations + 3 comment), because each macro exists twice behind `#if defined(__HAVE_BFLOAT__)` and both copies need the line |

Three things the survey did not name, found only by compiling:

* **`impl EncoderParam for i8`** (`src/utils.rs`). Without it `const_set` does
  not typecheck, so `zero_`, `fill_` and every factory stay refused.
* `copy2d::I8` and `sort.rs`'s two `DType` match tables, which are Rust
  name/size tables rather than MSL.
* `mlx_sort.metal`'s block-sort instantiations, which are a third list beside
  `sort.metal`'s `ARGSORT`.

Total: **51 added lines across 12 files**, none of them a shader body. The
survey's *shape* was right and its *size* was 40% low; it was right about the
thing the decision turned on.

#### The result, in the three grades, and the 284 is not the number

    builds     both crates compile; the MSL library loads on the device.
    reaches    0 of the 284 cells still fail at stage `operands`.
    AGREES     137.

**147 do not agree, and 99 of them never could.** §4.3a counted 284 cells at
stage `operands` and read that as the size of the prize. It is not: a cell can
only reach AGREES on `mps` if `int8` agrees on the **CPU** too, and 99 of the
284 have `int8_cpu` at BREAKS, REFUSES or REACHES. The real ceiling was
**185**, and 137 of those 185 were taken. This is the same error §4.3a itself
warned about one paragraph later — a number that names operators it has never
asked the question of — repeated by the document that raised it.

The 48 of the ceiling that remain, and the 99 that were never in it, cluster
into four causes and **none of them is an `int8` shader**:

| cause | cells | what it is |
|---|---|---|
| **A** | 98 | the `mps` host-readback gate (§7.16 and MPSATTN.md §3.1), which refuses these operators on **every** dtype. `int8` changed nothing here and was never going to |
| **B** | 38 | **upstream itself refuses `int8`** — `"round_cpu" not implemented for 'Char'`, `mean(): input must be floating point`, `masked_fill only supports boolean masks`, `where expected condition to be a boolean tensor`. There is no oracle, so there is nothing to agree with |
| **C** | 10 | candle has no `I8` kernel for that specific op: `mlx matmul doesn't support I8` (`mm`, `bmm`, `matmul`, `addmm`, `baddbmm`), `reduce op Max I8`, `unary usign I8`, and three `I8 <-> F64` casts which §3.1 refuses on Metal by name anyway |
| **D** | 1 | `convolution`, refused by this shim as floating-point only |

**C is the only one a further `candle-metal-kernels` change would move**, and
it is 10 cells across 7 operators — the smallest of the four, which is the
opposite of what §4.3a's headline implied.

#### What was vendored, and on what terms

`vendor/candle-metal-kernels/`, committed, on **exactly** the terms
`vendor/candle-core/` is on: the sha256-pinned published crate plus
`vendor/int8-candle-metal-kernels-0.11.0.patch`, regenerated by
`sh vendor/vendor_candle.sh` and verified byte for byte by `--check` in the
gate. `vendor_candle.sh` was extended to loop over both crates rather than
gaining a sibling script, and `test_int8.py` now runs the same four vendoring
tests against each: patch path relative and pointing at the tree `--check`
verifies, lock resolving from the fork and not the registry, a drifted tree
refused, a wrong crate refused. **`--check` reporting one tree instead of two
is itself a failure**, asserted, because that is how a second crate silently
stops being covered.

The existing patch is still named `int8-…-cpu.patch` although it now carries
Metal counters and Metal `I8`. Renaming it reaches `vendor_candle.sh`,
docs/numerics/INT8.md §1.2 and `test_int8.py`; it is **left alone**, and the
script says so where a reader meets it.

#### Nullification

Five mutants, each built through `vendor/install_shim.sh` so it reached the
**vendored** tree, run alone, and removed afterwards. `--check` was re-run
after the last one and both trees are byte for byte their patches again.

| mutant | what it breaks | result |
|---|---|---|
| M-1: `init_binary_k(bop, bop, i8, int8_t, int8_t)` deleted from both `init_binary` branches | the arithmetic binaries | RED — `Error while loading function: badd_i8` |
| M-2: the top-level `init_cast_all(i8, int8_t);` deleted | every cast *off* `i8` | RED in two tests — `cast_i8_f32`, and `cast_i8_i64` inside the comparisons |
| M-3: the three `WHERE_OP(int8_t, …)` lines deleted | the ternary family | RED — `where_u8_i8 was not found in the library` |
| M-4: `CpuStorage::I8` returned to the refusing arm in `candle-core` | §4.3a cause A, restored | RED in four tests — `unsupported dtype I8 for op to_dtype`, the exact pre-change message |
| M-5: **a host twin** — `to_dtype` for `I8` downloads, converts on the CPU and uploads | nothing a value can see | RED **only on the counters** — `read 2 tensor(s) back to the host (16 bytes)`. Every element was still correct and `.device` still said `mps` |

**M-5 is the one that matters**, and it is the experiment CLAUDE.md §2 records
as never having been run on Metal. It has now been: a host-computed twin of
the `int8` cast produced **correct values under an `mps` label**, and the
agreement assertions did not notice. `host_downloads == 0` did.

**Which tests survive which mutants.**
`test_the_inputs_actually_wrap` survives all five — it is a property of the
input data and touches no device, which is what it is for.
`test_int8_builds_on_mps_and_makes_the_round_trip` survives M-1, M-2, M-3 and
M-5 and dies only to M-4; it makes no counter claim, correctly, because a
buffer build legitimately uploads.
`test_int8_casts_agree_with_upstream_on_mps` is the only one that dies to both
M-2 and M-5.

#### What this round did not do, stated rather than left to be inferred

* **No `int8` reduction, matmul or `sign` on Metal** (cause C, 10 cells). Each
  is a real instantiation question and none was attempted.
* **`maximum` and `minimum` are not in `test_i8mps.py`**, and not for a dtype
  reason: they are on `_shim_mps_host_readback_ops()` for every dtype, so
  including them would fail this file for something it is not about.
* **The sweep is one shape per operator** (§5), so a cell graded AGREES here
  is graded at that shape. `test_i8mps.py` exercises inputs the sweep's cell
  does not contain — both `int8` endpoints in **both** operands, and inputs
  where `add`, `sub` and `mul` each wrap at least twice — because §7.9's
  `clamp` and §7.16's N-3 both passed a green suite on a cell that lacked the
  value that mattered.
* **Only the `int8_mps` column was re-swept**, for the 284 operators. The other
  fifteen columns moved too between `b4f89f3` and this round, but those moves
  are §7.16's refusals and other develop traffic, not this change, and are not
  attributed to it here.

<!-- DOCWATCH: symbol-in-file vendor/vendor_candle.sh candle-metal-kernels present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/Cargo.toml candle-metal-kernels present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-metal-kernels/src/lib.rs I8 present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-metal-kernels/src/metal_src/binary.metal "init_binary_k(bop, bop, i8, int8_t, int8_t)" present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-metal-kernels/src/metal_src/cast.metal "init_cast_all(i8, int8_t)" present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-metal-kernels/src/metal_src/ternary.metal where_u8_i8 present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-metal-kernels/src/utils.rs primitive!(i8) present -->
<!-- DOCWATCH: symbol-in-file vendor/candle-core/src/metal_backend/mod.rs cast_i8_f32 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_i8mps.py test_int8_builds_on_mps_and_makes_the_round_trip present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_i8mps.py test_int8_binary_operators_agree_with_upstream_on_mps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_i8mps.py test_int8_casts_agree_with_upstream_on_mps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_i8mps.py test_int8_where_agrees_with_upstream_on_mps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_i8mps.py test_the_inputs_actually_wrap present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_int8.py test_the_committed_metal_kernels_fork_is_the_pinned_crate_plus_the_patch present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_int8.py test_the_metal_kernels_check_refuses_a_tree_that_drifted_from_the_patch present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_int8.py test_the_metal_kernels_script_refuses_a_crate_that_is_not_the_pinned_one present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_int8.py test_the_lock_resolves_metal_kernels_from_the_fork_not_the_registry present -->

---

### 7.18 The constant gate, the half §4.3b did not reach — carried from `work/cmlsilence`

**Provenance, first.** Two rounds fixed the same defect in parallel. §4.3b
(cause D) landed `host_full` and closed the **float** dtypes of nine factories.
`work/cmlsilence`, on a base nine commits older, had independently routed the
same call sites through a helper of its own (`host_filled`) and gone further.
It was never committed; the integration round (`work/integA`, 2026-10-02)
carried it onto develop's structure. **Develop's `host_full` was kept and the
branch's `host_filled` dropped** — they are the same function, and `host_full`
is the one the gate had already measured. What was carried is only what
`host_full`'s round did not cover. The §4.3a correction about cause B above is
that branch's measurement and was carried, not re-measured.

**The shape, once more.** A kernel that needs a scalar operand wrote
`Tensor::full(v, shape, device).and_then(|t| t.fast_to(storage))`, and the
narrowing on the second half never ran on Metal, because the first half asked
the device for a dtype it does not have. Measured on this machine, against the
tree as develop left it, **three operators still died that way on every Metal
float dtype**, behind the seven §4.3b closed:

| operator | message on develop `78cf555` | after |
|---|---|---|
| `aten.fill_.Tensor` | `Metal contiguous to_dtype F32 F64 not implemented` | agrees, `float32`/`float16`/`bfloat16`/`int64` |
| `aten.normal_.default` | `Metal contiguous to_dtype F64 F32 not implemented` | agrees: same draws as `cpu` under the same seed |
| `aten.bernoulli_.float` | `Metal contiguous to_dtype F64 F32 not implemented` | agrees: same draws as `cpu` under the same seed |

and one cell §4.3c named: `aten.scalar_tensor.default` at `int32_mps`
(`Metal contiguous to_dtype I64 I32`) now agrees.

**What was carried, by kind.**

* `host_vec` — `normal_` and `bernoulli_.float` built their draws as
  `Tensor::from_vec(values_f64, shape, &metal)` and narrowed on the device.
  The draws are this crate's own RNG output, so narrowing them on the host is
  not a fallback; it is the only place that conversion exists.
* `widen_f64_host` in `scalar_arg` — a zero-dim **tensor** passed where a
  `Scalar` is taken was widened to `f64` *before* it was moved to the host.
  That is wider than the operator that exposed it: `mul.Scalar`, `add.Scalar`
  and `masked_fill.Scalar` handed a zero-dim `mps` tensor died the same way, and
  no sweep passes a tensor where a scalar is allowed.
* `fill_inplace`'s device path — once `fill_.Tensor` reached on Metal,
  forming the `Scalar` meant downloading the value tensor and uploading a
  constant built from it, a host readback of a dispatched tensor that
  `test_metalplace.py`'s in-place scan refuses. Where the value is already on
  an accelerator the fill is built from it *there*. The CPU path is untouched.
* **The integer arms** of `full`, `filled_block`, `scalar_tensor`,
  `masked_fill`, `fill_`, `where.Scalar`, `where.ScalarSelf`, `remainder`,
  `fmod` and `div`'s rounding modes go through `host_full` as their float arms
  already did. `host_full`'s `Cpu` arm is the old two calls, so the host answer
  is unchanged by construction.
* **Five call sites passed `host_full` no narrowing step** and narrowed after
  the move, on the device: `extremum_default`'s NaN seed, `nan_shaped_like`,
  and the scalar operands of `remainder_op`, `fmod_op` and `div_mode`. They
  pass their storage dtype now. On `mps` all five sit behind operators the
  host-readback gate refuses (`max`, `max.dim`, `remainder.Scalar`,
  `fmod.Scalar`, `floor_divide.Scalar`, `div.Scalar_mode` — measured), so this
  changes no cell today; it is what lets the next item exist.
* **`host_const` refuses an `f64` constant on Metal**, by name. Everything it
  does happens on the host, where `F64` always works, so a call site with no
  narrowing step got an `F64` buffer *allocated on Metal* — the capability
  claim by construction that `metal_dtype_gate` refuses everywhere else. The
  five sites above were exactly that, unreachable only by luck.

**What did not move, named.** `full.default` at `int32_mps` still breaks, on a
*different* candle symbol: the narrowed element now reaches the device, but
filling a shape from it is a strided copy and candle's Metal backend has no
`I32` one (`Metal copy_strided I32 not implemented`).
`test_the_integer_arm_narrows_on_the_host_too` pins it and goes red when it
falls. Every other `int32_mps` cell of these operators is refused by name at
the door, as before. `eye` and `linspace` (§4.3c "Still open") are untouched.
§6's table is **not** re-measured here; these cells are held by tests until the
next sweep moves them.

#### Nullification (integration round, each mutant built through `vendor/install_shim.sh`)

| mutant | result |
|---|---|
| M-1: `host_const`'s `f64`-on-Metal refusal replaced by `if false &&` | **Python suites all GREEN**; `cargo test` RED (`42 passed; 1 failed`) — `mod host_const_tests` is the only thing that kills it, as the branch reported |
| M-2: `host_vec` builds on the device and narrows there | RED — `test_the_rng_writers_...`: `normal_.default: Metal contiguous to_dtype F64 F32` |
| M-3: `scalar_arg` back to `widen_f64` | RED — `test_a_zero_dim_device_tensor_...`: `mul.Scalar: Metal contiguous to_dtype F32 F64` |
| M-4: `fill_inplace`'s device path disabled | RED in two files — `test_mpsconst`'s counter test (`read 4 byte(s) back`) and `test_metalplace`'s in-place readback scan |
| M-5: `scalar_tensor`'s integer arm back to `Tensor::full(i64, .., device)` | RED — `test_the_integer_arm_...`: `Metal contiguous to_dtype I64 I32` |
| M-6: `extremum_default`'s NaN seed back to `&[]` plus a device narrowing | **GREEN everywhere.** Not killable on this machine: the site is behind `max`, which `mps` refuses before it is reached. Recorded rather than papered over; M-1's guard is what would catch it if the refusal were ever lifted |

**One correction to the branch's own ledger.** `test_mpsconst.py` labelled
`aten.rs` 14791/14922 as exercised by its `mul.Scalar`/`div.Scalar` cases.
Those lines are `remainder_op` and `fmod_op`; the cases reach `arith_scalar`.
They are listed as unexercised now, with the measured refusal as the reason.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs host_vec present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs widen_f64_host present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs host_const_tests present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsconst.py test_constant_gate_agrees_with_upstream_on_mps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsconst.py test_the_constant_gate_operators_compute_on_the_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsconst.py test_the_rng_writers_narrow_on_the_host_and_draw_the_same_stream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsconst.py test_float64_on_mps_is_still_refused_in_upstreams_words present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsconst.py test_the_sites_this_round_converted_are_accounted_for present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsconst.py test_a_zero_dim_device_tensor_can_stand_in_for_a_scalar present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsconst.py test_the_integer_arm_narrows_on_the_host_too present -->
