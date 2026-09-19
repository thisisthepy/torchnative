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
CPU-only-integer story as A. So B is two causes, not 45 operators, and the
larger one is a gate placement question rather than a kernel.

**C is the family §4.3 originally named** — `mm`, `bmm`, `matmul`, `addmm`,
`baddbmm`, `convolution` — and it is the smallest real one: **6 operators.**
**D is the factories**, 9 of them. **F is one cell.**

    Work actually named here: 1 int8-on-Metal gate (A, and half of B's tail),
    1 float64-cast gate placement (B, 117 cells), 6 matmul ops (C),
    9 factories (D), 1 cell (F).

**What this clustering cannot see.** It groups by the *text* of the message, so
two different causes that happen to raise the same candle sentence are merged,
and one cause whose wording differs between call sites is split. It also
inherits every limit §5 states about the sweep — one shape per operator, and a
cell that refuses for the first reason it meets hides any second reason behind
it. Cause A hides the most: 284 operators have never been asked the question at
all on that cell.

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
| aten.argsort.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.argsort.stable | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
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
| aten.floor_divide.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.floor_divide.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
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
| aten.linalg_vector_norm.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
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
| aten.norm.ScalarOpt_dim | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
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
| aten.scatter_.src | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.scatter_.value | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
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
| aten.sort.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
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
| aten.topk.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REACHES |
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

**There is no Metal dispatch counter in this build.** `device.rs` has
`_cuda_counters()` and `_vulkan_counters()`; Metal has neither, and candle's
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
write does not add a readback, and `..._fill_..._different_gate` is about a
blocker upstream of the write door that row one cannot reach.

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
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsinplace.py test_fill_on_mps_is_blocked_by_a_different_gate_than_this_one present -->
<!-- DOCWATCH: op-implemented aten.zero_.default -->
<!-- DOCWATCH: op-implemented aten.add_.Tensor -->
<!-- DOCWATCH: op-implemented aten.copy_.default -->
<!-- DOCWATCH: op-implemented aten.clamp_min_.default -->
