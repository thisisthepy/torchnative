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

> **Answer first.**
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

### 4.1 `aten.view.dtype` answers on the CPU under an `mps` label

The one silent fallback in this sweep, and it is a real one:

```
input.device=mps:0  ->  aten.view.dtype  ->  result.device=cpu     (shim)
input.device=mps:0  ->  Tensor.view      ->  result.device=mps:0   (upstream)
```

Measured on all four dtype pairs tried (`float32->int32`, `int64->float64`,
`int32->float32`, `bool->uint8`). The result is correct and the device is wrong,
which is the quiet form of the failure docs/graph/NPU2.md is about — and because
the *output* is a cpu tensor, everything downstream of it silently leaves the
device too. It is recorded BREAKS in the table below.

**It is not fixed here.** It is outside the three changes this round was asked
to port, and a correct fix is a kernel change with its own test. It is reported
rather than quietly carried.

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

Measured 2026-09-16 by `rust/torch_c/pytests/dtype_device_matrix.py --drive`
against the artefact built from this tree. Reproduce with:

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
| aten.abs.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.abs_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.acos.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REACHES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.adaptive_avg_pool1d.default | AGREES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.adaptive_avg_pool2d.default | AGREES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.add.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.add.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.add_.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
| aten.add_.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
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
| aten.bernoulli_.float | AGREES | REACHES | REACHES | REACHES | REACHES | REACHES | REACHES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
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
| aten.ceil_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | REFUSES |
| aten.chunk.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.clamp.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.clamp_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | BREAKS |
| aten.clamp_min.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.clamp_min_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.clip.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.clone.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.col2im.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.constant_pad_nd.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.convolution.default | AGREES | BREAKS | REFUSES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.copy_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | REFUSES |
| aten.cos.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.cos_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.cumprod.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.cumsum.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.detach.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.diag.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.diff.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.div.Scalar_mode | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.div.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.div.Tensor_mode | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.div_.Scalar | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | REFUSES |
| aten.div_.Tensor | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.embedding.default | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.empty.memory_format | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | BREAKS | AGREES |
| aten.empty_like.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.empty_strided.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | BREAKS | AGREES |
| aten.eq.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.eq.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.equal.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.erf.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.erf_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.erfinv.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.exp.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.exp_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.expand.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.expm1.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.expm1_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.eye.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.eye.m | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.fill_.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | BREAKS | REFUSES | REFUSES | BREAKS |
| aten.fill_.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.flip.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.floor.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.floor_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | REFUSES |
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
| aten.linalg_qr.default | AGREES | REFUSES | REFUSES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.linalg_vector_norm.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.linspace.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.log.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.log2.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.log2_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.log_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.logical_and.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.logical_not.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.logsumexp.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.lstm.input | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.lt.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.lt.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.masked_fill.Scalar | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.masked_fill_.Scalar | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
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
| aten.mul_.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
| aten.mul_.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | BREAKS |
| aten.multinomial.default | REACHES | REACHES | REACHES | REACHES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.multiply.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.multiply.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.native_batch_norm.default | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS |
| aten.native_dropout.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.native_group_norm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.native_layer_norm.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.ne.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.ne.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.neg.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.neg_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
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
| aten.reciprocal_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.reflection_pad1d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.reflection_pad2d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.reflection_pad3d.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.relu.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.relu_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
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
| aten.round_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | AGREES | AGREES | REFUSES | REFUSES |
| aten.rsqrt.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.rsqrt_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.rsub.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | REFUSES |
| aten.scalar_tensor.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.scatter.src | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.scatter.value | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.scatter_.src | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | BREAKS |
| aten.scatter_.value | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | BREAKS |
| aten.scatter_reduce.two | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | AGREES | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.select.int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.sigmoid.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.sigmoid_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.sign.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
| aten.silu.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.sin.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.sin_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.sinc.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.slice.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.softplus.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.sort.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.split.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.split_with_sizes.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.sqrt.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.sqrt_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
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
| aten.sub_.Scalar | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
| aten.sub_.Tensor | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | REFUSES | REFUSES | REFUSES |
| aten.sum.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.sum.dim_IntList | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.t.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.t_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.tanh.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.tanh_.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.topk.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | AGREES | REFUSES | REFUSES | REACHES |
| aten.transpose.int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.tril.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.triu.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | AGREES |
| aten.unbind.int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.unflatten.int | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.unfold.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | REFUSES | REFUSES | BREAKS |
| aten.uniform_.default | REACHES | REACHES | REACHES | REACHES | REFUSES | REFUSES | REFUSES | REFUSES | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.unsqueeze.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.upsample_bicubic2d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.upsample_bilinear2d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.upsample_linear1d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.upsample_nearest1d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.upsample_nearest2d.default | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var.correction | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var.default | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.var.dim | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | BREAKS | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.view.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.view.dtype | AGREES | BREAKS | BREAKS | AGREES | AGREES | AGREES | REACHES | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
| aten.view_as.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | AGREES | AGREES | REFUSES | AGREES |
| aten.where.Scalar | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.where.ScalarOther | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.where.ScalarSelf | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | REFUSES |
| aten.where.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES | REFUSES |
| aten.where.self | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | BREAKS | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | AGREES |
| aten.zero_.default | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | AGREES | BREAKS | BREAKS | BREAKS | REFUSES | BREAKS | BREAKS | REFUSES | BREAKS |
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
