# NPUFUSE: a fused gated MLP and a dynamic row axis for the Intel NPU path

GitHub issue #3. Measured 2026-10-02 on an arm64 Mac (no Intel NPU) against
develop `59dc043`. Rules cited are `AGENTS.md`'s.

**One-paragraph summary.** `model.to(torchnative.device.npu)` on an Intel NPU
host now lowers each gated MLP, `down_proj(silu(gate_proj(x)) * up_proj(x))`,
as **one** OpenVINO graph instead of three `Linear` leaves with the SiLU and
the multiply done on the host. It also compiles every lowered module with a
**dynamic row axis** (`-1`), so one compiled model serves every prompt length
and `generate()` compiles nothing after `to()`. Both live on the private path
`to()` already used (`intelnpu._compile_model` → `_NPULinear`, plus a new
`_NPUGatedMLP`). No withdrawn public name came back. **None of it has run on
an NPU.** On this Mac it reaches IR acceptance, a compile count and
element-wise agreement on **OpenVINO's CPU plugin**. The NPU half is §5, which
the user runs on the Windows laptop.

## 0. What each claim reaches

The grades are `AGENTS.md` §16's: *builds* / *reaches* / *agrees*.

| Claim | Grade reached here | Where measured | In the gate? |
|---|---|---|---|
| `mlp_ir` is one graph: 3 MatMul, 1 Swish, 1 Multiply, SiLU on the **gate** | builds (IR text, counted) | `test_npufuse.py`, pure | yes |
| `-1` on every activation port and on no weight | builds | pure | yes |
| A real `LlamaMLP` lowers to ONE compiled OpenVINO model | **reaches**: one `ov_core_compile_model`, counted at two levels | real OpenVINO 2026.3.1, **CPU** plugin | no (skips by name) |
| One compiled model serves row counts 1, 3, 7, 12 and 5 | **reaches** (counters flat) | real runtime, CPU plugin | no |
| `generate()` at prompt lengths 3/5/9 compiles 0 models after lowering | **reaches** (counters flat) | real runtime, CPU plugin; also faked through `to(device.npu)` | faked half: yes |
| Fused MLP output = upstream torch's | **agrees** at f32 *execution* precision (ratio 0.92). At the plugin's default f16 execution it is within the derived tolerance, but **fails the 4x rule** (ratio 7.1) | real runtime, CPU plugin, upstream torch 2.13 in a separate subprocess | no |
| Unsupported patterns refuse by name | builds/reaches (pure) | six patterns, pure | yes |
| A refused dynamic axis is named, warned and visibly recompiles | reaches (faked device refusal) | faked | yes |
| Anything on an Intel **NPU** | **nothing** | §5, Windows | n/a |

The gate does not set `TORCHNATIVE_OPENVINO_C`, so every real-runtime row
**skips in the gate**. Each row says what it is missing. The real-runtime rows
were run by hand for this round, with the OpenVINO 2026.3.1 arm64 runtime kept
under the repository's `.caches` directory. §7 lists the two guarantees that
therefore *cannot go red in the gate on this machine*.

## 1. Where it lives, and what was not brought back

The archived branch `work/intelnpu2` (tag `archive/wip/bw-intelnpu2`) had both
capabilities, built on `compile_model` / `NPULinear`. Develop withdrew those
in `6415da8`, and they still refuse by name (`test_intelnpu.py`,
`test_ovpar.py`). This round **redesigned** both pieces rather than restoring
them:

* `intelnpu.mlp_ir(hidden, intermediate, batch)` is the archived IR, re-checked
  against a real runtime (§4).
* `intelnpu._NPUGatedMLP` subclasses the private `_NPULinear`. It overrides
  only the IR, the weights blob and construction. The shared core, the
  `EXECUTION_DEVICES` verdict, the counter and the byte crossing are inherited.
* `_compile_model` checks each module for the fusion **before** recursing into
  it. If it recursed first, it would lower the three projections as leaves,
  and the report would look *better* for it (more modules swapped).
* `plan_lowering` uses the same decision function (`_fusion_decision`). The
  plan and the lowering cannot disagree, and a test checks that they agree.
* The public surface grew by `mlp_ir` and `DYNAMIC_ROWS` (in `__all__`), and by
  two entries in `supported_ops()` / `SUPPORTED_MODULES`.

## 2. The fused gated MLP

**Why a subtree.** Lowered as three leaves, a `LlamaMLP` costs three compiled
models, three FFI crossings, and two host round trips for the SiLU and the
multiply. Lowered as `mlp_ir` it costs one model and one crossing. The
intermediates never leave the device. `docs/graph/QUANT2.md` §3 names the
limit: subtree replacement cannot see between modules, so the residual adds
and the norms stay on the CPU, and the report says so.

**The matcher checks behaviour as well as structure** (`_gated_mlp_refusal`).
Names and shapes are not enough. Gemma's MLP has the same three children with
a GELU. A module that applies the SiLU to `up` instead of `gate` has the same
names, shapes and op counts. Lowering either one would return confident wrong
numbers. So after the structural checks, the module runs once on a fixed
probe input, which draws no random numbers and so leaves the caller's RNG
alone. Its output is compared with `down(silu(gate(x)) * up(x))`, built from
the module's own three Linears. The comparison only counts if it **could have
failed**: the reference has to differ from the two nearest wrong functions
(SiLU on `up`, and no activation) by far more than the tolerance. If it does
not, the module is refused for that reason instead (`AGENTS.md` §17.5).

Each refusal names what is missing. They are pinned by
`test_each_unsupported_pattern_refuses_by_name_and_says_what_is_missing`:

| Pattern | Reason begins |
|---|---|
| GELU activation (`hidden_act="gelu_pytorch_tanh"`) | `it does not compute down(silu(gate(x)) * up(x))` |
| gate and up swapped inside `forward` | same |
| `mlp_bias=True` | `` `gate_proj` has a bias, and the fused IR emits none`` |
| `gate_proj` not a `torch.nn.Linear` | `` `gate_proj` is Identity, not torch.nn.Linear`` |
| `up_proj` shaped differently from `gate_proj` | `` `up_proj` has weight shape …`` |
| a dimension above `MAX_DIM` | `intermediate=131073 exceeds MAX_DIM=131072` |

**A refused fusion still lowers the Linears.** The module is listed in
`report["unfused"]` as `(path, reason)`. The walk then descends into it, so
its Linears still take the leaf path, and `to()` warns, naming the module
(`test_an_unfusable_mlp_is_named_and_its_linears_are_still_lowered`). The
module is no less offloaded than before this round. It is three compiled
models instead of one, and the warning says why.

## 3. The dynamic row axis

The row axis is every leading dimension of the activation, flattened (batch ×
sequence for a decoder). Before this round, `_NPULinear` compiled one static
IR per row count, and `generate()` sees a new row count for every prompt
length. `linear_ir` and `mlp_ir` now accept `batch=None`. That emits `-1` in
the Parameter's `shape` and on every activation port. The weights always keep
static shapes.

**Running a dynamic model.** The request's input tensor for a `-1` model has
**zero bytes** (measured: `ov_tensor_get_byte_size` reads 0), so there is
nothing to copy into yet. `OpenVINO.infer(compiled, bytes, shape)` first sets
the call's concrete shape on the request's own tensor
(`ov_tensor_set_shape`). The tensor keeps the element type the IR declared, so
this module still never binds `ov_element_type_e` for activations.

**Choosing the mode.** The first compile of each module tries the `-1` IR. If
the device compiles it, that model serves every row count
(`shape_mode="dynamic"`). If the device **refuses** it (`ov_core_compile_model`
fails), the refusal is kept verbatim on the module. The module then compiles
one static model per row count (`"static-per-length"`, the old behaviour).
That case is listed in `report["shape_fallbacks"]` and `to()` warns that
`generate()` will recompile per prompt length. A wrong `EXECUTION_DEVICES` on
a dynamic compile is **not** a reason to fall back. Placement is not a shape
question, and retrying would repeat the same wrong placement.

**The instrument** is `intelnpu._compile_counters()`. `ov_compile` counts
successful `ov_core_compile_model` calls inside `OpenVINO.compile_ir`.
`lowered` counts compiled models kept by a lowered module. A recompiled model
gives the same answer, so only a counter can show that a recompile happened
(`AGENTS.md` §16). The faked refusal test is also the positive control: with
the axis refused, the same `generate()` moves the counter by 30, so "moved by
0" is not vacuous.

**Open: whether the NPU accepts `-1`.** The NPU plugin is known to be
stricter about dynamic dimensions than the CPU plugin. This Mac cannot find
out. §5 step 6 reads `DYNAMIC_AXIS=` off the laptop.

## 4. Measured here (OpenVINO 2026.3.1, arm64 macOS, CPU plugin)

**IR acceptance and compile count.** A `transformers` `LlamaMLP` (H=64,
I=172), wrapped in a `Sequential` and lowered by `_compile_model(...,
device="CPU")`, gave `report["fused"] == ["0"]` and one compile at each
level. Five row counts then ran through that model with both counters flat. A
two-layer `LlamaForCausalLM` (11 lowered modules: 2 fused MLPs, 8 attention
projections, `lm_head`) compiled each module once. `generate()` at prompt
lengths 3, 5 and 9 added **0** compiles, and its tokens matched the unlowered
model's.

**Agreement, and a precision finding.** The reference is upstream torch 2.13.0
in a separate subprocess. It builds the weights and the inputs and computes
the answer in f32, f16 and f64. The IR is f16, so the oracle is **upstream's
own f16 error** against its f64 answer. Tolerance = p90 of that oracle,
floored at 8 ulp of f16 (`docs/numerics/AGREE.md` §2's rule at the IR's
precision). Then the 4x-ratio rule against the f64 truth.

| H × I, inputs | Execution precision | Worst rel. error | Tolerance | Ratio to upstream's own f16 error | Verdict |
|---|---|---|---|---|---|
| 64 × 172, 5 inputs | `INFERENCE_PRECISION_HINT=f32` | 6.05e-04 | 7.81e-03 | **0.92** | agrees |
| 64 × 172, 5 inputs | plugin default (f16 on ARM) | 4.67e-03 | 7.81e-03 | **7.14** | within tolerance, **fails the 4x rule** |
| 576 × 1536 (SmolLM2-135M layer 0), 3 prompts | `f32` | 5.32e-04 | 7.81e-03 | 1.00 | agrees |
| 576 × 1536 (SmolLM2-135M layer 0), 3 prompts | plugin default (f16) | 9.39e-03 | 7.81e-03 | 23.5 | **fails both** |

The SmolLM2 rows come from running `scripts/devices/intelnpu_fuse_verify.py`
here, with the CPU plugin remapped to stand in for the NPU. That was a smoke
test of the tool's plumbing. Its oracle is the **shim's** f16 against the
shim's f64, not upstream's, because that is what the tool can compute on the
laptop.

What the table says, and what it does not:

* **The graph computes the right function.** At f32 execution the same f16
  graph agrees at ratio ≤ 1 on both sizes. A wiring fault cannot do that:
  gate/up swapped is 1.2e-01 relative on the faked test.
* **The ARM CPU plugin's default execution precision is f16 accumulation, and
  it does not meet the bar.** The error grows with the reduction length
  (ratio 7 at I=172, 23 at I=1536). That is the plugin's arithmetic, not this
  lowering's. The tolerance was **not** widened. The gate test asserts
  agreement at f32 execution and only *reports* the default-precision ratio.
* **The tolerance is floor-dominated.** Upstream's own f16 error p90 is
  ~7e-4, below the 8-ulp floor of 7.81e-3, so the 4x ratio is the rule that
  does the work. Do not read "within tolerance" alone as agreement.
* **What the NPU does is unknown.** Its execution precision is the question
  §5 step 7 answers.

**An OpenVINO cache observation, not fixed here.** With the compile cache on,
a compile with `INFERENCE_PRECISION_HINT=f32` returned the blob cached from an
earlier compile of the same IR at the default precision: identical numbers,
ratio 7.14. With the cache off it gave 0.92. So OpenVINO 2026.3.1's cache did
not separate those two compiles here. The lowering passes no properties, so it
is not affected. A future caller that does pass them should know.

## 5. On the Windows laptop with the Intel NPU (the user runs this)

Nothing in this procedure has been run. Everything it reports is new
evidence.

1. Check out this branch on the laptop, in the same checkout used for
   `docs/devices/INTELNPU.md` §4.
2. Activate the environment where `import torch` is torchnative's shim. Check:
   `python -c "import torch; print(hasattr(torch._C, '_aten_implemented'))"`
   must print `True`. If it prints `False`, stop: you are measuring upstream
   torch.
3. `uv add openvino` (or `uv add "torchnative[npu]"`), if it is not
   already installed. Then
   `python -m torchnative.export.intelnpu NPU` must end with `PROVEN:`. If it
   prints `REFUSED` or `NOT PROVEN`, stop and send that output.
4. Optional, to keep the run's compile cache out of your user cache:
   `set TORCHNATIVE_CACHE_DIR=%CD%\.scratch\npucache`.
5. Run `python scripts/devices/intelnpu_fuse_verify.py --model Qwen/Qwen3-0.6B`.
   Any Llama/Qwen/Mistral-family causal LM works. Qwen3-0.6B is small enough to
   load twice: one copy stays on the CPU as the reference.
6. Copy these lines from the output:
   * `ONE_GRAPH_PER_MLP=` (expected `yes`). If it is `no`, the `NOT FUSED`
     lines say why.
   * `DYNAMIC_AXIS=` (`accepted` or `refused`). If it is `refused`, copy the
     `FALLBACK` line too: it carries OpenVINO's own message.
   * `EXECUTION_DEVICES=` (must be `[('NPU',)]`).
   * `COMPILES_DURING_GENERATE=` (expected `0` when the axis was accepted).
7. Copy every `mlp row` line and the `MLP_AGREEMENT=` line. These are the NPU's
   execution-precision numbers, comparable to the f16 and f32 rows of the §4
   table.
8. Copy `TOKENS_EQUAL_TO_CPU=` and `VERDICT=`, and the process exit status
   (`echo %ERRORLEVEL%`).
9. Optional: run the gate's real-runtime half against the laptop's runtime:
   `set TORCHNATIVE_OPENVINO_C=<path to openvino_c.dll>`, then
   `python tests/devices/npu/test_npufuse.py`. This exercises the **CPU**
   plugin on the laptop. It is not NPU evidence, but it checks the x86-64
   build of the same path.

Until steps 6 and 7 have been run, no claim in this document is about an NPU.

## 6. A pre-existing defect this round found: variadic calls on Apple arm64

`ov_core_compile_model` is variadic, and `load_openvino_c` declared only its
`restype`. Its comment said "argtypes covers the fixed prefix only", but no
`argtypes` was set. On Apple arm64, variadic arguments are passed on the
stack. ctypes only uses the variadic convention when it knows where the fixed
arguments end, which is `len(argtypes)`. So when the compile-cache properties
landed (`2591995`), the `CACHE_DIR` key and value went into registers, and
**every real-runtime test in `test_intelnpu.py` died with SIGSEGV on this
Mac**. Measured: six of six exited -11. Nobody saw it, because the gate does
not set `TORCHNATIVE_OPENVINO_C`. Windows x64 and Linux x86-64 pass variadic
arguments like fixed ones, so the user's laptop was not affected.

Fixed by declaring the five fixed arguments. `test_intelnpu.py` with the real
runtime now gives 49 ok, 1 SKIP (the NPU-hardware probe, by name), and 0 FAIL.
`test_npufuse.py` keeps the cache **on** in a throwaway directory, so its
real-runtime tests are this defect's regression check (§7, M12).

## 7. Nullifications

Each mutant was applied to the real source and the whole `test_npufuse.py`
was run with the real runtime. The file was then restored and checked
byte-identical against a backup.

| # | Mutant | Went RED |
|---|---|---|
| M1 | SiLU edge from `up`, not `gate` | IR-structure test; agreement test |
| M2 | dynamic attempt compiles a static IR | 6 tests, faked and real |
| M3 | `_row_dim(None)` returns 1 | 7 tests |
| M4 | dynamic model not reused | faked generate; both real compile-count tests |
| M5 | fusion never chosen | 6 tests |
| M6 | behaviour probe skipped | pattern-refusal test; unfusable-MLP test |
| M7 | fallback warning removed | refused-axis test |
| M8 | fallback not marked as a device refusal | refused-axis test |
| M9 | `lowered` counter not incremented | 3 tests |
| M10 | `ov_compile` counter not incremented | **real-runtime only** |
| M11 | gate/up order swapped in the blob | faked value test; agreement test |
| M12 | `argtypes` removed (§6) | **real-runtime only** (SIGSEGV) |
| M13 | unfused reason not reported | unfusable-MLP test |
| M14 | bias check removed | pattern-refusal test |

**Two of these cannot fail in the gate on this machine.** M10 and M12 are
caught only by the real-runtime tests, and those skip in the gate. They are
guarded only when someone runs `test_npufuse.py` with
`TORCHNATIVE_OPENVINO_C` set, as §5 step 9 does.

## 8. Not done

* **Nothing on an NPU** (§5).
* **The NPU's execution precision is unknown.** If it is f16 accumulation, the
  §4 default-precision rows predict that the fused MLP will not meet the 4x
  rule there. A `precision=` option on the openvino arm of `to()` would then
  be a real choice with two measured costs, as it already is on the CoreML
  arm. That decision belongs after step 7's numbers, not before.
* **Attention is still per-leaf.** The archived library also has a
  `LlamaAttention` fast path. This round fused only the MLP.
* **The status page**: done after this branch was brought up to develop: the
  "recompiling for the accelerator" row of
  [`docs/platform/STATUS.md`](../platform/STATUS.md) now carries one sentence on
  the fused MLP and the dynamic axis, at the grade measured here (CPU plugin).
