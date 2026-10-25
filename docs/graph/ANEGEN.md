# ANEGEN: `from_pretrained` -> `to(device.npu)` -> `generate` on the Apple Neural Engine

[`ANEDECODE3.md`](ANEDECODE3.md) built a two-layer decoder program that
`MLComputePlan` places on the Neural Engine, but nothing called it outside a
test: `to(device.npu)` still swapped `Linear` leaves one at a time, and at
batch 1 those are CPU-preferred, so a decode step reached the unit on nothing
([`NPU2.md`](NPU2.md) §8.2). This round connects the program to the path a
caller takes, measured on this Mac (M1, macOS 26, coremltools 9.0).

```python
import torch, torchnative
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
m = m.to(torchnative.device.npu)
ids = tok("The capital of France is", return_tensors="pt").input_ids
print(tok.decode(m.generate(ids, max_new_tokens=20, do_sample=False)[0]))
```

## 1. What `to(device.npu)` now does to a Llama-shaped model

At float16 (the default), `export.coreml._compile_model` first looks for a
`ModuleList` named `...layers` whose every element is a Llama-shaped decoder
layer (RMSNorm, q/k/v/o and gate/up/down `Linear` without bias, SiLU). Each run
of layers is grouped by weight count (`_group_decoder_layers`, 4.7M threshold),
and each group becomes one CoreML program that replaces the group's first layer;
the others become pass-throughs. SmolLM2-135M's 30 layers become 15 programs.
Anything else (a bias, another norm, float32) keeps the leaf path, and the
reason is in `report["decoder_refused"]`.

The KV rows stay in the `transformers` Cache. Each step reads the cached rows,
the program concatenates the new row inside (ANEDECODE2 §3), and the new row is
written back with `Cache.update`, so `generate` computes positions from a cache
that grows exactly as it would upstream. A `RangeDim` cannot be empty, so every
step prepends one zero row and a new `mask` input hides it. A prompt is fed one
position at a time through the same decode program. Batch 1 only; batch > 1
refuses by name.

## 2. The defect: RMSNorm overflowed at float16

The first end-to-end run produced `' , and in the, the, and, and, ...'`.
SmolLM2-135M's residual stream reaches **19279** at the first position (measured
per layer in upstream float32), and the program squared it before averaging:
19279² is far above float16's 65504, so the variance was inf. `_rms` now divides
by `max(max|x| / 16, 1)` before squaring the variance, divides eps by that
divisor's square, and multiplies the *unscaled* x by `rsqrt(...) / divisor`
(scaling x itself pushed a sink-only prompt's small components into float16
subnormals). The divisor is exactly 1 for ordinary activations, so they take
the unscaled arithmetic bit for bit. A fixed 1/256 was tried first
and failed `test_anetracer`'s float16 grade (1.6e-3 > 1.5e-3) through underflow;
the grade was not widened.

### 2.1 Between groups the hidden state stays float32

For a bfloat16 checkpoint (`dtype="auto"`), rounding the hidden state back to
bfloat16 at every group boundary added to the programs' own float16 error: on
`[1, 2, 3, 4, 5]` the lowered model was 11.16 from the float32 forward, the
checkpoint's own bfloat16 forward 10.28. Groups now hand float32 to each other
and the last group returns the dtype the first received (10.22, under 10.28).

## 3. What was measured

| claim | grade | evidence |
|---|---|---|
| every decoder layer is in a group, every group's `MLComputePlan` is `preferred: NeuralEngine` on every op | placement | `test_every_decoder_layer_is_in_a_neural_engine_group` |
| greedy tokens equal upstream float32's | **agrees** 20/20 | `test_generate_follows_upstream_greedy_until_a_near_tie` |
| the Neural Engine took work off the CPU | indirect | §4 |

The test accepts a divergence only after at least 8 agreeing tokens and only at
a near tie (the token chosen is upstream's runner-up within 0.5 logits). With
the fixed 1/256 divisor the run diverged at step 13, a 0.266-logit tie; with
`_rms` as landed it agrees on all 20.

**Nullified.** With the overflow fix disabled (`_RMS_HEADROOM = 1e9`), the
generate test fails at step 0 (`ours [1217] upstream [260]`); the placement test
stays green, which is why it is not the only check.

## 4. Neural Engine evidence other than `MLComputePlan`

`tests/bench/ane_generate_units.py` builds the same lowered model with
`compute_units=CPU_ONLY` and with `CPU_AND_NE` and decodes 32 tokens in fresh
subprocesses, alternating. Measured 2026-10-26, load 1.8-2.4, before §2.1's change:

| units | wall ms/token | process CPU ms/token |
|---|---|---|
| CPU_ONLY | 30.64, 31.23, 31.29 | 38.07, 38.53, 38.61 |
| CPU_AND_NE | 32.13, 31.49, 31.46 | 31.36, 30.85, 30.82 |

Process CPU time drops by about 7.5 ms/token (19%) in all three rounds while
wall time does not move, and all six runs produce the same tokens. That is
consistent with the group programs running off the CPU cores, but it is not a
direct reading: wall time is dominated by the per-step Python and tensor round
trip, and a CPU-time drop could also come from a different CPU kernel choice.
The direct reading needs root:

```sh
sudo powermetrics --samplers ane_power,cpu_power -i 500 -n 20
```

run while a `CPU_AND_NE` round decodes; a non-zero ANE power during the round,
and zero with `CPU_ONLY`, is the measurement.

## 5. Not established

| # | not established | why |
|---|---|---|
| 1 | ANE power during decode | needs sudo (§4) |
| 2 | any speed-up over the shim's CPU path | wall time is round-trip bound; not measured against `device.cpu` |
| 3 | models other than SmolLM2-135M | only this one was run; other Llama-shaped models take the same path |
| 4 | prompts longer than the program's `max_position_embeddings` | `RangeDim` upper bound |
| 5 | `lm_head` and the embedding on the unit | `lm_head` takes the leaf path; the embedding stays on the CPU |

## 6. Split the way AGENTS.md §17.3 asks

| | |
|---|---|
| **features added** | decoder-group lowering in `to(device.npu)` (`_lower_decoder_stacks`, `_decoder_group`); an optional `mask` input on `_build_decoder_subgraph_program` |
| **defects fixed** | RMSNorm overflow at float16 inside the decoder program |
| **tests added** | 2, in `tests/devices/coreml/test_anegenerate.py` |
| **tests changed** | 6 in `test_bf16ane.py`: four asserted the 211-leaf structure (now 15 groups + `lm_head`); the logit bar is now upstream's own bfloat16 error against float32 (AGENTS.md §16), because on `[1, 2, 3, 4, 5]` (every position an attention sink) the bfloat16 checkpoint is 9.66-10.28 from float32 itself; the two argmax assertions moved to a real sentence, because on that input the first-place token is precision noise (float32 7910, upstream bfloat16 407, shim bfloat16 9612) |
| **docs corrected** | 1, this file |
| **removed** | 0 |
