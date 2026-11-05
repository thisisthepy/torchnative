# 0.1.0b9: release notes

**GGUF files run from their own blocks on cpu and mps, and the two paths
llama.cpp is fastest at, CPU quantised prefill and mps attention, got their
first real kernels.** Prefill at 512 tokens on an M1, measured against
llama.cpp on the same files (`tests/bench/bench_llama_cpp.py`): SmolLM2-135M
Q8_0 on mps went from 629 to 2038 tok/s, and on the CPU from 211 to 500 tok/s.
llama.cpp is still 3x ahead on mps and 3.4x ahead on the CPU for that cell.

Entries are taken from `git log --no-merges v0.1.0b8..origin/develop`. Bench
numbers were taken on a shared machine; each table row below says the load.

---

## 1. Features added

- **GGUF blocks kept in the model (S2.5, #172, #176).**
  `from_pretrained(..., gguf_file=..., keep_blocks=True)` and
  `quantization_config=TorchnativeConfig(...)` put the file's own per-layer
  blocks into `QuantizedLinear` leaves without decoding them.
  `LlamaTensorProcessor` reverses llama.cpp's q/k row permutation, on dense
  tensors and directly on block rows, so llama-architecture files (SmolLM2)
  load. `keep_blocks=True` alone keeps every block the file stores, an untied
  `lm_head` included, because that is the model llama.cpp runs from the file.
- **Fused attention on mps (S7.3, #181).** Calls with no explicit mask, causal
  only when query and key lengths agree, and an instantiated head dim run as
  one Metal kernel (candle-metal-kernels' `steel_attention` and
  `sdpa_vector`): one compute encoder per call, where the unfused path opened
  23 to 27. Other calls keep the unfused path. The fused path does not compute
  logsumexp; it returns a meta tensor that refuses to be read.
- **CPU quantised matmul (#178).** One parallel region per matmul instead of
  one per activation row, and a NEON four-column Q4_0 dot. Outputs are
  bit-identical to 0.1.0b8.
- **Preallocated KV cache (S7.4, #182).**
  `torchnative.kernels.kv_cache.PreallocatedCache` writes keys and values in
  place and hands attention a view, so decode issues no `torch.cat` for the
  cache. It is opt-in (`generate(past_key_values=...)`). Its output is
  bit-identical to `DynamicCache` on cpu and mps.
- **Kernels and orchestration (S7.1, S7.2, S7.3 paged attention; #18, #169).**
  The mobile kernel bundle resolver, `TorchNativeAPI` deployment and lifetime
  policy, and paged attention on cpu and mps.

## 2. Defects fixed

- `module.to("mps")` left `QuantizedLinear` weights on the CPU, and the first
  forward panicked inside candle. The blocks now move byte for byte
  (`_C._quantized_to`), and a weight/activation device mismatch refuses by
  name (#176).
- Qwen3-0.6B-Q4_0 did not load with `keep_blocks=True`: the block carrier
  lacked `is_floating_point()`, and an untied block `lm_head` reached
  `nn.Parameter` (#176).
- Intel NPU: probe compiles are quiet, and an Embedding or RMSNorm the device
  refuses goes back to the CPU instead of failing `to()`.
- `kernels` 0.17 and 0.18 validator APIs both work in `get_kernel` (#171).

## 3. Measured but not implemented

- **Speed parity.** llama.cpp leads on every cell measured: 1.8 to 2.8x on mps
  decode, 3.0 to 3.5x on mps prefill, 3.3 to 4.6x on CPU prefill (cells measured
  at load 3 to 7; the Qwen3 Q4_0 CPU cells ran under load and are left out).
- **A decode gain from the preallocated cache.** At the bench's context
  (5-token prompt, 128 steps) the change was within run-to-run noise.
- **Q4_0 greedy text on the CPU.** SmolLM2-135M Q4_0 differs from llama.cpp's
  default CPU path. llama.cpp's own Metal path, and its CPU path with weight
  repacking or flash attention turned off, give torchnative's text. Qwen3-0.6B
  Q4_0 at 128 tokens differs at generated token 20, where the logit margin is
  0.084. Every Q8_0 cell and every mps cell matches llama.cpp.
- **The NNAPI and QNN device tests** did not run for this release; they wait on
  a connected device.

Carried over, not re-measured here:

- **`torch.compile` is still a permanent refusal**, for the structural reason
  in [`../graph/COMPILE.md`](../graph/COMPILE.md).
- **82 of 297 architectures** were never numerically judged. That count is
  [`../architectures/ARCH100.md`](../architectures/ARCH100.md)'s, it measured
  reachability, and it was not re-measured for this release.
- Intel NPU never ran on hardware (#21), and QNN `lower()` was not executed.

## 4. Documentation corrected

- `docs/SPEC.md`: the S2.5 and S7.3 rows state the kept-block and fused-attention
  behaviour, and a new S7.4 row states the preallocated cache.
- `tests/bench/bench_llama_cpp.py` no longer expects SmolLM2's GGUF file to
  refuse; it gains the `gguf_blocks` mode, `--tn-modes` and `--kv-cache`.

## 5. Gate at this head

Not claimed. The develop gate runs for #176 to #182 were in progress or queued
when these notes were written. The suites each change touched were run on the
Mac and are listed in its pull request.
