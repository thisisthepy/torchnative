# 0.1.0b8: release notes

**On Apple silicon, `to(device.npu)` now lowers SmolLM2-135M into 15 grouped
two-layer CoreML decoder programs, and `from_pretrained` -> `generate` matches
upstream float32 greedy on 20 of 20 tokens on an M1.** That is the one headline.
The rest of the release is a round of op coverage, training and correctness
work, each entry naming its issue or pull request where the commit does.

0.1.0b7 was tagged but never reached PyPI: its publish gate (run 40239265020) passed
2027 tests with 0 failures and stopped on one DOCWATCH marker with a wrong path. 0.1.0b8 is
0.1.0b7 plus the two fixes listed first in §2.

Entries are taken from `git log --no-merges v0.1.0b6..origin/develop`. The ANE
numbers are from PR #156 (issue #155); the record is
[`../graph/ANEGEN.md`](../graph/ANEGEN.md).

---

## 1. Features added

- **Apple ANE generate** (#155, #156). `to(device.npu)` on SmolLM2-135M turns
  its 30 decoder layers into 15 two-layer CoreML programs. The KV rows stay in
  the transformers `Cache`. Greedy `generate` through the shim equals upstream
  float32 greedy on 20 of 20 tokens, measured on an M1.
  - RMSNorm overflowed at float16 (the residual stream reaches 19279); that is
    fixed, and float32 is kept between groups.
  - `test_bf16ane` was updated to the new structure. Its logit bar is now
    upstream's own bfloat16 error, and its argmax claims use a real sentence.
  - What this does not show is listed in §3.
- **GGUF reader and a `gguf_file=` door on `from_pretrained`.** Upstream's
  config, name mapping and loading are kept; only the step that turns blocks
  into tensors is replaced. The door returns a dense model; keeping blocks
  inside a loaded model is not done. Against llama.cpp's gguf 0.17.1 the
  container metadata and all 620 tensor raw bytes are identical, and candle's
  dequantisation is bit-identical on all 620 real tensors. Refused by name:
  architectures whose tensor processor rewrites tensors (Llama's q/k
  permutation among them), `quantization_config=` or `state_dict=` with
  `gguf_file`, 25 GGML types and unknown type ids. The commit says 14 of 15
  `test_gguf.py` tests ran by hand and the 2.4 GB real-file test did not.
- **Q4 quality harness** (`tools/q4quality/measure.py`, `test_q4ref.py`). Q4_0
  is byte-identical to gguf-py's writer on three real Qwen3-0.6B layers. Q4_K's
  layout and reader match, but the writer does not: candle's unweighted
  quantiser gives 7.60/7.78/7.52% reconstruction error against 7.32/7.50/7.26%
  for llama.cpp's on the same layers. The model-level measurement was not done.
- **Training** (#48). `create_graph` through relu, gelu, layer_norm and
  embedding, and gradients with respect to a non-leaf. Earlier in the range:
  `autograd.Function`, tensor and node hooks, and `create_graph`.
- **Continuous batching** (#15). Greedy batched equals alone equals upstream;
  sampling is pinned to upstream.
- **`triu_`/`tril_`, `nansum`/`nanmean`, `Generator.get_state`/`set_state`**
  (#115), including `torch.get_rng_state`, `set_rng_state` and `fork_rng`, and
  int32 `tril`/`triu` on Metal. These were three named refusals from the
  Gemstone run on 0.1.0b6.
- **`Tensor.numpy()` and `__array__`** (#118): a cpu view sharing memory, with
  upstream's refusals.
- **MSVC STL tie order for `sort`/`argsort`/`topk`** (#81). Transcribed; the
  Windows oracle is pending, so this is not yet graded.
- **`index_put_` into an int32 `self` on mps** widens on the device (#129).
- **Thread safety audit work** (#114, #76). Bootstrap modes are thread-local,
  `STRIDED_BARRIER` is held across writes, `_GRAD_FN_CLASSES` and export and
  transformers module-level caches are synchronised, the CUDA stream is
  synchronised before quantized workspaces grow, `TensorBase` setters take
  `&self`, and dispatch registrations are published whole under a lock.
  `torch._C` now declares `gil_used = false`. The b6 note said the module kept
  the GIL; this commit reverses that. The 3.15t gate has not been run for this
  note.

## 2. Defects fixed

- **#159** (#160): `import torch` no longer installs a stub `torchaudio` module and warns
  about a Higgs shim when torchaudio is absent. The resample stub is opt-in through
  `torch._C._install_torchaudio_resample()`.
- **#162**: `docs/graph/GGUF.md`'s DOCWATCH marker names `quant.rs` by its repository path.

- **#113**: Metal float32 division is now correctly rounded.
- **#101**: Metal f32 `erf` and half `sin`/`cos` agree with upstream; the
  candle fork's Metal kernels compile without fast math by default, and the
  transcendentals are graded against float64 truth with the remaining
  fast-math gaps recorded.
- **#144**: eager backward through a freed graph raises upstream's
  `retain_graph` error.
- **#91**: Metal argsort on i8 ran the i16 kernel on a 1-byte buffer; it now
  runs the i8 kernel.
- **#46**: SPIR-V staleness compares recompiled bytes, not mtimes.
- **#47**: `verify_published` orders beta versions without `packaging`.
- **#54**: the aten op registry is closed and `verify_schemas` reads the real
  yaml.
- **#150**: the gradient-penalty block of `test_doublebackward` is pinned as a
  known x86_64 Linux divergence. It is a pin, not an agreement.
- CI: the metal-kernels patch is regenerated, the Windows oracle workflow is
  named, a develop run in progress is allowed to finish, and the DOCWATCH
  markers for BACKWARD10/9 point at the current layout.

## 3. Measured but not implemented

State of the ANE work, honestly:

- **ANE power was not measured directly.** The indirect evidence is process CPU
  time falling from 38.4 to 31.0 ms/token, with wall time equal at 31
  ms/token. To measure it, run
  `sudo powermetrics --samplers ane_power,cpu_power -i 500 -n 20`.
- **No ANE speed gain was measured.** Wall time per token is equal.
- **Only SmolLM2-135M was executed.** Other models are not claimed.
- **Decoder groups are batch 1 only.**
- **Intel NPU never ran on hardware** (#21).
- **QNN `lower()` was not executed**, and Hexagon still refuses by name.

Carried over, not re-measured here:

- **`torch.compile` is still a permanent refusal**, for the structural reason
  in [`../graph/COMPILE.md`](../graph/COMPILE.md).
- **82 of 297 architectures** were never numerically judged. That count is
  [`../architectures/ARCH100.md`](../architectures/ARCH100.md)'s, it measured
  reachability, and it was not re-measured for this release.
- The Windows tie order (#81) has no oracle yet.
- The 3.15t gate has not run for the `gil_used = false` change (#114, #51).

## 4. Documentation corrected

- [`../graph/ANEGEN.md`](../graph/ANEGEN.md) records the ANE decoder work.
- The README records ANE generate and pins the install to 0.1.0b7.
- Temporary files go to the session scratchpad or `.cache/`, not `.tmp/`.
- The DOCWATCH markers for BACKWARD10/9 point at the current layout.

## 5. Gate at this head

The 0.1.0b7 publish gate (run 40239265020) ran 152 of 152 suites: 2027 ok, 0 FAIL,
184 SKIP, and failed on one DOCWATCH marker (GGUF.md:76, wrong path), fixed by #162. Otherwise the gate is as at 0.1.0b6; no
other numbers are claimed here, and the b6 table is not repeated.

## 6. Platform status

Unchanged from [`RELEASE_0_1_0b6.md`](RELEASE_0_1_0b6.md) §6, except for the
rows below.

| platform | grade | what changed in 0.1.0b7 |
|---|---|---|
| Apple ANE (M1) | **agrees** on one model | SmolLM2-135M greedy `generate` equals upstream float32, 20 of 20 tokens (#155); power and speed not measured |
| Intel NPU | not run | never ran on hardware (#21) |
| Qualcomm QNN / Hexagon | refused by name | `lower()` not executed |
