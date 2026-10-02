# GGUF — reading Ollama's checkpoints through the real `from_pretrained`

Issue #17, milestone TN-M2b. Written 2026-10-03 on branch `work/gguf`. Host Apple M1 (8 cores,
16 GB), macOS darwin 25.5.0, CPython 3.13.0, transformers 5.15.1, candle-core 0.11.0, `gguf`
0.17.1. **The machine was at load 45–160 throughout**, so no number below is a timing claim
(AGENTS.md §16, rule 8). **Neither the gate nor `cargo` ran in this round**; nothing in Rust
changed. The measurements in §6 ran the new tests against the **main checkout's** built shim
(develop, same Rust as this branch), with this branch's `torchnative` package first on the path.

> **Conclusions first.**
>
> 1. **The entry point is upstream's spelling:**
>    `torchnative.transformers.AutoModelForCausalLM.from_pretrained(repo, gguf_file=...)`.
>    Upstream's GGUF config mapping, GGUF→HF tensor-name tables and loader run unchanged; only the
>    step that turns GGML blocks into tensors is ours (§1).
> 2. **GGUF's block layout is candle's block layout.** The file's bytes go into
>    `torch._C._quantized_from_blob` with no repacking and come back out of `_quantized_blob`
>    identical — measured on every tensor of two real Qwen3-0.6B files (620 tensors) and on
>    random blocks of all ten block types candle holds (§2).
> 3. **candle's dequantisation agrees with llama.cpp's `gguf` package bit for bit** on all 620
>    real tensors (F32, Q8_0, Q4_K, Q6_K) and on random blocks of Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
>    Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. No tolerance anywhere (§2, §6).
> 4. **The door works end to end on a tiny Qwen3 written by llama.cpp's own writer**: the real
>    `Qwen3ForCausalLM`, 13 of 13 tensors bit-identical to gguf-py's dequantisation across nine
>    types, forward finite (§6). Graded **agrees** on weights, **reaches** on the forward. The
>    same test on the real 0.6B file is written and **not run** (it loads 2.4 GB; the machine was
>    overloaded).
> 5. **What the door does not yet do: keep the blocks.** A model loaded through it is dense, as
>    upstream's GGUF path is dense. Mapping one tensor onto a `QuantizedLinear` byte for byte works
>    (`torchnative.gguf.quantized_linear`); doing it inside `from_pretrained` is §5.

---

## 1. The design, and the two that were not chosen

Three shapes were on the table. The constraint that decides between them is AGENTS.md §20 and
`docs/INTENT.md` §3: **the front end is fixed** — the user calls the real `from_pretrained`, the
same way upstream documents it.

### 1.1 What upstream's `gguf_file=` path actually needs (read, transformers 5.15.1)

`PreTrainedModel.from_pretrained(..., gguf_file=...)` (`modeling_utils.py`) does four things:

| step | what | needs |
|---|---|---|
| 1 | GGUF metadata → config (`AutoConfig.from_pretrained(..., gguf_file=)` → `load_gguf_checkpoint(return_tensors=False)`) | `gguf` package |
| 2 | model on `meta`; `get_gguf_hf_weights_map` pairs GGUF names with parameters | `gguf` (`get_tensor_name_map`) |
| 3 | each tensor: `gguf.quants.dequantize` in **numpy**, the architecture's `TensorProcessor`, then **`torch.from_numpy`** | `gguf`, numpy, `torch.from_numpy` |
| 4 | the ordinary loader (`_load_pretrained_model`) | `accelerate` (an up-front check: "accelerate is required when loading a GGUF file") |

On this stack, as installed:

- **`torch.from_numpy` refuses by name** — the shim has no numpy bridge (docs/devices/INTELNPU.md
  §1.5, asserted by `test_intelnpu.py`). Step 3 cannot run as written.
- `gguf` and `accelerate` are **not in the pinned venv**, and nothing may be installed there
  (AGENTS.md §15.2).
- Upstream **refuses `quantization_config` together with `gguf_file`** — a GGUF load upstream is
  dense, always.

### 1.2 The options

**(a) Make upstream's own path run.** Keeps every line upstream's, but needs a numpy bridge in the
shim (`torch.from_numpy` — a Rust change, an API decision with its own aliasing question, and the
reversal of a measured, documented refusal), plus `gguf` and `accelerate` importable. It also
dequantises in numpy, so quantised blocks can never reach torchnative's quantised modules through
it. **Not chosen as the whole answer**, but most of it survives: steps 1, 2 and 4 are exactly
upstream's in the chosen design.

**(c) Expose candle's `gguf_file.rs` through the extension.** The container is a header, a
key/value table and a descriptor table: ~300 lines of `struct` in Python. The only part of candle
that is *needed* — turning GGML block bytes into a `QTensor` — **is already exposed**, as
`_C._quantized_from_blob` (quant.rs), and it is the same operation candle's GGUF reader performs
(§2.1). Exposing `gguf_file.rs` would add a second Rust entrance into quantised storage and a
rebuild for no capability the Python reader lacks. **Not chosen.**

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/quant.rs _quantized_from_blob present -->

**(b) A reader in the `torchnative` package, feeding transformers. Chosen**, in this shape:

```
torchnative.transformers.<Auto*>.from_pretrained(repo, gguf_file=...)
  ├─ step 1  upstream: AutoConfig.from_pretrained(repo, gguf_file=...)
  ├─ step 2  upstream: get_gguf_hf_weights_map(meta skeleton, TensorProcessor)
  ├─ step 3  OURS:     torchnative.gguf.GGUFFile + load_tensor  (blocks → candle → float32)
  └─ step 4  upstream: <ModelClass>.from_pretrained(None, config=config, state_dict=...)
```

Step 4 uses upstream's documented `state_dict=` route (`pretrained_model_name_or_path=None`,
which `modeling_utils.py` requires for it), so the inner call never sees `gguf_file` and the
`accelerate` check is never reached. **That is a difference from upstream, stated rather than
hidden:** this door works without `accelerate`; upstream's does not.

`transformers.AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` — upstream's class, not
ours — is untouched and still stops at its own walls on this stack (`accelerate`, then
`torch.from_numpy`). The door is `torchnative.transformers`, whose contract is already "the
user's diff is the import line" (docs/api/TRANSFORMERS.md).

### 1.3 What the door refuses, by name (AGENTS.md §18)

| request | refusal | why |
|---|---|---|
| `gguf` not importable | `ImportError` naming `gguf>=0.10.0` | steps 1–2 are upstream's and import it; upstream's own requirement |
| an architecture whose `TensorProcessor` is not the identity (`llama`, `qwen2moe`, `qwen3moe`, `gemma2/3`, `t5`, `gpt2`, `bloom`, `mamba`, `nemotron`, `lfm2`, `gpt-oss`, `minimax-m2`) | `UnsupportedGGUF` naming the processor | those processors rewrite tensors in numpy (Llama's q/k permutation, MoE splits); not ported, so the weights would load wrong |
| `quantization_config=` with `gguf_file` | `UnsupportedGGUF` | upstream refuses it; here it would dequantise the blocks and quantise again — lossy twice |
| `state_dict=` with `gguf_file` | `ValueError` | upstream's refusal |
| a refused GGML type in a mapped tensor | `UnsupportedGGUF` naming the type (§1.4) | — |

### 1.4 GGML types

Loaded: `F32`, `F16`, `BF16` (dense, in their own dtype) and `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
`Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` (blocks). Refused by name, each with its reason in
`torchnative/gguf/reader.py` `GGML_TYPES`: `Q8_1`, `Q8_K` (activation-side formats), every
`IQ*` and `TQ*` and `MXFP4` (no candle `GgmlDType`), `I8`–`I64` and `F64` (not wired), the
removed repacked layouts (`Q4_0_4_4` … `IQ4_NL_8_8`), and any id not in the table (named by
number). `test_every_unsupported_type_refuses_by_name` pins both sets, so moving a type between
them is an edit to the test as well.

---

## 2. Block-layout compatibility — the evidence

### 2.1 By construction (read)

candle's own GGUF reader and `_C._quantized_from_blob` perform the same operation on the same
bytes:

- `gguf_file.rs` → `ggml_file.rs::qtensor_from_ggml` → `from_raw_data::<BlockT>`:
  `slice::from_raw_parts(ptr as *const BlockT, n_blocks).to_vec()`;
- `quant.rs::_quantized_from_blob` → `QStorage::from_data` → `GgmlDType::from_data`:
  `as_t_slice::<BlockT>(data).to_vec()`.

Both reinterpret the file's bytes as `#[repr(C)]` GGML block structs and copy them. candle's
block structs are ggml's, and candle reads GGUF files with them; there is no transcoding step
anywhere for a layout mismatch to hide in. Shape: GGUF's `ne` is fastest-first, candle reverses it
(`gguf_file.rs`, `dimensions.reverse()`), and so does this reader — a linear weight stored as
`ne = [in, out]` arrives as `(out, in)`, which is `nn.Linear`'s layout and what `QMatMul` reads.

### 2.2 Measured (this round; §6 for where)

| what | result |
|---|---|
| container: metadata (28 / 32 keys, incl. 151,936 tokenizer strings and 151,387 merges), 310 + 310 descriptors, `data_offset`, alignment, vs gguf-py `GGUFReader` | identical, both files (canonical-byte digests per key) |
| raw tensor bytes, our reader vs gguf-py | 620 / 620 identical (sha256) |
| `_quantized_blob(load_tensor(keep_blocks=True))` vs the file | 620 / 620 identical — 394 block tensors (197 Q8_0, 168 Q4_K, 29 Q6_K) |
| candle dequantisation vs `gguf.quants.dequantize`, float32 bytes | 620 / 620 identical |
| random blocks, all 10 block types, vs `gguf.quants.dequantize` | 10 / 10 identical; Q8_0/Q4_0/Q4_K also vs `ggml_ref.py` |
| a GGUF Q8_0 tensor as `QuantizedLinear`, lossless operands | blob identical; output bit-identical to dense `linear` |

This closes the gap `docs/graph/QUANT2.md` §8 recorded — "no GGUF written by llama.cpp to compare
against" — for the reader. ggml_ref.py's docstring names exactly this check as the one that would
close what it cannot.

### 2.3 Relayed from the parallel Q4-quality round (#16), not re-measured here

- **Q4_0's writer** is byte-identical to llama.cpp's: 0 of 4,718,592 bytes differ from gguf-py's
  writer on three real Qwen3-0.6B layers.
- **Q4_K**: layout and reader match (as above); candle's **writer** differs (unweighted
  `make_qkx1_quants` against llama.cpp's weighted `make_qkx2_quants`). That matters for
  *producing* GGUF, not for reading it.
- **`CANDLE_DEQUANTIZE_ALL` / `CANDLE_DEQUANTIZE_ALL_F16`** silently turn every quantised matmul
  into a dense one over dequantised weights (`vendor/candle-core/src/quantized/mod.rs`,
  `QMatMul::from_arc`). `test_a_gguf_linear_holds_the_file_bytes_and_multiplies_exactly`
  refuses to run with either set, because on its lossless operands the dense path gives the same
  bits.
- candle's aarch64 `+dotprod` build lazily keeps a **second, repacked copy** of Q4_K weights
  (`BlockQ4Kx8`) when `out_features % 8 == 0` — a memory cost, not a correctness one; the blob
  stays in GGUF layout.

### 2.4 One disagreement found, in the reference

`gguf` 0.17.1's `GGML_QUANT_SIZES` gives `Q8_1` **40** bytes (`4 + 4 + 32`, from when `d` and `s`
were `float`); ggml and candle store 36 (`ggml_half2 ds` + 32). `Q8_1` is refused regardless, and
`test_the_type_table_matches_gguf_py` asserts the disagreement exists, so the day gguf-py fixes it
the test says so.

---

## 3. The reference, and where `gguf` lives

`gguf` is installed **only** in `.caches/gguf-venv` (inside this repository, git-ignored; AGENTS.md
§15.1–§15.2), made with:

```sh
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .caches/gguf-venv
PIP_CACHE_DIR=$PWD/.caches/pip .caches/gguf-venv/bin/pip install "gguf==0.17.1"
# -> gguf 0.17.1, numpy 2.5.3, PyYAML 6.0.3, tqdm 4.70.1; no torch
```

It is used two ways, both deliberate:

1. **As the reference, in its own interpreter.** `rust/torch_c/pytests/gguf_ref.py` runs under
   `.caches/gguf-venv/bin/python` and emits digests as JSON. It shares no code with candle or
   `torchnative.gguf`, which is what makes agreement mean something (AGENTS.md §16: measured in a
   separate subprocess).
2. **To open the door in a subprocess.** The door needs `gguf` (§1.3). The door test appends the
   venv's `site-packages` **after** the gate interpreter's own, so spike-venv's numpy and
   transformers win and only `gguf` resolves from it. Nothing is installed into spike-venv.

Without the venv, every test that needs it **skips by name**. `TORCHNATIVE_GGUF_PYTHON`,
`TORCHNATIVE_GGUF_DIR` and `TORCHNATIVE_CACHES` override the locations; in a worktree the main
checkout's `.caches/` is found by walking out of `.worktrees/`.

---

## 4. Files fetched

Into `.caches/gguf/` (git-ignored), with `curl`, at pinned revisions; sha256 checked against the
Hub's LFS object ids. **No test downloads anything**; a missing file is a skip by name.

| file | source @ revision | bytes | sha256 | types |
|---|---|---|---|---|
| `Qwen3-0.6B-Q8_0.gguf` | `Qwen/Qwen3-0.6B-GGUF` @ `23749fefcc72` | 639,446,688 | `9465e63a…bb031` | F32 ×113, Q8_0 ×197 |
| `Qwen3-0.6B-Q4_K_M.gguf` | `unsloth/Qwen3-0.6B-GGUF` @ `50968a4468ef` | 396,705,472 | `ac2d9771…d524a` | F32 ×113, Q4_K ×168, Q6_K ×29 |

Qwen's own repository carries only the Q8_0 file; the k-quant file is unsloth's, chosen because
Q4_K_M is the format Ollama ships by default.

---

## 5. Not done

1. **Keeping the blocks inside a loaded model.** The door dequantises. The pieces exist —
   `quantized_linear` maps a GGUF tensor onto `QuantizedLinear` byte for byte, and
   `torchnative.quant.hf` already swaps leaves before weights land — but joining them inside
   `from_pretrained` needs a per-layer format (Q4_K_M mixes Q4_K and Q6_K) and an `adopt` that
   takes an already-quantised tensor instead of quantising. Not written; until it is,
   `quantization_config` with `gguf_file` refuses by name.
2. **Architectures with a rewriting `TensorProcessor`**, Llama first (its q/k permutation is a row
   permutation, which on blocks is a permutation of byte rows and so could stay byte-exact).
3. **Logits against upstream.** The forward is graded *reaches*. Upstream's own GGUF path cannot run
   in the pinned venv (`accelerate`), so an *agrees* grade needs upstream torch fed the same state
   dict and a derived tolerance (AGENTS.md §16).
4. **The real-file door test** (`test_from_pretrained_gguf_loads_qwen3_with_the_reference_weights`)
   is written and not run (2.4 GB of float32 on an overloaded machine).
5. **The gate** was not run; `test_gguf.py` has not been through `run.sh`.
6. The tokenizer: `AutoTokenizer.from_pretrained(repo, gguf_file=...)` is upstream's and needs only
   `gguf`; not exercised here.

---

## 6. What ran, where

All against `/Volumes/macMini/thisisthepy/torchnative/torchnative/src/main/torch` (main checkout,
develop, built shim) with this branch's package first on the path; `gguf_ref.py` under
`.caches/gguf-venv`.

| test | result | peak RSS |
|---|---|---|
| 12 synthetic/door-refusal tests | 12 ok | — |
| `test_real_file_q4_k_m_matches_the_reference_reader` | ok | 599 MB |
| `test_real_file_q8_0_matches_the_reference_reader` | ok | 850 MB |
| `test_from_pretrained_gguf_on_a_tiny_qwen3_written_by_gguf_py` | ok | — |
| `test_from_pretrained_gguf_loads_qwen3_with_the_reference_weights` | **not run** | — |

**Nullifications (AGENTS.md §17.5)**, each on a copy of the package, run against the same tests:

| mutant | red tests |
|---|---|
| shape not reversed | 4 |
| `keep_blocks` ignored | 2 |
| `data_offset` not aligned | 4 |
| `IQ4_NL` mapped to `q4_0` | 3 |
| `quantization_config` refusal removed | 1 |
| processor check removed | 1 |
| door: tensors round-tripped through bfloat16 | 1 (12 of 13 tensors differ) |
| door: one tensor left unmapped | 1 |
| control (no change) | 0 |

**One mis-measurement, caught and redone:** the first two door mutants came back *survived*. The
door test runs its load in a subprocess whose `PYTHONPATH` starts with the repository's own
package, so the mutated copy never reached it — the inverse misdiagnosis AGENTS.md §17.5 warns
about. With the mutant first on the subprocess's path, both went red and the control stayed green.
