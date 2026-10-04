# 0.1.0b5: release notes

**transformers' continuous batching runs on the cpu, and its greedy tokens are
upstream's.** Qwen3 (the test model since 2026-10-03) and streaming
`generate` agree with upstream, plain and with q8_0. mps `generate` still stops
before its first token; §3 names where.

The repository layout changed in this release (#43, #56). Paths in older
documents and in the 0.1.0b4 notes refer to the old layout.

---

## 1. Features added

- **Continuous batching on the cpu** (#13, #30). transformers 5.15.1's
  `generate_batch` (paged|sdpa and paged|eager) runs with nothing stubbed.
  Greedy tokens equal upstream's in all 8 measured cells: Qwen3 and Llama, both
  attention paths, float32 and bfloat16. What it needed:
  - a `torch.index_select` binding;
  - mixed int, slice and tensor indexing in `__getitem__`;
  - the three `torch.mps` memory queries (Metal's own numbers, plus a real count
    of live buffers);
  - `pin_memory=True` accepted on cpu (`is_pinned()` answers False);
  - `torch.Generator` objects as independent streams. The global cpu RNG
    already reproduced upstream's mt19937, and seeded generators draw what
    upstream draws.
- **Qwen3 agrees with upstream** (#23). On Qwen3-0.6B:
  - the state dict is bit-identical to the file;
  - logits are within bounds derived per model and dtype;
  - greedy `generate` is token for token in float32.

  In bfloat16 the shim parts from upstream at an exact tie: upstream's own
  margin between the two tokens is 0.000, and the float64 oracle prefers the
  shim's choice. The test accepts a low-precision divergence only at such a tie.
- **Streaming `generate`** (#11). `TextIteratorStreamer` yields upstream's tokens
  on Qwen3-0.6B and SmolLM2-135M, plain and with `TorchnativeConfig("q8_0")`.
  candle's q8_0 matmul also quantises the activation, so the q8_0 oracle does
  the same.
- **The backward-free adaptation configuration** (#9).
  `TORCHNATIVE_BACKWARD=off` makes stage-1 methods refuse by name at import,
  subclassing and `wrap`.

## 2. Defects fixed

- **float64 to bfloat16 rounded wrongly on every platform** (#28). For
  1+2^-8+2^-22, upstream gives 1.0078125 and the shim gave 1.0. Narrowing to
  half now follows c10 (through float32 for bf16; one rounding for f16 on
  aarch64).
- `torch.manual_seed` of a negative seed was off by one against upstream (#30).
- `test_docrefs` read untracked files and was quadratic on long lines (#8).
- 15 streaming tests silently skipped on a stale vendored-tree path; they run
  now (#43).

## 3. Measured but not implemented

- **mps `generate` stops before its first token** (#29). The walls, in the order
  they are hit:
  1. `isin` (refused as a host readback);
  2. the zero-byte Metal buffer `DynamicCache` allocates;
  3. `argmax`, `bitwise_*` and `max`.

  The fixes are written but not built. Gemstone's own suite on this Mac stops
  at the same first wall.
- **Continuous batching on mps** (#32): strided in-place writes, int32 compute
  and readback ops.
- **Linux x86_64:** the first full CI gate had 24 failures (#40). All 24 are
  triaged on `feat/linux-gate`, not yet landed: 5 are pinned by name as
  x86-only divergences, and the rest are fixed or skip by name.
- **`torch.compile` is still a permanent refusal**, for the structural reason in
  [`../graph/COMPILE.md`](../graph/COMPILE.md).
- **82 of 297 architectures** were never numerically judged. That count is
  [`../architectures/ARCH100.md`](../architectures/ARCH100.md)'s, it measured
  reachability, and it was not re-measured for this release.
- The schema check `verify_schemas` never ran (#54), and the SPIR-V staleness
  check compares mtimes (#46).

## 4. Documentation corrected

- `CLAUDE.md` is deleted; `AGENTS.md` is the one rules file, and every citation
  was rewritten through its section map (#22).
- The layout: `torchnative/{rust,python}`, `tests/` split by function, and no
  new root names (#43, #56). AGENTS.md §2 lists the approved root entries, and
  the gate checks them.
- Em-dash removed from tracked text outside vendored code; reader-facing
  install examples use `uv` (#45, #59).

## 5. Gate at this head

```
120/120 suites   1985 ok / 0 FAIL / 28 SKIP   (0.1.0b4 shipped at 1589 ok)
DOCWATCH         1538 / 1538
```

Measured in a worktree on this Mac. The 28 skips are all environmental: no
Android device, no OpenVINO runtime, no QNN interpreter, and no VOICE4 assets.

## 6. Platform status

Unchanged from [`RELEASE_0_1_0b3.md`](RELEASE_0_1_0b3.md) §6, except that the
gate now runs on GitHub's macOS runner (116 suites, 1792 ok, 0 FAIL) and on
Linux x86_64 (24 FAIL, §3).
