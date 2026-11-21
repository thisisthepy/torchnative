# 0.1.0b11: release notes

**Three pull requests since 0.1.0b10: two CPU performance changes aimed at LLM
decode and sampling, and the `isnan` family.** The owner approved releasing
0.1.0b11 on 2026-11-21 (the leader's record of the approval; the owner's words are
not quoted in this repository, AGENTS.md rule 7). The range is
`git log --merges --first-parent v0.1.0b10..develop`: #413, #416 and #417. The
build path is the one 0.1.0b10 introduced (pypackpack for seven of the nine abi3
wheels, setuptools for the rest, see
[`RELEASE_0_1_0b10.md`](RELEASE_0_1_0b10.md) section 4); nothing under
`.github/workflows/` changed in this range (`git diff v0.1.0b10..develop --stat
-- .github` lists only `.github/scripts/gate_suites.py`, six added lines).

**No speed was measured for this release, and no speed claim is made.** Both
performance changes were written against a Gemstone measurement of 0.1.0b9
(Gemstone #74, CI run 44792524320, per `docs/perf/MM_DECODE.md`); neither was
timed after the change. Gemstone #74 measures them after this release. The local
checks in every PR below were `cargo check` and `py_compile` only; the test
results are the gate's.

---

## 1. Features added

- **A skinny-gemm path for 1 to 8 row float32 `mm` on CPU (#413, Gemstone #74).**
  `torchnative/rust/torch_c/src/skinny_gemm.rs` handles a 2-D float32 CPU `mm`
  with 1 to 8 rows and depth of at least 32 (the transposed-weight layout of
  `Linear`): each weight column is streamed once against the cached rows and the
  work is threaded over output columns on candle's barrier pool. Every other
  `mm` keeps the old path. The reasoning is in `docs/perf/MM_DECODE.md`: from
  Gemstone's breakdown `aten.mm` is 74% of a decode step, and on non-Apple CPUs
  the `gemm` crate (0.19) takes a non-threaded gemv path at batch 1, stays under
  its 4M-flop threading threshold at batch 4, and spends three quarters of its
  16-row f32 microkernel on padding at 4 rows. That analysis is a prediction
  until the benchmark runs; the document says so itself.
- **`torch.isnan`, `torch.isposinf`, `torch.isneginf` and their `Tensor`
  methods (#416, closes #415).** One shared kernel body, same shape as `isinf`
  in b10 (#411): float inputs compute `x != x`, `x == +inf`, `x == -inf`;
  integer and bool inputs give all `False` without reading the operand; float8
  refuses by name; complex `isnan` refuses by name and complex `isposinf` and
  `isneginf` raise `RuntimeError` as upstream's meta check does. Overload and
  method rows from `native_functions.yaml`, golden cases, and the zero-element
  layout audit cover them.

## 2. Defects fixed

Both are performance defects, not wrong results (#417, Gemstone #74):

- **CPU `cumsum` ran as one launch per element.** It was n-1 narrow, add and
  clone launches plus one `cat` (about 49k candle launches for one 49152-entry
  row, from Gemstone's top-p sampling shape). It is now one host scan in the same
  addition order and accumulator (f64 for floats, wrapping i64 for integers), so
  the PR states results are unchanged. Integer and bool inputs promote to int64
  and `dtype=` is kept. A 0-d input no longer indexes an empty shape. A non-CPU
  tensor takes the old path, and the host scan refuses a non-CPU tensor by name
  (9d4ff598) and is a classified readback exemption in
  `tests/_support/test_shim.py`.
- **`sort`, `sort.stable` and `topk` lanes ran on one thread.** Lanes are now
  split across candle's barrier pool above 16384 elements; each lane still uses
  the transcribed `std::sort`, so tie order, NaN ordering and indices are
  unchanged by construction. "Unchanged" for both items is the PRs' statement
  backed by golden cases the gate runs, not something observed here.

## 3. Test added

- `tests/ops/test_skinny_gemm.py` (#413): `mm` against upstream across shapes,
  strides and edge sizes. `tests/bench/bench_skinny_mm.py`, the micro-benchmark,
  runs in CI.
- `tests/bindings/test_isnan.py` (#416): `isnan`, `isposinf` and `isneginf`
  against upstream 2.13.0, registered in `gate_suites`. Golden cases, and
  `tests/ops/test_empty_storage_slice.py` gains three lines for the audit.
- Golden cases for large, 3-d, transposed, `dtype=`, 0-d and empty `cumsum`, and
  for multi-lane and stable `sort` (#417); `tests/bench/bench_sort_cumsum.py`.
- Pins moved (#416): tag core count 139 to 140 (only `isnan` is core) and schema
  identities 402 to 405.

## 4. Deleted

- Nothing was removed from the public surface in this range.

## 5. Documentation corrected

- `docs/perf/MM_DECODE.md` (new, #413): the decode `mm` analysis, the kernel's
  design and counters, and the plan to measure it.
- README and `docs/locale/README_ko.md` pin `torchnative==0.1.0b11` in the
  `ppp core add` examples.

## 6. Measured but not implemented

- **No benchmark was run for this release.** The `skinny_gemm`, `cumsum` and
  `sort` changes are predictions about speed until
  `tests/bench/bench_skinny_mm.py` and `tests/bench/bench_sort_cumsum.py` run in
  CI and Gemstone #74 measures a published wheel.
- `skinny_gemm` covers 2-D float32 CPU `mm` only (1 to 8 rows, depth at least
  32); other dtypes, batched `matmul` shapes and larger row counts keep the old
  path.
- 11 CompositeImplicit ops remain in the pending pin (#374, carried from b10).
- The XNNPACK lowering on the shim (#306) is still open (carried from b10).
- Carried over from b10, which carried them from b9; not re-measured for this
  release:
  - **`torch.compile` through the stock path.** The b8 notes called it a
    permanent refusal for the structural reason in
    [`../graph/COMPILE.md`](../graph/COMPILE.md) (PEP 523 frame evaluation is
    not available to a Limited-API extension; measured 2026-09-06). #194
    marked that refusal unconfirmed and #208 added the PEP 523-free engine
    that SPEC S1.8 describes; `COMPILE.md` itself carries no banner and was
    not rewritten, and nothing in this range touched it.
  - **Architecture coverage.** [`../architectures/ARCH100.md`](../architectures/ARCH100.md)
    measured 215 of 297 `transformers` architectures forwarding, with 82 not
    forwarding for want of 31 operators. That document is a frozen baseline;
    its banner points at `docs/numerics/AGREE2.md` (2026-09-12), which
    measured 297 of 297 forwarding and 288 of 290 judgeable architectures
    agreeing numerically. Neither sweep was re-run for this release: 82 of 297
    is the ARCH100 count the release test names, and 288 of 290 is the latest
    number on file.
  - Intel NPU never ran on hardware (#21), the Windows tie order (#81) still
    has no oracle result recorded in these notes, and the on-device ExecuTorch
    route is unchanged.

## 7. Gate at this head

Not stated. The notes were written from the pull request descriptions and the
git history before the release run; no gate result for the head of this branch
is claimed, and the pypackpack build of the wheels was not run when this file was
written. The release run of `publish-pypi.yml` decides.
