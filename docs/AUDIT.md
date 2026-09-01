# Documentation audit (work/docaudit)

Audit of `docs/*.md` for claims that are checkable against the tree and have
gone false. Method, baseline, and per-file findings below. Written
incrementally so an interruption keeps the findings gathered so far.

Excluded from this audit (another agent is writing into these concurrently):
`docs/KERNELS26.md`, `docs/ARCH26.md`.

`README.md` is out of scope to edit (belongs to the coordinating session);
anything wrong found there is reported in the final summary, not fixed here.

## Baseline (ground truth established before auditing)

Commands run in the worktree, in the foreground, before touching any file
(so later "counts" claims can be checked against something real):

```
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-docaudit
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
PYTHON=$PY sh rust/torch_c/pytests/run.sh          -> EXIT=0, 268 "ok " lines, SELF-TEST PASS
$PY tools/golden/compare.py                        -> EXIT=0, SUMMARY: 5634/5634 cases passed, 0 failed, ops covered=148, pending case builders=1
$PY rust/torch_c/pytests/verify_schemas.py         -> EXIT=0, SUMMARY: 4392/4392 table entries matched upstream, 0 failed
```

torch version seen by verify_schemas.py: `torch 2.13.0`.

Implemented op count from `_aten_implemented()` (printed by both run.sh and
compare.py): **148** ops, full list captured in `/tmp/run_pytests.log` and
`/tmp/golden_compare.log` in this session.

Other numbers surfaced by verify_schemas.py, useful for cross-checking counts
elsewhere: overloads.json 200/200, methods.json 168/168,
native_functions.yaml re-printed 2584/2584, packet overload lists 118/118,
OpOverload.tags 148/148, CompositeImplicitAutograd registrations 744/744.

Ad hoc one-off checks (op implemented / spelling exists / etc.) are recorded
inline in each file's section below with the exact command used, so the
check is re-runnable.

Shell used for one-off python checks:
```
$PY -c "..."
```
with the same env vars as above exported first.

---

## Findings

(filled in per file below; format: File / Claim / Status / How checked)

### docs/ARCH20.md

- **Claim (§9, "A real gap" list, line ~573-583):** `gelu`, `silu`, `softplus` are listed
  alongside `abs`/`clamp`/etc. as names where "a public `torch.<name>` exists upstream, the
  kernel is here, and the name refuses," and the whole 25-name list is described as "not fixed
  here" / "a well-defined next round."
  **Status: FALSE, and stale.** `hasattr(torch, "gelu")` is `False` on real upstream torch
  2.13.0 (same for `silu`, `softplus`) — there is no bare `torch.<name>` to reach for those
  three, only `torch.nn.functional.<name>`/`torch._C._nn.<name>`, both already answered by this
  shim. Separately, "not fixed here" is stale: 22 of the 25 names now have an
  `overloads.json`/`methods.json` entry (confirmed via `rust/torch_c/src/overloads.json`'s own
  comment, and by grepping the 25 names as JSON keys — all 22 non-gelu/silu/softplus names are
  present); `reshape` is the one still-pending name (confirmed via
  `tools/golden/compare.py`'s current `PENDING` line naming only `aten.reshape.default`).
  **How checked:** `hasattr(torch, name)` for all 25 names against
  `/Volumes/macMini/caches/spike-venv/bin/python` (real torch 2.13.0); `grep -n '"<name>":'
  rust/torch_c/src/overloads.json` for each of the 25; `$PY tools/golden/compare.py` baseline run
  (PENDING line).
  **Fixed:** yes — added a `> **Correction (docs/SPELLINGS.md §5–§7): ...**` blockquote after
  the "well-defined next round" paragraph, matching this doc's existing correction style (the
  `_safe_softmax` correction a few paragraphs above it). Did not rewrite the original text.

- Section 0 (numbers), §11 (verification gates: 241 ok, 3302/133, 4295/4295) and §11.4
  (bit-identical prefill): these are historical before/after tables for *this document's own
  round* (explicit "before/after" column headers), not claims about the repo's state today.
  Current baseline is higher (268 ok, 5634/5634 ops=148, 4392/4392) because later rounds
  (SPELLINGS.md, TRIL.md, KERNELS26.md/ARCH26.md — the last two out of scope here) added more.
  Left as-is; not a stale claim, it's a historical diff correctly scoped to its own round.

- §6.2 ("What is still refused"): `pow(bool_t, 2)`, `pow(True, bool_t)` etc. still raise
  `NotImplementedError`. **Status: confirmed true, still current.**
  **How checked:** called `torch.pow(torch.tensor([True, False]), 2)` and
  `torch.pow(True, torch.tensor([True, False]))` against the loaded shim
  (`PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1`); both still raise
  `NotImplementedError` with the same message the doc describes.

- §10 (`gpt_bigcode`, TorchScript wall): re-ran the import chain description mentally against
  current tree structure; not independently re-verified by executing (would need `transformers`
  import of that specific model file) — reported, not checked, due to time budget. Low risk: the
  doc itself says "re-confirmed against the current build" as of its own writing and nothing in
  this session touched TorchScript.

### docs/SPELLINGS.md

Read in full (784 lines). This document is a chronological, self-correcting log of several
rounds of spelling-table work, and it already contains its own follow-up-correction section
(§8, "후속 정정 — §7.9가 남긴 세 항목 중 둘은 닫혔다") that revisits §7.9's "not done" list and
marks two of three items done, one still open. That is exactly the pattern this audit is
supposed to add where missing — here it was already present and accurate.

- §8's table claim "`clamp.Tensor` 커널 부재 — 그대로 (unchanged)" — **confirmed still true**:
  `aten.clamp.default`/`aten.clamp_.default` are in the current 148-op implemented list;
  `aten.clamp.Tensor` is not. How checked: grepped the implemented-ops list captured in
  `/tmp/run_pytests.log` for `clamp`.
- §7.1's `hasattr` measurements for `gelu`/`silu`/`softplus` (upstream False, `nn.functional`
  True) — **confirmed true**, same check as ARCH20.md above. This section is the "full per-name
  accounting" ARCH20.md §9's correction now points to.
- No false claims found in the sections read. Not exhaustively re-verified line by line (784
  lines of mostly narrative/measurement transcripts); spot-checked the numeric/existence claims
  most likely to be quoted by a future agent (the gelu/silu/softplus split, the clamp.Tensor gap,
  the §7.9→§8 closure table).

### docs/WHEEL.md

Long file (1261 lines), many dated rounds (§9 through §15, 2026-08-28 through 2026-08-31). Most
numeric tables are explicit before/after for their own round (e.g. §0's "168/168 shim tests,
2702/2702 ops=118" is that round's baseline, explicitly "이 작업 전후로 같은 값" — same before
and after *that* round, not a claim about today).

- §15 (2026-08-31, the most recent section — an audit of `tools/wheel/`'s own verification
  scripts) claims current self-test counts: `verify_ios_device.py --self-test` 8/8,
  `verify_cross.py --self-test` (android/device 11/11, manylinux 12/12, Windows 9/9),
  `verify_linux.py --self-test` 6/6, `build.py --self-test` LINUX 11/11 / VERIFY 4/4 /
  UPSTREAM-DIST-INFO 2/2.
  **Status: the ones checkable on this host are confirmed true.**
  **How checked:** ran `$PY tools/wheel/build.py --self-test` directly (foreground, exit 0) —
  output matched exactly: "LINUX SELF-TEST: PASS -- 11/11", "VERIFY SELF-TEST: PASS -- 4/4",
  "UPSTREAM-DIST-INFO SELF-TEST: PASS -- 2/2" (PREFLIGHT-CACHE 3/3 also matched, not separately
  claimed as changed in §15). Did not run `verify_ios_device.py`/`verify_cross.py`/
  `verify_linux.py --self-test` (they assert on Linux ELF / require prior cross-built artefacts
  not present in this worktree) — reported as checked-where-possible, not independently
  re-verified for the other three scripts.
- No false claims found in the sections read (§0, §15, spot checks elsewhere). Sections §1-§14
  were skimmed rather than read line-by-line given the file's length; not exhaustively audited.

### docs/GOLDEN.md

Read in full (297 lines). Entirely a historical record of one specific round (closing the
keyword-dispatch blind spot found by docs/DISPATCH.md §4.1): 2811→2843 cases, ops=119,
`run.sh` 177 ok / 34 "pre-existing unrelated" at the time. Current baseline is 5634/5634
cases, ops=148, 268 ok all passing — later rounds added far more; this doc's numbers are that
round's own snapshot, not presented as "today's numbers."

- §7 ("A wrong diagnosis, and what it actually was") is a deliberate, explicit self-correction —
  house style already applied correctly: it says outright that an earlier version of this same
  section was wrong, why, and what the actual cause was (worktree never had `vendor_torch.sh`
  run). Left untouched per the "leave deliberately-recorded mistakes" rule.
- **Minor, reported-not-fixed:** §8 ("Running it") gives copy-pasteable commands with inline
  comments stating expected output as `# 2843/2843, ops covered=119, exit 0` and
  `# 177 ok / 34 pre-existing unrelated (§7), exit 1`. Read at face value today, `run.sh` no
  longer exits 1 or shows 34 failures — current baseline is 268 ok, exit 0, no failures. Not
  edited: the whole document is explicitly framed as an analysis of one past round rather than a
  "how the repo behaves now" reference, and editing the inline comments would blur that framing
  without adding real information (the true current numbers live in the acceptance-criteria
  commands, not in this doc). Flagging in case a future reader treats §8 as copy-paste-and-expect
  literally.

### docs/DISPATCH.md

Skimmed in full (445 lines), read §4 and §6 closely.

- **Claim (§4.1, line ~313):** "The golden harness is blind to this entire code path" /
  "it is a standing gap." **Status: stale.** docs/GOLDEN.md (a direct sequel to this exact
  section, confirmed by GOLDEN.md's own opening line citing "docs/DISPATCH.md §4.1") closed most
  of this gap: 61 of `interned_name()`'s 74 arms are now exercised by keyword by at least one
  golden case (verified there by re-running the same tamper technique on a different arm,
  `"dim"`, and watching `compare.py` go red). 13 arms remain uncovered, one deliberately.
  **How checked:** read docs/GOLDEN.md §4-§5 in full (see above); did not independently re-run
  the `"dim"` tamper myself (GOLDEN.md already did, with exact before/after numbers and the
  restore-from-backup discipline).
  **Fixed:** yes — added an "Update (docs/GOLDEN.md): ..." blockquote directly after the
  "standing gap" paragraph, pointing to the sequel rather than rewriting DISPATCH.md's own
  (correct-at-the-time) narrative.
- §4's "Behaviour did not move" numbers (197 ok / 2811/2811 ops=119 / 4203/4203) and §6's "what
  is left" cost list (resolve+_bind ~0.7-1.0µs, `_tlv_get_addr` ~12%, the `match op` arm count):
  these are that round's own before/after and open-items list. Whether item 1 (resolve+_bind) is
  still open is really a question for docs/BIND.md, which a separate agent in this session is
  auditing in parallel — not re-checked here to avoid duplicate work.

