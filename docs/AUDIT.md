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

  **Update (this round):** that "separate agent" note is stale — this round's brief lists
  docs/BIND.md as this same audit's own territory and priority item 3. Auditing it below rather
  than skipping it a second time.

### docs/META.md

Long file (887 lines), one continuous document across two rounds (initial meta-device landing,
then a later "give it the kernels from_pretrained needs" round responding to a user bug report,
`3c9d000`). House style (round-scoped snapshots) applies throughout; most numeric tables are
explicitly framed as before/after for their own round.

- **Claim (§11.1, judgment block for the `3c9d000` round):** "골든 4284/4284, ops=139" /
  "`_aten_implemented()` 는 139 그대로" (closing line of §7.4, same round).
  **Status: FALSE — internally inconsistent with the same document's own §7.4 opening line**,
  which states "커널이 있는 op 148 개" for the *same* round (and the commit message of `3c9d000`
  itself says "Of 148 ops with a kernel"). One of the two numbers is a stale snapshot pasted next
  to a later one; 148 is the one that matches reality.
  **How checked:** live-probed the shim (`torch.mm`/`.sum`/`.view` on meta → `NotImplementedError:
  ... no meta kernel`; `.select`/`.tril`/`.expand`/`.gt` on meta → succeed), confirming §7.4's
  148-op-baseline categorisation (66 reachable / 82 not, table sums to 82) matches this session's
  own baseline (148 ops, confirmed already in the AUDIT.md baseline above). §11.1's 139 does not
  match anything current.
  **Fixed:** yes — added a `> **Correction (문서 감사): ...**` blockquote directly after the
  "139 그대로" sentence in §11.1's closing paragraph, explaining the mismatch and pointing to the
  148/5634 baseline instead of rewriting the original numbers.
- **Claim (§12, "못 한 것", pre-existing correction block on `m.to("cpu")` / `m.to_empty`):**
  "`m.to("cpu")` 는 여전히 거부되지만 이유가 다릅니다 ... `NotImplementedError: ... Please use
  torch.nn.Module.to_empty() instead ...`". **Status: confirmed still true** (this correction was
  already present before this audit, landed in `3c9d000`, not something this round added — but
  re-verified since it's exactly the "X refuses" claim shape this audit is supposed to test).
  **How checked:** built a `meta`-device `nn.Linear`, called `.to("cpu")` directly (not after
  `to_empty`, which mutates the module in place and would have made a second `.to("cpu")` trivially
  succeed) — raised the exact `NotImplementedError` text quoted above. `.to_empty(device="cpu")`
  on a fresh meta module also succeeded, confirming the other half of that correction too.
- §7.4's category table (reductions 22, view/shape 13, in-place 12, contraction 8, indexing 7,
  composite/activation 6, combine/split 4, other 10 = 82) and its "66 reachable" figure: spot-
  checked 4 unreached ops (`mm`, `sum`, `view` from the table; also tried `add` after `.t()` to hit
  the not-yet-meta `t.default`) and 4 reached ops (`select`, `tril`, `expand`, `gt`) directly
  against the live shim — all matched the table exactly.
- Section 9 (dispatch cost, +13.6 ns / +3.7%), §10 (107-item transcript diff, 41 match / 66
  mismatch broken down by category), §11 (round-1 judgment: 2258/2258 ops=96, pytests 91 ok),
  §8.2 (upstream-vs-shim transcript table for `with torch.device(...)`/`set_default_device`):
  not independently re-verified — these are dated, round-scoped measurement transcripts (explicit
  "측정일 2026-08-25" byline) rather than "current state" claims, consistent with the house style
  already confirmed correct in WHEEL.md/GOLDEN.md above. Not re-run given time budget; no reason
  found to suspect them specifically (no counter-evidence encountered while checking §7.4/§12).
- §12 "모르는 것" (unknowns) and "때운 것" (patched-over gaps) sections: these are explicit
  "we did not check this" / "we know this is wrong" admissions, not falsifiable claims — left as
  reported, not re-tested (auditing a "we don't know X" statement doesn't have a pass/fail).

### docs/SEQLEN.md

Very long (1417 lines), three rounds (`pow` fix §3, `amax` fix §7, scale+mask fusion §8), each
with its own before/after tables and its own numerics-safety argument. This document is already
unusually well self-corrected: §7.10 and §8.12 each carry a `> **TAKEN, in docs/KERNELS26.md /
docs/TRIL.md §N.**` blockquote pointing forward to where a "not done here" item was later done —
exactly the annotation pattern this audit is supposed to add, already present.

- **Claim (§7.10 correction blockquote):** `torch.amax`/`Tensor.amax` "now resolve and reach this
  kernel" (landed in docs/TRIL.md §2, entries in `overloads.json`/`methods.json`).
  **Status: confirmed true.** **How checked:** `grep '"amax"'` in both
  `rust/torch_c/src/overloads.json` and `rust/torch_c/src/methods.json` (both present); live call
  `torch.amax(t, dim=1)` and `t.amax(dim=1)` against the running shim — both return the kernel's
  answer (`tensor([5., 6.])` for the test input).
- **Claim (§8.12 correction blockquote):** `tensor.rs::transposed_contiguous`, "the same copy in
  32x32 cache blocks", landed in docs/KERNELS26.md §7, wired into the SDPA call site and into
  `aten.contiguous.default`. **Status: confirmed true — the function and both call sites exist.**
  **How checked:** `grep -rn transposed_contiguous rust/torch_c/src/` — function defined at
  `tensor.rs:2390`, referenced from `aten.rs:3152` (the SDPA k-transpose path, matching the
  `aten.rs:3143` comment citing this exact function) and from `tensor.rs`'s own unit tests. Did
  not re-run the cargo benchmark that produced the 5.25x/4.84x figures (out of scope: that
  measurement belongs to docs/KERNELS26.md, excluded from this audit).
- Every other numeric table in the file (§1 curve-fit coefficients, §3/§7/§8's old/new/upstream
  A/B tables, the gate-count tables in §6/§7.11/§8.10, the sha256 logit-digest tables): these are
  explicitly dated, round-scoped measurement transcripts, each already labelled with which round
  produced it and consistent with later rounds' own "before" column reproducing the prior round's
  "after" to 2-3 significant figures (i.e. the document already cross-checks itself round to
  round). Not independently re-measured — reproducing a performance curve is outside what a
  documentation audit can check against the tree, and re-running would risk exactly the
  "measurement contaminated by a concurrent agent" failure mode CLAUDE.md §"측정 작업은 단독으로
  돌린다" warns about, which this session is not isolated for (KERNELS26.md/ARCH26.md work is
  running concurrently). Spot-checked only the two claims above because they are "X now works"
  assertions checkable by existence rather than by re-measuring a curve.
- No false claims found. This file's own internal cross-checks (each round's "old" column
  reproducing the prior round's "new" column) already do a large part of what this audit would be
  checking for, and did not turn up a contradiction while reading.

### docs/BIND.md

Long file (1147 lines now), four rounds. §9.5 already carried a pre-existing
`> **Correction (문서 감사, 재확인): ...**` blockquote (landed in a prior commit, `b8c3ea1`, not
by this audit) re-pointing a stale smoke-test-count claim at the current 268/0 baseline — verified
still accurate (re-ran `run.sh`, 268 ok, matches this session's baseline).

- **Claim (§9.3, closing paragraph):** "The clean fix is one line in `tensor.rs` ... it is not
  written because `tensor.rs` is outside `bootstrap.py` + `docs/BIND.md`." **Status: FALSE, and
  found by cross-referencing the commit that landed this very section against its own commit
  message.** `git show b8c3ea1` (the commit that added this whole §9 to BIND.md) changes
  `tensor.rs`'s `dtype` getter from `PyDtype::new(self.tag)` to `crate::dtype::interned(py,
  self.tag)` in the *same commit*, and the commit message says outright: "The investigating agent
  ... wrote a Python-side property override instead ... this replaces it." So the documentation
  text (apparently carried over unedited from the investigating agent's draft) describes a fix
  that had already been superseded before the commit merged.
  **How checked:** `grep "fn dtype" rust/torch_c/src/tensor.rs` → confirms `interned(py, self.tag)`
  is what ships today; `grep _install_tensor_dtype_identity rust/torch_c/src/bootstrap.py` →
  confirms the Python override this section's code listing shows does not exist in the file;
  `git log -S"fn dtype(&self)" -- rust/torch_c/src/tensor.rs` → identifies `b8c3ea1` as the commit
  that made the change, and `git show b8c3ea1` → confirms both the code change and the commit
  message's own account of what replaced what.
  **Downstream effect:** §9.4's cost table ("roughly doubles the cost... ~0.07 → ~0.15 µs") is
  therefore measuring a code path that was never shipped — the commit message gives the real
  number for what did ship: "0.042 us against 0.070 before ... cheaper than the incorrect one."
  The correctness finding itself (§9.1/§9.2/§9.5 — `.dtype` identity was broken, `baddbmm` promoted
  to float64, now fixed) is unaffected and reconfirmed live below.
  **Fixed:** yes — added a `> **Correction (문서 감사): ...**` blockquote after §9.3's closing
  paragraph, pointing to the actual shipped fix and its real cost, without deleting the original
  (now-inaccurate) "not written" text.
- **Re-verified live** (the underlying correctness claim, independent of which fix shipped):
  `t.dtype is torch.float32` → `True`; `from torch._decomp.decompositions import baddbmm;
  baddbmm(ones(2,3,5), ones(2,3,4), ones(2,4,5)).dtype` → `torch.float32` (not `float64`). Both
  match §9.1/§9.2's claims exactly.
- **Claim (test existence, §9.5's correction block):** `test_decompose_lowers_baddbmm_default_
  now_that_the_dtype_is_a_singleton` replaced the old assertion. **Status: confirmed true** —
  `grep` finds both the old test name still present (as a different function, presumably renamed
  scope) and the new one defined at `rust/torch_c/pytests/test_shim.py:7625`.
- Gate-count tables in §4, §8.3, §8.4's prefill table, §9.4/§9.6/§9.7's profile counts: round-scoped
  snapshots consistent with house style, not re-verified individually given time budget — no
  contradiction found while checking the items above.
- §7's heading says "Android measurement (in progress)" but the section body is a complete,
  concluded measurement with a "Result" subsection and no open items — **reported, not fixed**:
  this is a stale heading word rather than a false claim (nothing in the body claims to be
  incomplete), and CLAUDE.md's fixing rule is for false mechanical claims, not phrasing.

### docs/DTYPE_PERF.md

Shorter (458 lines), single round, already self-annotated with its own reversal (§2's initial
microbench measured a layout the model never makes; §3-§4 catch it and say so explicitly at the
top of the file — the "§2 를 먼저 읽고 §3 에서 뒤집힙니다" callout). Gate counts (241 / 3302/3302
ops=133 / 4295/4295 / cargo test 10) are this round's own before/after, consistent with house
style and with the progression already seen in BIND.md/SEQLEN.md's own round-scoped numbers at
similar dates.

- **Claim (§7, "남은 것"):** the fused-gemv kernel for `lm_head` decode (`S=1`, where upstream is
  measured 6.2x faster) "이 회차에서 재지 않았습니다" (not measured this round) and is left as a
  named gap. **Status: still an open gap, not resolved by a later round.**
  **How checked:** `grep -rln "fused gemv\|융합 gemv\|fused_gemv\|gemv_trans"` across `docs/` and
  `rust/torch_c/src/` — only DTYPE_PERF.md and DTYPE.md (its own prior-round source) reference it;
  no later document or source symbol picks it up. Confirmed `widen_gemm_operand` (§5's actual
  fix) still exists in `rust/torch_c/src/aten.rs:1860`, so the fix this round *did* land is still
  in place; the fix this round explicitly declined to attempt is still undone.
- No false claims found. The document's numeric tables are internally consistent (§3's model-level
  numbers are what §4's layout explanation is built to explain, and §6's verification counts match
  the "before" numbers §5's fix started from), and its own retraction of §2 is already the kind of
  annotation this audit adds elsewhere.

### docs/VIEWS.md

758 lines, six sections landing incrementally (§1-§6), each with its own before/after gate counts
matching the next section's baseline exactly (§1→§2-3→§5→§6, each "before" reproducing the prior
section's "after"). Unusually rich in exactly the checkable-fact categories this audit targets:
several hard "still refuses" / "now works" claims. Live-tested the highest-stakes ones (the write-
through behaviour is a correctness property, not a performance number, so a false claim here would
be the "invented spelling" / "already landed" class of error the brief warns about).

- **Claim (§6, throughout):** in-place ops now write through views — `x[0] = v`, `x[1].fill_(3.0)`
  etc. are visible on the base tensor, where §4 says they previously were not.
  **Status: confirmed true.** **How checked:** `x = torch.zeros(5); x[1].fill_(3.0)` → base reads
  `[0.0, 3.0, 0.0, 0.0, 0.0]` — the write reached the original tensor through the `select.int` view.
- **Claim (§2-3):** `index_put_` now handles a bool mask (`x[mask] = values`) and non-1-D operands,
  where it previously refused. **Status: confirmed true.** **How checked:**
  `x2[torch.tensor([True,False,True,False])] = torch.tensor([9.,8.])` → `[9.0, 0.0, 8.0, 0.0]`,
  matching upstream semantics (positions 0 and 2 selected).
- **Claim (§1):** `aten.ge.Tensor` now has a kernel (previously resolved and then refused by name).
  **Status: confirmed true.** **How checked:** `torch.tensor([1,2,3]) >= torch.tensor([4,5,6])` →
  computes (`[False, False, False]`); `a >= a` → all `True` (the specific case §1 says
  distinguishes `Cmp::Ge` from `Cmp::Gt`).
- **Claim (§6.4, "the two that remain"):** `slice.Tensor` with step > 1 and `view.dtype` are still
  write-lost (materialised here, a view upstream) and this is structural (blocked by candle's
  `pub(crate)` storage boundary), not yet fixed. **Status: confirmed still true** — and this is the
  same divergence already independently confirmed in this session's baseline run of
  `tools/golden/compare.py` (its `KNOWN DIVERGENCE` section names exactly these same two ops with
  the same explanation), so it is cross-checked from two directions.
- **Claim (§6.5):** a partial-overlap `copy_` (`x[0:2].copy_(x[1:3])`) computes a defined answer
  here where upstream raises — a known, recorded, unfixed divergence ("cost decision", not a wall).
  **Status: confirmed true.** **How checked:** `x=torch.arange(4.); x[0:2].copy_(x[1:3])` →
  computed `[1.0, 2.0, 2.0, 3.0]` without raising (upstream raises "some elements of the input
  tensor and the written-to tensor refer to a single memory location").
- No false claims found. This document's own §4 (superseded verdict, correctly annotated with a
  blockquote pointing to §6) and its "counts after §N" tables at every section boundary already do
  most of what this audit checks for.

### docs/TORCHSCRIPT.md

397 lines, single round. Already carries two pre-existing `> **Correction ...**` blockquotes (not
from this audit) pointing at `docs/TRIL.md` for the `tril` kernel landing and the count moving
19/20 → 20/20.

- **Claim (top-of-file correction, and §9):** `aten.tril.default` is implemented and GPT-BigCode
  now constructs and forwards, making the sweep 20/20. **Status: confirmed true.**
  **How checked:** `"aten.tril.default" in torch._C._aten_implemented()` → `True`; built a small
  `gpt_bigcode` config via `AutoModelForCausalLM.from_config(...).eval()` and ran a forward — no
  error, correct logits shape.
- No false claims found otherwise; the document's own gate tables (§5) are round-scoped and
  consistent with the counts BIND.md/SEQLEN.md show at similar dates.

### docs/DYNAMO.md

737 lines, two parts: Part A (§0-9, what `_C._dynamo` needs when `torch.compile` is never called)
and Part B (§10-19, what happens when it actually is called). No code changed by this document's
own account (`git status --short` claim, verified: only `docs/DYNAMO.md` differs across its
commits). This is the one file in this pass where a **cross-document staleness** turned up — not a
number that drifted, but a conclusion invalidated by a *later* document's fix.

- **Claim (Part B summary table + §12, "Wall B"):** with the default (`inductor`) backend, after
  stubbing `pt2_archive_constants`, `torch.compile()` hits `SourceRangeFactory.make_range` via
  `torch.utils.mkldnn.MkldnnLinear`'s module-scope `@torch.jit.script_method` — described as "a
  different wall, unrelated to eval-frame" from the `backend='eager'` wall in §11/§15.
  **Status: FALSE as of today's default — superseded by a later, unrelated round.**
  `docs/TORCHSCRIPT.md` (commit `02758e8`, 2026-08-31 00:34) landed *after* this document's last
  commit (`4e6281f`, 2026-08-30 17:10) and added `os.environ.setdefault("PYTORCH_JIT", "0")` to
  `bootstrap.py` — upstream's own scripting-disabled default. Under that default,
  `@torch.jit.script_method` is `torch/jit/_script.py`'s `if not _enabled: return fn`, a no-op, so
  `MkldnnLinear` no longer reaches `make_range` at all.
  **How checked:** live-reproduced both conditions. With `pt2_archive_constants` stubbed and
  `PYTORCH_JIT` **unset** (today's actual default), `torch.compile(lambda x: x+1)()` sails past the
  TorchScript import and instead raises `NotImplementedError: ... eval_frame.set_skip_guard_eval_
  unsafe` — an eval-frame-family symbol, i.e. the *same* category of wall §11/§15 already judged
  structurally unreachable under abi3, not a different one. With `PYTORCH_JIT=1` forced (the
  condition this document was actually measured under), the exact `make_range` traceback in §12
  reproduces verbatim — confirming the original measurement was correct for its environment, and
  that the environment is what changed.
  **Cross-reference:** `docs/COMPAT.md` §5's own correction blockquote already independently
  reports this same TORCHSCRIPT.md-caused change for GPT-BigCode specifically ("this import wall
  is gone on both versions") — the two documents agree once both are read, they just were not
  cross-linked from DYNAMO.md's side until this pass.
  **Fixed:** yes — added a `> **Correction (문서 감사, 재측정): ...**` blockquote after §12's
  closing paragraph explaining the mismatch, what reproduces under today's default, and that the
  live-verified replacement wall is in the same family §15 already covers; also annotated the Part
  B summary table's Wall-B row to point at the correction, without rewriting §12's original
  (correct-at-the-time) measurement.
  **Does not affect:** §18's recommendation ("push capture, not Dynamo") — the correction notes
  explicitly that the reachable wall changed but the abi3-unreachability conclusion (§15) still
  holds for whichever eval-frame symbol is hit first.
- This is the first cross-document staleness found in this pass (as opposed to a document being
  stale against the live tree) — worth flagging because none of the other files audited so far had
  one; it is a real risk shape (a later, unrelated commit can invalidate an earlier document's
  scoped conclusion without either document being edited) rather than a copy-paste or transcription
  error like the ones found in META.md/BIND.md.
- Part A's §0-9 (the 137/52/2-name accounting, the self-check methodology) not independently
  re-verified — it is a `dir()`/instrumented-proxy census with its own self-check built in (§2.3,
  positive and negative controls), and re-running it would mean rebuilding the same instrumentation
  this document already built; spot-checking would not add confidence proportional to the time
  cost. §15's abi3 argument (reading upstream C source for `Py_BUILD_CORE`) not re-verified either
  — it is a source-reading argument about a cache directory outside this repo
  (`/Volumes/macMini/caches/pytorch-spike/pytorch`), not a claim about this repo's own tree.

### docs/COMPAT.md

547 lines, single commit (`8a1de11`), but already unusually dense with pre-existing
`> **Correction (re-verified live against the current build): ...**` blockquotes — at least six,
covering nearly every numeric claim in the document (the 4.x/5.x architecture counts, the specific
missing-op list, the `__getitem__` list-index gap, the `gpt_bigcode` import wall). These read as
the *original author* re-verifying their own document against a moved tree before committing, not
as this audit's work — but they are exactly the shape of correction this audit adds elsewhere, so
this file needed mostly re-verification rather than new fixes.

- **Claim (§5's correction, "still genuinely missing"):** `aten.all.default`, `aten.aminmax.default`,
  `aten.index_add_.default`, `aten.nonzero.default`, `aten.scatter_.value`, `aten.zeros.default`,
  `aten.roll.default` remain unimplemented. **Status: confirmed true.**
  **How checked:** grepped the current 148-op implemented list (captured in this session's baseline,
  `/tmp/golden_compare_2.log`) for `all`, `aminmax`, `nonzero`, `zeros`, `roll`, `index_add_` — none
  present; only `aten.scatter.src` exists, not `scatter_.value`.
- **Claim (§5's correction, "now works"):** `torch.square`, `torch.repeat_interleave` (as Python
  composites, no own `aten.*` entry), `aten.pow.Tensor_Tensor` float32×int32 promotion, and
  `x[:, [-1, 0]]` list-indexed `__getitem__` all now run. **Status: confirmed true.**
  **How checked:** called all four directly against the live shim: `torch.square(...)` computes,
  `torch.repeat_interleave(...)` computes, `x[:, [-1, 0]]` on a `(2,4)` tensor returns the correct
  gathered values.
- **Claim (§3.2 fix, still current):** `TensorBase.permute` and `Tensor.T` are wired (not kernel
  gaps, wiring gaps). **Status: confirmed true.** **How checked:** `x.permute(1,0)` and `x.T` both
  compute and agree.
- **Claim (§3.1 fix, still current):** `torch.autocast(device_type="cpu", enabled=False)` completes
  without raising. **Status: confirmed true**, called directly.
- No false claims found — everything spot-checked (a representative sample covering every section
  with a numeric or existence claim) matched. This is the second document in this pass (after
  SEQLEN.md) where the file's own prior self-correction already did most of what this audit checks
  for.

### docs/DECOMP.md

686 lines, single round (2026-08-30). Unlike SEQLEN.md/BIND.md/COMPAT.md, this document does not
frame its headline numbers as an explicit round-scoped snapshot — §0 ("이 문서가 답하는 것")
presents "37 개 중 9 개" as a present-tense answer, which is exactly the shape of claim that goes
stale silently as unrelated kernel work lands.

- **Claim (§0 and §4, headline):** of 37 non-Core-ATen, capture-reachable implemented ops, 9 lower
  to Core ATen via upstream's decomposition rules. **Status: FALSE today — stale by natural
  kernel-count growth, not a bug.** **How checked:** re-ran `rust/torch_c/pytests/decomp_sweep.py`
  against the current build (the script itself is unchanged since this document's single commit,
  confirmed via `git log`). Current output: `_aten_all_implemented() = 157` (not 129), `non-core =
  67` (not 52), capture-rejects `22` (not 15), **population 45** (not 37), **LOWERED 11** (not 9) —
  the 9 named in §4 plus `aten.baddbmm.default` (which §6/§7.2 already separately say was fixed and
  now lowers, but that fact was never folded back into §4's headline table) and
  `aten.floor_divide.default` (blocked in §4 on a missing `aten.div.Tensor_mode` kernel that has
  since been implemented by unrelated later work). REFUSED 25, CAPTURE_RAISED 1 (matches §4), and a
  new `NO_CASE` category (8 ops) that does not exist in this document's own classification scheme
  at all — `decomp_sweep.py` skips ops it has no golden case to synthesize an input from.
  **Fixed:** yes — added a single `> **Correction (문서 감사, 재측정 2026-09): ...**` blockquote
  right after §0's headline claim (rather than editing every downstream table in §4/§6/§7/§9,
  which would blur the "what this round opened" narrative the document is built around), giving
  the current numbers and pointing to `docs/AUDIT.md`'s baseline for gate counts.
- **Claim (§7.2.1):** "wall 3" (decomposition rule runs but disagrees with the recorded trace) has
  no live example as of 2026-08-30, confirmed by a sweep with a positive control (self-check that
  the detector can actually fire). **Status: confirmed still true.** **How checked:** the same
  `decomp_sweep.py` re-run above shows `DISAGREES: 0` in its output, consistent with the document's
  claim not having regressed.
- §9's gate counts (`run.sh` 176, `compare.py` 2744/2744 ops=118, `verify_schemas.py` 4200/4200) are
  covered by the same correction — not re-annotated individually, since they are downstream of the
  same "written before later kernel work landed" cause and the correction already says so.
- No other false claims found; §1-§3 (the four wiring-defect fixes: packet overload collapse, CIA
  registration enumeration, kwarg-name mismatch, and the three `overloads.json` entries) are
  mechanism descriptions and fix narratives, not counts that drift, and were not re-verified
  line-by-line given time budget — no contradiction found while re-running the sweep for the
  headline number.

### docs/CAPTURE.md

448 lines, an earlier round than DECOMP.md (its own §5's Core-ATen count, 108 implemented/70 core,
is explicitly the pre-decomposition-pass motivation, correctly framed as historical — §10 already
has a strikethrough-style "~~1. 분해 패스.~~ **섰습니다**" update pointing at docs/DECOMP.md).

- **Claim (§9's "분해 패스" row):** repeats DECOMP.md's "37 개 중 9 개" figure for how many
  non-core ops the decomposition pass lowers. **Status: stale for the same reason found in
  DECOMP.md** (kernel count grew from 129 to 157 `_aten_all_implemented()` since DECOMP.md was
  written, taking the population from 37 to 45 and LOWERED from 9 to 11).
  **Fixed:** yes — appended a short pointer to the same table cell rather than re-deriving the
  numbers a second time, since DECOMP.md's own correction (added earlier in this pass) is the
  canonical location for the detail.
- **Claim (§2, "실제로 나오는 것"):** capturing `nn.Sequential(Linear, ReLU, Linear)` records
  `Linear` as `aten.t.default` + `aten.addmm.default`, and `relu`'s argument arrives entirely as a
  keyword (`{'self': %1}`) rather than positionally, because `_torch_level_function` binds
  everything by name before dispatch. **Status: confirmed true, byte for byte.**
  **How checked:** ran the exact capture live — `_C._capture_begin([x]); y = m(x);
  tr = _C._capture_end([y])` — and the printed node list matches the document's node-by-node
  transcript exactly, including `aten.relu.default` showing `args=[]`, `kwargs={'self': %1}`.
- **Claim (§4, control-flow rejection):** any Python branch on a tensor value is caught by
  rejecting `aten._local_scalar_dense.default` by name, with a specific reason string.
  **Status: confirmed true.** **How checked:** called `.item()` inside an active capture region;
  `_capture_reason()` returned text matching the document's description verbatim ("a Python branch
  taken on that value is not in the record, so the trace would be one arm of a branch replayed
  unconditionally").
- §7's A/B dispatch-cost measurement is explicitly self-flagged in the document as taken under a
  contaminated (load 8-13) condition and the document already narrows its own conclusion
  accordingly — not re-measured here (re-measuring a microbenchmark is out of scope for a
  documentation audit, and the document's own honesty about the contamination is exactly the
  house-style behavior this audit looks for elsewhere).
- No other false claims found.

### docs/DESIGN.md

Longest priority file (1147+ lines), the project's design-rationale document. Per this audit's
own rule, its reasoning/design sections (§1-§8 mostly, the A-vs-B tensor-engine decision, the TTL/
TTA/TTT taxonomy, the kernel-strategy argument) were **left alone** — they are arguments, not
checkable facts, and CLAUDE.md's rule is explicit about not rewriting those. What was checked is
the narrower set of *status* claims about this repo's own tree, which is where this document
turned up its most consequential finding this pass.

- **Claim (§11.1, both the original text and its own 2026-08-25 correction blockquote):**
  "`from_pretrained` 와 실제 체크포인트 경로는 여전히 한 번도 실행되지 않았습니다" (the
  `from_pretrained` path and real checkpoint loading have still never been executed, as of the
  in-document 2026-08-25 update, blocked on `torch._C.is_autocast_enabled`).
  **Status: FALSE today — this is exactly the "refusal names something as missing that already
  landed" failure shape the brief specifically warns about**, and it is the highest-severity finding
  of this whole pass: a design document telling a reader that the project's core end-to-end path
  (loading a real Hub checkpoint and running it) has never worked, when it has.
  **How checked:** ran it directly — `AutoModelForCausalLM.from_pretrained("HuggingFaceTB/
  SmolLM2-135M")` against the live shim, downloaded and loaded real Hub weights, forward pass
  produced logits of the correct shape, and `.generate(max_new_tokens=5)` produced 8 tokens with
  no error. Independently corroborated by `docs/META.md` §7-§8.3 (already audited earlier in this
  pass), which ran the same `from_pretrained` path with an `AutoModelForCausalLM` for llama3-rope
  and checked logits against upstream to 1e-5.
  **Root cause of the gap:** `docs/COMPAT.md` (audited earlier, also in this session) implemented
  `torch._C._is_autocast_available` and closed exactly the wall this section names — but nothing
  ever came back to update DESIGN.md's §11.1, which still reads as current.
  **Fixed:** yes — added a nested `> > **Correction (문서 감사, 재측정 2026-09): ...**` blockquote
  directly inside the existing 2026-08-25 blockquote (matching its own nesting convention), giving
  the live re-verification and pointing to COMPAT.md as the commit that closed the wall, while
  leaving the original historical account of what was blocked and why intact.
- **Claim (§9 "이 저장소"):** `torchnative/api/__init__.py:4` has a `SyntaxError` (`*args, *kwargs`
  instead of `**kwargs`) and `torchnative/nn/federated/__init__.py:1` has a dead import
  (`DistributedDataFederated`, no such module), both blocking first import.
  **Status: FALSE today — both fixed.** **How checked:** `py_compile.compile(..., doraise=True)`
  on `torchnative/api/__init__.py` succeeds (no SyntaxError — the file is now docstring-only with
  imports deferred, matching a design note already elsewhere in the file about deferred `torch`
  imports); `import torchnative.nn.federated` and `import torchnative.api` both succeed directly.
  **Fixed:** yes — added a short correction paragraph after the table noting both are resolved,
  without tracking down which commit did it (out of scope for a documentation pass).
- **Claim (§11.1, "열려 있는 성능 결함", 2026-08-28):** `linear`'s per-call `.contiguous()` copies
  a 113 MB weight on every forward pass (30x slower than upstream at the `lm_head` shape), and this
  was explicitly **not fixed** — a deliberate accuracy-vs-speed tradeoff left open, by name,
  matching how `docs/SDPA.md` §12 handled a similar case.
  **Status: FALSE today — fixed, and landed as the default, not gated behind an opt-in.**
  **How checked:** `docs/LINEAR.md` (2026-08-29, its own file, not previously part of this audit's
  priority list but pulled in because DESIGN.md pointed at it) diagnosed the same defect, measured
  that landing the fix moves bits *only toward* upstream agreement (0 regressions, 35 cases newly
  agreeing with upstream out of 507), and left the accept/gate decision open, uncommitted, for the
  coordinating session. `git log -S"fn batched_matmul" -- rust/torch_c/src/aten.rs` finds commit
  `2e00ec3` ("Perf: Fold instead of broadcasting, and stop copying the weight every call"), and
  `gemm_with_layout_fallback`/`batched_matmul` are present in the current `aten.rs` as the default
  path (`.contiguous()` now only runs as a fallback when candle refuses a strided operand) —
  confirmed by reading the current source, not merely inferred from the commit existing.
  **Fixed:** yes — added a correction blockquote pointing to `docs/LINEAR.md` and the landing
  commit, noting LINEAR.md §6 already names five remaining walls (e.g. bf16/f16 weights still
  widening per call) for anyone who reads past this correction.
- Everything else in §0-§10 (the A-vs-B tensor-engine decision, the TTL/TTA/TTT taxonomy, the
  kernel-strategy argument, the `theRiverLethe`/`test-time-adapters` findings in §9's first two
  tables) either (a) references source in sibling repositories not present in this worktree and
  therefore not checkable against this tree, or (b) is reasoning/design argument rather than a
  checkable fact — both out of scope for fixing per this audit's rules, and not reported as
  findings since nothing here contradicted them.
- This is the second cross-document staleness found in this pass (after DYNAMO.md/TORCHSCRIPT.md)
  and by far the most consequential — a reader trusting §11.1 today would conclude the project's
  core value proposition (real HF checkpoints running on-device) has never been demonstrated, when
  it has been demonstrated twice over (COMPAT.md, META.md) by documents this same document does
  not reference.

