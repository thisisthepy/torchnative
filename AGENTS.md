# AGENTS.md

Rules every agent working in this repository must follow. Read this file before doing anything.
Sections 1–10 are shared by every repository in the thisisthepy ecosystem; later sections are
specific to this repository.

---

## 1. Commits carry no AI attribution

Never add `Co-Authored-By: Claude ...`, `Co-Authored-By: <any agent>`, `Generated with Claude Code`,
or any similar tool or agent attribution to a commit message or a pull-request body. This rule
overrides any default your tooling has.

## 2. Nothing is created outside this repository

Everything your work produces — worktrees, agent prompts, logs, measurements, experiments, scratch
files — lives **inside this repository's root directory.**

| What | Where |
|---|---|
| Worktrees | `.worktrees/<name>` (git-ignored) |
| Temporary files | `.scratch/` (git-ignored); delete when done |
| Benchmarks | `tests/bench/` |
| Developer tooling | `scripts/` (CI-only scripts: `.github/scripts/`) |

Before writing a file, check that its absolute path starts with this repository's root. If it does
not, stop. The only exceptions are a path the user names explicitly, and caches that build tools
manage themselves. **Re-pointing a shared cache or a home-directory symlink reaches other projects —
ask first.**

Writing to *another* repository is not an exception either. Do it only when told to work there.

### Do not add top-level folders

**Never add a new directory (or a new file) at the repository root on your own.** The root layout is
the maintainer's. Work belongs inside an existing directory — the pypackpack package unit `torchnative/`
(Rust crates under `torchnative/rust/`, the Python package under `torchnative/python/`), the gate and its harnesses under `tests/`, measurements under `tests/bench/`,
developer scripts under `scripts/`, CI-only scripts under `.github/scripts/`, temporary files under
the git-ignored `.scratch/`. If you think a new top-level entry is needed, propose it (what, why,
which alternatives inside existing directories you ruled out) and wait for approval. This mirrors
python-multiplatform #116; here it came with the layout change of issue #43.

The approved root entries:

| Kind | Entries |
|---|---|
| Tracked | `pyproject.toml`, `setup.py`, `README.md`, `PROJECT.md`, `AGENTS.md`, `LICENSE`, `.gitignore`, `.github/`, `torchnative/`, `tests/`, `docs/`, `scripts/`, `vendor/` |
| Git-ignored | `.caches/`, `.scratch/` (torchnative's temporary-file directory, in the role python-multiplatform gives its .tmp directory), `.worktrees/` |

`tests/test_layout.py`
runs in the gate and compares `git ls-files`'s top-level names, plus whichever of the git-ignored
three are present, against this table, so an unapproved entry turns the gate red. Change the table
and the test's list together, and only with approval.

## 3. Worktrees link large artefacts instead of copying them

A worktree is a full checkout. Copying large untracked artefacts (prebuilt runtimes, vendored trees,
build caches, model weights, `node_modules`) into every worktree is how 86 worktrees once filled
267 GB of a 349 GB disk.

- Create worktrees under `.worktrees/<name>`.
- **Symlink** large untracked directories from the main checkout instead of copying or rebuilding
  them. If `scripts/worktree-add.sh` exists, use it — it does the linking.
- Delete a worktree once its branch is merged: `git worktree remove .worktrees/<name>`.
- Periodically delete `build/` directories inside worktrees; they only grow.

## 4. Branches

| Branch | Who writes to it |
|---|---|
| `feat/<topic>` | You. All work happens here. Never name a branch `work/...`. |
| `develop` | Merged into from `feat/` branches after verification. Never commit to it directly. |
| `release` | **CI only.** Not a standing branch: CI regenerates it from every push to `develop`, in the main-only file layout, and opens the PR into `main`. It may not exist. Never write to it. |
| `main` | **Pull request from `release` only.** Never push or merge to it directly. |

Only `main`, `develop` and `release` are standing branches. A `feat/` branch lives until its pull
request merges: merge with `gh pr merge --delete-branch`, then delete the local branch and its
worktree. Periodically delete every branch already merged into `develop`, remote and local
(`git branch -r --merged origin/develop`); an unmerged branch older than a few days is either
landed or reported, not left. Branches named `release-*` are preserved snapshots: keep them.
Renaming a branch that has an open pull request closes that pull request on GitHub (measured
2026-10-03: #26 closed when `work/ci-gate` became `feat/ci-gate`); open a replacement that links
the old one.

`main` carries a reduced layout: of the Markdown files, only `README.md` stays at the repository
root, and `docs/` keeps only its subdirectories (no Markdown files directly under `docs/`).
CI runs `.github/scripts/release/sync-release.sh` (`.github/workflows/release-sync.yml`) to produce that layout; do not hand-edit `release` or `main`.

### Issues and pull requests

Every new feature goes through an issue and a pull request:

1. Before starting, search the repository's issues (`gh issue list --state all --search "<keywords>"`).
2. If no issue covers the work, open one (`gh issue create`) stating what and why, and the
   completion criterion — which tests must pass.
3. Work on a `feat/<topic>` branch, push every commit, and open a pull request into `develop`
   whose body contains `Closes #<number>`.
4. Merge into `develop` through that pull request (`gh pr merge`), not by a local merge, so the
   issue is linked.
5. Then close the issue yourself: `gh issue close <number> --comment "Landed in develop via #<PR>"`.
   GitHub's `Closes #N` only fires when a pull request merges into the default branch (`main`),
   and these pull requests merge into `develop`.

## 5. Intent → Spec → Test → Code

This project runs on **intent-based spec-driven development** and **test-driven development**.

1. `docs/INTENT.md` states what the project is for. It is the boundary. **The spec may not go
   beyond the intent.**
2. `docs/SPEC.md` states what the project does. A behaviour change starts as a spec change.
3. Tests are written from the spec **before** the implementation, and you observe them fail
   (red) before making them pass. Report the red output.
4. Code is written to make the tests pass.

If a request conflicts with `docs/INTENT.md`, say so instead of implementing it.

## 6. User-authored files are specification

Files the user wrote by hand — notebooks, example build files, sample apps — are the specification.
Read them **first**. Never delete, rewrite, or `git add` them without being told to. Generated
documentation (roadmaps, design notes) is a record of work, not a requirement; when the two
disagree, the user's file wins.

## 7. Show a conclusion before acting on it

Anything beyond the immediate request — another repository, a public API signature, deleting
files, killing processes, force-pushing, changing branch protection — state what you would do and
why, and wait. Investigating, measuring, and reporting are always fine.

**Push every commit right away.** After you commit — on a work branch or on `develop` — push it to
the remote immediately; no confirmation is needed. Never push to `main` or `release` by hand, and
never force-push without the user's explicit approval.

When a rule and backward compatibility conflict, **the rule wins.** List the callers that break and
fix them; do not keep the forbidden thing "so nothing breaks".

## 8. Verification that can fail

- Never read a build's exit code through a pipe (`| tail`, `| grep`). Redirect to a file, then read
  `$?`. A background command ending in `echo` always reports 0.
- Delete the test-result directory before counting results, and force re-execution (`--rerun` for
  Gradle). Stale XML otherwise reports an old, larger number.
- Run independent test modules as **separate** invocations. One invocation can hide an ordering
  dependency.
- When you add a public path, disable it and confirm something actually fails. If nothing fails,
  nothing uses it.
- **Do not trust an agent's report.** Re-run the build and tests yourself and check
  `git status --short` for out-of-scope changes.
- **Never `git add -A`.** Stage explicit paths. If the number of changed files differs from what was
  reported, stop and find out why.
- Measurements run alone, unfiltered, after checking `uptime`.

## 9. Reporting

Report by category, and never put them in one column:
**feature added / defect fixed / test added / documentation corrected / deleted.**
A rising test count is not progress when the tests assert an absence. Before writing "nothing left
to implement", say what you counted against.

## 10. Agents

- A headless agent (`claude -p`, `agy -p`) has **no next turn**. Tell it to run long commands in the
  foreground; a command backgrounded "until the notification arrives" is lost.
- Pass the model explicitly. Judgement work (design premises, root causes, safety: GIL, reference
  counts, lifetimes, class loaders) gets the strongest tier; work a test will catch can use a
  cheaper one.
- Give every agent prompt the absolute paths it may write to, and repeat rule 2 in it.
- **Subagents do not run heavy local builds.** Subagents write code, design, investigate, review
  and document. Gradle builds, cargo builds, the test gate and model runs are done by the session
  itself — one at a time on this machine — or by CI (GitHub Actions) on a pushed branch. Several
  sessions share one machine; parallel local builds slow every one of them.

---

# torchnative-specific rules

Sections 11 and later apply to this repository only. They replace the Korean `CLAUDE.md`, which
was deleted (issue #22, confirmed by the user on 2026-10-03; Claude Code reads `AGENTS.md`). Every
citation of it in `docs/`, code comments, tests and workflows has been rewritten to point here; the
table in [Appendix A](#appendix-a--claudemd-section-map) remains as a historical map of the old
numbers.

§22 holds the hard rules that used to be pasted into every agent prompt from `tools/agent_rules.txt`.
That file was folded into §22 and deleted (issue #43); point agents at this file instead.

## 11. What this project is

**torchnative leaves the PyTorch/`transformers` ecosystem untouched and replaces only `torch._C`
with a native Rust extension.** It is not a reimplementation. The real `transformers` is
imported, the real `from_pretrained` and `generate` run, and only the computation underneath
is ours.

```
torchnative/rust/torch_c/        the `torch._C` extension (Rust, pyo3 abi3-py313, candle-core)
  src/aten.rs          operator kernels; every op enters through one door, `_aten_dispatch`
  src/bootstrap.py     Python bootstrap baked into the extension with `include_str!`
torchnative/rust/vulkan_probe/   standalone Vulkan probe crate
torchnative/rust/wasm_probe/     standalone wasm probe crate
torchnative/python/torchnative/    the Python package (quant, export, adapt, delta, device, ...)
torchnative/python/torch/          upstream's VENDORED tree: generated, git-ignored, never edit it
tests/                 the gate (run.sh) and its suites, test_*.py
tests/golden/          value-comparison harness against upstream
tests/docwatch/        the documentation checker (DOCWATCH)
tests/bench/               measurement harnesses
scripts/vendor/        vendor_torch.sh, install_shim.sh, vendor_candle.sh, gen_surface.py, probe.py
scripts/wheel/         cross builds and wheel verification
scripts/devices/       on-device harnesses (Android parity, Intel NPU, ...)
scripts/scan/, scripts/colab/   upstream-source scanner; Colab notebook
vendor/                the candle-core / candle-metal-kernels forks and their patches
.github/scripts/       CI-only scripts; .github/scripts/release/ is release-sync
docs/<folder>/         round-by-round records of what was measured (index: docs/README.md)
```

**No root `Cargo.toml` workspace.** The crates stay independent, each with its own `Cargo.lock`,
`.cargo/config.toml` and `target/`. A workspace would move every member's build output to one root
`target/` (an unapproved root entry, §2) and break the per-crate `torchnative/rust/torch_c/target` that
`scripts/vendor/install_shim.sh`, `tests/run.sh`, `scripts/wheel/build.py` and the cross builds read;
it would replace the per-crate lock files with one, so `vulkan_probe` and `wasm_probe` would resolve
against `torch_c`'s graph; and `[patch.crates-io]` (the `vendor/candle-*` forks) and `[profile.*]` are
honoured only at a workspace root, so `torch_c`'s patch entries and `vulkan_probe`'s release profile
would be ignored. Measured on 2026-10-03 with a trial root `[workspace]` and `cargo metadata`:
`target_directory` moved from `torchnative/rust/torch_c/target` to `<root>/target`, and cargo warned
"patch for the non root package will be ignored" and "profiles for the non root package will be
ignored".

- **One door.** Every operator goes through `_aten_dispatch`. Do not build a bypass.
- **`Repr` enum** (`tensor.rs`): `Dense | Quantized | Vulkan | Complex | Meta`. `tensor()` refuses
  every non-`Dense` variant, and that refusal propagates automatically to roughly 400 call sites.
  **The point of the design is to make a silent fallback unrepresentable.** When you add a
  variant, do not use a wildcard `match` arm — a compile error there is the desired property.

## 12. Commits, merges and agents

- **Sub-agents do not commit.** `git commit`, `git push`, `git merge` and `git checkout -- <path>`
  are the coordinating session's actions only. An agent backs a file up with `cp` and leaves its
  work in place.
- **Never `git add -A`** (rule 8). It once committed a partially resolved merge **with the conflict
  markers still in it.**
- **Merges happen in the coordinating session only.** `git merge` silently refuses when there are
  uncommitted changes, so grepping its output for `CONFLICT` misses the refusal. Read the result.
- **Rebuild after every merge.** Each branch passing alone does not mean the merge passes. Real
  case: one round confirmed on its base that `zeros_like` refuses; a parallel branch added the meta
  kernel for it. Both were right, the text merged cleanly, and the resulting list was
  **behaviourally false**. It was caught only because the test counted op *names*, not a total.

## 13. The gate

```sh
PATH="$HOME/.cargo/bin:$PATH" PYTHON="$PWD/.caches/spike-venv/bin/python" \
    bash tests/run.sh > .scratch/gate.log 2>&1; echo "EXIT=$?"
```

In a worktree, `.caches/` does not exist; point `PYTHON` at the main checkout's
`/Volumes/macMini/thisisthepy/torchnative/.caches/spike-venv/bin/python`.

To include the Vulkan tests (without these they are skipped **by name**):

```sh
TORCHNATIVE_VULKAN_DYLD="$HOME/Library/Android/sdk/emulator/lib64/vulkan" \
VK_DRIVER_FILES="$HOME/Library/Android/sdk/emulator/lib64/vulkan/libkosmickrisp_icd.json"
```

**What the gate runs:** crate unit tests → every Python suite in `tests/` (43 files
at the time `CLAUDE.md` recorded the baseline) → golden self-test → the documentation checker
(DOCWATCH, over `docs/**/*.md` and `README.md`).

**Baseline (2026-09-07, `dd2e0a3`):** 1122 ok / 0 FAIL · DOCWATCH 1011/1011 · golden 11420/11420
ops=302 failed=0 pending=0 · `cargo test` 30/30 · golden self-test PASS. Always say **where** a
baseline was measured (§13.2).

- **Never read the exit code through a pipe** (rule 8). In a background run, a script ending in
  `echo` exits 0; read the `EXIT=` line from the log.
- **Adding an op to `_aten_implemented()` requires a case builder in `tests/golden/cases.py`.**
  Without one, the `golden_cases_failed eq 0` marker turns the gate red. That marker exists because
  a failing golden case rode **three commits** with the gate green — `ge` floors on *passed* and
  *total* cannot see `passed < total`.
- **DOCWATCH markers use `ge`.** `eq` on a shared global count turns red the moment a later round
  legitimately raises it. The only exceptions are the two counts that must be zero and stay zero,
  `golden_pending` and `golden_cases_failed`, where `ge` is useless (`failed ge 0` holds for every
  number). `test_release.py` enforces this.

### 13.1 A host twin is caught only by counters — and that evidence comes from Vulkan alone

Swap a kernel for a twin that computes on the host and **every value still matches**; the
agreement tests stay green and the `.device` label is unchanged. Each time, only a dispatch
counter caught it (`ran 0 compute shaders, expected exactly 1`, `{0 dispatches, 1 up, 2 down}`).

The three rounds that actually performed that tamper were all **Vulkan** (`7dff9f0`, `35e002f`,
`22d9158`), so the evidence exists **only for a backend with counters.** The experiment cannot be
run on Metal, because candle's `MetalDevice` exposes nothing countable — which is exactly why this
failure mode **cannot be excluded** on `mps`. Do not cite it as cross-backend evidence.

The count of those rounds was once inflated twice without anyone re-counting: commit `1f97c9f` said
"four" (it counted itself, and it had no counters), and a later brief said "five". **When you quote
a quotation, count the original.**

### 13.2 A worktree's gate and the main checkout's gate do not measure the same thing

Same commit, different `SKIP` counts: 24 in a worktree, 25 in the main checkout. The difference is
`test_toolguard_wheel_staging`: the main checkout holds a real cross-build artefact under
`torchnative/rust/torch_c/target/x86_64-unknown-linux-gnu/release`, so that test correctly declines to
overwrite it; a fresh worktree has no such directory, so the test runs.

- When you hand an agent a baseline, say **where** it was measured.
- Never compare a worktree count with a main-checkout count directly.
- When counts differ, diff the `SKIP` lines first — environment differences are named there.

### 13.3 One green run is not a baseline for the CoreML suites (measured 2026-09-19)

The same tree, run six times with nothing rebuilt: `ok, ok, ok, FAIL, FAIL, ok`, and the two
failures were in **different suites** (`test_coremlops`, `test_anedecode`). Both were
`MLComputePlan` declining to answer, not a verdict.

CoreML's global state **moves in response to activity**: over 91.7 minutes and 1392 observations
in fresh subprocesses there were six flips, **all inside gate runs** (one to four minutes after each
start), and zero in 30.8 quiet minutes. Independent samplers and gate fixtures agreed cell by cell,
so it is global, not per-process. Evidence: `docs/graph/NPU2.md` §9.8.

- Reading one green run as a clean baseline is wrong **about one time in three.** For a round that
  touches CoreML, two runs are not enough.
- Before calling a red run a regression, read its `PLAN` lines: they print which (op, shape,
  configuration) cell went silent, which separates external drift from a real regression.
- **Silence is not a CPU verdict.** `preferred: unknown` · `supported: []` is a refusal to answer.
  It is **deterministic per (op, shape, configuration) cell** — no cell flipped in 800
  observations — so do not treat it as a probability. Two rounds in a row assumed independence
  (across shapes, then across configurations) and both retracted.

## 14. Worktrees and the gate

### 14.1 Build the vendored tree before gating a worktree

A worktree created by `git worktree add` has **no** `torchnative/python/torch/`. Gated in that
state, more than 100 tests fail, all with `torch._C has no _aten_implemented` — **the gate becomes
meaningless.** On 2026-09-13 two rounds (ANE decode, Vulkan) judged their work on such a tree: one
reported "0 new failures" while 119 were already failing and burying any new one; the other
explained the failures away as "build order". Both were wrong.

**Before distributing work**, the coordinating session runs, in every worktree:

```sh
PYTHON=/Volumes/macMini/thisisthepy/torchnative/.caches/spike-venv/bin/python bash scripts/vendor/vendor_torch.sh
PYTHON=/Volumes/macMini/thisisthepy/torchnative/.caches/spike-venv/bin/python bash scripts/vendor/install_shim.sh
```

Do not delegate this to an agent: the permission classifier sometimes blocks it, and then an agent
that follows the rules stops and reports while one that does not works around it. Neither is the
outcome you want.

### 14.2 An agent's completion notice does not mean its worktree is quiet

The `Agent` tool's completion notice means **that agent's turn ended**, not that nothing is running
in its worktree; the agent may still be waiting on a gate it started.

On 2026-09-17 the coordinating session took such a notice as "finished" and ran `git stash`,
`merge`, `pop`, a rebuild and a gate in the same worktree. The other gate died at **944 ok** with
`cat: test_shim.py.log: No such file` — a number that looks like a partial pass, which made it
worse — and the two gates shared one stage directory, so the later run's `rm -rf "$suite_logs"`
deleted the earlier run's logs while it was running.

Before touching a worktree, check that no gate is running in it:

```sh
ps -eo command | grep '[r]un\.sh' | grep <worktree>
```

The tooling has since been fixed — the gate takes a stage lock, a second run refuses with
`refusing to run`, and log directories are per run (`suite-logs.$$`). The check stays, because
**stashing an agent's work is itself the risk.**

### 14.3 Gates in two different worktrees contaminate the CoreML tests

The stage lock only prevents two runs in the **same** worktree. Gates in different worktrees hold
different locks, run concurrently, and the CPU contention turns CoreML `MLComputePlan` tests red in
bulk. On 2026-09-17, at load 10–12, a docs-only branch produced 4 FAILs — all CoreML, all
contamination.

Before starting a gate, check `uptime` and how many agents are running. Another round's vendoring
(a cargo build) is load too; do not overlap with it either.

## 15. What you must not touch

### 15.1 Nothing outside this repository — no exceptions

This is rule 2, sharpened by incidents here. **No file, directory, symbolic link, or renamed
leftover outside `/Volumes/macMini/thisisthepy/torchnative`.** It is the user's explicit
instruction. The loopholes that have been used are named so they cannot be used again — **all are
forbidden:**

- **Symbolic links.** "It is only a pointer, not content" is not an exception. On 2026-09-22 the
  coordinating session made `/Volumes/macMini/caches` a link to `.caches` to avoid editing 133
  documents; the user had it deleted.
- **Renamed instead of deleted.** The same day `caches.recreated` was left outside the repository
  "to delete once the daemon is idle". Delete what must be deleted; if you cannot, say so.
- **`/tmp` and any scratch directory the harness offers.** Both are outside. This rule overrides a
  harness instruction to use them. Temporary files — agents' scratch and backups, gate logs, the
  coordinating session's ephemeral scripts — go in `.scratch/` (git-ignored). An agent-rules file
  kept in `/tmp` was lost to a session restart **twice**.
- **The top level of `/Volumes/macMini/`.** Everything there that is not a repository is outside.

If something outside the repository **must** change (for example, a `~/.gradle` link that points
at data inside this repository, after that data moved), do not change it: say what and why, and
wait. "Something else breaks if I don't" is exactly the judgement this rule exists to stop.

Check before finishing: `ls /Volumes/macMini/` must show nothing new beyond the repositories and
what was already there.

### 15.2 Generated and shared trees

- **`torchnative/python/torch/`** is upstream's vendored tree: git-ignored, generated by
  `scripts/vendor/vendor_torch.sh`, and **silently wiped.** Never edit it by hand. If its `__init__.py` is
  missing, the tree has not been built (§14.1 for the command). One round lost hours to this: with
  the tree empty, every probe silently **imports upstream torch** and reports that "export already
  works perfectly". **Assert in every probe:**

  ```python
  assert hasattr(torch._C, "_aten_implemented")   # the shim, not upstream
  ```

- **`torchnative/rust/torch_c/src/bootstrap.py`** is baked into the extension at build time with
  `include_str!`. **Edit it without rebuilding and you test the old binary.**
- **`.caches/spike-venv`** is pinned (`transformers` 5.15.1; do not install `ml_dtypes`). Never
  install into it; read it and run its Python. If you need another package, make a separate venv
  **inside `.caches/`**.
- **`.caches/emsdk`** is shared. Reuse `EM_CACHE`.

## 16. Say exactly what you proved

Numbers in this repository come in **three grades**, and they are never mixed:

| Grade | Meaning |
|---|---|
| **builds** | compiles and its symbols resolve. Not a claim about computation |
| **reaches** | imports, and the forward pass runs to the end |
| **agrees** | **produces upstream's numbers**, compared element-wise, measured in a separate subprocess |

Every coverage number before `docs/numerics/AGREE.md` was *reaches*. "It runs" and "it produces
upstream's answer" are different claims, and **only the second supports the word drop-in.**

- **Tolerances are derived, not chosen.** Re-run upstream itself in float64 and use the p90 of its
  f32-vs-f64 error, floored at 8 ulp. **Re-derive per model**: BigVGAN's self-oracle error was 57×
  the population constant, and the constant would have reported a **48× divergence that did not
  exist.** Never widen a tolerance to make something pass; change the arithmetic instead.
- **"It ran on the GPU/NPU" must be asserted at runtime**, never inferred from a correct answer.
  The CoreML round learned that **its models were running on the CPU** only by reading
  `MLComputePlan`'s per-op answer — the results were right either way. The Intel and Qualcomm stacks
  fall back silently too. `docs/devices/MPSATTN.md` §3.1 records how source-scan evidence can be
  defeated; use evidence that is not a description of the source, such as counters.
- **Dispatch counters are the only instrument that detects a silent host fallback**
  (`_cuda_counters()`, `_vulkan_counters()`, `_metal_counters()`). Assert on them and state the
  bracket you measured — counters from different brackets read like regressions side by side.
- **One cell shape cannot speak for a column.** `clamp` was wrong on Metal from the day `mps`
  landed while every table graded it *agrees*, because the cell was one shape and that shape had no
  NaN.
- **Measurements run alone** (rule 8). On a loaded machine one commit read anywhere from 672 to
  1076 ns. Check `uptime` first and do not filter the suite; filtering changes what is measured.

## 17. Check periodically that you are doing what was asked

These subsections are numbered to match the old `CLAUDE.md` §5.1–§5.7 exactly (§17.*n* = old
§5.*n*), because those numbers were cited throughout the repository and still appear in history. Do not renumber them.

Individually reasonable steps have repeatedly summed to something the instruction did not ask for.
What follows is a checklist, not a confession.

### 17.1 The specification is what the user gave, not a document

What the user handed over directly — scripts, notebooks, example projects — is the specification
(rule 6). `docs/` and roadmaps are **outputs of the work, not requirements.** When they disagree,
**the specification wins.**

Before starting, find and read the user's files. If there are none, ask where the specification is.
Do not answer from a directory listing: seeing only `coreml.py` and `nnapi.py` in `export/`, a round
answered "Intel is not started" — but Intel was recorded as a design precedent in
`docs/graph/QUANT2.md` §3.

### 17.2 When the user states a structure, it is a constraint, not a suggestion

Before choosing a design, check explicitly that **the user's structure and yours are compatible.**
If you believe your way is better, **say so instead of implementing it.**

### 17.3 A test count is not progress

Report split into: feature added / defect fixed / test added / documentation corrected / deleted
(rule 9). One round added ten kernels and moved the architecture pass count **from 0 to 0.** Another
round's 71 tests all asserted an *absence* and added no capability. **Report "N of M", not "N
added".**

### 17.4 The criterion you write decides the answer

When you write an audit or investigation prompt, add one line: **what this criterion cannot find.**
When an agent's conclusion matches your hypothesis, **be more suspicious, not less** — you supplied
the criterion. Ask in terms of capability: "what now provides what this file used to provide?"

Coverage lists are this trap. All 29 ops BigVGAN dispatches were in `_aten_implemented()`, so the
first answer was "no gaps" — but the **meta kernels** for `sum`/`view` were missing and the model
stopped. The list answered "does a kernel exist"; the question was "does this model run".

### 17.5 A check that cannot fail is not a check

**The most-cited rule in this repository.** It recurred:

- `cargo test` was not in the gate, so **31 tests ran for nobody**, and one of them was red on its
  own alignment assertion **without telling anyone.**
- The golden self-test and the documentation checker were outside the gate for the same reason.
- A golden case failed for three commits, **invisible because the only markers were two `ge`
  floors.**
- A tolerance guard was `assert tol == FACTOR * oracle` — **a tautology restating tol's
  definition**, green even with the factor raised 100×.
- A queue-drain guard existed **in two places that masked each other**, so neither could be
  nullified alone.
- 21 fixtures used raw addresses instead of the public API, so **emptying that API entirely still
  passed.**

**Nullify everything.** Break the implementation on purpose and confirm the test goes red. **A
nullification that is not caught is a more valuable finding than a feature** — always report it.
Build mutants through `scripts/vendor/install_shim.sh` so they reach the vendored tree and not only the
stage; a subprocess probe reading the unmutated `.so` produces the inverse misdiagnosis.

**Do not use a model run as the only check.** A model exercises a fraction of the kernels — one
nullification turned the kernel tests red while the **model replay stayed green.**

Three blind spots of the finite-difference (FD) oracle are recorded: on real models it disagrees
with upstream autograd by 600×, it misses training-mode batch-norm defects, and **it differentiates
one input only, so it never sees weight gradients.**

### 17.6 Backward compatibility ranks below the rules

When a rule and backward compatibility conflict, **the rule wins** (rule 7). List the callers that
break and fix them. If you catch yourself thinking "to avoid breaking anything, X must stay", first
check **whether X is the forbidden thing.**

### 17.7 Show a conclusion before acting on it

Anything beyond the immediate request — another repository, an API signature change, deleting
files, killing processes, **a PyPI upload** — state what and why, and wait (rule 7). Investigation,
measurement, reporting, and landing verified work inside the assigned scope proceed as normal.

**Never upload to PyPI without the user's explicit approval.** Credentials live in `~/.pypirc` /
`TWINE_PASSWORD`; **never read or print them.**

## 18. A refusal names itself

What is not supported **refuses with its own name and the reason.** Never invent a plausible
answer.

**A promised refusal is worse than no refusal.** Release notes said four collectives "refuse by
name"; in fact they never refused and **each rank silently returned its own input.** Readers had
been told there was nothing to check.

**Do not weaken a refusal to make progress.** If you cannot proceed, report the name of the wall.

**Re-measure before quoting a gap statement.** Four of the gap statements in a release note's §5
were already false, each wrong in a different way: one false when written, one two half-claims
wrong in opposite directions, one contradicting its own paragraph, one true but misleading.

## 19. Parallel work and devices

- **Worktrees live inside this repository** at `.worktrees/<name>` (git-ignored):
  `git worktree add -b feat/<name> .worktrees/tn-<name> develop`. Then §14.1. The repository itself
  sits on the external volume (`/Volumes/macMini/thisisthepy/torchnative`), so this still avoids the
  internal SSD, and nothing is scattered outside the repository (§15.1).
- **Agent rules live in this file** (§22 and the sections it names), a repository-relative path, so
  every worktree has them automatically. Never keep them in `/tmp` or elsewhere outside the
  repository.
- `build/` and `target/` only grow; delete them periodically.
- **Concurrency limits** (8 cores · 16 GB): 3–4 agents that build or test, plus 3–4 that only read
  or analyse.
- **Headless agents have no next turn** (rule 10). Put this in the prompt: *"Run long commands in
  the foreground. Do not background a command and wait — you have no next turn to wake up in."*

| Device | Notes |
|---|---|
| Android emulators 5554 / 5556 | **Shared with other projects.** `ANDROID_SERIAL` is mandatory; work only inside `/data/local/tmp/bw_*`; never install apps |
| Physical device (Galaxy Tab S9 Ultra, SM8550 / Snapdragon 8 Gen 2, API 36) | Has a Hexagon NPU. Wireless ADB |
| Intel NPU laptop (Windows) | The user runs it. Provide numbered step-by-step procedures |
| NVIDIA GPU | The user's Windows machine, or Colab (`colab` CLI is logged in) |

**Never stop what you did not start; always stop what you did start.** Never use an unscoped
`pkill`. `adb` is global through `~/.zshenv`; if a non-interactive shell cannot find it,
`export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"`.

## 20. Accelerator architecture

**The front end is fixed; only the back end is swapped.**

```
front end (fixed)   real transformers · from_pretrained · generate
                    └ torchnative swaps leaves / submodules in place
back end (swapped)  Apple → CoreML  |  Android → ExecuTorch / QNN  |  Windows → Intel (OpenVINO)
```

A `.pte` or a vendor blob is **an implementation detail hidden behind a module**, never an object
the user holds. `docs/graph/QUANT2.md` §3 defines `quantize_` in exactly this module-replacement
shape. **Choosing a back end is a question of which one reaches the vendor NPU best, never a
question of API shape.**

An NPU does not take ops one at a time; it takes **whole subgraphs.** That is why the capture layer
(`decompose.py` → `refold.py`) comes first and the delegates are its consumers.

**Devices come in two kinds.** Those candle already has a backend for (Metal, CUDA) are variants of
its `Device` enum and need **no `Repr` arm, no dispatcher arm and no kernel of ours** — one line in
`PyDevice::resolve()`. Those it lacks (Vulkan) need everything written, from shaders to the upload
path. **Check which kind it is before adding a `Repr` arm.**

## 21. Known constraints

- **`torch.compile` is refused permanently.** abi3 blocks PEP 523 (`docs/graph/COMPILE.md`, which
  also records the correction that abi3 blocks the frame hook, not Dynamo itself).
- **NNAPI is deprecated in Android 15.** Its successor is **per-vendor LiteRT delegates**, not one
  OS API. `nnapi.py` still works; do not delete it.
- **No `timeout` command** (coreutils is not installed). Use each tool's own timeout.
- **`find` silently returns nothing for relative times** (`-newermt '-15 minutes'`). Use `stat`.
- **`cargo` is not on the non-interactive `PATH`:** `export PATH="$HOME/.cargo/bin:$PATH"`. An exit
  127 inside a build script leaves a stale artefact, so the next test measures the wrong build.
- **Docker** is at `~/.docker/bin/docker` and its daemon is **aarch64 Linux**: manylinux aarch64
  builds and runs natively. `linux/amd64` is QEMU — **never measure on it.**
- In zsh, `$path` is tied to `$PATH`. Never name a loop variable `path`; reading into it destroys
  `PATH` and every later command fails with "command not found".

## 22. Rules for agent rounds

`tools/agent_rules.txt` used to be pasted into every agent prompt as a condensed copy of §12, §15,
§16, §17.5 and §21 plus the rules below. It was folded into this section and deleted (issue #43), so
there is one copy. **Point every agent prompt at this file** — "read AGENTS.md first and follow it;
§2, §12, §15, §16, §17.5, §21 and §22 are hard rules, and violating any invalidates the round" — and
repeat rule 2 and the absolute paths it may write to (rule 10). What follows is what that file
said that is not already stated above.

- **Backups:** before editing a file you may need to restore, `cp` it into `.scratch/` of the main
  checkout, not into the worktree (where it shows up in `git status`) and never outside the
  repository. Restoring is `cp` back, not `git checkout -- <path>` (§12).

- **Never suppress stderr on a command whose success you rely on.** `2>/dev/null` once hid a
  failing `rm`, and the coordinating session read the failure as success.
- **Liveness checks:** `pgrep -f "tests/run.sh"` matches your own polling loop; use
  `kill -0 <captured pid>`. A wait loop must match **both** `tests/run.sh` and
  `docwatch/check_docs.py` — `run.sh` `exec`s into the checker at the end, so matching `run.sh`
  alone reports "clear" while the gate is still running (DOCWATCH alone has taken 14+ minutes).
- **No scratch files in the worktree root.** Files named `test_*.py` are collected as gate suites.
  Keep backups and scratch in `.scratch/`.
- **Stay in your territory.** Other agents work in other worktrees.
- **TDD, with proof.** Write the failing test before the implementation. A round reporting "tests
  added: 0" has not met the bar unless it is a measurement-only round that says so.
  - An assertion-free test is worse than no test. Never make a test pass by changing what it tests.
  - A suite with no `if __name__ == "__main__":` block defines tests and calls none; the gate counts
    it as passing and the ok count sits exactly at baseline. **Confirm the ok count rises by the
    number of tests you wrote.**
  - A test defined below `raise SystemExit(_main())` is never collected. That hid a headline claim
    for a week.
- **End every report with `git status --short`, verbatim**, and state plainly what you did not
  finish, and any count that did not move.

---

## Appendix A — `CLAUDE.md` section map

`CLAUDE.md` (Korean) was deleted on 2026-10-03 (issue #22). It had been cited about 330 times across
`docs/`, code comments, tests and workflows, mostly as `CLAUDE.md §5.5` and `CLAUDE.md §5.3`. Those
citations were rewritten through this table; it is kept as a historical map, so that an old quotation
of a `CLAUDE.md` section number (in a commit message, an archived log) can still be resolved.

| `CLAUDE.md` | `AGENTS.md` |
|---|---|
| §0 이 프로젝트가 무엇인가 | §11 |
| §1 커밋 규정 | §1 (no attribution), §12 (who commits), rule 8 (`git add -A`) |
| §2 게이트 | §13 |
| §2 › 호스트 쌍둥이는 … 카운터로만 잡힌다 | §13.1 |
| §2 › 워크트리의 게이트와 메인 저장소의 게이트는 … | §13.2 |
| §2 › 초록 한 번은 CoreML 스위트에서 기준선이 아니다 | §13.3 |
| §3 건드리면 안 되는 것 | §15 |
| §3.0 이 저장소 밖에는 아무것도 만들지 않는다 | §15.1 (and rule 2) |
| §3 (vendored tree, `bootstrap.py`, `.caches`) | §15.2 |
| §4 무엇을 증명했는지 정확히 말한다 | §16 |
| §5 지시받은 것을 하고 있는지 주기적으로 확인한다 | §17 |
| §5.1 사양은 문서가 아니라 사용자가 준 것이다 | §17.1 |
| §5.2 사용자가 구조를 말하면 그것이 제약이지 제안이 아니다 | §17.2 |
| §5.3 테스트 수는 진척이 아니다 | §17.3 |
| §5.4 내가 쓴 기준이 답을 정해버린다 | §17.4 |
| §5.5 실패할 수 없는 검증은 검증이 아니다 | §17.5 |
| §5.6 하위 호환은 규칙보다 아래다 | §17.6 |
| §5.7 결론을 실행에 옮기기 전에 보여준다 | §17.7 |
| §6 거부는 이름을 댄다 | §18 |
| §7 병렬 실행과 기기 | §19 (merging: §12) |
| §8 가속기 구조 | §20 |
| §9 알려진 제약 | §21 |
| (python-multiplatform) worktree 를 만들면 벤더 트리부터 세운다 | §14.1 |
| (python-multiplatform) 에이전트 완료 알림은 … 조용하다는 뜻이 아니다 | §14.2 |
| (python-multiplatform) 게이트를 두 worktree 에서 동시에 돌리면 … | §14.3 |
| `tools/agent_rules.txt` (folded into §22 and deleted, #43) | §12, §15, §16, §17.5, §21, §22 |

Mapping rule that was applied: `CLAUDE.md §5.N` → `AGENTS.md §17.N`; `CLAUDE.md §N` for N ≠ 5 → the
row above. A bare "CLAUDE.md 5.5" (no `§`) was the same citation.
