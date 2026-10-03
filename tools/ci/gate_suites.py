"""Which gate suite runs on which CI runner -- and which stays local-only.

Issue #24. `.github/workflows/gate.yml` runs `rust/torch_c/pytests/run.sh` on
GitHub-hosted runners with `TORCHNATIVE_GATE_RUNNER` set to one of `RUNNERS`;
`run.sh` hands it to `suite_ledger.py --runner`, which reads this file.

**The rule this file exists for** (AGENTS.md §17.5): the suites run on CI plus
the suites left local-only must equal the full gate. So every `test_*.py` in
`rust/torch_c/pytests/` is named in `SUITES` -- explicitly, not by a default --
and `check()` refuses a suite file that is missing here, or a name here that
is not a file. That refusal fires in three places:

  * `test_cigate.py`, which is itself a gate suite, so a round that adds a
    suite without classifying it goes red **locally**, before any push;
  * `suite_ledger.py --runner`, before a single suite runs on CI;
  * `gate_suites.py report`, which builds the PR summary.

A suite planned for a runner is run there whole. A suite **not** planned for a
runner is not executed there; the ledger prints

    SKIP <suite>: (whole suite) -- not run on CI runner '<runner>': <reason>

so it is counted in the ledger's SKIP column and named, the same shape as
`_skip.py`'s per-test lines. Inside a suite that does run, tests that need a
device the runner lacks skip by name through the suites' own `_skip` /
`vulkan_coverage` helpers, as they already do on a Mac without Vulkan.

How a suite earns `LOCAL_ONLY` here: it would either FAIL on a hosted runner
for a reason that is about the runner and not the code, or -- worse -- print
`ok` having tested nothing, because its skip is a hand-rolled `print` that the
ledger cannot see (the `_skip.py` docstring has the defect). The reasons are
in `WHY_NOT`, and the report copies them onto the PR.

What this classification cannot see: whether a suite planned for a runner
actually works there. Before #24 nobody had run this gate on Linux, so the
plan below is a static reading of the suites, not a measurement. The first CI
run is the measurement. A suite that fails there for a platform reason moves
to a narrower runner tuple **with a reason**; it is never deleted from
`SUITES`.

Usage:
    python tools/ci/gate_suites.py check  [--pytests DIR]
    python tools/ci/gate_suites.py list
    python tools/ci/gate_suites.py report --log linux=PATH --log macos=PATH \
        [--out SUMMARY.md]
"""

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PYTESTS = REPO / "rust" / "torch_c" / "pytests"

#: The CI runners a gate job can name in `TORCHNATIVE_GATE_RUNNER`.
RUNNERS = {
    "linux": "GitHub-hosted ubuntu-24.04 (x86_64). No GPU, no Apple "
             "frameworks, no Vulkan driver, no adb, no Intel NPU.",
    "macos": "GitHub-hosted macos-15 (arm64, a VM). Builds the Apple "
             "variant of the shim (Accelerate + Metal). No Neural Engine; "
             "GitHub documents MPS as unsupported in these VMs -- what Metal "
             "itself does there is printed by the job's probe step.",
}

BOTH = ("linux", "macos")
LOCAL_ONLY = ()

_ANE_SILENT = (
    "needs CoreML's MLComputePlan answering for the Apple Neural Engine "
    "(AGENTS.md §13.3), which no hosted runner has; and its fixture reports a "
    "missing coremltools with a hand-rolled print followed by `ok`, so on a "
    "runner it would print `ok` having run nothing. Local-only until its skips "
    "go through `_skip`, and even then it would only skip.")

_DEV_TOOLCHAIN = (
    "needs the release machine's toolchain, which a hosted runner does not "
    "have: the pinned interpreter at .caches/spike-venv (transformers 5.15.1, "
    "AGENTS.md §15.2), the rust cross targets and the target CPython "
    "distributions under tools/wheel.")

#: Every gate suite, and the runners it runs on. `()` means local-only.
SUITES = {
    "test_absmps.py": BOTH,
    "test_agree.py": BOTH,
    "test_agree2.py": BOTH,
    "test_aliasinc.py": BOTH,
    "test_ane_subgraph_components.py": LOCAL_ONLY,
    "test_anedecode.py": LOCAL_ONLY,
    "test_anepath.py": BOTH,
    "test_anesubgraph.py": LOCAL_ONLY,
    "test_anetracer.py": LOCAL_ONLY,
    "test_argform.py": BOTH,
    "test_asyncwork.py": BOTH,
    "test_benchguard_ab_upstream.py": BOTH,
    "test_bf16ane.py": BOTH,
    "test_bfallback.py": BOTH,
    "test_bind2.py": BOTH,
    "test_bind3.py": BOTH,
    "test_bind4.py": BOTH,
    "test_bind5.py": BOTH,
    "test_bindings.py": BOTH,
    "test_bkeyset.py": BOTH,
    "test_canine.py": BOTH,
    "test_cigate.py": BOTH,
    "test_cipub.py": BOTH,
    "test_collect2.py": BOTH,
    "test_complex.py": BOTH,
    "test_constset.py": BOTH,
    "test_convbackend.py": BOTH,
    "test_coremlops.py": LOCAL_ONLY,
    "test_cplx2.py": BOTH,
    "test_cuda.py": BOTH,
    "test_devicens.py": BOTH,
    "test_dispatch.py": BOTH,
    "test_docrefs.py": BOTH,
    "test_dtmdev.py": BOTH,
    "test_dtypedev.py": BOTH,
    "test_emptyplan.py": BOTH,
    "test_equal_allclose.py": BOTH,
    "test_export.py": BOTH,
    "test_export4.py": BOTH,
    "test_export5.py": BOTH,
    "test_export6.py": BOTH,
    "test_fft.py": BOTH,
    "test_gatelock.py": BOTH,
    "test_gemmint.py": BOTH,
    "test_gloopin.py": BOTH,
    "test_glu.py": BOTH,
    "test_higgs.py": BOTH,
    "test_i8mps.py": BOTH,
    "test_indexsel.py": BOTH,
    "test_int8.py": BOTH,
    "test_intelnpu.py": BOTH,
    "test_intmps.py": BOTH,
    "test_last7.py": BOTH,
    "test_liftfresh.py": BOTH,
    "test_metaemb.py": BOTH,
    "test_metafam.py": BOTH,
    "test_metakey.py": BOTH,
    "test_metalcount.py": BOTH,
    "test_metalplace.py": BOTH,
    "test_metastride.py": BOTH,
    "test_methodspell.py": BOTH,
    "test_mpsattn.py": BOTH,
    "test_mpsconst.py": BOTH,
    "test_mpsfwd.py": BOTH,
    "test_mpsinplace.py": BOTH,
    "test_mpsrefuse.py": BOTH,
    "test_npmarshal.py": BOTH,
    "test_npu2.py": BOTH,
    "test_npublob.py": BOTH,
    "test_npudim.py": BOTH,
    "test_npufuse.py": BOTH,
    "test_npuvendor.py": BOTH,
    "test_npuwire.py": BOTH,
    "test_ovcache.py": BOTH,
    "test_ovpar.py": BOTH,
    "test_pad.py": BOTH,
    "test_platver_harnesses.py": BOTH,
    "test_publish.py": BOTH,
    "test_qnn_ops.py": BOTH,
    "test_qnn_plan.py": BOTH,
    "test_qnn.py": BOTH,
    "test_qnnci.py": BOTH,
    "test_qnnprobe.py": BOTH,
    "test_qwen3.py": BOTH,
    "test_release.py": BOTH,
    "test_remeasure2.py": BOTH,
    "test_repeat.py": BOTH,
    "test_rnn.py": BOTH,
    "test_scatter.py": BOTH,
    "test_setitem.py": BOTH,
    "test_shim.py": BOTH,
    "test_skipvis.py": BOTH,
    "test_split_probe.py": BOTH,
    "test_stage0.py": BOTH,
    "test_stagetype.py": BOTH,
    "test_stream.py": BOTH,
    "test_strided.py": BOTH,
    "test_structseq.py": BOTH,
    "test_tail1.py": BOTH,
    "test_tail2.py": BOTH,
    "test_tail3.py": BOTH,
    "test_tail4.py": BOTH,
    "test_tmpleak.py": BOTH,
    "test_tnnamespace.py": BOTH,
    "test_tntransformers.py": BOTH,
    "test_toolguard_run_preflight.py": LOCAL_ONLY,
    "test_toolguard_wheel_staging.py": LOCAL_ONLY,
    "test_train.py": BOTH,
    "test_varmean.py": BOTH,
    "test_viewdtype.py": BOTH,
    "test_vmap.py": BOTH,
    "test_voice3.py": BOTH,
    "test_voice4.py": BOTH,
    "test_vulkan4.py": BOTH,
    "test_vulkancov.py": BOTH,
    "test_wheelmatrix.py": LOCAL_ONLY,
}

#: Why a suite is kept off a runner. Required for every suite whose tuple is
#: not every runner, and forbidden for one that runs everywhere.
WHY_NOT = {
    "test_coremlops.py": _ANE_SILENT,
    "test_anedecode.py": _ANE_SILENT,
    "test_anesubgraph.py": _ANE_SILENT,
    "test_anetracer.py": _ANE_SILENT,
    "test_ane_subgraph_components.py": (
        "imports coremltools at module scope, so without it the suite dies "
        "at import, and what it measures is subgraph placement on the Apple "
        "Neural Engine, which no hosted runner has. It also has no `_skip` "
        "runner of its own."),
    "test_toolguard_run_preflight.py": _DEV_TOOLCHAIN + (
        " Measured on run 37078300696 (macos): 3 of 3 tests FAIL with "
        "`known-good interpreter missing: .../.caches/spike-venv/bin/python`."),
    "test_toolguard_wheel_staging.py": _DEV_TOOLCHAIN + (
        " Measured on run 37078300696 (macos): its one test FAILs with "
        "FileNotFoundError on the same interpreter."),
    "test_wheelmatrix.py": _DEV_TOOLCHAIN + (
        " Measured on run 37078300696 (macos): 20 of 22 ok; "
        "`test_every_target_names_a_rust_target_that_is_installed` needs the "
        "nine cross targets the release machine has, and "
        "`test_build_pys_own_self_test_passes` asserts `LINUX SELF-TEST: PASS`, "
        "which build.py prints only where the target CPython distribution "
        "exists (it skips loudly otherwise). The 20 lose CI coverage until "
        "those two take the runner's toolchain into account; follow-up."),
}


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------

def check(on_disk):
    """Problems with the manifest against the suite files present, by name."""
    problems = []
    disk = set(on_disk)
    named = set(SUITES)
    for name in sorted(disk - named):
        problems.append(
            f"{name} is a gate suite in neither the CI list nor the local-only "
            f"list -- classify it in tools/ci/gate_suites.py SUITES")
    for name in sorted(named - disk):
        problems.append(
            f"{name} is named in tools/ci/gate_suites.py but is not a file in "
            f"rust/torch_c/pytests/ -- remove it or restore the suite")
    for name, runners in sorted(SUITES.items()):
        unknown = sorted(set(runners) - set(RUNNERS))
        if unknown:
            problems.append(f"{name} names unknown runner(s) {unknown}")
        if set(runners) != set(RUNNERS) and not WHY_NOT.get(name, "").strip():
            problems.append(f"{name} is kept off a runner with no reason in WHY_NOT")
    return problems


def ci_suites():
    """Suites at least one CI runner executes."""
    return sorted(n for n, r in SUITES.items() if r)


def local_only():
    """Suites no CI runner executes: they are verified only by the local gate."""
    return sorted(n for n, r in SUITES.items() if not r)


def plan(runner, on_disk):
    """`(run, skipped, problems)` for one runner.

    `skipped` maps each suite this runner does not execute to the reason. An
    unknown runner raises: a typo in the workflow must not plan for nothing.
    """
    if runner not in RUNNERS:
        raise ValueError(
            f"unknown CI runner {runner!r}; tools/ci/gate_suites.py knows "
            f"{sorted(RUNNERS)}")
    problems = check(on_disk)
    run, skipped = [], {}
    for name in sorted(on_disk):
        runners = SUITES.get(name)
        if runners is None:
            continue
        if runner in runners:
            run.append(name)
        else:
            skipped[name] = WHY_NOT.get(name, "no reason recorded")
    return run, skipped, problems


def skip_line(name, runner, why):
    """The ledger line for a suite not executed on `runner`."""
    stem = name[:-3] if name.endswith(".py") else name
    return f"SKIP {stem}: (whole suite) -- not run on CI runner '{runner}': {why}"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

_HEADER = re.compile(r"^--- (test_\w+\.py) ---$")
_WHOLE = "(whole suite) -- not run on CI runner"


def classify_log(text, on_disk):
    """Per suite: counts and a verdict, read from one runner's gate log.

    PASS        ran, every test ok
    PARTIAL     ran, some tests ok and some skipped by name
    UNVERIFIED  ran, but every test skipped: it verified nothing here
    NOT RUN     not executed on this runner (local-only here)
    FAIL        a test failed
    EMPTY       ran and printed no ok/FAIL/SKIP line at all
    MISSING     the log has no section for it
    """
    sections, current = {}, None
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            current = m.group(1)
            sections[current] = []
            continue
        if line.startswith("SUITE LEDGER:") or line.startswith("CI PLAN:"):
            current = None
        if current is not None:
            sections[current].append(line)
    rows = {}
    for name in on_disk:
        body = sections.get(name)
        if body is None:
            rows[name] = {"verdict": "MISSING", "ok": 0, "fail": 0, "skip": 0}
            continue
        ok = sum(1 for ln in body if re.match(r"^ok\s", ln))
        fail = sum(1 for ln in body if re.match(r"^FAIL\s", ln))
        skips = [ln for ln in body if re.match(r"^SKIP\s", ln)]
        if any(_WHOLE in ln for ln in skips):
            verdict = "NOT RUN"
        elif fail:
            verdict = "FAIL"
        elif ok and skips:
            verdict = "PARTIAL"
        elif ok:
            verdict = "PASS"
        elif skips:
            verdict = "UNVERIFIED"
        else:
            verdict = "EMPTY"
        rows[name] = {"verdict": verdict, "ok": ok, "fail": fail,
                      "skip": len(skips), "skips": skips}
    return rows


def _grab(text, prefix):
    lines = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    return lines[-1] if lines else None


def render_report(logs, on_disk):
    """`(markdown, ok)`. `logs` maps runner -> log text, or None if absent.

    `ok` is False when the partition is broken or a runner's log is missing
    or lacks a suite: those are the ways a subset reads as the full gate.
    Test failures are the gate jobs' verdict, and are shown, not re-judged.
    """
    ok = True
    out = ["<!-- torchnative-gate-report -->", "## Gate on CI (issue #24)", ""]
    problems = check(on_disk)
    ci, local = ci_suites(), local_only()
    out.append(
        f"**Full gate: {len(on_disk)} suites** = {len(ci)} run on at least one "
        f"CI runner + {len(local)} local-only. The local gate "
        f"(`rust/torch_c/pytests/run.sh` on the Mac) is still the only run "
        f"that covers all {len(on_disk)}.")
    out.append("")
    if problems:
        ok = False
        out.append("### PARTITION BROKEN -- a suite is in neither list")
        out += [f"- {p}" for p in problems]
        out.append("")

    all_rows = {}
    out.append("| runner | exit | ran | not run here | PASS | PARTIAL | "
               "UNVERIFIED (all skipped) | FAIL | missing/empty | ledger |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for runner in RUNNERS:
        text = logs.get(runner)
        if text is None:
            ok = False
            out.append(f"| {runner} | -- | no log | | | | | | | the gate job "
                       f"produced no log: nothing on this runner is verified |")
            continue
        rows = classify_log(text, on_disk)
        all_rows[runner] = rows
        n = {v: sum(1 for r in rows.values() if r["verdict"] == v)
             for v in ("PASS", "PARTIAL", "UNVERIFIED", "NOT RUN", "FAIL",
                       "EMPTY", "MISSING")}
        if n["MISSING"]:
            ok = False
        exit_line = _grab(text, "EXIT=") or "EXIT=?"
        ledger = _grab(text, "SUITE LEDGER:") or "no SUITE LEDGER line"
        ran = len(on_disk) - n["NOT RUN"] - n["MISSING"]
        out.append(
            f"| {runner} | {exit_line[5:]} | {ran} | {n['NOT RUN']} | "
            f"{n['PASS']} | {n['PARTIAL']} | {n['UNVERIFIED']} | {n['FAIL']} | "
            f"{n['MISSING'] + n['EMPTY']} | `{ledger[len('SUITE LEDGER: '):][:60]}` |")
    out.append("")
    for runner in all_rows:
        text = logs[runner]
        for prefix in ("CI PLAN:", "VULKAN COVERAGE:", "DOCWATCH:"):
            line = _grab(text, prefix)
            out.append(f"- **{runner}** `{line}`" if line else
                       f"- **{runner}** no `{prefix}` line (the gate stopped before it)")
    out.append("")

    out.append(f"### Local-only: verified by no CI runner ({len(local)})")
    out.append("")
    for name in local:
        out.append(f"- `{name}` -- {WHY_NOT.get(name, '')}")
    out.append("")

    attention = []
    for runner, rows in all_rows.items():
        for name, r in sorted(rows.items()):
            if r["verdict"] in ("FAIL", "UNVERIFIED", "EMPTY", "MISSING"):
                attention.append(f"| `{name}` | {runner} | {r['verdict']} | "
                                 f"{r['ok']} | {r['fail']} | {r['skip']} |")
    out.append("### Suites that verified nothing, or failed, on a runner")
    out.append("")
    if attention:
        out.append("| suite | runner | verdict | ok | FAIL | SKIP |")
        out.append("|---|---|---|---|---|---|")
        out += attention
    else:
        out.append("None.")
    out.append("")

    out.append("<details><summary>Every suite, every runner</summary>")
    out.append("")
    out.append("| suite | " + " | ".join(RUNNERS) + " |")
    out.append("|---|" + "---|" * len(RUNNERS))
    for name in on_disk:
        cells = []
        for runner in RUNNERS:
            r = all_rows.get(runner, {}).get(name)
            if r is None:
                cells.append("no log")
            else:
                cells.append(f"{r['verdict']} ({r['ok']} ok / {r['fail']} FAIL / "
                             f"{r['skip']} SKIP)")
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")
    out.append("")
    out.append("</details>")
    out.append("")
    out.append(
        "A SKIP is unverified on that runner, never passed. UNVERIFIED means the "
        "suite ran and every one of its tests skipped by name.")
    return "\n".join(out) + "\n", ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _disk(pytests):
    return sorted(p.name for p in pathlib.Path(pytests).glob("test_*.py"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--pytests", default=str(PYTESTS))
    sub.add_parser("list")
    r = sub.add_parser("report")
    r.add_argument("--pytests", default=str(PYTESTS))
    r.add_argument("--log", action="append", default=[], metavar="RUNNER=PATH")
    r.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "check":
        problems = check(_disk(args.pytests))
        for p in problems:
            print("gate_suites: FAIL -- " + p)
        if not problems:
            print(f"gate_suites: {len(SUITES)} suites classified: "
                  f"{len(ci_suites())} on CI, {len(local_only())} local-only")
        return 1 if problems else 0

    if args.cmd == "list":
        for runner in RUNNERS:
            run = [n for n, rs in SUITES.items() if runner in rs]
            print(f"{runner}: {len(run)} suites")
        print(f"local-only ({len(local_only())}):")
        for n in local_only():
            print(f"  {n}")
        return 0

    logs = {runner: None for runner in RUNNERS}
    for entry in args.log:
        runner, _, path = entry.partition("=")
        if runner not in RUNNERS:
            print(f"gate_suites: unknown runner {runner!r} in --log", file=sys.stderr)
            return 2
        p = pathlib.Path(path)
        logs[runner] = p.read_text(errors="replace") if p.is_file() else None
    md, ok = render_report(logs, _disk(args.pytests))
    if args.out:
        pathlib.Path(args.out).write_text(md)
    print(md)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
