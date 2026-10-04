"""The CI gate (issue #24): what runs where, and that nothing drops out.

`.github/workflows/test.yml` runs `run.sh` on hosted runners, where a Metal
GPU, the Neural Engine, a Vulkan loader, `adb` and the Intel NPU are all
absent. A suite that cannot run there is skipped **by name**, and the
CI report lists it as local-only rather than as passed. The list of which
suite runs on which runner is `.github/scripts/gate_suites.py`.

The requirement this file holds down (AGENTS.md §17.5): **the suites run on CI
plus the suites left local-only must equal the full gate.** This repository
has already been fooled by a gate that printed `SKIP=0` while 24 tests
skipped silently, so the obvious failure here is a CI that turns green on a
subset while people read it as the whole gate. Three things can cause that,
and each one has a test below:

* a new `test_*.py` lands and is in neither list -- `gate_suites` must refuse
  it by name, both here (this suite runs in every gate, local included) and in
  `suite_ledger.py --runner`, before a single suite runs on CI;
* a suite that is not planned for a runner is dropped there without a trace --
  it must print a `SKIP` line naming it and the reason, so the ledger counts it;
* the report renders a run with skipped or missing suites as a pass -- it
  must say UNVERIFIED and NOT RUN for them, and refuse a missing log.

What this file cannot see: whether a suite planned for a runner really works
there. The Linux runner had never run this gate before #24, so the plan in
`gate_suites.py` is a static classification. The first CI run is what tests
it; a suite that FAILs there for a platform reason has to be moved to
local-only **with the reason**, never dropped from `SUITES`.

Nothing here imports torch or the shim. It reads `.github/scripts/gate_suites.py`,
runs `suite_ledger.py` on throwaway suites in a temporary directory, and reads
the workflow as text and YAML.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

import _skip

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = REPO / ".github" / "scripts" / "gate_suites.py"
LEDGER = HERE.parent / "suite_ledger.py"
WORKFLOW = REPO / ".github" / "workflows" / "test.yml"
RUN_SH = HERE.parent / "run.sh"
CHECK_DOCS = REPO / "tests" / "docwatch" / "check_docs.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # `dataclasses` looks the module up in sys.modules while building a class.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gs():
    return _load(MANIFEST, "_gate_suites_under_test")


def _on_disk():
    return sorted(p.relative_to(HERE.parent).as_posix()
                  for p in HERE.parent.rglob("test_*.py")
                  if "__pycache__" not in p.parts)


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------

def test_every_suite_on_disk_is_in_the_manifest_and_nothing_else_is():
    gs = _gs()
    problems = gs.check(_on_disk())
    assert problems == [], "\n".join(problems)


def test_ci_suites_plus_local_only_suites_is_the_full_gate():
    gs = _gs()
    disk = set(_on_disk())
    ci = set(gs.ci_suites())
    local = set(gs.local_only())
    assert ci | local == disk, sorted(disk ^ (ci | local))
    assert not (ci & local), sorted(ci & local)


def test_a_new_suite_in_neither_list_is_refused_by_name():
    gs = _gs()
    problems = gs.check(_on_disk() + ["test_brand_new_round.py"])
    assert any("test_brand_new_round.py" in p for p in problems), problems
    problems = gs.check([n for n in _on_disk() if n != "_support/test_shim.py"])
    assert any("_support/test_shim.py" in p for p in problems), problems


def test_every_suite_kept_off_a_runner_says_why():
    gs = _gs()
    for name, runners in gs.SUITES.items():
        assert set(runners) <= set(gs.RUNNERS), (name, runners)
        if set(runners) != set(gs.RUNNERS):
            why = gs.WHY_NOT.get(name, "")
            assert len(why.strip()) > 20, (
                f"{name} is kept off {sorted(set(gs.RUNNERS) - set(runners))} "
                f"with no reason: {why!r}")
    stray = sorted(set(gs.WHY_NOT) - set(gs.SUITES))
    assert not stray, stray


def test_a_reason_without_an_exclusion_is_refused():
    """A WHY_NOT entry for a suite that runs everywhere reads as an exclusion
    that is not one -- a reader of the report would believe the wrong list."""
    gs = _gs()
    for name, runners in gs.SUITES.items():
        if set(runners) == set(gs.RUNNERS):
            assert name not in gs.WHY_NOT, name


def test_the_plan_for_a_runner_names_every_suite_exactly_once():
    gs = _gs()
    disk = _on_disk()
    for runner in gs.RUNNERS:
        run, skipped, problems = gs.plan(runner, disk)
        assert problems == [], problems
        assert sorted(run + list(skipped)) == disk, runner
        for name, why in skipped.items():
            assert why.strip(), name
    try:
        gs.plan("windows-arm", disk)
    except ValueError as e:
        assert "windows-arm" in str(e), e
    else:
        raise AssertionError("an unknown runner name was planned for, not refused")


# ---------------------------------------------------------------------------
# The ledger, driven on throwaway suites
# ---------------------------------------------------------------------------

_MANIFEST_FIXTURE = '''
RUNNERS = {"linux": "fixture linux", "macos": "fixture macos"}
SUITES = {
    "test_runs.py": ("linux", "macos"),
    "test_mac_only.py": ("macos",),
    "test_local.py": (),
}
WHY_NOT = {
    "test_mac_only.py": "fixture: needs something only the macos runner has",
    "test_local.py": "fixture: needs a device no hosted runner has",
}
'''


def _fixture_dir(tmp, extra=()):
    d = pathlib.Path(tmp) / "pytests"
    d.mkdir()
    sentinel = pathlib.Path(tmp) / "ran"
    sentinel.mkdir()
    for name in ("test_runs.py", "test_mac_only.py", "test_local.py") + tuple(extra):
        (d / name).write_text(
            "import pathlib\n"
            f"pathlib.Path({str(sentinel)!r}, {name!r}).write_text('x')\n"
            "print('ok   test_one')\n")
    manifest = pathlib.Path(tmp) / "manifest.py"
    gs_src = MANIFEST.read_text()
    # The fixture manifest reuses gate_suites' functions with fixture tables.
    manifest.write_text(gs_src + "\n" + _MANIFEST_FIXTURE)
    return d, sentinel, manifest


def _ledger(d, manifest, logs, *extra):
    suites = sorted(str(p) for p in d.glob("test_*.py"))
    return subprocess.run(
        [sys.executable, str(LEDGER), "--logs", str(logs), "--pytests", str(d),
         "--python", sys.executable, *extra, "--manifest", str(manifest),
         "--", *suites],
        capture_output=True, text=True)


def test_the_ledger_skips_by_name_what_the_runner_cannot_run():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        d, sentinel, manifest = _fixture_dir(tmp)
        proc = _ledger(d, manifest, pathlib.Path(tmp) / "logs", "--runner", "linux")
        out = proc.stdout
        assert proc.returncode == 0, out + proc.stderr
        assert sorted(p.name for p in sentinel.iterdir()) == ["test_runs.py"], (
            "a suite not planned for this runner was executed anyway: "
            + str(sorted(p.name for p in sentinel.iterdir())))
        for name in ("test_mac_only", "test_local"):
            line = [ln for ln in out.splitlines() if ln.startswith(f"SKIP {name}:")]
            assert line, f"no SKIP line names {name}:\n{out}"
            assert "fixture:" in line[0], line[0]
        assert "suites=3/3" in out, out
        assert "SKIP=2" in out, out
        plan = [ln for ln in out.splitlines() if ln.startswith("CI PLAN:")]
        assert plan and "runner=linux" in plan[0] and "ran=1/3" in plan[0], out
        assert "test_local.py" in plan[0] and "test_mac_only.py" in plan[0], plan


def test_the_ledger_refuses_a_suite_the_manifest_does_not_name():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        d, sentinel, manifest = _fixture_dir(tmp, extra=("test_unlisted.py",))
        proc = _ledger(d, manifest, pathlib.Path(tmp) / "logs", "--runner", "linux")
        assert proc.returncode != 0, proc.stdout
        assert "test_unlisted.py" in proc.stdout, proc.stdout
        assert not any(sentinel.iterdir()), (
            "the ledger ran suites before refusing an unplanned one")


def test_the_ledger_refuses_an_unknown_runner():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        d, sentinel, manifest = _fixture_dir(tmp)
        proc = _ledger(d, manifest, pathlib.Path(tmp) / "logs", "--runner", "lnux")
        assert proc.returncode != 0, proc.stdout
        assert "lnux" in proc.stdout + proc.stderr
        assert not any(sentinel.iterdir())


def test_without_a_runner_the_ledger_runs_everything_as_before():
    """The local gate does not set a runner; nothing is skipped there."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        d, sentinel, manifest = _fixture_dir(tmp)
        proc = _ledger(d, manifest, pathlib.Path(tmp) / "logs")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert len(list(sentinel.iterdir())) == 3, proc.stdout
        assert "SKIP=0" in proc.stdout, proc.stdout
        assert "CI PLAN:" not in proc.stdout, proc.stdout


def test_run_sh_hands_the_runner_to_the_ledger():
    text = RUN_SH.read_text()
    assert "TORCHNATIVE_GATE_RUNNER" in text
    assert '--runner "$TORCHNATIVE_GATE_RUNNER"' in text, (
        "run.sh no longer passes TORCHNATIVE_GATE_RUNNER to suite_ledger.py, so "
        "a CI runner would run suites it cannot run")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _log(sections, tail=""):
    out = []
    for name, body in sections.items():
        out.append(f"--- {name} ---")
        out.extend(body)
    return "\n".join(out) + "\n" + tail


def test_the_report_never_renders_a_skipped_suite_as_passed():
    gs = _gs()
    disk = ["test_a.py", "test_b.py", "test_c.py", "test_d.py"]
    text = _log({
        "test_a.py": ["ok   t1", "ok   t2"],
        "test_b.py": ["SKIP test_b: t1 -- no mps device"],
        "test_c.py": ["SKIP test_c: (whole suite) -- not run on CI runner 'linux': needs ANE"],
        "test_d.py": ["ok   t1", "FAIL t2: AssertionError: x"],
    }, "SUITE LEDGER: suites=4/4 ok=3 FAIL=1 SKIP=2\nEXIT=1\n")
    rows = gs.classify_log(text, disk)
    assert rows["test_a.py"]["verdict"] == "PASS", rows
    assert rows["test_b.py"]["verdict"] == "UNVERIFIED", rows
    assert rows["test_c.py"]["verdict"] == "NOT RUN", rows
    assert rows["test_d.py"]["verdict"] == "FAIL", rows


def test_the_report_names_a_suite_missing_from_a_log():
    gs = _gs()
    rows = gs.classify_log(_log({"test_a.py": ["ok   t1"]}), ["test_a.py", "test_b.py"])
    assert rows["test_b.py"]["verdict"] == "MISSING", rows


def test_the_report_refuses_a_missing_log_and_lists_local_only_suites():
    gs = _gs()
    md, ok = gs.render_report({"linux": None, "macos": None}, _on_disk())
    assert not ok, "a report with no gate logs at all came out ok"
    assert "no log" in md.lower(), md
    # In the local-only section itself, with its reason -- every suite also
    # appears in the per-suite table, so `name in md` could not fail.
    section = md.split("### Local-only", 1)[1].split("\n### ", 1)[0]
    assert gs.local_only(), "no local-only suite: this test would check nothing"
    for name in gs.local_only():
        line = [ln for ln in section.splitlines() if f"`{name}`" in ln]
        assert line, f"local-only suite {name} is not listed in the local-only section"
        assert gs.WHY_NOT[name][:40] in line[0], line[0]


def test_the_report_counts_add_up_to_the_full_gate():
    gs = _gs()
    disk = _on_disk()
    sections = {}
    for name in disk:
        sections[name] = ["ok   t"]
    md, ok = gs.render_report({"linux": _log(sections, "EXIT=0\n"),
                               "macos": _log(sections, "EXIT=0\n")}, disk)
    line = [ln for ln in md.splitlines() if "full gate" in ln.lower()]
    assert line, md
    assert str(len(disk)) in line[0], line


# ---------------------------------------------------------------------------
# DOCWATCH's `smoke_ok` on a runner that skipped tests
# ---------------------------------------------------------------------------

def test_smoke_ok_is_unmeasured_on_a_ci_runner_that_skipped_tests_and_counted_locally():
    """`count smoke_ok ge 480` was measured on a Mac where every test_shim
    test ran. On a CI runner the Metal/CoreML/Vulkan tests skip by name, so
    the count is a different quantity there: SKIP with the reason, not FAIL
    and not PASS. Locally (no runner) nothing changes -- a test that started
    skipping on the Mac still lowers the count and still fails the marker."""
    cd = _load(CHECK_DOCS, "_check_docs_under_test")
    stdout = "ok   a\nok   b\nSKIP test_shim: c -- no mps device\n"
    assert cd.smoke_verdict(stdout, 0, runner=None) == 2
    try:
        cd.smoke_verdict(stdout, 0, runner="linux")
    except cd.LiveFactsSkip as e:
        assert "linux" in str(e) and "1" in str(e), e
    else:
        raise AssertionError("smoke_ok on a CI runner with skips was evaluated")
    assert cd.smoke_verdict("ok   a\nok   b\n", 0, runner="linux") == 2
    try:
        cd.smoke_verdict("", 1, runner=None)
    except cd.LiveFactsError:
        pass
    else:
        raise AssertionError("a test_shim run with no ok lines was counted")


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------

def _yaml():
    try:
        import yaml
    except ImportError:
        raise _skip.Skip("pyyaml is not installed in this interpreter")
    doc = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML reads the bare key `on` as the boolean True.
    if True in doc and "on" not in doc:
        doc["on"] = doc.pop(True)
    return doc


def test_the_workflow_runs_on_prs_to_develop_and_on_dispatch():
    doc = _yaml()
    on = doc["on"]
    assert "workflow_dispatch" in on, on
    assert "develop" in on["pull_request"]["branches"], on


def test_every_runner_in_the_manifest_has_a_gate_job_that_names_it():
    gs = _gs()
    doc = _yaml()
    named = {}
    for job_id, job in doc["jobs"].items():
        env = job.get("env", {}) or {}
        if "TORCHNATIVE_GATE_RUNNER" in env:
            named[env["TORCHNATIVE_GATE_RUNNER"]] = job_id
    assert set(named) == set(gs.RUNNERS), (named, sorted(gs.RUNNERS))
    for runner, job_id in named.items():
        job = doc["jobs"][job_id]
        assert not job.get("continue-on-error"), job_id
        steps = job["steps"]
        gate = [s for s in steps if "tests/run.sh" in str(s.get("run", ""))]
        assert len(gate) == 1, (job_id, gate)
        assert not gate[0].get("continue-on-error"), job_id
        assert "|| true" not in gate[0]["run"], job_id


def test_the_report_job_runs_even_when_a_gate_is_red():
    doc = _yaml()
    report = doc["jobs"]["report"]
    assert "always()" in str(report.get("if", "")), report.get("if")
    text = WORKFLOW.read_text()
    assert ".github/scripts/gate_suites.py report" in text


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_cigate")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
