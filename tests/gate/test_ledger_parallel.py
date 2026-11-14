"""The ledger with `--jobs`: the same totals, sooner, and a hang has a name (issue #335).

`tests/suite_ledger.py` ran every suite one after another. It can now run the
suites `.github/scripts/gate_suites.py` calls parallel-safe several at a time
and the rest afterwards, alone. The owner's condition for that (2026-11-15,
relayed by the leader) is that nothing is dropped or skipped to get faster:
the ledger numbers must be the same before and after.

These tests run the real ledger on a directory of small fake suites, with a
manifest of their own, so no shim and no real suite is involved.

**What makes these tests able to fail.**

* `test_the_totals_and_the_aggregate_equal_the_one_job_run`: the ledger line
  and the whole printed aggregate (suite sections in the caller's order) of a
  `--jobs 3` run are compared with a `--jobs 1` run of the same directory.
  Dropping a suite, printing sections in completion order or counting a suite
  twice changes one of them.
* `test_parallel_safe_suites_overlap_and_serial_suites_never_do`: each fake
  suite records when it started and ended. If the parallel-safe ones never
  overlap, `--jobs` does nothing and the test fails; if a serial one overlaps
  anything, the serial list is not honoured and the test fails.
* `test_a_suite_past_the_timeout_is_killed_and_counted_as_a_fail_by_name`: a
  suite that sleeps far past the timeout must end the run in seconds, with a
  FAIL line naming it and a non-zero exit. Without the kill this test takes
  the full sleep and its own bound fails it.
* `test_the_times_are_reported_and_written`: the `SUITE TIME:` lines and
  `suite-times.tsv` must name every executed suite.
* `test_one_job_runs_every_suite_alone_without_asking_the_manifest`: with one
  job a manifest that cannot classify suites must not matter.

**What this cannot find.** Whether a real suite is parallel-safe. That is a
reading of the suites, recorded in `gate_suites.py`, and the evidence for it
is a CI run whose ledger line equals the one-job run's.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_LEDGER = _TESTS_DIR / "suite_ledger.py"

_SUITE = '''\
import os, sys, time
name = os.path.basename(__file__)
stamp = os.environ["LEDGER_TEST_STAMPS"]
with open(os.path.join(stamp, name + ".start"), "w") as fh:
    fh.write(repr(time.time()))
time.sleep({sleep})
print("ok   " + name + " first")
print("ok   " + name + " second")
{extra}
with open(os.path.join(stamp, name + ".end"), "w") as fh:
    fh.write(repr(time.time()))
raise SystemExit({rc})
'''

_MANIFEST = '''\
RUNNERS = {"fake": "a runner for the ledger's own test"}

def plan(runner, on_disk):
    return sorted(on_disk), {}, []

def skip_line(name, runner, why):
    return "SKIP " + name + ": (whole suite) -- " + why

def serial_reason(name):
    return None if name.startswith("side/") else "serial in this test"
'''


def _tree(specs):
    """A temp dir with fake suites. `specs`: {relative path: (sleep, rc, extra)}."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="ledger-par-"))
    pytests = root / "pytests"
    for rel, (sleep, rc, extra) in specs.items():
        path = pytests / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_SUITE.format(sleep=sleep, rc=rc, extra=extra), encoding="utf-8")
    (root / "manifest.py").write_text(_MANIFEST, encoding="utf-8")
    (root / "stamps").mkdir()
    return root


def _ledger(root, tag, *extra, manifest=True):
    """Run the ledger over `root`'s suites. `(exit code, stdout, logs dir)`."""
    pytests = root / "pytests"
    logs = root / ("logs-" + tag)
    stamps = root / "stamps"
    shutil.rmtree(stamps)
    stamps.mkdir()
    suites = sorted(str(p) for p in pytests.rglob("test_*.py"))
    env = dict(os.environ)
    env["LEDGER_TEST_STAMPS"] = str(stamps)
    env.pop("TORCHNATIVE_GATE_JOBS", None)
    env.pop("TORCHNATIVE_SUITE_TIMEOUT_S", None)
    cmd = [sys.executable, str(_LEDGER), "--logs", str(logs), "--pytests", str(pytests)]
    if manifest:
        cmd += ["--manifest", str(root / "manifest.py")]
    cmd += list(extra) + ["--"] + suites
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr, logs


def _spans(root):
    spans = {}
    for start in (root / "stamps").glob("*.start"):
        name = start.name[: -len(".start")]
        end = root / "stamps" / (name + ".end")
        spans[name] = (float(start.read_text()), float(end.read_text()))
    return spans


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _ledger_line(out):
    lines = [ln for ln in out.splitlines() if ln.startswith("SUITE LEDGER: suites=")]
    assert len(lines) == 1, out
    return re.sub(r"\(printed and re-read from .*", "", lines[0])


def _aggregate(out):
    """The suite sections as printed: headers and ok/FAIL/SKIP lines, in order."""
    return [ln for ln in out.splitlines()
            if ln.startswith(("--- ", "ok ", "FAIL ", "SKIP "))]


_SPECS = {
    "side/test_a.py": (0.8, 0, ""),
    "side/test_b.py": (0.8, 0, 'print("SKIP " + name + " third -- a reason")'),
    "side/test_c.py": (0.8, 1, 'print("FAIL " + name + " third: on purpose")'),
    "solo/test_d.py": (0.4, 0, ""),
    "solo/test_e.py": (0.4, 0, ""),
}


def test_the_totals_and_the_aggregate_equal_the_one_job_run():
    root = _tree(_SPECS)
    try:
        rc1, out1, _ = _ledger(root, "one", "--jobs", "1")
        rc3, out3, _ = _ledger(root, "three", "--jobs", "3")
        assert rc1 == rc3 == 1, (rc1, rc3, out3)
        assert _ledger_line(out1) == _ledger_line(out3), (out1, out3)
        assert "suites=5/5 ok=10 FAIL=1 SKIP=1" in _ledger_line(out3), out3
        assert _aggregate(out1) == _aggregate(out3), (out1, out3)
        headers = [ln for ln in _aggregate(out3) if ln.startswith("--- ")]
        assert headers == ["--- test_%s.py ---" % c for c in "abcde"], headers
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ok   ledger_parallel: jobs=3 prints the one-job aggregate and ledger line "
          "(suites=5/5 ok=10 FAIL=1 SKIP=1)")


def test_parallel_safe_suites_overlap_and_serial_suites_never_do():
    root = _tree(_SPECS)
    try:
        rc, out, _ = _ledger(root, "three", "--jobs", "3")
        spans = _spans(root)
        assert len(spans) == 5, (spans, out)
        side = ["test_a.py", "test_b.py", "test_c.py"]
        solo = ["test_d.py", "test_e.py"]
        pairs = [(x, y) for i, x in enumerate(side) for y in side[i + 1:]]
        overlapping = [p for p in pairs if _overlap(spans[p[0]], spans[p[1]])]
        assert overlapping, f"no two parallel-safe suites ran at the same time: {spans}"
        for name in solo:
            for other in spans:
                if other != name:
                    assert not _overlap(spans[name], spans[other]), (
                        f"{name} is serial and ran while {other} did: {spans}")
        assert "SUITE PLAN: jobs=3, 3 suite(s) run side by side, then 2 alone" in out, out
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"ok   ledger_parallel: {len(overlapping)} pair(s) of parallel-safe suites "
          "overlapped, the 2 serial suites overlapped nothing")


def test_a_suite_past_the_timeout_is_killed_and_counted_as_a_fail_by_name():
    root = _tree({
        "side/test_quick.py": (0.1, 0, ""),
        "solo/test_stuck.py": (120, 0, ""),
    })
    try:
        started = time.monotonic()
        rc, out, logs = _ledger(root, "t", "--jobs", "2", "--suite-timeout", "2")
        took = time.monotonic() - started
        assert took < 60, f"the ledger waited {took:.0f} s for a suite with a 2 s timeout"
        assert rc == 1, out
        assert "FAIL test_stuck: (whole suite) -- timed out after 2 s and was killed" in out, out
        assert "suites=2/2 ok=2 FAIL=1 SKIP=0" in _ledger_line(out), out
        assert "SUITE LEDGER: suites killed at the timeout: test_stuck.py" in out, out
        assert "timed out after 2 s" in (logs / "test_stuck.py.log").read_text()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"ok   ledger_parallel: a 120 s suite under a 2 s timeout ended the run in "
          f"{took:.0f} s as FAIL test_stuck, exit 1")


def test_the_times_are_reported_and_written():
    root = _tree(_SPECS)
    try:
        rc, out, logs = _ledger(root, "three", "--jobs", "3")
        timed = re.findall(r"^SUITE TIME: +([0-9.]+) s  (test_\w+\.py)  \((.+)\)$", out, re.M)
        assert sorted(n for _, n, _ in timed) == ["test_%s.py" % c for c in "abcde"], out
        modes = {n: m for _, n, m in timed}
        assert modes["test_a.py"] == "side by side" and modes["test_d.py"] == "alone", modes
        assert all(float(s) >= 0.3 for s, _, _ in timed), timed
        assert re.search(r"^SUITE TIMES: executed=5 sum=\d+ s wall=\d+ s jobs=3 timeout=2700 s$",
                         out, re.M), out
        rows = (logs / "suite-times.tsv").read_text().splitlines()
        assert rows[0] == "seconds\tmode\tsuite" and len(rows) == 6, rows
        assert not list(logs.glob("suite-times*.log")), "the times file must not look like a suite log"
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ok   ledger_parallel: 5 SUITE TIME lines, one SUITE TIMES line and a 5-row suite-times.tsv")


def test_one_job_runs_every_suite_alone_without_asking_the_manifest():
    root = _tree(_SPECS)
    try:
        # No --manifest: the default one does not know these suites. One job
        # must not need it (the local gate never read a manifest).
        rc, out, _ = _ledger(root, "one", manifest=False)
        assert "suites=5/5 ok=10 FAIL=1 SKIP=1" in _ledger_line(out), out
        assert "SUITE PLAN:" not in out, out
        spans = _spans(root)
        names = sorted(spans)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                assert not _overlap(spans[x], spans[y]), (x, y, spans)
        order = sorted(names, key=lambda n: spans[n][0])
        assert order == names, f"one job must run in the caller's order: {order}"
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ok   ledger_parallel: with one job the 5 suites ran alone, in the caller's order")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:1500]}")
    raise SystemExit(1 if failures else 0)
