"""The macOS gate in three shards: the same totals, and no suite lost or run twice.

Leader decision 2026-11-15 (no new workflow, no new dependency): the macOS job
of `.github/workflows/test.yml` is three jobs of one run, because one runner
took the work of three (run 43670318805: side-by-side pool 84 suites in 1751 s,
lane "mps" 1964 s, lane "distributed" 2359 s, 62 exclusive suites 687 s, about
5 minutes of setup per job and about 4 minutes of DOCWATCH). The conditions:

1. the report's combined `suites=N/N` and ok/FAIL/SKIP for the three shards
   equal what one runner prints;
2. a check that no suite runs in no shard, and none in two.

These tests run the real `tests/suite_ledger.py` on small fake suites, read the
real `.github/scripts/gate_suites.py` table, and feed `merge_shards` synthetic
shard logs. No shim and no real suite is involved.

**What makes these tests able to fail.**

* `test_three_shards_add_up_to_the_unsharded_ledger`: the sums of the three
  shards' ok/FAIL/SKIP are compared with a shard-less run of the same tree, and
  the executed suites with what ran there. A shard that also executes another
  shard's suite, one that drops a suite, or one that prints the whole-suite
  SKIP of a suite it does not own (the total would then be 5, not 3) changes
  one of them. The SHARD line is checked to be none of ok, FAIL, SKIP.
* `test_every_real_suite_has_exactly_one_shard`: reads the real `SUITES` table
  and the suites on disk. A `shard()` that returns something outside `SHARDS`,
  two shards owning one suite, an empty shard, or a lane no shard owns fails it.
* `test_the_report_merges_three_shard_logs_into_one_runner`, and the three
  refusals after it: a suite executed by no shard, by two shards, in a shard the
  manifest does not give it, a missing shard log, and a digest that disagrees
  with the log are each named. Removing the check for any of them turns its test
  red.

**What this cannot find.** Whether a real suite behaves the same on a runner
that has run fewer suites before it (a suite reading something an earlier
suite left behind). That is what the first CI run of the three shards measures.
"""

import atexit
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_REPO = _TESTS_DIR.parent
_LEDGER = _TESTS_DIR / "suite_ledger.py"
_MANIFEST = _REPO / ".github" / "scripts" / "gate_suites.py"


def _gs():
    spec = importlib.util.spec_from_file_location("_gate_suites_shards_test", _MANIFEST)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _disk():
    return sorted(p.relative_to(_TESTS_DIR).as_posix() for p in _TESTS_DIR.rglob("test_*.py")
                  if "__pycache__" not in p.parts)


# ---------------------------------------------------------------------------
# The ledger, driven on fake suites
# ---------------------------------------------------------------------------

# Folder names are the real ones, so the real `shard()` decides: bindings/ is
# the side-by-side pool and devices/mps/ the mps lane (shard 1), distributed/
# the distributed lane (shard 2), release/ and gate/ exclusive (shard 3).
_FAKE = {
    "bindings/test_pool_a.py": ("", ("macos",)),
    "bindings/test_pool_b.py": ('print("SKIP test_pool_b: third -- a reason")', ("macos",)),
    "devices/mps/test_mps_a.py": ("", ("macos",)),
    "distributed/test_dist_a.py": ("", ("macos",)),
    "distributed/test_dist_local.py": ("", ()),
    "release/test_rel_a.py": ('print("FAIL test_rel_a third: on purpose")\nraise SystemExit(1)', ("macos",)),
    "release/test_rel_local.py": ("", ()),
    "gate/test_gate_a.py": ("", ("macos",)),
}

_SUITE = '''\
import os
name = os.path.basename(__file__)
with open(os.environ["SHARD_TEST_SENTINEL"], "a") as fh:
    fh.write(name + "\\n")
print("ok   " + name + " first")
print("ok   " + name + " second")
{extra}
'''

_FIXTURE_TABLES = '''
RUNNERS = {"macos": "fixture macos"}
SUITES = %r
WHY_NOT = {name: "fixture: a device no hosted runner has, for the shard test" for name, r in SUITES.items() if not r}
WHY_NOT_ON = {}
'''


def _tree():
    made = tempfile.mkdtemp(prefix="ledger-shards-")
    atexit.register(shutil.rmtree, made, ignore_errors=True)
    root = pathlib.Path(made)
    for rel, (extra, _) in _FAKE.items():
        path = root / "pytests" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_SUITE.format(extra=extra), encoding="utf-8")
    # The real manifest functions (shard, plan, skip_line) over fixture tables.
    suites = {name: runners for name, (_, runners) in _FAKE.items()}
    (root / "manifest.py").write_text(
        _MANIFEST.read_text(encoding="utf-8") + _FIXTURE_TABLES % (suites,), encoding="utf-8")
    return root


def _ledger(root, tag, shard=None):
    """Run the ledger over the fake tree. `(exit code, stdout, logs dir, executed names)`."""
    sentinel = root / ("ran-" + tag)
    sentinel.write_text("")
    logs = root / ("logs-" + tag)
    suites = sorted(str(p) for p in (root / "pytests").rglob("test_*.py"))
    env = dict(os.environ)
    env["SHARD_TEST_SENTINEL"] = str(sentinel)
    for name in ("TORCHNATIVE_GATE_JOBS", "TORCHNATIVE_GATE_SHARD", "TORCHNATIVE_SUITE_TIMEOUT_S"):
        env.pop(name, None)
    cmd = [sys.executable, str(_LEDGER), "--logs", str(logs), "--pytests", str(root / "pytests"),
           "--manifest", str(root / "manifest.py"), "--runner", "macos", "--jobs", "2"]
    if shard:
        cmd += ["--shard", shard]
    proc = subprocess.run(cmd + ["--"] + suites, capture_output=True, text=True, env=env, timeout=300)
    return (proc.returncode, proc.stdout + proc.stderr, logs,
            sorted(sentinel.read_text().split()))


def _ledger_numbers(out):
    m = re.search(r"^SUITE LEDGER: suites=(\d+)/(\d+) ok=(\d+) FAIL=(\d+) SKIP=(\d+) ", out, re.M)
    assert m, out
    return tuple(int(x) for x in m.groups())


def _headers(out):
    return [ln for ln in out.splitlines() if re.match(r"^--- test_\w+\.py ---$", ln)]


def test_three_shards_add_up_to_the_unsharded_ledger():
    """Condition 1: the shards' totals and executed suites equal one runner's.

    Able to fail: see the module docstring. The expected numbers are written
    out (12 ok, 1 FAIL, 3 SKIP over 8 suites), so a base run that itself drops
    something is caught too, and each shard must own a different, non-empty
    part so a ledger that ignores `--shard` (every shard = the base run) fails
    the sum.
    """
    root = _tree()
    gs = _gs()
    rc0, out0, _, ran0 = _ledger(root, "base")
    assert rc0 == 1, out0
    assert _ledger_numbers(out0) == (8, 8, 12, 1, 3), out0
    assert len(ran0) == 6, ran0   # the two local-only suites did not run
    sums, executed, rcs = [0, 0, 0], [], []
    owned_all = []
    for k in gs.SHARDS:
        rc, out, logs, ran = _ledger(root, "s%d" % k, "%d/%d" % (k, len(gs.SHARDS)))
        rcs.append(rc)
        suites, total, ok, fail, skip = _ledger_numbers(out)
        assert (suites, total) == (8, 8), ("every shard accounts for every suite", out)
        sums = [a + b for a, b in zip(sums, (ok, fail, skip))]
        executed += ran
        assert _headers(out) == _headers(out0), ("a shard dropped or reordered a header", out)
        owned = (logs / "suite-shard.txt").read_text().split()
        assert owned == sorted(r for r in _FAKE if gs.shard(r) == k), (k, owned)
        owned_all += owned
        elsewhere = [ln for ln in out.splitlines() if ln.startswith("SHARD ")]
        assert len(elsewhere) == 8 - len(owned), (k, elsewhere)
        assert all(re.match(r"^SHARD \w+: \(whole suite\) -- runs in shard [123] of 3$", ln)
                   for ln in elsewhere), elsewhere
        line = [ln for ln in out.splitlines() if ln.startswith("SUITE SHARD: ")]
        assert line == ["SUITE SHARD: k=%d n=3 owned=%d names=%s"
                        % (k, len(owned), gs.names_digest(owned))], line
        assert " shard=%d/3 owned=%d executed=%d" % (k, len(owned), len(ran)) in out, out
        skips = [ln.split(":")[0] for ln in out.splitlines() if ln.startswith("SKIP ")]
        mine = [ln for ln in skips if ln.endswith(("dist_local", "rel_local"))]
        assert mine == {2: ["SKIP test_dist_local"], 3: ["SKIP test_rel_local"]}.get(k, []), (k, skips)
    assert sums == [12, 1, 3], ("the shards' ok/FAIL/SKIP do not sum to one runner's", sums)
    assert sorted(executed) == ran0, ("the shards ran other suites than one runner does", executed, ran0)
    assert sorted(owned_all) == sorted(_FAKE), "a suite is in no shard or in two"
    assert rcs == [0, 0, 1], rcs
    print("ok   ledger_shards: shards 1/2/3 sum to the unsharded ledger (8/8 suites, ok=12 FAIL=1 SKIP=3), "
          "6 suites executed once each, every SKIP printed by its owning shard only")


def test_every_real_suite_has_exactly_one_shard():
    """Condition 2 on the real table and the real tree.

    Able to fail: a `shard()` outside `SHARDS` (or a ValueError for a lane no
    shard owns), a suite on disk with no shard, a shard with no suite (a job
    that would run nothing), or two shards owning one suite. The lane test
    adds a lane to a copy of the table and expects the refusal to name it.
    """
    gs = _gs()
    names = sorted(set(_disk()) | set(gs.SUITES))
    owned = {k: [] for k in gs.SHARDS}
    for name in names:
        k = gs.shard(name)
        assert k in gs.SHARDS, (name, k)
        owned[k].append(name)
    assert sum(len(v) for v in owned.values()) == len(names), "a suite is in two shards or none"
    assert len(set(sum(owned.values(), []))) == len(names), "a suite is in two shards"
    for k, v in owned.items():
        assert v, "shard %d owns nothing" % k
    assert all(gs.lane(n) == "distributed" for n in owned[2]) and owned[2], owned[2]
    assert all(gs.serial_reason(n) is not None and gs.lane(n) is None for n in owned[3]), owned[3]
    assert not [n for n in owned[1] if gs.serial_reason(n) is not None and gs.lane(n) != "mps"], owned[1]
    assert gs.check(_disk()) == [], gs.check(_disk())
    gs.LANES["devices/zzz/"] = "zzz"
    try:
        gs.shard("devices/zzz/test_x.py")
    except ValueError as exc:
        assert "zzz" in str(exc), exc
    else:
        raise AssertionError("a lane no shard owns was given a shard")
    print("ok   ledger_shards: %d suites, shards own %s, one each; an unowned lane is refused by name"
          % (len(names), "/".join(str(len(owned[k])) for k in gs.SHARDS)))


# ---------------------------------------------------------------------------
# The report merger, on synthetic shard logs
# ---------------------------------------------------------------------------

_DISK = ["bindings/test_a.py", "distributed/test_b.py", "release/test_c.py"]   # shards 1, 2, 3


def _shard_log(gs, k, owned, exit_line="EXIT=0", digest=None, extra=()):
    lines = []
    for rel in _DISK:
        stem = rel.rsplit("/", 1)[-1][:-3]
        lines.append("--- %s ---" % rel.rsplit("/", 1)[-1])
        if rel in owned:
            lines += ["ok   t1", "ok   t2"]
        else:
            lines.append("SHARD %s: (whole suite) -- runs in shard %d of 3" % (stem, gs.shard(rel)))
    lines.append("SUITE SHARD: k=%d n=3 owned=%d names=%s"
                 % (k, len(owned), digest or gs.names_digest(owned)))
    lines.append("SUITE LEDGER: suites=3/3 ok=%d FAIL=0 SKIP=0 (printed and re-read from x; they agree)"
                 " shard=%d/3 owned=%d executed=%d" % (2 * len(owned), k, len(owned), len(owned)))
    lines += list(extra) + [exit_line]
    return "\n".join(lines) + "\n"


def _good(gs):
    return [("s1", _shard_log(gs, 1, [_DISK[0]])), ("s2", _shard_log(gs, 2, [_DISK[1]])),
            ("s3", _shard_log(gs, 3, [_DISK[2]], extra=["DOCWATCH: 1/1"]))]


def test_the_report_merges_three_shard_logs_into_one_runner():
    """Able to fail: totals that are not the sum, a suite section taken from
    the shard that did not run it, or an EXIT line that hides a failed shard."""
    gs = _gs()
    text, problems = gs.merge_shards(_good(gs), _DISK)
    assert problems == [], problems
    assert "SUITE LEDGER: suites=3/3 ok=6 FAIL=0 SKIP=0 (combined from shards 1,2,3 of 3)" in text, text
    assert text.splitlines()[-1] == "EXIT=0", text
    assert "DOCWATCH: 1/1" in text, text
    rows = gs.classify_log(text, _DISK)
    assert {r["verdict"] for r in rows.values()} == {"PASS"}, rows
    assert all(r["ok"] == 2 for r in rows.values()), rows
    bad = _good(gs)
    bad[1] = ("s2", _shard_log(gs, 2, [_DISK[1]], exit_line="EXIT=1"))
    text, problems = gs.merge_shards(bad, _DISK)
    assert text.splitlines()[-1] == "EXIT=1", text
    print("ok   ledger_shards: three shard logs merge to suites=3/3 ok=6 FAIL=0 SKIP=0 and the first failing EXIT")


def _named(problems, *words):
    return [p for p in problems if all(w in p for w in words)]


def test_the_report_refuses_a_suite_no_shard_ran_and_one_two_shards_ran():
    """Condition 2 in the report. Able to fail: drop the per-suite check and
    both refusals disappear; drop the manifest comparison and the third does."""
    gs = _gs()
    none = _good(gs)
    none[2] = ("s3", _shard_log(gs, 3, []))
    _, problems = gs.merge_shards(none, _DISK)
    assert _named(problems, "release/test_c.py", "no shard"), problems
    twice = _good(gs)
    twice[0] = ("s1", _shard_log(gs, 1, [_DISK[0], _DISK[2]]))
    _, problems = gs.merge_shards(twice, _DISK)
    assert _named(problems, "release/test_c.py", "shards [1, 3]"), problems
    wrong = _good(gs)
    wrong[0] = ("s1", _shard_log(gs, 1, [_DISK[0], _DISK[2]]))
    wrong[2] = ("s3", _shard_log(gs, 3, []))
    _, problems = gs.merge_shards(wrong, _DISK)
    assert _named(problems, "release/test_c.py", "ran in shard 1", "shard 3"), problems
    md, ok = gs.render_report({"linux": None, "macos": "x\n", "windows": None}, _DISK, problems)
    assert not ok and "SHARD MERGE FAILED" in md and "release/test_c.py" in md, md
    print("ok   ledger_shards: a suite in no shard, in two shards and in the wrong shard are each named")


def test_the_report_refuses_a_missing_shard_log_and_a_log_that_disagrees_with_itself():
    """Able to fail: a merge that accepts two logs, one whose digest line names
    other suites than its sections hold, or a ledger whose totals are not its
    sections' (a log edited after the ledger read it)."""
    gs = _gs()
    _, problems = gs.merge_shards([_good(gs)[0], _good(gs)[2], ("gone/gate.log", None)], _DISK)
    assert _named(problems, "gone/gate.log", "missing"), problems
    assert _named(problems, "no log for shard 2/3"), problems
    assert not _named(problems, "no shard"), ("per-suite noise while a shard is missing", problems)
    forged = _good(gs)
    forged[1] = ("s2", _shard_log(gs, 2, [_DISK[1]], digest="0" * 64))
    _, problems = gs.merge_shards(forged, _DISK)
    assert _named(problems, "s2", "SUITE SHARD line"), problems
    edited = _good(gs)
    edited[0] = ("s1", _shard_log(gs, 1, [_DISK[0]]).replace("ok   t2\n", "", 1))
    _, problems = gs.merge_shards(edited, _DISK)
    assert _named(problems, "s1", "ledger says"), problems
    _, problems = gs.merge_shards(_good(gs)[:2] + [("s3", "no shard line here\n")], _DISK)
    assert _named(problems, "s3", "SUITE SHARD"), problems
    print("ok   ledger_shards: a missing shard log, a forged digest, an edited log and a non-shard log are named")


def test_the_report_command_takes_the_three_logs_and_fails_on_a_missing_one():
    """The comma form of `--log macos=A,B,C` end to end. Able to fail: a CLI
    that reads only the first path, or one that exits 0 with a shard missing."""
    gs = _gs()
    made = tempfile.mkdtemp(prefix="ledger-shards-cli-")
    atexit.register(shutil.rmtree, made, ignore_errors=True)
    root = pathlib.Path(made)
    for rel in _DISK:
        (root / "pytests" / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "pytests" / rel).write_text("")
    paths = []
    for k, (_, text) in enumerate(_good(gs), 1):
        p = root / ("shard%d.log" % k)
        p.write_text(text)
        paths.append(str(p))

    def report(listed):
        proc = subprocess.run(
            [sys.executable, str(_MANIFEST), "report", "--pytests", str(root / "pytests"),
             "--log", "macos=" + ",".join(listed)], capture_output=True, text=True, timeout=120)
        return proc.returncode, proc.stdout + proc.stderr

    rc, out = report(paths)
    assert "gate_suites: FAIL -- macos:" not in out, out
    assert "suites=3/3 ok=6 FAIL=0 SKIP=0" in out, out
    rc, out = report(paths[:2] + [str(root / "absent.log")])
    assert rc != 0, out
    assert "gate_suites: FAIL -- macos: shard log" in out and "absent.log" in out, out
    rc, out = report(paths[:1])
    assert rc != 0 and "no log for shard 2/3" in out, out
    print("ok   ledger_shards: report --log macos=A,B,C reads three logs and exits non-zero when one is missing")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:1500]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
