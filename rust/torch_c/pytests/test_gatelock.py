"""The gate must not be able to lose a whole suite and still exit 0.

Two runs of `run.sh` in the same checkout derived the same `$stage` from the
repository path and therefore shared **one** `$stage/suite-logs`, which the
second run began by `rm -rf`-ing while the first was still writing into it.
That produced both halves of a false green on the same day:

  * one run died with `cat: .../test_shim.py.log: No such file or directory`
    at 944 ok / 0 FAIL -- a number shaped exactly like a partial pass; and
  * three runs dropped `test_bf16ane.py`'s entire section from the aggregate,
    21 tests, **and still exited 0**. The aggregate read 1669 where the suite
    logs summed to 1690.

A gate that loses a suite while reporting success is the defect class this
repository keeps finding, arrived at in the instrument itself. The counts it
has published are only worth what its accounting is worth, so the accounting
is what this file pins:

1. **A lost or unreadable suite log fails loudly, naming the suite.** Not a
   raw `cat:` shell error taking the run down at a plausible-looking number,
   and never a section that is simply absent from the output with nobody
   counting.
2. **The reported total is reconcilable.** `suite_ledger.py` counts what it
   printed and then re-reads every log from disk and counts again; if the two
   disagree, or a log vanished between the two, the run fails. The suite count
   is asserted against the number of `test_*.py` files, so a vanished suite is
   arithmetic rather than absence.
3. **Concurrent runs cannot destroy each other.** `run.sh` stages its logs in
   a per-run directory (PID-suffixed) and takes a lock on `$stage`, so a
   second gate in the same stage refuses to start rather than corrupting the
   first. Refusing is honest; racing is not.

The ledger cases below drive `suite_ledger.py` against a throwaway directory
of toy suites, so the collision can be *simulated* rather than described: one
toy suite deletes another's log exactly the way the `rm -rf` did. The two
`run.sh` cases use a fake `cargo` on PATH, as `test_toolguard_run_preflight.py`
does, so the lock is exercised without a real build.
"""

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "suite_ledger.py"
RUN_SH = HERE / "run.sh"

GOOD_PYTHON = "/Volumes/macMini/caches/spike-venv/bin/python"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _toy_suite(path: Path, oks: int, body: str = "") -> None:
    """A suite that prints `oks` ok lines, plus whatever `body` does first."""
    path.write_text(
        "import os, sys\n"
        f"{body}\n"
        f"for i in range({oks}):\n"
        "    print('ok   test_%d' % i)\n"
    )


def _run_ledger(tmp: Path, suites, logs: Path = None, extra=()):
    logs = logs or (tmp / "logs")
    argv = [sys.executable, str(LEDGER), "--logs", str(logs),
            "--pytests", str(tmp), *extra, "--",
            *[str(s) for s in suites]]
    proc = subprocess.run(argv, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# 1. the happy path, so the failures below mean something
# --------------------------------------------------------------------------

def test_a_clean_run_reconciles_and_reports_every_suite():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, n in (("test_a.py", 3), ("test_b.py", 5)):
            _toy_suite(tmp / name, n)
        rc, out = _run_ledger(tmp, sorted(tmp.glob("test_*.py")))
        assert rc == 0, f"a clean run must pass:\n{out}"
        assert "SUITE LEDGER: suites=2/2" in out, out
        assert "ok=8" in out, f"the ledger must total the ok lines:\n{out}"
        assert out.count("ok   test_0") == 2, (
            "each suite's own output must still be printed verbatim:\n" + out)


# --------------------------------------------------------------------------
# 2. property 1 -- a lost suite log fails loudly, naming the suite
# --------------------------------------------------------------------------

def test_a_log_deleted_by_a_concurrent_run_fails_naming_the_suite():
    """The real collision: something deletes an already-written log.

    `test_b.py` here does what the second gate's `rm -rf "$suite_logs"` did to
    the first gate's directory. Before this guard the loss was either a bare
    `cat:` error at a plausible count or nothing at all.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logs = tmp / "logs"
        _toy_suite(tmp / "test_a.py", 3)
        _toy_suite(tmp / "test_b.py", 2,
                   body=f"os.remove({str(logs / 'test_a.py.log')!r})")
        rc, out = _run_ledger(tmp, sorted(tmp.glob("test_*.py")), logs=logs)
        assert rc != 0, f"a vanished suite log must fail the gate:\n{out}"
        assert "test_a.py" in out, (
            "the failure must name the suite that was lost:\n" + out)
        assert "cat:" not in out, (
            "the loss must be reported by the gate, not leak out as a raw "
            "shell error at a plausible-looking count:\n" + out)


def test_an_unreadable_log_is_not_silently_skipped():
    """Unreadable is a loss too, and must not be ignored to keep the run going.

    Chmod 000 rather than deletion: the temptation when fixing the `cat`
    failure is `cat ... || true`, which turns a loud corruption into a quiet
    one. This case fails if that is what happened.
    """
    if os.geteuid() == 0:
        return  # root reads anything; the case cannot be staged
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logs = tmp / "logs"
        _toy_suite(tmp / "test_a.py", 3)
        _toy_suite(tmp / "test_b.py", 2,
                   body=f"os.chmod({str(logs / 'test_a.py.log')!r}, 0)")
        rc, out = _run_ledger(tmp, sorted(tmp.glob("test_*.py")), logs=logs)
        assert rc != 0, f"an unreadable suite log must fail the gate:\n{out}"
        assert "test_a.py" in out, out


# --------------------------------------------------------------------------
# 3. property 2 -- the reported total must be reconcilable
# --------------------------------------------------------------------------

def test_a_log_that_changes_after_it_was_printed_fails_reconciliation():
    """1669 vs 1690: the printed aggregate and the logs must agree.

    A later suite appends to an earlier suite's log, so the disk sum is no
    longer what was printed. A gate whose total cannot be reconciled is a
    gate whose totals are not evidence.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logs = tmp / "logs"
        _toy_suite(tmp / "test_a.py", 3)
        _toy_suite(
            tmp / "test_b.py", 2,
            body=("open(%r, 'a').write('ok   test_ghost\\n')"
                  % str(logs / "test_a.py.log")))
        rc, out = _run_ledger(tmp, sorted(tmp.glob("test_*.py")), logs=logs)
        assert rc != 0, (
            "the aggregate disagreed with the suite logs and the gate still "
            f"passed -- the 1669/1690 shape:\n{out}")
        assert "test_a.py" in out, out


def test_the_suite_count_is_asserted_against_the_number_of_suite_files():
    """A suite that never ran is arithmetic, not absence.

    The caller's list is short by one file that exists in the directory --
    what a partially expanded glob, or a suite quietly dropped from the run,
    looks like from here.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name in ("test_a.py", "test_b.py", "test_c.py"):
            _toy_suite(tmp / name, 1)
        rc, out = _run_ledger(tmp, [tmp / "test_a.py", tmp / "test_b.py"])
        assert rc != 0, f"a suite missing from the run must fail:\n{out}"
        assert "test_c.py" in out, (
            "the failure must name the suite that never ran:\n" + out)


def test_a_stray_log_from_another_run_fails_rather_than_being_counted():
    """Extra logs in the directory mean two runs shared it. Refuse."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        logs = tmp / "logs"
        logs.mkdir()
        (logs / "test_somebody_elses.py.log").write_text("ok   test_x\n")
        _toy_suite(tmp / "test_a.py", 1)
        rc, out = _run_ledger(tmp, [tmp / "test_a.py"], logs=logs)
        assert rc != 0, f"a foreign log in the log directory must fail:\n{out}"
        assert "test_somebody_elses.py.log" in out, out


def test_a_failing_suite_still_fails_the_gate():
    """The original contract: a non-zero suite is a non-zero gate."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _toy_suite(tmp / "test_a.py", 1, body="sys.exit(1)")
        rc, out = _run_ledger(tmp, [tmp / "test_a.py"])
        assert rc != 0, f"a failing suite must fail the gate:\n{out}"


# --------------------------------------------------------------------------
# 4. property 3 -- concurrent runs cannot destroy each other
# --------------------------------------------------------------------------

def test_run_sh_stages_logs_per_run_and_does_not_wipe_a_shared_directory():
    text = RUN_SH.read_text()
    assert 'suite_logs="$stage/suite-logs"' not in text, (
        "run.sh assigns the shared suite-log directory again -- that is the "
        "path whose `rm -rf` deleted a concurrent gate's logs. Clearing a "
        "per-run ($$-suffixed) directory is fine; clearing a shared one is "
        "the defect.")
    assert 'suite_logs="$stage/suite-logs.$$"' in text, (
        "run.sh must stage suite logs in a per-run directory so two gates in "
        "one stage cannot destroy each other's logs")
    assert "$stage/lock" in text and "kill -0" in text, (
        "run.sh must take a lock on the stage so a second gate refuses to "
        "start rather than corrupting the first")
    assert "pytests/test_*.py" in text, (
        "run.sh must still hand the ledger the full suite glob")


def _fake_cargo_dir(tmp: Path, test_rc: int) -> Path:
    bindir = tmp / "fakebin"
    bindir.mkdir()
    cargo = bindir / "cargo"
    cargo.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = build ]; then exit 0; fi\n"
        f"exit {test_rc}\n"
    )
    cargo.chmod(cargo.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


def _run_run_sh(tmp: Path, stage: Path, test_rc: int = 3):
    """`run.sh` far enough to reach the stage lock, with nothing really built."""
    target = tmp / "target"
    (target / "release").mkdir(parents=True)
    (target / "release" / "lib_C.dylib").write_bytes(b"not a dylib")
    vendor = tmp / "vendor"
    (vendor / "torch").mkdir(parents=True)
    env = dict(os.environ)
    env["PATH"] = f"{_fake_cargo_dir(tmp, test_rc)}:{env['PATH']}"
    env["PYTHON"] = GOOD_PYTHON
    env["CARGO_TARGET_DIR"] = str(target)
    env["TORCHNATIVE_VENDOR_DIR"] = str(vendor)
    env["TORCH_C_STAGE"] = str(stage)
    proc = subprocess.run(["/bin/sh", str(RUN_SH)], env=env,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_a_second_gate_in_the_same_stage_refuses_rather_than_racing():
    if not Path(GOOD_PYTHON).exists():
        return
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stage = tmp / "stage"
        (stage / "lock").mkdir(parents=True)
        (stage / "lock" / "pid").write_text(f"{os.getpid()}\n")
        rc, out = _run_run_sh(tmp, stage)
        assert rc != 0, f"a second gate in a held stage must refuse:\n{out}"
        assert "refusing" in out, out
        assert str(os.getpid()) in out, (
            "the refusal must name the run that holds the stage:\n" + out)


def test_a_lock_left_by_a_dead_run_does_not_wedge_the_stage():
    """Refusing is honest; refusing forever because a run crashed is not."""
    if not Path(GOOD_PYTHON).exists():
        return
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stage = tmp / "stage"
        (stage / "lock").mkdir(parents=True)
        (stage / "lock" / "pid").write_text("999999\n")  # above macOS PID max
        rc, out = _run_run_sh(tmp, stage, test_rc=3)
        assert "refusing to run -- another gate" not in out, (
            "a stale lock from a dead run wedged the stage:\n" + out)
        assert rc == 3, (
            "the run should have proceeded past the stale lock and stopped at "
            f"the fake `cargo test` (exit 3), got {rc}:\n{out}")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
