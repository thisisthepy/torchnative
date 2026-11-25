"""`tests/golden/compare.py`'s oracle-crash guard (#369, CI run 45056531650).

Run 45056531650's Windows runner (Intel family 6 model 207, AVX-512) died with
STATUS_ILLEGAL_INSTRUCTION inside upstream torch's own float32
`aten._grouped_mm`, the oracle side of the first golden case, so the golden
self-test and the standalone comparison ended without a SUMMARY and the gate
went red with no case named. `compare.py` now runs as a child process that
records which side, op and case it is about to run; `_drive` reads that record
when the child dies natively, leaves the op out when the death was upstream's,
and runs again.

**What makes each test able to fail.** `_drive` is driven with a stand-in
child (a `python -c` script) that kills itself with SIGILL on chosen ops, so
the guard is exercised without the shim and without the hardware that crashes:

* an upstream death on two ops: both left out, named, the run completes;
* a death on the shim side: fatal, returned, never left out;
* a child that dies on every op: stops at the limit instead of looping;
* an ordinary failure (exit 1): returned as it is, no re-run;
* `_native_death` on Windows NTSTATUS codes, POSIX signals and plain codes.

**What this cannot find.** Whether the real upstream call crashes on a given
CPU; that is the runner's, and the CI log names it (`ORACLE CRASH:` lines).
A native death while a case builder runs is reported as fatal, not left out.
"""

import os
import pathlib
import signal
import subprocess
import sys

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(REPO / "tests" / "golden"))
sys.path.insert(0, str(REPO / "tests" / "_support"))

import _skip  # noqa: E402
import compare  # noqa: E402

SCRATCH = REPO / ".scratch"

#: The stand-in for `compare.py` as a child: it walks ops a..e, records each
#: as `compare._mark` does, and dies with SIGILL on the ops in BAD whose side
#: is SIDE, unless `_drive` has told it to leave them out.
_CHILD = r'''
import os, signal, sys
bad = set(sys.argv[1].split(",")) if sys.argv[1] else set()
side = sys.argv[2]
left_out = set(filter(None, os.environ["TORCHNATIVE_GOLDEN_ORACLE_CRASHED"].split(",")))
assert os.environ["TORCHNATIVE_GOLDEN_CHILD"] == "1"
for op in "abcde":
    if op in left_out:
        continue
    with open(os.environ["TORCHNATIVE_GOLDEN_PROGRESS"], "w") as f:
        f.write(side + "\t" + op + "\tcase of " + op)
    if op in bad:
        os.kill(os.getpid(), signal.SIGILL)
print("CHILD DONE left_out=" + ",".join(sorted(left_out)), flush=True)
'''


def _posix_only():
    if sys.platform == "win32":
        _skip.skip("the stand-in child kills itself with a POSIX signal")
        return False
    return True


def _drive(bad, side, limit=compare._ORACLE_CRASH_LIMIT):
    SCRATCH.mkdir(exist_ok=True)
    progress = str(SCRATCH / f"oracle-crash-test.{os.getpid()}")
    out = SCRATCH / f"oracle-crash-test.{os.getpid()}.out"
    try:
        with open(out, "w") as f:
            sys.stdout.flush()
            saved = os.dup(1)
            os.dup2(f.fileno(), 1)
            try:
                rc = compare._drive([sys.executable, "-c", _CHILD, ",".join(bad), side],
                                    dict(os.environ), progress, limit=limit)
            finally:
                sys.stdout.flush()
                os.dup2(saved, 1)
                os.close(saved)
        return rc, out.read_text()
    finally:
        for p in (progress, str(out)):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


def test_an_upstream_death_leaves_the_op_out_and_the_run_completes():
    if not _posix_only():
        return
    rc, text = _drive(["b", "d"], "torch")
    assert rc == 0, (rc, text)
    assert text.count("ORACLE CRASH:") == 2, text
    assert "running b :: case of b" in text and "running d :: case of d" in text, text
    assert "CHILD DONE left_out=b,d" in text, text
    print("ok   golden_oracle_crash: two upstream deaths left out by name, the third run completed")


def test_a_death_on_the_shim_side_is_fatal():
    if not _posix_only():
        return
    rc, text = _drive(["c"], "shim")
    assert compare._native_death(rc), (rc, text)
    assert "FATAL: the shim side died natively" in text and "running c :: case of c" in text, text
    assert "ORACLE CRASH" not in text and "CHILD DONE" not in text, text
    print("ok   golden_oracle_crash: a native death in the shim is fatal and named")


def test_the_guard_gives_up_at_its_limit():
    if not _posix_only():
        return
    rc, text = _drive(list("abcde"), "torch", limit=2)
    assert compare._native_death(rc), (rc, text)
    assert text.count("ORACLE CRASH:") == 2 and "giving up" in text, text
    print("ok   golden_oracle_crash: stops after the limit instead of re-running forever")


def test_an_ordinary_failure_is_returned_without_a_rerun():
    SCRATCH.mkdir(exist_ok=True)
    progress = str(SCRATCH / f"oracle-crash-test-plain.{os.getpid()}")
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path[:0] = [%r]; import compare, os\n"
         "n = [0]\n"
         "import subprocess\n"
         "real = subprocess.run\n"
         "def counting(*a, **k):\n"
         "    n[0] += 1\n"
         "    return real(*a, **k)\n"
         "subprocess.run = counting\n"
         "rc = compare._drive([sys.executable, '-c', 'raise SystemExit(1)'], dict(os.environ), %r)\n"
         "print('RC', rc, 'RUNS', n[0])" % (str(REPO / "tests" / "golden"), progress)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RC 1 RUNS 1" in proc.stdout, proc.stdout
    print("ok   golden_oracle_crash: exit 1 is returned as it is, one run")


def test_native_death_reads_ntstatus_and_signals():
    assert compare._native_death(0xC000001D)  # STATUS_ILLEGAL_INSTRUCTION (run 45056531650)
    assert compare._native_death(0xC0000005)  # access violation
    assert compare._native_death(0xC0000409)  # fast fail
    assert compare._native_death(-signal.SIGILL)
    assert compare._native_death(-signal.SIGSEGV)
    for rc in (0, 1, 2, 132):
        assert not compare._native_death(rc), rc
    print("ok   golden_oracle_crash: NTSTATUS failure codes and signals are native deaths; 0/1/2/132 are not")


if __name__ == "__main__":
    failures = _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")],
        suite="test_golden_oracle_crash",
    )
    raise SystemExit(1 if failures else 0)
