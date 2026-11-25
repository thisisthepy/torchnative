"""`tests/suite_ledger.py`'s memory watchdog (CI runs tn6 to tn11).

Five Linux gate runs lost the runner with no log ("lost communication",
tn11: "received a shutdown signal" 2.5 minutes into the suites). tn11's
progress lines, the only text that survived, show `test_cia_frame.py` started
and still running at the loss while about 20 later suites finished. The
ledger now runs a watchdog on Linux that prints available memory and every
running suite's tree RSS to the progress descriptor, and kills the largest
suite's process group, failing it by name, when available memory falls below
a floor.

**What makes each test able to fail.** `run_suite` runs a real stand-in suite
(a script that starts a child and sleeps) in its own session, with a
`MemWatch` whose memory readers are fakes:

* below the floor: the suite and its child are dead within seconds (not the
  60 s they sleep), the log has the FAIL line naming the watchdog, the exit
  code is not 0, and the progress text says which suite and why;
* above the floor: nothing is killed, the suite ends on its own with 0, and a
  heartbeat line names it;
* the victim is the running suite with the largest tree;
* on Linux, the real readers answer numbers (skipped by name elsewhere).

**What this cannot find.** Whether memory is what loses the Linux runner; the
next CI run's `GATE MEM:` lines answer that.
"""

import os
import pathlib
import subprocess
import sys
import threading
import time

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "tests" / "_support"))

import _skip  # noqa: E402
import suite_ledger as sl  # noqa: E402

WORK = REPO / ".scratch" / f"memwatch-test.{os.getpid()}"

_SUITE = r'''
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[0] + ".child", "w").write(str(child.pid))
print("ok   stand-in started", flush=True)
time.sleep(float(os.environ.get("STANDIN_SLEEP", "60")))
child.kill()
print("ok   stand-in finished", flush=True)
'''


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); ask ps for its state.
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def _run(avail_mb, sleep_s, rss=lambda pid: (5000, 2)):
    WORK.mkdir(parents=True, exist_ok=True)
    suite = WORK / "test_standin.py"
    suite.write_text(_SUITE)
    log = WORK / "test_standin.py.log"
    lines = []
    watch = sl.MemWatch(lines.append, interval=0.2, heartbeat=0.0, floor_mb=1024,
                        meminfo=lambda: (avail_mb, 16000), tree_rss=rss)
    sl._MEM_KILLED.clear()
    watch.start()
    started = time.monotonic()
    try:
        rc, took, timed_out = sl.run_suite(sys.executable, suite, log,
                                           dict(os.environ, STANDIN_SLEEP=str(sleep_s)), 120,
                                           own_session=True)
    finally:
        watch.stopped.set()
        watch.join(5)
    child = int((WORK / "test_standin.py.child").read_text())
    return rc, time.monotonic() - started, timed_out, log.read_text(), lines, child, watch


def _cleanup():
    for p in sorted(WORK.glob("*")):
        p.unlink()
    if WORK.exists():
        WORK.rmdir()


def test_below_the_floor_the_largest_suite_and_its_children_are_killed_by_name():
    if not hasattr(os, "killpg"):
        _skip.skip("POSIX process groups")
        return
    try:
        rc, took, timed_out, text, lines, child, watch = _run(avail_mb=300, sleep_s=60)
        assert took < 30, took
        assert rc != 0 and not timed_out, (rc, timed_out)
        assert "FAIL test_standin: (whole suite) -- killed by the memory watchdog: available memory 300 MB" \
            in text, text
        assert watch.kills == ["test_standin.py"], watch.kills
        assert any("GATE MEM: test_standin.py killed by the memory watchdog" in ln for ln in lines), lines
        deadline = time.monotonic() + 5
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(child), "the suite's child survived the kill"
    finally:
        _cleanup()
    print("ok   ledger_memwatch: below the floor the suite and its child die within seconds, FAIL by name")


def test_above_the_floor_nothing_is_killed_and_the_heartbeat_names_the_suite():
    try:
        rc, took, timed_out, text, lines, child, watch = _run(avail_mb=8000, sleep_s=1.5)
        assert rc == 0 and not timed_out, (rc, text)
        assert watch.kills == [] and "memory watchdog" not in text, (watch.kills, text)
        assert any(ln.startswith("GATE MEM: available 8000 MB of 16000; running: test_standin.py 5000 MB/2 proc")
                   for ln in lines), lines
    finally:
        _cleanup()
    print("ok   ledger_memwatch: above the floor nothing is killed; the heartbeat names the running suite")


def test_the_victim_is_the_largest_tree():
    class P:
        def __init__(self, pid):
            self.pid = pid
    sizes = {1: (300, 1), 2: (4000, 3), 3: (900, 2)}
    killed = []
    watch = sl.MemWatch(lambda _l: None, floor_mb=1024, meminfo=lambda: (100, 16000),
                        tree_rss=lambda pid: sizes[pid], kill=lambda proc: killed.append(proc.pid))
    sl._MEM_KILLED.clear()
    with sl._RUNNING_LOCK:
        saved = dict(sl._RUNNING)
        sl._RUNNING.clear()
        sl._RUNNING.update({"a.py": (P(1), time.monotonic()), "b.py": (P(2), time.monotonic()),
                            "c.py": (P(3), time.monotonic())})
    try:
        watch.poll(float("-inf"))
        watch.poll(float("-inf"))
    finally:
        with sl._RUNNING_LOCK:
            sl._RUNNING.clear()
            sl._RUNNING.update(saved)
        sl._MEM_KILLED.clear()
    # b.py is still the largest at the second poll (the fake never shrinks):
    # it is killed once, not twice.
    assert killed == [2], killed
    print("ok   ledger_memwatch: the largest tree is the one killed, once")


def test_the_real_readers_answer_on_linux():
    if not os.path.exists("/proc/meminfo"):
        _skip.skip("/proc/meminfo is Linux's")
        return
    avail, total = sl._meminfo_mb()
    assert 0 < avail <= total, (avail, total)
    mb, count = sl._tree_rss_mb(os.getpid())
    assert mb > 0 and count >= 1, (mb, count)
    print(f"ok   ledger_memwatch: /proc readers answer ({avail} of {total} MB; this process {mb} MB)")


if __name__ == "__main__":
    failures = _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")],
        suite="test_ledger_memwatch",
    )
    raise SystemExit(1 if failures else 0)
