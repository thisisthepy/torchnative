"""Run every suite, print it, and then prove the printed total is the real one.

`run.sh` used to do this in five lines of shell::

    rm -rf "$suite_logs"; mkdir -p "$suite_logs"
    for suite in .../test_*.py; do
        ... > "$suite_logs/$name.log" 2>&1 || suite_failed=1
        cat "$suite_logs/$name.log"
    done

and `$suite_logs` was derived from the *repository path*, so two gates in one
worktree shared it and the second one's `rm -rf` deleted the first one's logs
mid-run. Both halves of the resulting false green were observed on the same
day (2026-09-17):

  * one run died on `cat: .../test_shim.py.log: No such file or directory` at
    **944 ok / 0 FAIL** -- a number that reads like a partial pass, because
    `set -eu` turned a corrupted run into an ordinary-looking early exit; and
  * three runs dropped `test_bf16ane.py`'s whole section -- **21 tests** --
    from the aggregate and **still exited 0**. The aggregate read 1669 where
    the suite logs summed to 1690.

Nothing in that loop could notice. `cat` is the only reader, a missing section
is simply absent, and no count is ever compared with anything.

So the loop moved here, where the accounting can exist:

* **Every suite file is accounted for.** The caller passes its glob expansion;
  this script *independently* globs `test_*.py` in the pytests directory and
  refuses if the two sets differ. A suite that silently stopped running is
  then arithmetic (`suites=83/84`, named) rather than absence.
* **A lost or unreadable log fails loudly, naming the suite.** Not `cat:` on
  stderr, and emphatically not `cat ... || true`: silencing the read error
  would convert a loud corruption into the quiet one this file exists to stop.
* **The total is reconcilable.** Each suite's counts are taken twice -- once
  from the bytes printed into the aggregate, once by re-reading the log from
  disk after every suite has finished -- and a disagreement fails the run,
  naming the suite. That second pass is what catches a log destroyed *after*
  it was printed, which is exactly what the concurrent `rm -rf` did.

The `SUITE LEDGER:` line at the end is therefore the only total worth quoting
from a gate run: it is the one that has been checked against the files.

`run.sh` additionally stages these logs in a per-run, PID-suffixed directory
and holds a lock on `$stage`, so the collision cannot be staged in the first
place. This script is the second line: it assumes nothing about who else is
on the machine and verifies what it reports.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_OK = re.compile(r"^ok\s")
_FAIL = re.compile(r"^FAIL\s")
_SKIP = re.compile(r"^SKIP\s")


def tally(text):
    """(ok, fail, skip) counted the way a reader counts a gate log."""
    ok = fail = skip = 0
    for line in text.splitlines():
        if _OK.match(line):
            ok += 1
        elif _FAIL.match(line):
            fail += 1
        elif _SKIP.match(line):
            skip += 1
    return ok, fail, skip


def _read_log(path):
    """The log's text, or a reason it could not be read. Never swallowed."""
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", "replace"), None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", required=True,
                    help="directory to write one <suite>.log per suite into")
    ap.add_argument("--pytests", required=True,
                    help="the suite directory, globbed independently of the "
                         "caller's list so a dropped suite is detectable")
    ap.add_argument("--python", default=None,
                    help="interpreter for the suites (default: this one)")
    ap.add_argument("--suite-env", action="append", default=[],
                    metavar="NAME=VALUE",
                    help="environment entry for each suite subprocess")
    ap.add_argument("suites", nargs="*", help="the caller's glob expansion")
    args = ap.parse_args(argv)

    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    python = args.python or sys.executable
    env = dict(os.environ)
    for entry in args.suite_env:
        name, _, value = entry.partition("=")
        env[name] = value

    given = [Path(s) for s in args.suites]
    problems = []

    # 1. The caller's list against the directory. A suite that stopped being
    #    run -- a half-expanded glob, a file renamed out of the pattern -- is
    #    invisible from inside the loop, so it is checked before the loop.
    on_disk = sorted(p.name for p in Path(args.pytests).glob("test_*.py"))
    listed = sorted(p.name for p in given)
    if listed != on_disk:
        missing = [n for n in on_disk if n not in listed]
        extra = [n for n in listed if n not in on_disk]
        if missing:
            problems.append(
                "suite files present in %s but NOT RUN: %s"
                % (args.pytests, ", ".join(missing)))
        if extra:
            problems.append(
                "suites run that are not files in %s: %s"
                % (args.pytests, ", ".join(extra)))

    printed = {}   # suite -> (ok, fail, skip) as it went into the aggregate
    ran_rc = {}
    suite_failed = 0

    for suite in given:
        name = suite.name
        log_path = logs / (name + ".log")
        print(f"--- {name} ---", flush=True)
        with open(log_path, "wb") as fh:
            rc = subprocess.call([python, str(suite)], stdout=fh,
                                 stderr=subprocess.STDOUT, env=env)
        ran_rc[name] = rc
        if rc != 0:
            suite_failed = 1
        text, why = _read_log(log_path)
        if text is None:
            # The `cat` failure, named and owned. Not silenced: the run is
            # over, because a suite whose output cannot be read has not been
            # observed, whatever its exit status was.
            print(f"SUITE LEDGER: FATAL -- {name}: its log "
                  f"({log_path}) could not be read: {why}\n"
                  "Another process deleted or replaced it while this gate was "
                  "running. Nothing in this run's totals can be trusted; "
                  "re-run it alone.", flush=True)
            return 1
        sys.stdout.write(text)
        sys.stdout.flush()
        printed[name] = tally(text)

    # 2. Read every log again, from disk, now that every suite has finished.
    #    This is the pass that catches a log destroyed or rewritten *after*
    #    it was printed -- the 1669-vs-1690 shape.
    total_printed = [0, 0, 0]
    total_disk = [0, 0, 0]
    for name, counts in printed.items():
        text, why = _read_log(logs / (name + ".log"))
        if text is None:
            problems.append(
                f"{name}: its log vanished after it was printed ({why}) -- "
                "another run deleted it; this run's totals are not evidence")
            for i in range(3):
                total_printed[i] += counts[i]
            continue
        again = tally(text)
        for i in range(3):
            total_printed[i] += counts[i]
            total_disk[i] += again[i]
        if again != counts:
            problems.append(
                f"{name}: printed ok/FAIL/SKIP {counts} but its log now "
                f"reads {again} -- the aggregate and the suite logs disagree")

    # 3. Nothing else may be in the log directory. An extra file means two
    #    runs shared it, which is the condition this whole file is about.
    expected_logs = {p.name + ".log" for p in given}
    stray = sorted(p.name for p in logs.glob("*.log")
                   if p.name not in expected_logs)
    if stray:
        problems.append(
            "log directory %s holds logs this run did not write: %s -- a "
            "second gate is staging into it" % (logs, ", ".join(stray)))

    failed_suites = sorted(n for n, rc in ran_rc.items() if rc != 0)
    print("SUITE LEDGER: suites=%d/%d ok=%d FAIL=%d SKIP=%d "
          "(printed and re-read from %s; they agree)"
          % (len(printed), len(on_disk) or len(given),
             total_printed[0], total_printed[1], total_printed[2], logs)
          if not problems else
          "SUITE LEDGER: suites=%d/%d ok=%d FAIL=%d SKIP=%d -- NOT RECONCILED"
          % (len(printed), len(on_disk) or len(given),
             total_printed[0], total_printed[1], total_printed[2]),
          flush=True)
    if failed_suites:
        print("SUITE LEDGER: suites exiting non-zero: "
              + ", ".join(failed_suites), flush=True)
    for line in problems:
        print("SUITE LEDGER: FAIL -- " + line, flush=True)
    if problems:
        print("SUITE LEDGER: this run lost or altered suite output while it "
              "was running. A gate that cannot account for its own logs "
              "cannot report a total.", flush=True)
        return 1
    return 1 if suite_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
