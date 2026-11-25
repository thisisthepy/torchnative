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
  this script *independently* walks `test_*.py` under the pytests directory (recursively) and
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

* **On a CI runner (`--runner`, issue #24) nothing drops out unnamed.** The
  plan comes from `.github/scripts/gate_suites.py`. A suite in neither its CI list nor
  its local-only list stops the run before any suite executes. A suite not
  planned for the runner is not executed, and its log holds one line --
  `SKIP <suite>: (whole suite) -- not run on CI runner '<r>': <reason>` -- so
  it is in the SKIP column and named, and the `CI PLAN:` line lists every one.
  Without `--runner` (the local gate) every suite runs, as before.

* **Suites may run side by side, and the ledger is the same (issue #335).**
  `--jobs N` (or `TORCHNATIVE_GATE_JOBS`; 1 unless set, so the Mac's gate runs
  exactly as before) runs the suites `.github/scripts/gate_suites.py` calls
  parallel-safe N at a time, and every other suite afterwards, one at a time
  and alone. A suite is still one subprocess with one log, the aggregate is
  still printed in the caller's order, and the totals are still counted twice,
  so `suites=N/N ok= FAIL= SKIP=` cannot differ from a one-at-a-time run of
  the same tree for any reason except a suite behaving differently under
  load, which is what the serial list exists for.
* **Lanes (issue #343).** A serial suite the manifest puts in a lane (`gate_suites.lane`)
  runs one at a time with the other suites of its lane, while the lanes and
  the side-by-side suites run at once; a serial suite with no lane runs
  afterwards with nothing beside it.
* **Shards (leader decision 2026-11-15).** `--shard K/N` (run.sh passes
  `TORCHNATIVE_GATE_SHARD`; unset, the local gate is unchanged) runs only the
  suites `.github/scripts/gate_suites.py shard()` gives shard K, so the macOS
  gate can be three jobs of one run. The macOS job on one runner took 3 jobs of
  work (run 43670318805: side-by-side pool 1751 s, lane mps 1964 s, lane
  distributed 2359 s, exclusive suites 687 s, plus setup and DOCWATCH). A suite
  of another shard is not executed; its log holds one line,
  `SHARD <suite>: (whole suite) -- runs in shard J of N`, which is neither ok,
  FAIL nor SKIP, so the totals count it nowhere, and the aggregate still has
  every suite's header, so `suites=N/N` is the same on every shard. The
  whole-suite SKIP for a suite the runner cannot run is printed only by the
  shard that owns it, so the three shards' ok/FAIL/SKIP add up to one runner's.
  `SUITE SHARD: k= n= owned= names=<sha256 of the sorted owned names>` is what
  `gate_suites.py report` reads to check that every suite ran in exactly one
  shard; the owned names are also written to `suite-shard.txt` beside the logs.

* **A suite that hangs is named, not waited for.** `--suite-timeout S` (or
  `TORCHNATIVE_SUITE_TIMEOUT_S`, default 2700) kills a suite that runs longer
  and appends a `FAIL <suite>: (whole suite) -- timed out ...` line to its log,
  so the hang is in the FAIL column and the job is not held for hours.
* **Where the time went is printed.** `SUITE TIME:` lines name the ten slowest
  suites, `SUITE TIMES:` gives the sum and the wall clock, and every suite's
  seconds are written to `suite-times.tsv` beside the logs.

* **A suite that exhausts memory is named and killed, not the runner.** On
  Linux (`/proc/meminfo`; off with `TORCHNATIVE_GATE_MEMWATCH=0`) a watchdog
  thread prints `GATE MEM:` lines to the progress descriptor every
  `TORCHNATIVE_GATE_MEMWATCH_HEARTBEAT_S` (default 30): available memory and
  each running suite with the RSS of its whole process tree. When available
  memory falls below `TORCHNATIVE_GATE_MEMWATCH_FLOOR_MB` (default the larger
  of 1024 MB and 8% of the total), it kills the process group of the running
  suite with the largest tree and that suite's log gets a FAIL line saying
  so. CI runs tn6, tn7, tn9, tn10 and tn11 lost the Linux runner ("lost
  communication" / "received a shutdown signal") with no log; tn11's
  progress lines, the only text that survived, show `test_cia_frame.py`
  started and still running at the loss while about 20 later suites
  finished. Suites run in their own session while the watchdog is on, so the
  kill reaches their subprocesses; a timeout kills the group too.

* **Progress can be read while the gate runs.** With
  `TORCHNATIVE_GATE_PROGRESS_FD=N` one line per finished suite goes to file
  descriptor N as it finishes. `test.yml` points it at the step's own output,
  so a job that is cancelled or times out still shows which suites ran and
  how long each took (CI run 43645421136: the macOS job was cancelled at 51
  minutes and left nothing to read, because the gate log is printed at the end).

`run.sh` additionally stages these logs in a per-run, PID-suffixed directory
and holds a lock on `$stage`, so the collision cannot be staged in the first
place. This script is the second line: it assumes nothing about who else is
on the machine and verifies what it reports.
"""

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_OK = re.compile(r"^ok\s")
_FAIL = re.compile(r"^FAIL\s")
_SKIP = re.compile(r"^SKIP\s")


_MANIFEST = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "gate_suites.py"


def _load_manifest(path):
    spec = importlib.util.spec_from_file_location("_gate_suites", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _env_int(name, default):
    """An integer from the environment; a value that is not one is refused."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"SUITE LEDGER: FATAL -- {name}={raw!r} is not an integer")


#: name -> Popen of each suite running now, for the memory watchdog.
_RUNNING = {}
_RUNNING_LOCK = threading.Lock()
#: name -> why the watchdog killed it.
_MEM_KILLED = {}


def _meminfo_mb():
    """(MemAvailable, MemTotal) in MB from /proc/meminfo, or None."""
    try:
        with open("/proc/meminfo") as fh:
            vals = {}
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemAvailable", "MemTotal"):
                    vals[key] = int(rest.split()[0]) // 1024
        return vals["MemAvailable"], vals["MemTotal"]
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _tree_rss_mb(pid):
    """(RSS in MB, process count) of `pid` and all its descendants, from /proc."""
    children, rss = {}, {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry) as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            ppid, pages = int(fields[1]), int(fields[21])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
        rss[int(entry)] = pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
    total, count, todo = 0, 0, [pid]
    while todo:
        p = todo.pop()
        total += rss.get(p, 0)
        count += 1 if p in rss else 0
        todo.extend(children.get(p, ()))
    return total, count


class MemWatch(threading.Thread):
    """The memory watchdog (see the module docstring). `meminfo` and
    `tree_rss` are injectable so the kill path is testable off Linux."""

    def __init__(self, progress, interval=5.0, heartbeat=30.0, floor_mb=None,
                 meminfo=_meminfo_mb, tree_rss=_tree_rss_mb, kill=None):
        super().__init__(daemon=True, name="gate-memwatch")
        self.progress, self.interval, self.heartbeat = progress, interval, heartbeat
        self.floor_mb, self.meminfo, self.tree_rss = floor_mb, meminfo, tree_rss
        self.kill = kill or (lambda proc: os.killpg(proc.pid, 9))
        self.stopped = threading.Event()
        self.kills = []

    def snapshot(self):
        with _RUNNING_LOCK:
            running = dict(_RUNNING)
        rows = []
        for name, (proc, started) in running.items():
            mb, count = self.tree_rss(proc.pid)
            rows.append((mb, name, count, time.monotonic() - started, proc))
        return sorted(rows, key=lambda r: (-r[0], r[1]))

    def poll(self, last_beat):
        mem = self.meminfo()
        if mem is None:
            return last_beat
        avail, total = mem
        floor = self.floor_mb if self.floor_mb is not None else max(1024, total * 8 // 100)
        rows = self.snapshot()
        now = time.monotonic()
        if now - last_beat >= self.heartbeat:
            self.progress("GATE MEM: available %d MB of %d; running: %s" % (
                avail, total, ", ".join("%s %d MB/%d proc %.0f s" % (n, mb, c, t)
                                        for mb, n, c, t, _ in rows) or "none"))
            last_beat = now
        if avail < floor and rows:
            mb, name, count, took, proc = rows[0]
            why = ("killed by the memory watchdog: available memory %d MB fell below the "
                   "floor %d MB (of %d) while it held %d MB in %d process(es) after %.0f s"
                   % (avail, floor, total, mb, count, took))
            with _RUNNING_LOCK:
                if name in _MEM_KILLED:
                    return last_beat
                _MEM_KILLED[name] = why
            self.progress("GATE MEM: %s %s" % (name, why))
            self.kills.append(name)
            try:
                self.kill(proc)
            except OSError:
                pass
        return last_beat

    def run(self):
        last_beat = float("-inf")
        while not self.stopped.wait(self.interval):
            last_beat = self.poll(last_beat)


def _memwatch_wanted():
    return os.path.exists("/proc/meminfo") and os.environ.get("TORCHNATIVE_GATE_MEMWATCH", "1") != "0"


def run_suite(python, suite, log_path, env, timeout, own_session=False):
    """Run one suite into its log. `(exit code, seconds, timed out)`.

    A suite that outlives `timeout` seconds (0: no limit) is killed, and the
    reason goes into its log as a FAIL line, so the ledger counts it and a
    reader finds it by name.
    """
    started = time.monotonic()
    timed_out = False
    # The suite sees the timeout it runs under, so `_skip.run_tests` can arm
    # a stack dump before the kill below (which loses the suite's buffered
    # stdout and leaves only "timed out" to read).
    if timeout:
        env = dict(env, TORCHNATIVE_SUITE_TIMEOUT_S=str(timeout))
    name = Path(suite).name
    with open(log_path, "wb") as fh:
        proc = subprocess.Popen([python, str(suite)], stdout=fh,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=own_session)
        with _RUNNING_LOCK:
            _RUNNING[name] = (proc, started)
        try:
            rc = proc.wait(timeout=timeout or None)
        except subprocess.TimeoutExpired:
            if own_session:
                try:
                    os.killpg(proc.pid, 9)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.wait()
            timed_out = True
            rc = 124
        finally:
            with _RUNNING_LOCK:
                _RUNNING.pop(name, None)
                mem_why = _MEM_KILLED.get(name)
    if mem_why:
        with open(log_path, "ab") as fh:
            fh.write(("\nFAIL %s: (whole suite) -- %s; the tests it had not reached did "
                      "not run\n" % (Path(suite).stem, mem_why)).encode())
        if rc == 0:
            rc = 137
    seconds = time.monotonic() - started
    if timed_out:
        with open(log_path, "ab") as fh:
            fh.write((
                "\nFAIL %s: (whole suite) -- timed out after %d s and was "
                "killed (TORCHNATIVE_SUITE_TIMEOUT_S); the tests it had not "
                "reached did not run\n" % (Path(suite).stem, timeout)).encode())
    return rc, seconds, timed_out


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
    ap.add_argument("--runner", default=None,
                    help="a CI runner named in .github/scripts/gate_suites.py RUNNERS. "
                         "Suites not planned for it are not executed; each is "
                         "printed as a SKIP line naming it and the reason. "
                         "Unset (the local gate): every suite runs, as before")
    ap.add_argument("--manifest", default=str(_MANIFEST),
                    help="the suite classification (default: %(default)s)")
    ap.add_argument("--jobs", type=int,
                    default=_env_int("TORCHNATIVE_GATE_JOBS", 1),
                    help="how many parallel-safe suites run at once (default: "
                         "TORCHNATIVE_GATE_JOBS, else 1). Suites the manifest "
                         "does not call parallel-safe always run alone")
    ap.add_argument("--suite-timeout", type=int,
                    default=_env_int("TORCHNATIVE_SUITE_TIMEOUT_S", 2700),
                    help="seconds after which a suite is killed and counted "
                         "as a FAIL by name; 0 for no limit (default: "
                         "TORCHNATIVE_SUITE_TIMEOUT_S, else 2700)")
    # Only the flag, never the environment: tests/run.sh passes `--shard` from
    # TORCHNATIVE_GATE_SHARD, and a ledger a suite starts for itself (the gate
    # tests run fake ledgers) inherited that variable and sharded its own fake
    # manifest (bundle run 43677317812, shard 3: "module '_gate_suites' has no
    # attribute 'SHARDS'", 6 tests).
    ap.add_argument("--shard", default="",
                    metavar="K/N",
                    help="run only the suites of shard K of N (default: every "
                         "suite; tests/run.sh passes TORCHNATIVE_GATE_SHARD here). The others "
                         "get one SHARD line and are not executed")
    ap.add_argument("suites", nargs="*", help="the caller's glob expansion")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        print(f"SUITE LEDGER: FATAL -- --jobs {args.jobs} is not a positive number", flush=True)
        return 1
    if args.suite_timeout < 0:
        print(f"SUITE LEDGER: FATAL -- --suite-timeout {args.suite_timeout} is negative", flush=True)
        return 1

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
    #    Suites live in function subfolders (tests/ops/, tests/devices/mps/,
    #    ...), so the directory is walked recursively and suites are compared
    #    by path relative to it. Logs and the ledger are still keyed by the
    #    file's basename, which is what a reader recognises -- so a basename
    #    that occurs twice is refused: two suites would share one log.
    root = Path(args.pytests).resolve()

    def _rel(p):
        try:
            return p.resolve().relative_to(root).as_posix()
        except ValueError:
            return p.name

    on_disk = sorted(_rel(p) for p in root.rglob("test_*.py")
                     if "__pycache__" not in p.parts)
    listed = sorted(_rel(p) for p in given)
    for label, names in (("on disk", on_disk), ("listed", listed)):
        base = [n.rsplit("/", 1)[-1] for n in names]
        dup = sorted({b for b in base if base.count(b) > 1})
        if dup:
            problems.append("suite basenames %s are not unique (%s): their "
                            "logs would collide" % (label, ", ".join(dup)))
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

    # 1b. On a CI runner (issue #24), the plan for it. Refused before any
    #     suite runs if a suite is in neither the CI list nor the local-only
    #     list: a CI that goes green on a subset nobody chose is the failure
    #     this exists to stop (.github/scripts/gate_suites.py's docstring).
    not_here = {}
    gs = None
    if args.runner is not None:
        try:
            gs = _load_manifest(args.manifest)
            _run, not_here, plan_problems = gs.plan(args.runner, on_disk)
        except Exception as exc:                          # noqa: BLE001
            print(f"SUITE LEDGER: FATAL -- cannot plan for CI runner "
                  f"{args.runner!r}: {type(exc).__name__}: {exc}", flush=True)
            return 1
        if plan_problems:
            for line in plan_problems:
                print("SUITE LEDGER: FAIL -- " + line, flush=True)
            print("SUITE LEDGER: refusing to run on CI runner %r: the suites "
                  "run on CI plus the local-only suites would not be the full "
                  "gate." % args.runner, flush=True)
            return 1

    # 1b'. Which shard this run is (leader decision 2026-11-15). A suite of
    #      another shard is decided here, before `not_here`: the SKIP line for a
    #      suite the runner cannot run belongs to the shard that owns it.
    elsewhere = {}
    shard_k = shard_n = None
    if args.shard.strip():
        try:
            k_text, _, n_text = args.shard.partition("/")
            shard_k, shard_n = int(k_text), int(n_text)
            if gs is None:
                gs = _load_manifest(args.manifest)
            if shard_n != len(gs.SHARDS) or shard_k not in gs.SHARDS:
                raise ValueError(f"the manifest has shards {tuple(gs.SHARDS)}")
            for rel in on_disk:
                owner = gs.shard(rel)
                if owner not in gs.SHARDS:
                    raise ValueError(f"{rel} has shard {owner!r}")
                if owner != shard_k:
                    elsewhere[rel] = owner
        except Exception as exc:                              # noqa: BLE001
            print(f"SUITE LEDGER: FATAL -- cannot use --shard {args.shard!r}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return 1

    # 1c. Which suites may share the machine (issue #335). Only asked when
    #     more than one job is wanted; with one job every suite runs alone, in
    #     the caller's order, as it always did. A manifest that cannot answer
    #     is not guessed for: every suite then runs alone and the line says so.
    serial_why = {}
    lane_of = {}   # serial suite -> its lane; a serial suite with none is exclusive
    if args.jobs > 1:
        try:
            if gs is None:
                gs = _load_manifest(args.manifest)
            lane_fn = getattr(gs, "lane", None)
            for suite in given:
                why = gs.serial_reason(_rel(suite))
                if why is not None:
                    serial_why[suite.name] = why
                    which = lane_fn(_rel(suite)) if lane_fn else None
                    if which is not None:
                        lane_of[suite.name] = which
        except Exception as exc:                              # noqa: BLE001
            print(f"SUITE LEDGER: --jobs {args.jobs} ignored, every suite runs "
                  f"alone: the manifest cannot say which suites are "
                  f"parallel-safe ({type(exc).__name__}: {exc})", flush=True)
            serial_why = {suite.name: "manifest unreadable" for suite in given}
            lane_of = {}

    printed = {}   # suite -> (ok, fail, skip) as it went into the aggregate
    ran_rc = {}
    seconds = {}   # suite -> wall-clock seconds, executed suites only
    timed_out = []
    suite_failed = 0

    # The aggregate is printed in the caller's order whatever order the suites
    # finish in: a suite's section goes out once every suite before it has.
    # `header_out` is the one case where the header precedes the run, a suite
    # running alone that is next in line, which is every suite when jobs is 1;
    # so a one-job gate still shows the name of the suite it is waiting on.
    lock = threading.Lock()
    finished = set()
    header_out = set()
    position = [0]
    fatal = []

    def emit_ready():
        while position[0] < len(given) and not fatal:
            name = given[position[0]].name
            if name not in finished:
                return
            log_path = logs / (name + ".log")
            if name not in header_out:
                print(f"--- {name} ---", flush=True)
            text, why = _read_log(log_path)
            if text is None:
                # The `cat` failure, named and owned. Not silenced: the run is
                # over, because a suite whose output cannot be read has not
                # been observed, whatever its exit status was.
                fatal.append(
                    f"SUITE LEDGER: FATAL -- {name}: its log "
                    f"({log_path}) could not be read: {why}\n"
                    "Another process deleted or replaced it while this gate was "
                    "running. Nothing in this run's totals can be trusted; "
                    "re-run it alone.")
                return
            sys.stdout.write(text)
            sys.stdout.flush()
            printed[name] = tally(text)
            position[0] += 1

    progress_fd = _env_int("TORCHNATIVE_GATE_PROGRESS_FD", 0)

    def progress(text):
        if progress_fd > 0:
            try:
                os.write(progress_fd, (text + "\n").encode())
            except OSError:
                pass  # a closed descriptor must not change the gate's verdict

    memwatch = None
    if _memwatch_wanted():
        memwatch = MemWatch(progress,
                            heartbeat=_env_int("TORCHNATIVE_GATE_MEMWATCH_HEARTBEAT_S", 30),
                            floor_mb=(_env_int("TORCHNATIVE_GATE_MEMWATCH_FLOOR_MB", 0) or None))
        memwatch.start()
        progress("GATE MEM: watchdog on (floor %s MB, heartbeat %d s)" % (
            memwatch.floor_mb or "max(1024, 8%)", memwatch.heartbeat))

    def mode_of(name):
        if args.jobs == 1 or (name in serial_why and name not in lane_of):
            return "alone"
        if name in lane_of:
            return "lane " + lane_of[name]
        return "side by side"

    def execute(suite, alone):
        name = suite.name
        log_path = logs / (name + ".log")
        if alone:
            with lock:
                if position[0] < len(given) and given[position[0]].name == name:
                    print(f"--- {name} ---", flush=True)
                    header_out.add(name)
        rc, took, killed = run_suite(python, suite, log_path, env,
                                     args.suite_timeout, own_session=memwatch is not None)
        with lock:
            ran_rc[name] = rc
            seconds[name] = took
            if killed:
                timed_out.append(name)
            finished.add(name)
            progress("GATE PROGRESS: %d/%d %s exit=%d %.1f s (%s)%s" % (
                len(seconds), len(to_run), name, rc, took, mode_of(name),
                " TIMED OUT" if killed else ""))
            emit_ready()

    wall_started = time.monotonic()
    to_run = []
    for suite in given:
        name = suite.name
        rel = _rel(suite)
        if rel in elsewhere:
            # Another shard's suite: not executed and not counted here. The
            # line is none of ok, FAIL, SKIP, so the totals do not see it.
            stem = name[:-3] if name.endswith(".py") else name
            with open(logs / (name + ".log"), "wb") as fh:
                fh.write(("SHARD %s: (whole suite) -- runs in shard %d of %d\n"
                          % (stem, elsewhere[rel], shard_n)).encode())
            ran_rc[name] = 0
            finished.add(name)
        elif rel in not_here:
            # Not executed, and not silent: the line goes into the suite's
            # log like any output, so the re-read below reconciles it too.
            with open(logs / (name + ".log"), "wb") as fh:
                fh.write((gs.skip_line(name, args.runner, not_here[rel])
                          + "\n").encode())
            ran_rc[name] = 0
            finished.add(name)
        else:
            to_run.append(suite)

    together = [s for s in to_run if args.jobs > 1 and s.name not in serial_why]
    lanes = {}
    for s in to_run:
        if s.name in lane_of:
            lanes.setdefault(lane_of[s.name], []).append(s)
    in_lanes = {s.name for group in lanes.values() for s in group}
    alone = [s for s in to_run if s not in together and s.name not in in_lanes]
    if args.jobs > 1:
        print("SUITE PLAN: jobs=%d, %d suite(s) run side by side, %d in %d "
              "lane(s) (%s) at the same time, then %d alone, one at a time"
              % (args.jobs, len(together), len(in_lanes), len(lanes),
                 ", ".join("%s %d" % (k, len(v)) for k, v in sorted(lanes.items()))
                 or "none", len(alone)),
              flush=True)
    with lock:
        emit_ready()

    def run_lane(group):
        for s in group:
            if fatal:
                return
            execute(s, False)

    if together or lanes:
        # The side-by-side pool and one thread per lane, all at once. A lane
        # runs its own suites in the caller's order, one at a time.
        with ThreadPoolExecutor(max_workers=args.jobs + len(lanes)) as pool:
            futures = [pool.submit(run_lane, g) for _, g in sorted(lanes.items())]
            with ThreadPoolExecutor(max_workers=args.jobs) as side:
                futures += [side.submit(execute, s, False) for s in together]
                for future in futures:
                    future.result()
    for suite in alone:
        if fatal:
            break
        execute(suite, True)
    wall = time.monotonic() - wall_started
    if fatal:
        print(fatal[0], flush=True)
        return 1
    if any(rc != 0 for rc in ran_rc.values()):
        suite_failed = 1

    # Where the time went. Not in the ok/FAIL/SKIP columns: none of these
    # lines starts with one of those words.
    with open(logs / "suite-times.tsv", "w", encoding="utf-8") as fh:
        fh.write("seconds\tmode\tsuite\n")
        for name in sorted(seconds, key=lambda n: -seconds[n]):
            fh.write("%.1f\t%s\t%s\n" % (seconds[name], mode_of(name).replace(" ", "-"), name))
    for name in sorted(seconds, key=lambda n: -seconds[n])[:10]:
        print("SUITE TIME: %7.1f s  %s  (%s)" % (seconds[name], name, mode_of(name)), flush=True)
    print("SUITE TIMES: executed=%d sum=%.0f s wall=%.0f s jobs=%d timeout=%s"
          % (len(seconds), sum(seconds.values()), wall, args.jobs,
             ("%d s" % args.suite_timeout) if args.suite_timeout else "none"),
          flush=True)
    if timed_out:
        print("SUITE LEDGER: suites killed at the timeout: "
              + ", ".join(sorted(timed_out)), flush=True)

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

    shard_note = ""
    if shard_k is not None:
        owned = sorted(r for r in on_disk if r not in elsewhere)
        # The same digest `gate_suites.names_digest` recomputes in the report.
        digest = hashlib.sha256("\n".join(owned).encode()).hexdigest()
        with open(logs / "suite-shard.txt", "w", encoding="utf-8") as fh:
            fh.write("".join(r + "\n" for r in owned))
        print("SUITE SHARD: k=%d n=%d owned=%d names=%s"
              % (shard_k, shard_n, len(owned), digest), flush=True)
        shard_note = " shard=%d/%d owned=%d executed=%d" % (
            shard_k, shard_n, len(owned), len(seconds))
    if args.runner is not None:
        print("CI PLAN: runner=%s ran=%d/%d not-run=%d (UNVERIFIED on this "
              "runner, local-only here): %s"
              % (args.runner, len(printed) - len(not_here), len(printed),
                 len(not_here), ", ".join(sorted(not_here)) or "none"),
              flush=True)
    failed_suites = sorted(n for n, rc in ran_rc.items() if rc != 0)
    print("SUITE LEDGER: suites=%d/%d ok=%d FAIL=%d SKIP=%d "
          "(printed and re-read from %s; they agree)"
          % (len(printed), len(on_disk) or len(given),
             total_printed[0], total_printed[1], total_printed[2], logs)
          + shard_note
          if not problems else
          "SUITE LEDGER: suites=%d/%d ok=%d FAIL=%d SKIP=%d -- NOT RECONCILED"
          % (len(printed), len(on_disk) or len(given),
             total_printed[0], total_printed[1], total_printed[2]) + shard_note,
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
