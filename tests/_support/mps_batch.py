"""Batched children for the per-case Metal suites (issue #362).

Six suites under `tests/devices/mps/` (mpscb, mpsextreme, mpsindexput,
mpsindex, mpssetitem, mpsint32) grade every case in a fresh Python child per
side: one upstream oracle child and one shim child per case, about 1020
processes and about 1430 s on the macOS gate's critical path, because every
child imports torch and every shim child recompiles candle's Metal libraries.
The cases themselves are 3x4 tensors.

This module runs the same per-case scripts, unchanged, in far fewer children.
A case script still reads its payload from `sys.argv[1]` and still prints
its JSON answer as its last stdout line; the child driver below sets
`sys.argv`, captures stdout and `exec`s the script once per case.

The rules (leader approval 2026-11-16, with conditions):

* **Upstream (A).** One child runs every case, in order. Each case is its own
  `exec` and its own script-level `try/except`, so one case's exception does
  not reach the next. A case that kills the child (or that raises past its
  script's own handler) is re-run alone in a fresh child, the old way, and a
  fresh child continues from the next case.
* **Shim (B).** One child runs cases in order. A candle Metal panic poisons
  the encoder's mutex for the rest of a process (test_mpsgen.py), so after
  any case that *reports an error* the child is retired and a fresh child
  continues from the next case. A reported error from a case that was not
  the first in its child is re-run alone in a fresh child (the old way) and
  that answer is the case's verdict, so an error cannot be an artefact of an
  earlier case in the same process. A case that kills the child is re-run
  alone the same way.
* **Tags.** Every case carries a tag; every answer is keyed by tag, never by
  position. A missing tag, a duplicated tag or an unknown tag fails by name.
* **Counter isolation (C), shim side.** Before each case the child drains
  the Metal queue (`torch._C._shim_metal_synchronize()`), then takes an
  empty sentinel bracket (counters, drain, counters) that must read all
  zeros, or the run fails by name. After the case it drains again and keeps
  the whole case's counter delta (uploads, downloads, encoders) in
  `CASE_COUNTERS[label]`, beside the script's own bracket.

Switches:

* `TORCHNATIVE_MPS_ONE_PROCESS_PER_CASE=1`: the old way, one child per side
  per case, nothing batched.
* `COMPARE_MODES` (below): run both ways and fail by name, listing every case
  whose answers differ.

What this cannot find (AGENTS.md §17.4): state a case leaves behind that
neither an error nor the counters show (a changed global default, a seeded
generator, a cached kernel that computes differently the second time). The
compare mode is the instrument for that on the first CI run; once it is
off, such a leak is visible only as a changed verdict.
"""

import json
import os
import subprocess
import sys

#: Compare the batched answers with the one-process-per-case answers, case by
#: case, and fail by name on any difference. ON for now: turn it OFF (set the
#: default below to "0") once one CI run has shown the two modes equal.
COMPARE_MODES = os.environ.get("TORCHNATIVE_MPS_COMPARE_MODES", "1") == "1"

ONE_PROCESS_PER_CASE = os.environ.get("TORCHNATIVE_MPS_ONE_PROCESS_PER_CASE") == "1"

#: Per child, as the old per-case `_run` allowed per case.
TIMEOUT = 1800

#: The marker on every line the child driver writes for the parent.
MARK = "@@MPSCASE "

#: label -> the whole case's counter delta, measured between two drains.
CASE_COUNTERS = {}


def child_env(upstream, vendor_dir):
    """The environment the old per-suite `_run` built: upstream children see
    no PYTHONPATH (site-packages torch only); shim children see the vendored
    tree."""
    env = dict(os.environ)
    env.pop("TORCH_C_ARTEFACT", None)
    if upstream:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    else:
        env["PYTHONPATH"] = vendor_dir
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    return env


def run_one(script, payload, env, timeout=TIMEOUT):
    """The old way, exactly: one child for one case, its last stdout line."""
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(payload)],
                          capture_output=True, text=True, timeout=timeout, env=env)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise AssertionError("child produced no JSON (rc=%d):\n%s"
                             % (proc.returncode, proc.stderr[-3000:]))
    return json.loads(lines[-1])


# The child driver. Job on stdin: {"script", "cases": [[tag, payload], ...],
# "isolate", "probe"}. It writes one MARK line per event: {"tag", "start"}
# before a case, then {"tag", "result"} or {"tag", "crash"}, and
# {"tag", "sentinel"} when the empty bracket was not empty.
_DRIVER = r'''
import builtins, contextlib, importlib, io, json, sys, traceback
job = json.loads(sys.stdin.read())
real = sys.stdout
MARK = "@@MPSCASE "
def emit(obj):
    real.write(MARK + json.dumps(obj) + "\n")
    real.flush()
code = compile(job["script"], "<case script>", "exec")
probe = importlib.import_module(job["probe"]) if job["isolate"] else None
def counters():
    c = probe._metal_counters()
    return {k: v for k, v in c.items() if k != "built"}
def drain():
    probe._shim_metal_synchronize()
for tag, payload in job["cases"]:
    emit({"tag": tag, "start": True})
    base = None
    if probe is not None:
        drain()
        a = counters()
        drain()
        base = counters()
        moved = {k: base[k] - a[k] for k in a if base[k] != a[k]}
        if moved:
            emit({"tag": tag, "sentinel": moved})
            break
    sys.argv = ["-c", json.dumps(payload)]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__name__": "__main__", "__builtins__": builtins})
    except BaseException:
        emit({"tag": tag, "crash": traceback.format_exc()[-3000:]})
        break
    lines = buf.getvalue().strip().splitlines()
    if not lines:
        emit({"tag": tag, "crash": "the case printed nothing"})
        break
    rec = {"tag": tag, "result": json.loads(lines[-1])}
    if probe is not None:
        drain()
        after = counters()
        rec["case_counters"] = {k: after[k] - base[k] for k in base}
    emit(rec)
'''


def _child(script, cases, env, isolate, probe, timeout):
    """One driver child over `cases` ([(tag, payload)]). Returns the parsed
    MARK records and the child's stderr tail."""
    job = json.dumps({"script": script, "cases": cases, "isolate": isolate,
                      "probe": probe})
    proc = subprocess.Popen([sys.executable, "-c", _DRIVER], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    try:
        out, err = proc.communicate(job, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        err = (err or "") + "\n[mps_batch] child killed after %ss" % timeout
    records = [json.loads(line[len(MARK):]) for line in out.splitlines()
               if line.startswith(MARK)]
    return records, "rc=%s\n%s" % (proc.returncode, (err or "")[-3000:])


def _batched(script, payloads, labels, env, upstream, probe, timeout):
    tags = ["c%04d" % i for i in range(len(payloads))]
    by_tag = dict(zip(tags, range(len(payloads))))
    results = {}
    isolate = not upstream
    i = 0
    while i < len(payloads):
        cases = [[tags[k], payloads[k]] for k in range(i, len(payloads))]
        records, err = _child(script, cases, env, isolate, probe, timeout)
        started = []
        retire_after = None
        for rec in records:
            tag = rec.get("tag")
            if tag not in by_tag:
                raise AssertionError("mps_batch: a child answered for unknown tag %r" % (tag,))
            k = by_tag[tag]
            if "start" in rec:
                if k in started or tag in results:
                    raise AssertionError("mps_batch: case %s started twice" % labels[k])
                started.append(k)
                continue
            if "sentinel" in rec:
                raise AssertionError(
                    "mps_batch: the empty bracket before case %s moved the Metal counters "
                    "%s; the drain does not isolate the cases" % (labels[k], rec["sentinel"]))
            if tag in results:
                raise AssertionError("mps_batch: case %s answered twice" % labels[k])
            if not started or started[-1] != k:
                raise AssertionError(
                    "mps_batch: an answer tagged %s arrived while %s was running"
                    % (labels[k], labels[started[-1]] if started else "no case"))
            if "crash" in rec:
                retire_after = k
                break
            result = rec["result"]
            if "case_counters" in rec:
                CASE_COUNTERS[labels[k]] = rec["case_counters"]
            if isolate and isinstance(result, dict) and "error" in result:
                if k != i:
                    # Not the first case in this child: the verdict comes from
                    # a clean process, the old way.
                    result = run_one(script, payloads[k], env, timeout)
                results[tag] = result
                retire_after = k
                break
            results[tag] = result
        if retire_after is None:
            if not started:
                raise AssertionError("mps_batch: a child started no case from %s:\n%s"
                                     % (labels[i], err))
            last = started[-1]
            if tags[last] not in results:
                retire_after = last   # it died inside this case
            elif last == len(payloads) - 1:
                break
            else:
                raise AssertionError(
                    "mps_batch: a child stopped after case %s without a reason:\n%s"
                    % (labels[last], err))
        k = retire_after
        if tags[k] not in results:
            # Died or crashed inside case k: re-run it alone, the old way (which
            # raises by name as the old `_run` did if it cannot answer either).
            results[tags[k]] = run_one(script, payloads[k], env, timeout)
        i = k + 1
    missing = [labels[by_tag[t]] for t in tags if t not in results]
    if missing:
        raise AssertionError("mps_batch: no answer for %s" % ", ".join(missing))
    return [results[t] for t in tags]


def _canon(result):
    return json.dumps(result, sort_keys=True)


def run_cases(script, payloads, upstream, env, labels=None, probe="torch._C",
              timeout=TIMEOUT):
    """Answers for `payloads`, in their order (matched by tag inside).

    `upstream` picks the policy (A or B above); `env` is the child
    environment (`child_env`). `probe` names the module whose
    `_metal_counters()` and `_shim_metal_synchronize()` isolate shim cases;
    it is a parameter only so the helper's own test can run without torch.
    """
    payloads = list(payloads)
    if labels is None:
        labels = ["case %d %s" % (i, _canon(p)[:160])
                  for i, p in enumerate(payloads)]
    if len(set(labels)) != len(labels):
        raise AssertionError("mps_batch: duplicated case labels")
    if ONE_PROCESS_PER_CASE:
        return [run_one(script, p, env, timeout) for p in payloads]
    got = _batched(script, payloads, labels, env, upstream, probe, timeout)
    if COMPARE_MODES:
        old = [run_one(script, p, env, timeout) for p in payloads]
        differ = ["%s: batched %s, one process per case %s"
                  % (labels[i], _canon(g)[:400], _canon(o)[:400])
                  for i, (g, o) in enumerate(zip(got, old)) if _canon(g) != _canon(o)]
        if differ:
            raise AssertionError(
                "mps_batch: %d of %d %s cases differ between the batched and the "
                "one-process-per-case modes:\n%s"
                % (len(differ), len(payloads), "upstream" if upstream else "shim",
                   "\n".join(differ)))
    return got


class Cases:
    """A suite's `_run(payload, upstream)` over prefetched, batched answers.

    The suite calls `prefetch([(payload, upstream), ...])` with every request
    its loop will make, then its unchanged loop calls `run(payload, upstream)`,
    which returns the answer for that exact payload. A request that was not
    prefetched fails by name instead of quietly starting its own child.
    `mark_shim` adds `shim=not upstream` to each payload, as the old `_run`s
    of five suites did.
    """

    def __init__(self, script, vendor_dir, mark_shim=True):
        self.script = script
        self.vendor_dir = vendor_dir
        self.mark_shim = mark_shim
        self.answers = {}

    def _key(self, payload, upstream):
        if self.mark_shim:
            payload = dict(payload, shim=not upstream)
        return payload, (bool(upstream), _canon(payload))

    def prefetch(self, requests):
        for side in (True, False):
            todo = {}
            for payload, upstream in requests:
                if bool(upstream) != side:
                    continue
                full, key = self._key(payload, upstream)
                if key not in self.answers:
                    todo[key] = full
            if not todo:
                continue
            keys = list(todo)
            got = run_cases(self.script, [todo[k] for k in keys], side,
                            child_env(side, self.vendor_dir),
                            labels=["%s case %d %s" % ("upstream" if side else "shim", n,
                                                       k[1][:200])
                                    for n, k in enumerate(keys)])
            self.answers.update(zip(keys, got))

    def run(self, payload, upstream):
        _, key = self._key(payload, upstream)
        if key not in self.answers:
            raise AssertionError("mps_batch: %s case was not prefetched (the test's "
                                 "prefetch list and its loop disagree): %s"
                                 % ("upstream" if upstream else "shim", key[1][:300]))
        return self.answers[key]
