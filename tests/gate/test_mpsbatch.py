"""The batched Metal-suite children, tested without torch (issue #362).

`tests/_support/mps_batch.py` runs the per-case scripts of six Metal suites in
a few children instead of one child per side per case. These tests drive it
with fake case scripts and a fake counter module, so no shim, no torch and no
Metal device is involved, and they run on every runner.

What they pin:

* answers are matched by tag: an answer arriving under another case's tag, a
  duplicated answer, or a child that stops without a reason fails by name;
* the shim policy: cases share a child until one reports an error; an error
  that was not the first case in its child is re-graded alone in a fresh
  child; a child that dies mid-case has that case re-run alone, and every
  later case runs in a fresh child;
* the upstream policy: one child for every case, errors included;
* the empty sentinel bracket: a counter module whose counters move between
  two drains fails the run by name;
* the compare mode lists, by name, exactly the cases whose answers differ
  between the batched and the one-process-per-case runs;
* `TORCHNATIVE_MPS_ONE_PROCESS_PER_CASE` mode starts one child per case.

Nullifications a reviewer should run on CI (they need the shim on a Mac,
which these tests deliberately do not use):

* N1: remove the drain and the sentinel bracket from `mps_batch._DRIVER` (or
  make `_shim_metal_synchronize` a no-op while a case leaves work in flight)
  -> a counter-bracket assertion in one of the six suites, or the sentinel
  check here (`test_a_moving_sentinel_fails_by_name`) goes red.
* N2: inject a panic into one case of a suite (for example raise inside the
  block of one mpscb row) -> that case is reported by name and every later
  case is graded from a fresh child (`test_an_error_retires_the_child_*`
  pins the mechanism).
* N3: shift the oracle results by one (emit each answer under the previous
  case's tag in `_DRIVER`) -> the tag check fails by name
  (`test_an_answer_under_the_wrong_tag_fails_by_name` pins the check).

What this cannot find (AGENTS.md §17.4): whether the real shim leaves state
behind between cases that changes a verdict. That is what the compare mode
measures on the first CI run.
"""

import atexit
import os
import shutil
import tempfile

import mps_batch

_TMP = tempfile.mkdtemp(prefix="mpsbatch-test-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

with open(os.path.join(_TMP, "probe_still.py"), "w", encoding="utf-8") as _f:
    _f.write("def _metal_counters():\n"
             "    return {'built': True, 'host_downloads': 0, 'host_uploads': 0}\n"
             "def _shim_metal_synchronize():\n"
             "    return None\n")
with open(os.path.join(_TMP, "probe_moving.py"), "w", encoding="utf-8") as _f:
    _f.write("N = [0]\n"
             "def _metal_counters():\n"
             "    N[0] += 1\n"
             "    return {'built': True, 'host_downloads': N[0], 'host_uploads': 0}\n"
             "def _shim_metal_synchronize():\n"
             "    return None\n")

# The case script: reads its payload from sys.argv[1] like the real ones and
# prints its JSON answer last. It reports its pid and how many cases its
# process has run, and can report an error, die, or die only the first time.
_SCRIPT = r"""
import json, os, sys
A = json.loads(sys.argv[1])
sys._mpsbatch_n = getattr(sys, "_mpsbatch_n", 0) + 1
print("noise before the answer")
if A.get("die_once") and not os.path.exists(A["die_once"]):
    open(A["die_once"], "w").close()
    os._exit(3)
out = {"n": A["n"], "pid": os.getpid()}
if A.get("count"):
    out["seen"] = sys._mpsbatch_n
if A.get("error"):
    out = {"error": "RuntimeError: case %d" % A["n"], "pid": os.getpid()}
print(json.dumps(out))
"""


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = _TMP
    return env


class _Modes:
    """Set the two switches for one test, then restore them."""

    def __init__(self, one=False, compare=False):
        self.want = (one, compare)

    def __enter__(self):
        self.saved = (mps_batch.ONE_PROCESS_PER_CASE, mps_batch.COMPARE_MODES)
        mps_batch.ONE_PROCESS_PER_CASE, mps_batch.COMPARE_MODES = self.want

    def __exit__(self, *exc):
        mps_batch.ONE_PROCESS_PER_CASE, mps_batch.COMPARE_MODES = self.saved
        return False


def _raises(fn, *needles):
    try:
        fn()
    except AssertionError as exc:
        text = str(exc)
        missing = [n for n in needles if n not in text]
        assert not missing, (missing, text)
        return text
    raise AssertionError("expected an AssertionError naming %s" % (needles,))


def test_answers_come_back_in_payload_order_from_one_upstream_child():
    payloads = [dict(n=i, error=(i == 2)) for i in range(5)]
    with _Modes():
        got = mps_batch.run_cases(_SCRIPT, payloads, True, _env())
    assert [g.get("n", 2) for g in got] == [0, 1, 2, 3, 4], got
    assert "error" in got[2], got
    assert len({g["pid"] for g in got}) == 1, ("upstream must use one child", got)
    print("ok   mpsbatch: upstream: 5 cases, errors included, one child, answers in order")


def test_an_error_retires_the_child_and_is_regraded_alone():
    payloads = [dict(n=i, error=(i == 2)) for i in range(5)]
    with _Modes():
        got = mps_batch.run_cases(_SCRIPT, payloads, False, _env(), probe="probe_still")
    pids = [g["pid"] for g in got]
    assert pids[0] == pids[1], pids
    assert pids[2] not in (pids[0], pids[3]), ("the error is re-graded alone", pids)
    assert pids[3] == pids[4] and pids[3] != pids[0], ("a fresh child after the error", pids)
    assert got[2]["error"] == "RuntimeError: case 2", got[2]
    assert [g.get("n", 2) for g in got] == [0, 1, 2, 3, 4], got
    print("ok   mpsbatch: shim: an error retires its child, is graded alone, later cases fresh")


def test_an_error_first_in_its_child_is_not_rerun():
    payloads = [dict(n=0, error=True), dict(n=1, error=True), dict(n=2)]
    with _Modes():
        got = mps_batch.run_cases(_SCRIPT, payloads, False, _env(), probe="probe_still")
    assert len({g["pid"] for g in got}) == 3, got
    print("ok   mpsbatch: shim: consecutive errors each get their own fresh child")


def test_a_case_that_kills_the_child_is_rerun_alone_and_later_cases_are_fresh():
    marker = os.path.join(_TMP, "died-once")
    payloads = [dict(n=0), dict(n=1, die_once=marker), dict(n=2), dict(n=3)]
    for upstream in (True, False):
        if os.path.exists(marker):
            os.remove(marker)
        with _Modes():
            got = mps_batch.run_cases(_SCRIPT, payloads, upstream, _env(), probe="probe_still")
        assert [g["n"] for g in got] == [0, 1, 2, 3], got
        pids = [g["pid"] for g in got]
        assert len({pids[0], pids[1], pids[2]}) == 3, (upstream, pids)
        assert pids[2] == pids[3], (upstream, pids)
    print("ok   mpsbatch: a death mid-case: re-run alone, later cases from a fresh child")


def test_an_answer_under_the_wrong_tag_fails_by_name():
    real = mps_batch._child

    def shifted(script, cases, env, isolate, probe, timeout):
        records, err = real(script, cases, env, isolate, probe, timeout)
        answers = [r for r in records if "result" in r]
        tags = [r["tag"] for r in answers]
        for r, tag in zip(answers, tags[1:] + tags[:1]):
            r["tag"] = tag
        return records, err

    mps_batch._child = shifted
    try:
        with _Modes():
            _raises(lambda: mps_batch.run_cases(_SCRIPT, [dict(n=i) for i in range(3)],
                                                True, _env()),
                    "arrived while", "case 1")
    finally:
        mps_batch._child = real

    def doubled(script, cases, env, isolate, probe, timeout):
        records, err = real(script, cases, env, isolate, probe, timeout)
        return records + [r for r in records if "result" in r][:1], err

    mps_batch._child = doubled
    try:
        with _Modes():
            _raises(lambda: mps_batch.run_cases(_SCRIPT, [dict(n=i) for i in range(3)],
                                                True, _env()),
                    "answered twice", "case 0")
    finally:
        mps_batch._child = real
    print("ok   mpsbatch: a shifted or duplicated answer fails by name")


def test_a_moving_sentinel_fails_by_name():
    with _Modes():
        _raises(lambda: mps_batch.run_cases(_SCRIPT, [dict(n=0)], False, _env(),
                                            probe="probe_moving"),
                "empty bracket", "case 0", "host_downloads")
    print("ok   mpsbatch: counters moving across an empty bracket fail by name")


def test_compare_mode_names_exactly_the_cases_that_differ():
    payloads = [dict(n=0), dict(n=1, count=True), dict(n=2, count=True)]
    script = _SCRIPT.replace('"pid": os.getpid()}', '"pid": 0}')
    with _Modes(compare=True):
        text = _raises(lambda: mps_batch.run_cases(script, payloads, True, _env()),
                       "2 of 3 upstream cases differ", "case 1", "case 2")
    assert "case 0" not in text, text
    with _Modes(compare=True):
        got = mps_batch.run_cases(script, [dict(n=0), dict(n=1)], True, _env())
    assert [g["n"] for g in got] == [0, 1], got
    print("ok   mpsbatch: compare mode names the 2 of 3 cases that differ, passes when equal")


def test_one_process_per_case_mode_starts_a_child_per_case():
    with _Modes(one=True):
        got = mps_batch.run_cases(_SCRIPT, [dict(n=i) for i in range(3)], True, _env())
    assert len({g["pid"] for g in got}) == 3, got
    print("ok   mpsbatch: one-process-per-case mode: 3 cases, 3 children")


def test_cases_refuses_a_request_that_was_not_prefetched():
    cases = mps_batch.Cases(_SCRIPT, _TMP, mark_shim=False)
    with _Modes():
        cases.prefetch([(dict(n=0), True)])
    assert cases.run(dict(n=0), True)["n"] == 0
    _raises(lambda: cases.run(dict(n=1), True), "was not prefetched")
    print("ok   mpsbatch: Cases answers prefetched requests and names a missing one")


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
