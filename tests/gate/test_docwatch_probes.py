"""A DOCWATCH live probe runs once per gate, whatever its verdict (issue #335).

`tests/docwatch/check_docs.py` answers `count <name> <op> <N>` markers from
live probes: subprocesses that run a suite or a harness and read its output.
Each probe kept its answer in a cache, but only when it had an answer. A probe
whose verdict was "unmeasured here" (`LiveFactsSkip`) or "could not be
produced" (`LiveFactsError`) raised before the cache was written, so the next
marker ran the subprocess again.

Measured on CI run 43623378387 (2026-11-14): on every hosted runner
`test_shim.py` skips some tests by name, so `smoke_ok` is a skip there, and
the 24 `count smoke_ok` markers ran `test_shim.py` 24 times. The Windows log
shows it: the 24 SKIP lines quote two different counts ("446 ok", "447 ok"),
which one run could not produce. The gate step took 113 minutes on Windows,
where the suites themselves take about one.

**What makes these tests able to fail.** They count how often the probe's
subprocess is started. With the cache written only on success, the skip and
the error tests count 3 where they assert 1 (observed before the fix, see the
issue). The success test passed before and after: it holds the behaviour that
must not change.

**What this cannot find.** How long a probe takes, and whether a probe's
verdict is right. It replaces `subprocess.run` inside `check_docs`, so no
suite is executed here.
"""

import atexit
import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile
import types

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_CHECK_DOCS = _TESTS_DIR / "docwatch" / "check_docs.py"


def _load():
    spec = importlib.util.spec_from_file_location("_check_docs_under_test", _CHECK_DOCS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRun:
    """Stands in for `subprocess.run`: counts the starts, returns one answer."""

    def __init__(self, stdout, returncode):
        self.calls = 0
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return types.SimpleNamespace(stdout=self.stdout, stderr="", returncode=self.returncode)


def _smoke(stdout, returncode, runner):
    """(module, fake, LiveFacts) with a real file as the artefact."""
    cd = _load()
    fake = _FakeRun(stdout, returncode)
    cd.subprocess = types.SimpleNamespace(run=fake)
    tmp = tempfile.mkdtemp(prefix="docwatch-probe-")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    artefact = os.path.join(tmp, "_C.abi3.so")
    with open(artefact, "wb") as fh:
        fh.write(b"not a real extension; the probe only copies it")
    env = {"TORCH_C_ARTEFACT": artefact}
    if runner:
        env["TORCHNATIVE_GATE_RUNNER"] = runner
    return cd, fake, cd.LiveFacts(sys.executable, env), tmp


def _cleanup(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def test_a_probe_that_skips_runs_once_for_every_marker_that_asks():
    cd, fake, live, tmp = _smoke("ok   a\nok   b\nSKIP c: (reason)\n", 0, "linux")
    try:
        reasons = []
        for _ in range(3):
            try:
                live.smoke_ok()
            except cd.LiveFactsSkip as exc:
                reasons.append(str(exc))
            else:
                raise AssertionError("a run that skipped a test by name on a CI runner gave a count")
        assert len(set(reasons)) == 1 and "2 ok" in reasons[0], reasons
        assert fake.calls == 1, (
            f"test_shim.py was started {fake.calls} times for 3 markers; a skip "
            "verdict must be kept like a count is")
    finally:
        _cleanup(tmp)
    print("ok   docwatch_probes: a skipped probe is started once for 3 markers")


def test_a_probe_that_errors_runs_once_for_every_marker_that_asks():
    cd, fake, live, tmp = _smoke("", 1, None)
    try:
        for _ in range(3):
            try:
                live.smoke_ok()
            except cd.LiveFactsError:
                pass
            else:
                raise AssertionError("a run with no ok lines and a non-zero exit gave a count")
        assert fake.calls == 1, f"the failing probe was started {fake.calls} times for 3 markers"
    finally:
        _cleanup(tmp)
    print("ok   docwatch_probes: a probe that errors is started once for 3 markers")


def test_a_probe_that_answers_still_runs_once_and_gives_the_count():
    cd, fake, live, tmp = _smoke("ok   a\nok   b\nok   c\n", 0, None)
    try:
        assert [live.smoke_ok() for _ in range(3)] == [3, 3, 3]
        assert fake.calls == 1, fake.calls
    finally:
        _cleanup(tmp)
    print("ok   docwatch_probes: an answering probe is started once and counts 3 ok lines")


def test_the_probe_report_names_each_probe_its_runs_and_its_seconds():
    cd, fake, live, tmp = _smoke("ok   a\n", 0, None)
    try:
        live.smoke_ok()
        live.smoke_ok()
        lines = cd.probe_report(live)
        assert len(lines) == 1, lines
        assert lines[0].startswith("DOCWATCH PROBE: smoke_ok ran 1 time(s), "), lines
        assert lines[0].endswith(" s"), lines
    finally:
        _cleanup(tmp)
    print("ok   docwatch_probes: the report line names the probe, 1 run and its seconds")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
