"""A Vulkan skip must not be countable as a pass (docs/devices/VULKAN5.md §2).

These tests need no Vulkan loader, on purpose: the defect they hold down only
exists on a machine that has none. They run the real runner and the real skip
helpers of `test_vulkan4.py` and `test_shim.py` against a probe that reports
"unavailable", and check what a reader of the log would see.
"""

import contextlib
import io
import os
import subprocess
import sys

import vulkan_coverage as cov

HERE = os.path.dirname(os.path.abspath(__file__))

_NO_LOADER = {"available": False, "device": None, "type": None,
              "queue_family": None, "loader": None,
              "error": "failed to load the Vulkan loader: tried libvulkan.dylib (no such file)"}
_LOADER = {"available": True, "device": "Test GPU", "type": "INTEGRATED_GPU",
           "queue_family": 0, "loader": "/x/libvulkan.dylib", "error": None}


class _FakeC:
    def __init__(self, real, probe):
        self._real, self._probe = real, probe

    def _vulkan_probe(self):
        return dict(self._probe)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _reset():
    cov._state.update(current=None, used=set(), skipped={}, device=None)


def _run_with_probe(module, probe, tests):
    _reset()
    real = module._C
    module._C = _FakeC(real, probe)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            failures = cov.run_tests(tests)
    finally:
        module._C = real
        _reset()
    return failures, buf.getvalue()


def _suites():
    import test_shim
    import test_vulkan4
    return ((test_vulkan4, lambda: test_vulkan4._vulkan_or_skip("x")),
            (test_shim, lambda: test_shim._vulkan_or_skip("x")))


def test_a_vulkan_skip_prints_SKIP_and_never_ok_in_either_suite():
    for module, helper in _suites():
        def test_needs_vulkan():
            if helper() is None:
                return
            raise AssertionError("the probe said unavailable; must not get here")

        failures, out = _run_with_probe(module, _NO_LOADER, [("test_needs_vulkan", test_needs_vulkan)])
        assert failures == 0, out
        assert "ok   test_needs_vulkan" not in out, (module.__name__, out)
        assert "SKIP test_needs_vulkan: " in out, (module.__name__, out)
        # The loader's own words reach the log, not a paraphrase.
        assert "failed to load the Vulkan loader" in out, out
        assert "VULKAN: ran=0 ok=0 failed=0 skipped=1 device=-" in out, out


def test_a_vulkan_test_that_ran_is_counted_as_ran():
    for module, helper in _suites():
        def test_uses_vulkan():
            assert helper() is not None

        def test_does_not_touch_vulkan():
            pass

        failures, out = _run_with_probe(
            module, _LOADER,
            [("test_uses_vulkan", test_uses_vulkan),
             ("test_does_not_touch_vulkan", test_does_not_touch_vulkan)])
        assert failures == 0, out
        assert "ok   test_uses_vulkan" in out, out
        # A test that never asked for the device is in neither column.
        assert "VULKAN: ran=1 ok=1 failed=0 skipped=0 device=Test GPU" in out, (module.__name__, out)


def test_a_vulkan_test_that_failed_is_ran_and_failed_not_ok():
    for module, helper in _suites():
        def test_breaks():
            helper()
            raise AssertionError("wrong answer")

        failures, out = _run_with_probe(module, _LOADER, [("test_breaks", test_breaks)])
        assert failures == 1, out
        assert "VULKAN: ran=1 ok=0 failed=1 skipped=0" in out, out


def test_the_gate_summary_says_UNVERIFIED_when_nothing_ran():
    log = "ok   test_a\nSKIP test_b: no vulkan -- dlopen said no\nVULKAN: ran=0 ok=0 failed=0 skipped=1 device=-\n"
    text, code = cov.summarize({"test_vulkan4.py.log": log, "test_shim.py.log": log})
    assert "VULKAN COVERAGE: 0 ran (0 ok, 0 failed) / 2 skipped" in text, text
    assert "UNVERIFIED" in text, text
    assert "test_vulkan4.py.log::test_b: no vulkan -- dlopen said no" in text, text
    assert code == 0, "without REQUIRE a machine with no loader must still pass the gate"

    text, code = cov.summarize({"s": log}, require=True)
    assert code == 1 and "TORCHNATIVE_REQUIRE_VULKAN=1" in text, text


def test_the_gate_summary_distinguishes_partial_and_full():
    part = "VULKAN: ran=3 ok=3 failed=0 skipped=2 device=Apple M1\n"
    full = "VULKAN: ran=5 ok=5 failed=0 skipped=0 device=Apple M1\n"
    text, code = cov.summarize({"a": part}, require=True)
    assert "PARTIAL" in text and code == 1, text
    text, code = cov.summarize({"a": full}, require=True)
    assert code == 0 and "UNVERIFIED" not in text and "PARTIAL" not in text, text
    assert "5 ran (5 ok, 0 failed) / 0 skipped -- device: Apple M1" in text, text
    # A log with no VULKAN line contributes nothing -- it is not "0 skipped
    # therefore verified".
    text, code = cov.summarize({"other": "ok   test_x\n"}, require=True)
    assert "UNVERIFIED" in text and code == 1, text


def test_the_command_line_exit_code_honours_REQUIRE():
    """What `run.sh` actually invokes, as a process -- exit code read directly."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test_vulkan4.py.log")
        with open(path, "w") as fh:
            fh.write("VULKAN: ran=0 ok=0 failed=0 skipped=4 device=-\n")
        script = os.path.join(HERE, "vulkan_coverage.py")
        env = dict(os.environ)
        env.pop("TORCHNATIVE_REQUIRE_VULKAN", None)
        plain = subprocess.run([sys.executable, script, path], capture_output=True, text=True, env=env)
        env["TORCHNATIVE_REQUIRE_VULKAN"] = "1"
        strict = subprocess.run([sys.executable, script, path], capture_output=True, text=True, env=env)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert "UNVERIFIED" in plain.stdout, plain.stdout
    assert strict.returncode == 1, strict.stdout + strict.stderr


def test_both_vulkan_suites_use_the_counting_runner():
    """The runner is only worth anything if the suites with Vulkan tests use it.

    Checked by behaviour, not by grepping for the call: each suite's `_main`
    is run over one fake test with the probe reporting unavailable.
    """
    for module, helper in _suites():
        def test_fake():
            helper()

        real_main_globals = {k: v for k, v in vars(module).items() if k.startswith("test_")}
        for k in real_main_globals:
            delattr(module, k)
        module.test_fake = test_fake
        real = module._C
        module._C = _FakeC(real, _NO_LOADER)
        buf = io.StringIO()
        _reset()
        try:
            with contextlib.redirect_stdout(buf):
                rc = module._main()
        finally:
            del module.test_fake
            for k, v in real_main_globals.items():
                setattr(module, k, v)
            module._C = real
            _reset()
        out = buf.getvalue()
        assert rc == 0, out
        assert "SKIP test_fake" in out and "ok   test_fake" not in out, (module.__name__, out)
        assert "VULKAN: ran=0 ok=0 failed=0 skipped=1" in out, (module.__name__, out)


def _docwatch():
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    sys.path.insert(0, os.path.join(root, "tools", "docwatch"))
    import check_docs
    return check_docs


def test_a_docwatch_vulkan_claim_is_SKIP_not_PASS_where_nothing_ran():
    """The DOCWATCH half: an agreement marker on a host with no loader."""
    cd = _docwatch()
    from pathlib import Path

    claim = cd.Claim(doc=Path("x.md"), line=1, kind="count",
                     args=["vulkan_tests_ok", "ge", "19"], context="")
    live = cd.LiveFacts(sys.executable, {})
    live._vulkan_cache = {"ran": 0, "ok": 0, "failed": 0, "skipped": 19, "device": "-"}
    (result,) = cd.evaluate([claim], live)
    assert result.status == "SKIP", result.format()
    assert "unmeasured" in result.detail, result.format()

    live._vulkan_cache = {"ran": 19, "ok": 19, "failed": 0, "skipped": 0, "device": "Apple M1"}
    (result,) = cd.evaluate([claim], live)
    assert result.status == "PASS", result.format()

    live._vulkan_cache = {"ran": 19, "ok": 17, "failed": 2, "skipped": 0, "device": "Apple M1"}
    (result,) = cd.evaluate([claim], live)
    assert result.status == "FAIL", result.format()


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
