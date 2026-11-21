"""Shared "this test could not run" reporting, so a skip can never read as ok.

The shape of the defect this exists for: a test whose fixture is missing
(no vendored shim, no upstream torch, an env var unset, ...) printed a
hand-rolled ``print("   (skipped: <reason>)")`` line and then `return`ed
normally. The runner loop in each suite's ``__main__`` treats "returned
without raising" as success and prints ``ok   {name}``. `suite_ledger.py`'s
``tally()`` only counts lines matching ``^SKIP\\s`` -- three leading spaces
and no ``SKIP`` prefix means the line is invisible to it. So the skip landed
in **neither** the ok nor the SKIP column: it vanished. A test gutted to
``pass`` behind such a guard would print an identical log.

``vulkan_coverage.py`` solved exactly this for the Vulkan suites first (see
its docstring); this module is the same shape, generalised for every other
"missing fixture" skip in this directory. It intentionally does not import
or extend ``vulkan_coverage`` -- that module is Vulkan-suite territory and
other rounds are actively editing it; duplicating the ~20 lines here is
cheaper than coordinating a shared edit.

Usage in a test body, in place of the old ``print(...); return``::

    if fixture is None:
        _skip.skip("no vendored shim installed")
        return

Usage in the suite's runner (in place of a bare ``for name, fn in ...: fn()``
loop that prints ``ok`` on any non-raising return)::

    failures = _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items())
         if name.startswith("test_")],
        suite="test_whatever",
    )

``run_tests`` prints ``SKIP <suite>: <name> -- <reason>`` for a test that
called ``skip()`` and returned without raising, instead of ``ok``. Never
both: the ``else`` branch below is exclusive, matching the shape
``vulkan_coverage.run_tests`` already uses so the two skip mechanisms behave
identically to a reader of the log.
"""

import functools
import os
import platform
import sys

_state = {"current": None, "skipped": {}}


def skip(reason):
    """Record why the test now running cannot proceed.

    The caller must still `return` right after calling this -- `skip()`
    only records the reason; it does not itself stop execution. (A test
    that wants execution to stop immediately should `raise Skip(reason)`
    instead; `run_tests` treats that identically to `skip()` + `return`.)
    """
    name = _state["current"]
    if name is not None:
        _state["skipped"].setdefault(name, reason)


class Skip(Exception):
    """Raise this instead of `skip()` + `return` when there is no single
    return point to fall through to (e.g. deep in a helper called from
    several places in the test)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _arm_hang_dump():
    """Print every thread's Python stack to stderr before the gate's per-suite
    timeout kills this process, so a hang names its thread and the op it is
    inside. The kill loses buffered stdout, and the gate then has only "timed
    out" to show (run 44880397455: test_threadstate killed at 2700 s with no
    FAIL line). Armed only when the gate exports `TORCHNATIVE_SUITE_TIMEOUT_S`,
    at 80 % of it; `faulthandler`'s watchdog thread needs no GIL, so a
    deadlocked interpreter still dumps. Returns the cancel function."""
    try:
        seconds = float(os.environ.get("TORCHNATIVE_SUITE_TIMEOUT_S", ""))
    except ValueError:
        return lambda: None
    if seconds <= 0:
        return lambda: None
    import faulthandler
    faulthandler.dump_traceback_later(max(30.0, seconds * 0.8), repeat=False,
                                      file=sys.stderr, exit=False)
    return faulthandler.cancel_dump_traceback_later


def run_tests(items, suite):
    """Run `(name, fn)` pairs; print ok / SKIP / FAIL; return the failure count."""
    cancel_dump = _arm_hang_dump()
    failures = 0
    for name, fn in items:
        _state["current"] = name
        try:
            fn()
        except Skip as s:
            print(f"SKIP {suite}: {name} -- {s.reason}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            if name in _state["skipped"]:
                print(f"SKIP {suite}: {name} -- {_state['skipped'][name]}")
            else:
                print(f"ok   {name}")
        _state["current"] = None
    cancel_dump()
    return failures


def on_x86_64_linux():
    """The platform of issue #40's first Linux gate (and of the hosted runner)."""
    return sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64")


def on_x86_64_windows():
    """The Windows MSVC gate runner (issue #81), where the golden harness
    consults the same x86_64 pins as on Linux (issue #300)."""
    return sys.platform == "win32" and platform.machine() in ("x86_64", "AMD64")


def known_x86_64_linux_divergence(issue, what):
    """Pin, by name, a test that fails on x86_64 Linux for a *real* reason.

    For the class of failure that is neither a test assuming macOS arm64 nor a
    runner limit: the shim reproduces one upstream build's arithmetic (the
    arm64 one) and the x86_64 wheel's capability-dispatched kernels give other
    bits (issue #40 splits `A1`..`A4`; each has its own body and is not fixed).

    Not a blanket skip. The test body **runs** on every platform:

    * on macOS arm64 (and anywhere that is not x86_64 Linux) an `AssertionError`
      is a failure exactly as before -- nothing there is weakened;
    * on x86_64 Linux an `AssertionError` becomes a `SKIP` that carries `what`,
      `issue` and the first line of the measured difference, so the ledger
      counts it by name and the log shows how far apart the two sides were;
    * on x86_64 Linux a body that *passes* is an `ok`: the day the divergence
      is fixed, the pin simply stops being used, and the ledger's SKIP count
      goes down by one.

    Only `AssertionError` is demoted. A crash, a missing probe or an import
    error on x86 is still a FAIL, which is what keeps this from hiding more
    than the one measured thing.
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except AssertionError as exc:
                if not on_x86_64_linux():
                    raise
                first = (str(exc).strip().splitlines() or [repr(exc)])[0][:220]
                raise Skip(
                    f"known x86_64 Linux divergence from upstream ({what}; {issue}); "
                    f"measured: {first}"
                ) from exc
        return wrapper
    return decorate


def known_divergence(issue, what, match):
    """Pin, by name, a test whose body asserts upstream's behaviour where the
    shim is known to differ (leader rule, 2026-11-15: no silent skip).

    The body **runs** everywhere. An `AssertionError` whose text contains
    `match` (the measured symptom) becomes a `SKIP` naming `what` and `issue`;
    any other failure is still a FAIL, so a new, different defect is not
    hidden. A body that passes is an `ok`: the day the divergence is fixed the
    pin stops being used and the SKIP count goes down by one.
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except AssertionError as exc:
                if match not in str(exc):
                    raise
                raise Skip(
                    f"known divergence from upstream ({what}; {issue}); "
                    f"measured: {match}"
                ) from exc
        return wrapper
    return decorate
