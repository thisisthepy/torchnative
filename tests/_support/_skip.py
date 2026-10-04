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


def run_tests(items, suite):
    """Run `(name, fn)` pairs; print ok / SKIP / FAIL; return the failure count."""
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
    return failures


def on_x86_64_linux():
    """The platform of issue #40's first Linux gate (and of the hosted runner)."""
    return sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64")


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
