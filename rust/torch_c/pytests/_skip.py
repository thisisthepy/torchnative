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
