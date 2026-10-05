"""State that upstream keeps per thread must not leak between threads (#76).

Upstream keeps grad mode (`c10/core/GradMode.h`) and the `torch.device(...)`
mode stack per thread. The shim held both process-wide, so `torch.no_grad()`
in one thread silently stopped another thread's `grad_fn` recording. That is
wrong with the GIL on, not only on a free-threaded interpreter: the GIL
serialises bytecode, it does not make a global per thread.

Each test parks one thread inside a context manager (a no-grad region, a device
mode) and asserts from another thread that the state it sees is its own. The
same tests pass against upstream torch, which is how they were checked before
the fix.

What this cannot see: a race that leaves the right answer. These tests check
isolation, not memory safety; the RNG stream free race of #76 needs the
free-threaded hammer in tests/api/test_freethreaded.py.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_support"))
sys.path.insert(0, os.path.join(str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())), "torchnative", "python"))

import torch  # noqa: E402
import _skip  # noqa: E402


def _park(enter, seen):
    """Run `enter()` in a thread, record what it sees inside, hold it open."""
    inside, release = threading.Event(), threading.Event()

    def body():
        with enter():
            seen["inside"] = torch.is_grad_enabled()
            inside.set()
            release.wait(10)
        seen["after"] = torch.is_grad_enabled()

    t = threading.Thread(target=body)
    t.start()
    assert inside.wait(10), "parked thread never entered"
    return t, release


def test_no_grad_in_another_thread_does_not_stop_recording_here():
    seen = {}
    t, release = _park(torch.no_grad, seen)
    try:
        x = torch.ones(2, requires_grad=True)
        y = x * 2
        here = torch.is_grad_enabled()
    finally:
        release.set()
        t.join(10)
    assert seen["inside"] is False, seen
    assert here is True, "grad mode leaked from another thread's no_grad"
    assert y.grad_fn is not None, "grad_fn not recorded: another thread's no_grad applied here"


def test_overlapping_no_grad_regions_restore_each_threads_own_mode():
    # Thread A enters no_grad, then the main thread enters and leaves its own.
    # With one global flag, the main thread's exit restores A's False (or A's
    # exit restores the main thread's), and someone is left with the wrong mode.
    seen = {}
    t, release = _park(torch.no_grad, seen)
    try:
        with torch.no_grad():
            assert torch.is_grad_enabled() is False
        after_main = torch.is_grad_enabled()
    finally:
        release.set()
        t.join(10)
    assert after_main is True, "main thread left with another thread's mode"
    assert seen["after"] is True, seen


def test_a_device_mode_in_another_thread_does_not_apply_here():
    inside, release = threading.Event(), threading.Event()

    def body():
        with torch.device("meta"):
            inside.set()
            release.wait(10)

    t = threading.Thread(target=body)
    t.start()
    try:
        assert inside.wait(10)
        dev = torch.empty(2).device
    finally:
        release.set()
        t.join(10)
    assert dev.type == "cpu", f"another thread's torch.device('meta') applied here: {dev}"


if __name__ == "__main__":
    assert hasattr(torch._C, "_aten_implemented") or os.environ.get("TORCHNATIVE_ORACLE") == "1", \
        "imported upstream torch, not the shim"
    raise SystemExit(1 if _skip.run_tests(
        [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)],
        "api/test_threadstate") else 0)
