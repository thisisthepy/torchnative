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

# The shim imports upstream extension libraries that resolve only with global
# symbols, as every other suite that imports torch sets.
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
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


def test_ending_a_capture_on_one_thread_does_not_stop_another_threads():
    # #76: the "is anyone recording" flag was one bool, cleared by whichever
    # thread ended its capture first, while the recorders are per thread. The
    # other thread's later ops then skipped recording and vanished from its
    # trace. Here the main thread captures and ends while thread A is mid-capture.
    C = torch._C
    if not hasattr(C, "_capture_begin"):
        raise _skip.Skip("no capture API (upstream torch)")
    began, release, out = threading.Event(), threading.Event(), {}

    def body():
        x = torch.ones(3)
        C._capture_begin([x])
        began.set()
        release.wait(10)
        y = x.neg()
        out["ops"] = [n["op"] for n in C._capture_end(y).nodes]

    t = threading.Thread(target=body)
    t.start()
    try:
        assert began.wait(10)
        a = torch.ones(2)
        C._capture_begin([a])
        C._capture_end(a.exp())
    finally:
        release.set()
        t.join(10)
    assert any("neg" in op for op in out.get("ops", [])), \
        f"thread A's op was dropped after another thread ended its capture: {out}"


_REGISTRATIONS_CHILD = r"""
import json, sys, threading
import torch
assert hasattr(torch._C, "_aten_implemented"), "the child imported upstream, not the shim"
sys.setswitchinterval(1e-6)
key = "CompositeImplicitAutograd"
n = 8
barrier = threading.Barrier(n)
seen = [None] * n

def body(i):
    barrier.wait()
    seen[i] = torch._C._dispatch_get_registrations_for_dispatch_key(key)

threads = [threading.Thread(target=body, args=(i,)) for i in range(n)]
for t in threads:
    t.start()
for t in threads:
    t.join(60)
final = torch._C._dispatch_get_registrations_for_dispatch_key(key)
print(json.dumps({"seen": [len(s) if s is not None else None for s in seen],
                  "same": [s == final for s in seen],
                  "final": len(final), "unique": len(set(final))}))
"""


def test_dispatch_registrations_built_by_eight_threads_at_once_are_whole():
    # #76 B6: the table was filled lazily into the shared dict, and the
    # `if _DISPATCH_REGISTRATIONS: return` check is true as soon as the keys
    # are inserted, before any name is. A second thread got back a partial
    # table, and two threads filling at once appended every name twice. Each
    # run is a fresh subprocess because the table is built once per process.
    import json
    import subprocess
    if not hasattr(torch._C, "_aten_implemented"):
        raise _skip.Skip("upstream torch: no file-built registration table")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path[:2])
    for attempt in range(3):
        proc = subprocess.run([sys.executable, "-c", _REGISTRATIONS_CHILD],
                              capture_output=True, text=True, timeout=600, env=env)
        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        assert proc.returncode == 0 and lines, proc.stderr[-2000:]
        out = json.loads(lines[-1])
        assert out["final"] > 0, out
        assert out["unique"] == out["final"], (
            f"run {attempt}: the table holds duplicate names, two threads filled it at once: {out}")
        assert all(out["same"]), (
            f"run {attempt}: a thread got back a different table than the finished one: {out}")


def test_a_constant_stamped_while_mutably_borrowed_is_refused_not_skipped():
    # #114, R5. The freshness stamp read a tensor with `try_borrow`, and a
    # tensor held by a `&mut self` method read as "no storage", which
    # refuses nothing: the constant was never checked again. Without the GIL
    # the `&mut` holder is another thread; here it is deterministic and needs
    # no free-threaded interpreter, because a setter drops the value it
    # replaces while it still holds the borrow, so the old value's `__del__`
    # runs inside it. That `__del__` ends the capture, which stamps `w`.
    C = torch._C
    if not hasattr(C, "_capture_begin"):
        raise _skip.Skip("no capture API (upstream torch)")
    x = torch.ones(3)
    w = torch.full((3,), 2.0)
    C._capture_begin([x])
    y = x * w
    got = {}

    class EndCaptureOnDrop:
        def __del__(self):
            try:
                got["trace"] = C._capture_end(y)
            except BaseException as e:  # __del__ swallows exceptions; keep it
                got["error"] = f"{type(e).__name__}: {e}"

    w._backward_hooks = EndCaptureOnDrop()
    w._backward_hooks = None  # the old value dies inside the setter
    assert "trace" in got, got
    trace = got["trace"]
    w.add_(1.0)  # the constant moves after the capture
    try:
        out = trace.replay([x])
    except RuntimeError as e:
        assert "constant 0" in str(e), str(e)
        assert "could not be read" in str(e) or "in-place" in str(e), str(e)
    else:
        raise AssertionError(
            f"a trace whose constant moved after capture replayed to "
            f"{out[0].tolist()} instead of refusing: the constant was stamped while "
            f"borrowed and the stamp was skipped")


def test_a_tensor_setter_holds_no_borrow_while_the_value_it_replaces_dies():
    # #114, R23. `TensorBase`'s setters (`requires_grad`, `_backward_hooks`,
    # `_shim_grad`, the door's `from_op`/`retains_grad`) took `&mut self`, so
    # for their duration any other use of the tensor -- another thread's op,
    # or a `__del__` the setter itself triggered by dropping the old value --
    # raised `PanicException: Already mutably borrowed` (measured on the
    # commit before this one: replaying a trace over `w` in that `__del__`).
    # Upstream raises nothing there. The setters take `&self` now, and drop
    # the old value after releasing the slot's lock.
    C = torch._C
    if not hasattr(C, "_capture_begin"):
        raise _skip.Skip("no capture API (upstream torch)")
    x = torch.ones(3)
    w = torch.full((3,), 2.0)
    C._capture_begin([x])
    trace = C._capture_end(x * w)
    got = {}

    class UseOnDrop:
        def __del__(self):
            try:
                got["replay"] = trace.replay([x])[0].tolist()
                got["add"] = (w + 1).tolist()
                got["hooks"] = w._backward_hooks
                w.requires_grad = True  # a second setter, re-entrant
                got["requires_grad"] = w.requires_grad
            except BaseException as e:
                got["error"] = f"{type(e).__name__}: {e}"

    w._backward_hooks = UseOnDrop()
    w._backward_hooks = None
    assert "error" not in got, got
    assert got == {"replay": [2.0, 2.0, 2.0], "add": [3.0, 3.0, 3.0],
                   "hooks": None, "requires_grad": True}, got


def test_backward_on_another_thread_than_the_forward():
    # #78: upstream's graph hangs off the tensors, so a forward recorded in one
    # thread can be differentiated from another. The shim's eager tape was per
    # thread and the main thread found none: "backward through the graph a
    # second time", which names the wrong cause as well as refusing.
    x = torch.ones(3, requires_grad=True)
    out = {}

    def body():
        out["y"] = (x * 3).sum()

    t = threading.Thread(target=body)
    t.start()
    t.join(10)
    out["y"].backward()
    assert x.grad is not None and x.grad.tolist() == [3.0, 3.0, 3.0], x.grad


def test_backward_while_the_recording_thread_is_still_alive():
    # The same as above with the worker parked rather than finished: a thread
    # pool keeps its threads, so a tape handed over only at thread exit would
    # not cover it.
    x = torch.ones(3, requires_grad=True)
    out, ready, release = {}, threading.Event(), threading.Event()

    def body():
        out["y"] = (x * x).sum()
        ready.set()
        release.wait(10)

    t = threading.Thread(target=body)
    t.start()
    try:
        assert ready.wait(10)
        out["y"].backward()
    finally:
        release.set()
        t.join(10)
    assert x.grad is not None and x.grad.tolist() == [2.0, 2.0, 2.0], x.grad


def test_two_threads_train_independently_at_once():
    # Each thread's backward must free only its own graph: with one shared tape
    # taken whole, the first backward would consume the other thread's nodes.
    # The barrier puts both forwards on the tapes before either backward.
    rounds, barrier, errors = 5, threading.Barrier(2), []

    def body(scale):
        try:
            for r in range(rounds):
                x = torch.ones(4, requires_grad=True)
                y = (x * float(scale + r)).sum()
                barrier.wait(10)
                y.backward()
                barrier.wait(10)
                want = [float(scale + r)] * 4
                if x.grad is None or x.grad.tolist() != want:
                    errors.append((scale, r, None if x.grad is None else x.grad.tolist()))
        except Exception as exc:  # noqa: BLE001 -- reported below, by thread
            errors.append((scale, repr(exc)))
            barrier.abort()

    ts = [threading.Thread(target=body, args=(s,)) for s in (2, 7)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert not errors, errors


def test_grad_fn_classes_and_return_type_namedtuples_have_one_type_across_threads():
    # #114 B7 and R10: concurrent first calls for an op's grad_fn class or
    # a namedtuple return type (`max`, `min`, `sort`, `topk`, `linalg_qr`) must
    # expose one shared class object across all threads.
    barrier = threading.Barrier(8)
    grad_types, max_types, min_types, sort_types, topk_types, qr_types = [], [], [], [], [], []
    lock = threading.Lock()
    errors = []

    def worker():
        try:
            x = torch.ones(4, requires_grad=True)
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            barrier.wait(10)
            y = torch.sin(x)
            mx = torch.max(a, dim=0)
            mn = torch.min(a, dim=0)
            st = torch.sort(a, dim=0)
            tk = torch.topk(a, k=1, dim=0)
            qr = torch.linalg.qr(a)
            with lock:
                grad_types.append(type(y.grad_fn))
                max_types.append(type(mx))
                min_types.append(type(mn))
                sort_types.append(type(st))
                topk_types.append(type(tk))
                qr_types.append(type(qr))
        except BaseException as exc:
            with lock:
                errors.append(repr(exc))
            barrier.abort()

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
        assert not t.is_alive(), "worker thread timed out"
    assert not errors, errors
    for kinds in (grad_types, max_types, min_types, sort_types, topk_types, qr_types):
        assert len(kinds) == 8 and len({id(k) for k in kinds}) == 1, kinds


if __name__ == "__main__":
    assert hasattr(torch._C, "_aten_implemented") or os.environ.get("TORCHNATIVE_ORACLE") == "1", \
        "imported upstream torch, not the shim"
    raise SystemExit(1 if _skip.run_tests(
        [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)],
        "api/test_threadstate") else 0)
