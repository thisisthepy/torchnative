"""`import torch` on Android needs `_multiprocessing` and `_posixshmem` (#437).

Android's CPython builds neither: there is no SysV IPC and no usable POSIX named
semaphore. `torch/multiprocessing/__init__.py` imports
`multiprocessing.resource_tracker` unconditionally, which imports both, so
`import torch` stopped there on the Galaxy Z Flip4 under CPython 3.15.0t
(measured 2026-11-25, `ModuleNotFoundError: Standard library module
'_multiprocessing' was not found`). The device harnesses worked around it with
stubs gated on `BW_STUB_MULTIPROCESSING=1`; an app has no such variable.

The shim's bootstrap now stands in for the two modules on Android only, and
only when they are absent. `_multiprocessing` is empty: `resource_tracker`
guards it with `hasattr(_multiprocessing, "sem_unlink")`, and absence is the
honest answer. `_posixshmem`'s two functions refuse by name, so a real use of
shared memory fails loudly instead of leaking.

What this cannot see: the device. These tests call the installer with a fake
platform on the host; examples/android's run is the on-device check.
"""

import sys

from test_shim import _C
import _skip


def _install(platform, available=lambda name: False):
    modules = {}
    names = _C._shim_android_ipc_stand_ins(platform, modules, available)
    return names, modules


def test_android_gets_both_stand_ins_when_they_are_absent():
    names, modules = _install("android")
    assert names == ["_multiprocessing", "_posixshmem"], names
    assert not hasattr(modules["_multiprocessing"], "sem_unlink")


def test_shared_memory_refuses_by_name():
    _, modules = _install("android")
    for fn in ("shm_open", "shm_unlink"):
        try:
            getattr(modules["_posixshmem"], fn)("/x")
        except OSError as exc:
            assert "Android" in str(exc) and fn in str(exc), str(exc)
        else:
            raise AssertionError(f"_posixshmem.{fn} did not refuse")


def test_other_platforms_are_left_alone():
    for platform in ("darwin", "linux", "win32", "ios", "emscripten"):
        names, modules = _install(platform)
        assert names == [] and modules == {}, (platform, names)


def test_a_real_module_is_never_replaced():
    names, modules = _install("android", available=lambda name: name == "_multiprocessing")
    assert names == ["_posixshmem"], names
    assert "_multiprocessing" not in modules


def test_the_host_import_did_not_install_anything():
    if sys.platform == "android":
        raise _skip.Skip("this host is Android")
    import _multiprocessing
    assert hasattr(_multiprocessing, "sem_unlink") or sys.platform == "win32", _multiprocessing


def _main():
    items = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    return 1 if _skip.run_tests(items, suite="test_android_ipc") else 0


if __name__ == "__main__":
    raise SystemExit(_main())
