"""`import torch` on Android and under Pyodide needs `_multiprocessing` and `_posixshmem` (#437).

Neither Android's CPython nor Pyodide's (Emscripten) builds them: Android has no
SysV IPC and no usable POSIX named semaphore, and Emscripten has no processes.
`torch/multiprocessing/__init__.py` imports `multiprocessing.resource_tracker`
unconditionally, which imports both, so `import torch` stopped there:

- on the Galaxy Z Flip4 under CPython 3.15.0t (measured 2026-11-25,
  `ModuleNotFoundError: Standard library module '_multiprocessing' was not found`);
- under Pyodide 314.0.8 in Node, with torchnative 0.1.0b11's
  `pyemscripten_2026_0_wasm32` wheel installed from PyPI by micropip (measured
  2026-11-27, `ModuleNotFoundError: No module named '_multiprocessing'` from
  `multiprocessing/resource_tracker.py`, line 45).

The device harnesses worked around it with stubs the caller installs; an app or a
web page has no such hook before `import torch`.

The shim's bootstrap stands in for the two modules on those two platforms only,
and only when they are absent. `_multiprocessing` is empty: `resource_tracker`
guards it with `hasattr(_multiprocessing, "sem_unlink")`, and absence is the
honest answer. `_posixshmem`'s two functions refuse by name, so a real use of
shared memory fails loudly instead of leaking.

What this cannot see: the targets. These tests call the installer with a fake
platform on the host; examples/android's run and examples/web's run are the
on-target checks.
"""

import sys

from test_shim import _C
import _skip

STAND_IN_PLATFORMS = {"android": "Android", "emscripten": "Emscripten"}


def _install(platform, available=lambda name: False):
    modules = {}
    names = _C._shim_ipc_stand_ins(platform, modules, available)
    return names, modules


def test_each_platform_without_ipc_gets_both_stand_ins_when_they_are_absent():
    for platform in STAND_IN_PLATFORMS:
        names, modules = _install(platform)
        assert names == ["_multiprocessing", "_posixshmem"], (platform, names)
        assert not hasattr(modules["_multiprocessing"], "sem_unlink"), platform


def test_shared_memory_refuses_by_name_and_platform():
    for platform, label in STAND_IN_PLATFORMS.items():
        _, modules = _install(platform)
        for fn in ("shm_open", "shm_unlink"):
            try:
                getattr(modules["_posixshmem"], fn)("/x")
            except OSError as exc:
                assert label in str(exc) and fn in str(exc), (platform, str(exc))
            else:
                raise AssertionError(f"{platform}: _posixshmem.{fn} did not refuse")


def test_other_platforms_are_left_alone():
    for platform in ("darwin", "linux", "win32", "ios", "wasi"):
        names, modules = _install(platform)
        assert names == [] and modules == {}, (platform, names)


def test_a_real_module_is_never_replaced():
    for platform in STAND_IN_PLATFORMS:
        names, modules = _install(platform, available=lambda name: name == "_multiprocessing")
        assert names == ["_posixshmem"], (platform, names)
        assert "_multiprocessing" not in modules, platform


def test_the_host_import_did_not_install_anything():
    if sys.platform in STAND_IN_PLATFORMS:
        raise _skip.Skip(f"this host is {sys.platform}")
    assert _C._shim_ipc_installed == [], _C._shim_ipc_installed
    import _multiprocessing
    assert hasattr(_multiprocessing, "sem_unlink") or sys.platform == "win32", _multiprocessing


def _main():
    items = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    return 1 if _skip.run_tests(items, suite="test_ipc_stand_ins") else 0


if __name__ == "__main__":
    raise SystemExit(_main())
