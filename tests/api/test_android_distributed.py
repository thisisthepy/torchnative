"""`torch.distributed` is not built on Android, as upstream's Android builds (#437).

`torch.distributed.is_available()` is `hasattr(torch._C, "_c10d_init")`
(`torch/distributed/__init__.py:28`). The shim answers it on every host. On the
Galaxy Z Flip4 (CPython 3.15.0t, 2026-11-25) that answer sent transformers
5.24.1 into `torch.distributed.checkpoint` from `modeling_utils.py`, and
upstream's distributed tree then stopped twice on Android:

- `multiprocessing.synchronize`: "This platform lacks a functioning sem_open
  implementation".
- `torch/distributed/elastic/multiprocessing/redirects.py`: ctypes loads
  `libc.so.6`, a glibc name Android does not have.

Upstream's Android builds set `USE_DISTRIBUTED=0`, so there `_c10d_init` is
absent and the tree takes its own "not available" path. The shim now does the
same on Android only. That is the leader's decision (2026-11-25, option (a)),
not the owner's words. With `_c10d_init` deleted after import, the common task
ran to the end on the phone and agreed with upstream on cpu and npu.

What this cannot see: the device. These tests ask the predicate with a fake
platform on the host and check the host's own install. examples/android's run
is the on-device check.
"""

import sys

from test_shim import _C
import _skip


def test_android_is_the_one_platform_without_distributed():
    assert _C._shim_distributed_built("android") is False
    for platform in ("darwin", "linux", "win32", "ios", "emscripten"):
        assert _C._shim_distributed_built(platform) is True, platform


def test_the_host_answers_as_its_platform_does():
    built = _C._shim_distributed_built()
    assert built is (sys.platform != "android"), (sys.platform, built)
    assert hasattr(_C, "_c10d_init") is built, (sys.platform, built)
    assert ("_c10d_init" in _C._shim_off_switches) is (not built), _C._shim_off_switches


def _main():
    items = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    return 1 if _skip.run_tests(items, suite="test_android_distributed") else 0


if __name__ == "__main__":
    raise SystemExit(_main())
