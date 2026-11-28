"""Write to logcat. Python's stdout goes nowhere in an Android app, so the example
reports through liblog's `__android_log_write`, which every Android process has."""

import ctypes
from typing import Callable, cast

TAG = b"TorchnativeExample"
_INFO = 4
_liblog = ctypes.CDLL("liblog.so")
# int __android_log_write(int prio, const char* tag, const char* text)
_write = cast(Callable[[int, bytes, bytes], int], _liblog.__android_log_write)


def log(text: object) -> None:
    for line in str(text).splitlines() or [""]:
        _write(_INFO, TAG, line.encode("utf-8", "replace"))
