"""Write to logcat. Python's stdout goes nowhere in an Android app, so the example
reports through liblog's `__android_log_write`, which every Android process has."""

import ctypes

TAG = b"TorchnativeExample"
_INFO = 4
_liblog = ctypes.CDLL("liblog.so")


def log(text):
    for line in str(text).splitlines() or [""]:
        _liblog.__android_log_write(_INFO, TAG, line.encode("utf-8", "replace"))
