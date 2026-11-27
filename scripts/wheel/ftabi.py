"""The free-threaded wheel variant (issue #51): what is derived, and what is refused.

Pure and importable without building anything. `build.py --abi ft` calls it; so
does `tests/release/test_ftwheel.py`.

A free-threaded CPython (3.15t) has no stable ABI, so the abi3 wheel cannot load
there. The ft wheel carries a version-pinned extension name and a `cpXYZ-cpXYZt`
tag. Everything version-shaped is read out of the target interpreter's
sysconfig (`Py_GIL_DISABLED`, `VERSION`, `SOABI`, `EXT_SUFFIX`) and cross-checked,
so a GIL interpreter, or one whose suffix disagrees with its version, is refused
by name instead of producing a wheel that fails at import.

What this cannot see: whether the produced wheel loads. That needs a 3.15t
interpreter and is `tests/*/test_freethreaded.py`'s subprocess.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass


class FtRefusal(Exception):
    """The free-threaded variant cannot be built or derived here; the text says why."""


@dataclass(frozen=True)
class FtIdentity:
    python_tag: str   # cp315
    abi_tag: str      # cp315t
    ext_suffix: str   # .cpython-315t-x86_64-linux-gnu.so
    member: str       # torch/_C<ext_suffix>

    @property
    def tag_prefix(self) -> str:
        return f"{self.python_tag}-{self.abi_tag}"


_VARS = ("Py_GIL_DISABLED", "VERSION", "SOABI", "EXT_SUFFIX")


def ft_identity(variables: dict) -> FtIdentity:
    """Derive the ft tag and extension name from an interpreter's sysconfig vars."""
    flag = variables.get("Py_GIL_DISABLED")
    if flag is None:
        raise FtRefusal(
            "Py_GIL_DISABLED is not set in this interpreter's sysconfig; it is "
            "not known to be a free-threaded build, and neither true nor false "
            "is assumed")
    try:
        free = int(flag) == 1
    except (TypeError, ValueError):
        raise FtRefusal(f"Py_GIL_DISABLED is {flag!r}, which is not 0 or 1") from None
    if not free:
        raise FtRefusal(
            "this interpreter is not free-threaded (Py_GIL_DISABLED is "
            f"{flag!r}); an extension built for it is not a cp3XXt wheel")
    m = re.fullmatch(r"3\.(\d+)", str(variables.get("VERSION", "")))
    if not m:
        raise FtRefusal(f"VERSION is {variables.get('VERSION')!r}, expected 3.<minor>")
    minor = m.group(1)
    want = f"3{minor}t"
    for key in ("SOABI", "EXT_SUFFIX"):
        value = variables.get(key)
        if not isinstance(value, str) or want not in value:
            raise FtRefusal(
                f"{key} is {value!r}, which does not carry {want}; VERSION "
                f"{variables.get('VERSION')} with Py_GIL_DISABLED=1 implies it, "
                "so the interpreter's sysconfig disagrees with itself")
    suffix = variables["EXT_SUFFIX"]
    return FtIdentity(f"cp3{minor}", f"cp3{minor}t", suffix, "torch/_C" + suffix)


def ft_wheel_name(name: str, ident: FtIdentity) -> str:
    """Swap the python and abi tags of `<dist>-<ver>-<py>-<abi>-<plat>.whl`."""
    if not name.endswith(".whl"):
        raise FtRefusal(f"{name} is not a wheel filename")
    parts = name[:-4].split("-")
    if len(parts) != 5:
        raise FtRefusal(
            f"{name} has {len(parts)} dash-separated parts, not the five of "
            "<dist>-<version>-<python>-<abi>-<platform>; refusing to guess")
    parts[2], parts[3] = ident.python_tag, ident.abi_tag
    return "-".join(parts) + ".whl"


def ft_wheel_metadata(wheel_file: bytes, ident: FtIdentity) -> bytes:
    """Rewrite every `Tag:` line of a WHEEL file, keeping the platform part."""
    out = []
    for line in wheel_file.splitlines():
        if line.startswith(b"Tag: "):
            plat = line.rsplit(b"-", 1)[1]
            line = b"Tag: " + ident.tag_prefix.encode() + b"-" + plat
        out.append(line + b"\n")
    return b"".join(out)


def interpreter_variables(python: str) -> dict:
    """The four sysconfig variables of a runnable interpreter."""
    code = ("import json,sysconfig;print(json.dumps({k:sysconfig.get_config_var(k) "
            f"for k in {_VARS!r}}}))")
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise FtRefusal(f"{python} could not report its sysconfig: {r.stderr.strip()}")
    return json.loads(r.stdout)


#: Targets a free-threaded wheel can be derived for. Linux cross targets read
#: python-build-standalone's `<triple>-freethreaded` distribution. Android arm64
#: reads python-multiplatform's 3.15t prefix, the runtime toolchain's apps ship
#: (`PythonSdk.PUBLISHED = ["3.15.0t"]`), so the abi3 wheel cannot load there
#: (issue #437, examples/android).
FT_TARGETS = ("host", "linux-x86_64", "linux-aarch64", "android-arm64-v8a", "ios-arm64-sim")

#: Derivable and built here, but not in publish-pypi.yml's `build-ft` job: a
#: cp315t Android wheel on PyPI is a new published artefact, which waits for the
#: owner's approval (AGENTS.md section 7).
FT_LOCAL_ONLY = ("android-arm64-v8a", "ios-arm64-sim")

#: Where a target's 3.15t prefix comes from, named when it is missing. The
#: Android one is python-multiplatform's own release asset, the same file (and
#: sha256) its Gradle build pins in `python-checksums.properties`.
FT_PREFIX_HINTS = {
    "android-arm64-v8a": (
        "download https://github.com/thisisthepy/python-multiplatform/releases/download/"
        "cpython-3.15.0t-mobile/python-3.15.0t-aarch64-linux-android.tar.gz "
        "(sha256 40995c530c3e454d829b4fdea8404d8c0b0e58f60c2d41d400daa1fb0ab65991), "
        "and move its prefix/ to .caches/target-python/aarch64-linux-android-freethreaded"),
    # The iOS one is the same release's XCframework (#437). Its simulator slice keeps the
    # stdlib shared and the sysconfig module per arch, so the prefix is composed from it.
    "ios-arm64-sim": (
        "download https://github.com/thisisthepy/python-multiplatform/releases/download/"
        "cpython-3.15.0t-mobile/python-3.15.0t-iOS-XCframework.tar.gz "
        "(sha256 b8c0a4b32d31738f6ead7ace28acee0cb2a590bd0ca7015e9c32af613445e6b5); in "
        ".caches/target-python/aarch64-apple-ios-sim-freethreaded put lib/python3.15t "
        "(Python.xcframework/lib/python3.15t plus the simulator slice's "
        "lib-arm64/python3.15t/_sysconfigdata_t_ios_arm64-iphonesimulator.py), "
        "Python.framework, bin and include/python3.15t (the framework's Headers), all from "
        "Python.xcframework/ios-arm64_x86_64-simulator"),
}

#: The cargo command that builds a target's ft extension, named when it is
#: missing or stale. API 24 is the prefix's own ANDROID_API_LEVEL.
FT_REBUILD_HINTS = {
    "android-arm64-v8a": (
        "(cd torchnative/rust/torch_c && CARGO_TARGET_DIR=$TORCHNATIVE_FT_CARGO_TARGET_DIR "
        "PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.15 "
        "PYO3_CROSS_LIB_DIR=.caches/target-python/aarch64-linux-android-freethreaded/lib "
        "cargo ndk -t arm64-v8a --platform 24 build --release --no-default-features "
        "--features executorch)"),
    "ios-arm64-sim": (
        "(cd torchnative/rust/torch_c && CARGO_TARGET_DIR=$TORCHNATIVE_FT_CARGO_TARGET_DIR "
        "PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.15 "
        "PYO3_CROSS_LIB_DIR=.caches/target-python/aarch64-apple-ios-sim-freethreaded/lib "
        "PYO3_CONFIG_FILE=<a file of implementation=CPython, version=3.15, shared=true, "
        "abi3=false, lib_name=Python, pointer_width=64, build_flags=Py_GIL_DISABLED, "
        "suppress_build_script_link_lines=true; docs/platform/WHEEL.md 7.1 has the 3.13 one> "
        "IPHONEOS_DEPLOYMENT_TARGET=13.0 "
        "cargo build --target aarch64-apple-ios-sim --release --no-default-features)"),
}

_WINDOWS = ("Windows CPython ships no _sysconfigdata, and WindowsTarget is written "
            "against python3.lib / python313.dll, so there is no 3.15t identity "
            "to read and no import library to link")
_ANDROID_X86 = ("the android-x86_64 target itself is refused (build.py: no "
                "x86_64-linux-android CPython to derive a tag from), and that holds "
                "for the free-threaded variant too")

FT_REFUSALS = {
    "windows-x86_64": f"cannot build for 3.15t: {_WINDOWS}",
    "windows-arm64": f"cannot build for 3.15t: {_WINDOWS}",
    "ios-arm64": ("cannot build for 3.15t yet: python-multiplatform's iOS XCframework has a "
                  "free-threaded ios-arm64 slice, but the device target is not wired here "
                  "(its framework link and signing differ from the simulator's; #437)"),
    "android-x86_64": f"cannot build for 3.15t (free-threaded): {_ANDROID_X86}",
    "wasm32-emscripten": ("cannot build for 3.15t: Pyodide is not a "
                          "free-threaded interpreter"),
}
