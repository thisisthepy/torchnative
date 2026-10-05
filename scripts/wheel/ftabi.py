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
#: python-build-standalone's `<triple>-freethreaded` distribution.
FT_TARGETS = ("host", "linux-x86_64", "linux-aarch64")

_WINDOWS = ("Windows CPython ships no _sysconfigdata, and WindowsTarget is written "
            "against python3.lib / python313.dll, so there is no 3.15t identity "
            "to read and no import library to link")
_NO_DIST = ("no free-threaded 3.15t distribution exists for this platform "
            "(BeeWare's 3.15-b0 is GIL-only)")

FT_REFUSALS = {
    "windows-x86_64": f"cannot build for 3.15t: {_WINDOWS}",
    "windows-arm64": f"cannot build for 3.15t: {_WINDOWS}",
    "ios-arm64": f"cannot build for 3.15t: {_NO_DIST}",
    "ios-arm64-sim": f"cannot build for 3.15t: {_NO_DIST}",
    "android-arm64-v8a": f"cannot build for 3.15t: {_NO_DIST}",
    "android-x86_64": f"cannot build for 3.15t: {_NO_DIST}",
    "wasm32-emscripten": ("cannot build for 3.15t: Pyodide is not a "
                          "free-threaded interpreter"),
}
