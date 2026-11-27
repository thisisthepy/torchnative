"""The rustc flags that link the crate unit tests against this interpreter's libpython.

`tests/run.sh` uses this on Linux (#442). `torch_c` depends on pyo3 with the
`extension-module` feature, which leaves libpython out of every link: right for
the extension module, which the loading interpreter supplies, and wrong for the
unit-test binary, which is a plain executable (pyo3 FAQ, issue #340). GNU ld
refuses its CPython references at link time (the aarch64 runner, run
45924641519: 1100 `undefined reference` lines). The x86_64 runner linked only
because rustc links that target with rust-lld by default since 1.90. macOS
resolves the same references at load time instead (`run.sh`, issue #302).

    python tests/_support/libpython_link.py   # prints the flags, or refuses

The flags name the interpreter's own shared libpython (`LIBDIR`/`LDLIBRARY`)
and put its directory on the rpath, so the binary loads it without
LD_LIBRARY_PATH. A static interpreter has no library to link, and cargo splits
RUSTFLAGS on spaces, so a path with whitespace cannot be passed: both refuse by
name rather than produce a link line that fails 1100 times.
"""

import sys
import sysconfig


class Refused(RuntimeError):
    pass


def rustflags(config_vars):
    """Flags for `CARGO_TARGET_<host>_RUSTFLAGS`, from `sysconfig.get_config_vars()`."""
    if not config_vars.get("Py_ENABLE_SHARED"):
        raise Refused("the interpreter is a static build (Py_ENABLE_SHARED=0), so it has no "
                      "libpython to link the crate unit tests against")
    libdir = config_vars.get("LIBDIR")
    if not libdir:
        raise Refused("sysconfig names no LIBDIR for the interpreter's libpython")
    ldlibrary = config_vars.get("LDLIBRARY") or ""
    if not (ldlibrary.startswith("lib") and ".so" in ldlibrary):
        raise Refused(f"sysconfig LDLIBRARY is {ldlibrary!r}, not a shared libpython (lib*.so)")
    name = ldlibrary[len("lib"):ldlibrary.index(".so")]
    if any(c.isspace() for c in libdir + name):
        raise Refused(f"the libpython path {libdir!r} contains whitespace, which cargo's "
                      "RUSTFLAGS cannot carry")
    return " ".join(f"-C link-arg={a}" for a in (f"-L{libdir}", f"-l{name}", f"-Wl,-rpath,{libdir}"))


def main():
    try:
        print(rustflags(sysconfig.get_config_vars()))
    except Refused as err:
        print(f"libpython_link: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
