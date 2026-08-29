#!/usr/bin/env python3
"""What can be checked about the Linux x86-64 wheel without a Linux machine.

    python tools/wheel/verify_linux.py dist/torchnative-*manylinux*.whl
    python tools/wheel/verify_linux.py --self-test

`tools/wheel/verify_ios_device.py` is the pattern this follows, and the first
thing to say is **where it does not reach as far.**

That script proves something strong about a wheel it cannot run. A Mach-O built
for a device uses the *two-level namespace*: every undefined symbol records the
library it is bound to, because the linker wrote it there. So each import can be
resolved against that specific library and nothing else, and a `_Py*` symbol
that happened to also exist in libSystem does not paper over a missing one in
Python.framework.

**ELF has no two-level namespace.** `ld.so` resolves an undefined symbol by
searching every object in the process's global scope, in load order. The image
therefore does not, in general, say where anything comes from -- and a check
that asked "does this symbol exist in some library I have" would be answering a
different question from the one that matters.

There is exactly one exception, and it carries a lot of the weight:

    symbol *versioning*. `.gnu.version` gives every `.dynsym` entry an index;
    for an undefined symbol that index points into `.gnu.version_r`, which is
    grouped **by library**. glibc versions all of its exports, so every libc
    import does name its library and its minimum glibc, exactly as a Mach-O
    import names its dylib.

CPython versions none of its exports, so no `Py*` import names anything. Those
are checked as a union against the target distribution's `libpython3.13.so`,
which is the weaker form and is reported as such.

So the ladder, honestly:

    Mach-O device (iOS)   per-library binding for *every* undefined symbol
    ELF     (Linux)       per-library binding for versioned symbols (glibc);
                          union-against-libpython for the rest (CPython);
                          nothing at all for a symbol that is neither

What this does NOT say
----------------------

**It does not say the wheel works on Linux.** Beyond the iOS script's list, two
gaps are specific here and both are properties of this machine, not of the
method:

  * **the glibc half is unresolvable.** Versioned imports name `libc.so.6` and a
    `GLIBC_x.y`, and there is no glibc on this machine to check them against --
    no container runtime either (docs/LINUX.md §7). What is verified is that the
    requirement is *internally consistent* and within the manylinux policy; that
    the symbols exist in a real glibc of that version is taken on the linker's
    word.
  * **there is no artefact yet.** No toolchain here can cross-compile the crate
    to Linux (docs/LINUX.md §2, §4), so this script has never been run against a
    torchnative wheel. `--self-test` runs it against real Linux x86-64 CPython
    extension modules from the target distribution instead, which exercises the
    resolver but says nothing about our extension.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binfmt import describe, elf_dynamic, elf_info, elf_symbols  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Same default and same variable as tools/wheel/build.py, so the libpython the
# imports are resolved against is the one the tag was derived from.
TARGET_PYTHON_ROOT = Path(os.environ.get(
    "TORCHNATIVE_TARGET_PYTHON", "/Volumes/macMini/caches/target-python"))
LINUX_PYTHON = TARGET_PYTHON_ROOT / "x86_64-unknown-linux-gnu"

MEMBERS = ("torch/_C.abi3.so", "torch/lib/libtorch_global_deps.so")


def _fail(msg: str) -> None:
    sys.exit(f"tools/wheel/verify_linux.py: {msg}")


def libpython() -> tuple[Path, set[str]]:
    """The target distribution's libpython, and everything it exports."""
    found = sorted(LINUX_PYTHON.glob("lib/libpython3.*.so.*"))
    if not found:
        _fail(
            f"no lib/libpython3.*.so.* under {LINUX_PYTHON}\n"
            "  Without it the CPython half of the symbol check cannot run at "
            "all. That is\n"
            "  the check being unavailable, not the wheel being fine.\n"
            "  TORCHNATIVE_TARGET_PYTHON selects the distribution root."
        )
    symbols = elf_symbols(found[0].read_bytes())
    if symbols is None:
        _fail(f"{found[0]} is not a readable 64-bit ELF ({describe(found[0].read_bytes())})")
    return found[0], symbols["defined"]


def resolve(data: bytes, what: str, exports: set[str],
            extra: dict[str, set[str]] | None = None) -> int:
    """Report where each undefined symbol of `data` can come from.

    Returns the number that nothing accounts for. Weak undefined symbols are
    counted separately and are not failures -- see `binfmt.elf_symbols`.
    """
    symbols = elf_symbols(data)
    if symbols is None:
        _fail(f"{what} is not a readable 64-bit ELF ({describe(data)})")

    versioned: dict[str, list[tuple[str, str]]] = {}
    unversioned: list[str] = []
    weak: list[str] = []
    for name, library, version, is_weak in symbols["undefined"]:
        if is_weak:
            weak.append(name)
        elif library is not None:
            versioned.setdefault(library, []).append((name, version or "?"))
        else:
            unversioned.append(name)

    print(f"  {what}")
    print(f"    {len(symbols['undefined'])} undefined "
          f"({len(symbols['defined'])} exported)")

    # Per-library, from the ELF's own version records. This half is as strong as
    # the iOS check in its binding and weaker in its resolution: the library is
    # named by the file, but no such library is on this machine to look in.
    for library in sorted(versioned):
        entries = versioned[library]
        floors = sorted({v for _, v in entries})
        print(f"    {len(entries):>4}  -> {library}  (bound by .gnu.version_r; "
              f"needs {', '.join(floors)})")
        print(f"          not resolved here -- no {library} on this machine "
              "(docs/LINUX.md §6)")

    # Union, against libpython plus whatever else the caller supplied. Weaker on
    # purpose and labelled so, because ELF gives nothing better for these.
    pools = {"libpython (target distribution)": exports}
    pools.update(extra or {})
    unresolved = []
    accounted: dict[str, int] = {}
    for name in unversioned:
        for label, pool in pools.items():
            if name in pool:
                accounted[label] = accounted.get(label, 0) + 1
                break
        else:
            unresolved.append(name)
    if unversioned:
        print(f"    {len(unversioned):>4}  unversioned -- ELF records no library "
              "for these; checked as a union")
        for label, count in sorted(accounted.items()):
            print(f"          {count:>4}  found in {label}")
    if weak:
        print(f"    {len(weak):>4}  weak, allowed to stay unresolved "
              f"({', '.join(sorted(weak)[:3])}...)")
    print(f"    {len(unresolved):>4}  unresolved")
    for name in unresolved[:12]:
        print(f"          {name}")
    if len(unresolved) > 12:
        print(f"          ... and {len(unresolved) - 12} more")
    return len(unresolved)


def check_wheel(wheel: Path) -> None:
    path, exports = libpython()
    print(f"libpython: {path} ({len(exports):,} exported symbols)")
    print(f"{wheel.name}")

    bad = 0
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        for member in MEMBERS:
            hit = next((n for n in names if n.endswith(member)), None)
            if hit is None:
                _fail(f"{wheel.name} has no {member}")
            data = zf.read(hit)
            info = elf_info(data)
            if info is None or (info["bits"], info["machine"], info["type"]) != \
                    (64, "x86_64", "dyn"):
                _fail(f"{wheel.name}::{hit} is {describe(data)}, expected "
                      "ELF 64-bit x86_64 dyn")
            dynamic = elf_dynamic(data) or {}
            print(f"\n  {hit}: {describe(data)}")
            print(f"    DT_NEEDED {dynamic.get('needed')}")
            bad += resolve(data, hit, exports)

    print()
    print("ladder (docs/LINUX.md §6):")
    print("  built              yes -- the archive holds ELF x86-64 shared objects")
    print("  tagged             yes -- see tools/wheel/build.py LinuxTarget")
    print("  symbols resolve    CPython half yes; glibc half taken on the "
          "linker's word,")
    print("                     there being no glibc on this machine")
    print("  imports on Linux   UNKNOWN -- needs a Linux machine or a container")
    print("  computes           UNKNOWN -- same")
    if bad:
        _fail(f"{bad} symbol(s) nothing accounts for")
    print("\nPASS -- every non-weak undefined symbol is accounted for")


def self_test() -> int:
    """Run the resolver against real Linux x86-64 CPython extension modules.

    Not our artefact -- none exists (docs/LINUX.md §4). These are the extension
    modules shipped inside the target CPython distribution, which are the same
    shape as `torch/_C.abi3.so` would be: ELF x86-64 shared objects that import
    `Py*` unversioned from libpython and libc symbols versioned from glibc.

    `_dbm` is the plain case. `_tkinter` is the one that matters more: 70 of its
    unversioned imports come from Tcl/Tk, *not* from libpython, so it is the case
    where a union check over the wrong pool would report failures -- and it is
    checked here with the Tcl libraries supplied, and separately without them, to
    show that the resolver distinguishes the two.
    """
    dynload = LINUX_PYTHON / "lib" / "python3.13" / "lib-dynload"
    dbm = dynload / "_dbm.cpython-313-x86_64-linux-gnu.so"
    tk = dynload / "_tkinter.cpython-313-x86_64-linux-gnu.so"
    tcl = sorted(LINUX_PYTHON.glob("lib/libtcl9*.so"))

    print("SELF-TEST of the ELF symbol resolver")
    if not (dbm.exists() and tk.exists() and tcl):
        print(f"  ! skipped -- {LINUX_PYTHON} does not hold the fixtures.")
        print("    This is a SKIP, not a pass: the resolver is unexercised.")
        return 0

    path, exports = libpython()
    print(f"  libpython: {path.name} ({len(exports):,} exported)\n")

    checks: list[tuple[str, bool, str]] = []

    # 1. The plain case: every non-weak import is accounted for by libpython
    #    and by glibc's version records.
    left = resolve(dbm.read_bytes(), dbm.name, exports)
    checks.append(("a real Linux CPython extension resolves completely",
                   left == 0, f"{left} unresolved, expected 0"))

    # 2. The wrong-pool case, run first *without* Tcl so the failure is visible.
    #    A union check that could not fail would pass this too.
    print()
    left_without = resolve(tk.read_bytes(), tk.name + "  [without Tcl]", exports)
    checks.append(("...and _tkinter does NOT, when Tcl is left out",
                   left_without > 0,
                   f"{left_without} unresolved, expected more than 0"))

    # 3. Same file, with the libraries it names actually supplied.
    pools = {}
    for lib in tcl:
        symbols = elf_symbols(lib.read_bytes())
        if symbols:
            pools[lib.name] = symbols["defined"]
    print()
    left_with = resolve(tk.read_bytes(), tk.name + "  [with Tcl]", exports, pools)
    checks.append(("...and does resolve once Tcl is supplied",
                   left_with == 0,
                   f"{left_with} unresolved, expected 0"))

    # 4. The per-library binding really is coming out of the file, not guessed.
    symbols = elf_symbols(dbm.read_bytes()) or {"undefined": []}
    libc_bound = [n for n, lib, _v, _w in symbols["undefined"]
                  if lib == "libc.so.6"]
    checks.append((f"{len(libc_bound)} imports name libc.so.6 via .gnu.version_r",
                   len(libc_bound) > 0,
                   "no import was bound to a library -- .gnu.version parsing is "
                   "not working, and every symbol would fall to the weaker "
                   "union check without saying so"))

    # 5. ...and the CPython ones name nothing, which is the limit this script
    #    exists to be honest about.
    py_bound = [lib for n, lib, _v, _w in symbols["undefined"]
                if n.startswith("Py") or n.startswith("_Py")]
    checks.append(("CPython imports name no library (the ELF limit)",
                   py_bound and all(lib is None for lib in py_bound),
                   f"expected all None, got {sorted(set(py_bound))}"))

    print()
    bad = 0
    for label, ok, detail in checks:
        bad += not ok
        print(f"  {'ok    ' if ok else 'WRONG '}{label}")
        if not ok:
            print(f"          {detail}")
    if bad:
        print(f"SELF-TEST: FAIL -- {bad}/{len(checks)} wrong")
    else:
        print(f"SELF-TEST: PASS -- {len(checks)}/{len(checks)} cases, on real "
              "Linux ELF from the target distribution.")
        print("  This exercises the resolver. It says nothing about "
              "torch/_C.abi3.so,")
        print("  which does not exist on this machine (docs/LINUX.md §4).")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wheel", nargs="?", type=Path)
    ap.add_argument("--self-test", action="store_true",
                    help="exercise the resolver on the target distribution's "
                         "own extension modules; inspects no wheel")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)
    if args.wheel is None:
        ap.error("a wheel is required unless --self-test is given")
    if not args.wheel.exists():
        _fail(f"{args.wheel} does not exist")
    check_wheel(args.wheel)


if __name__ == "__main__":
    main()
