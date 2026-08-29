"""Read what a binary says about the platform it was built for.

A cross-built wheel has exactly one interesting failure mode: it looks right and
carries the wrong machine code. Nothing in `pip`, `setuptools` or `wheel` checks
that, because on the host they never have to -- the compiler and the tag come
from the same `sysconfig`. Once the two are decoupled the tag becomes a claim,
and this module is what the claim is checked against.

`file(1)` would answer most of these questions, but only in prose, and only for
a path on disk; the bytes here come out of a zip archive. Both formats are read
directly instead:

  Mach-O   architecture, `LC_BUILD_VERSION` (which platform, which minimum OS)
           and the `LC_LOAD_DYLIB` list. The platform field is the load-bearing
           one -- an arm64 `iphoneos` dylib and an arm64 `macos` dylib differ in
           nothing else that a size or an architecture check would notice.
  ELF      class, endianness, machine and type. Android's `.so` files carry no
           API level, so that half of the tag cannot be verified from the
           artefact and is taken from the interpreter it is built against.
           Linux is the other way round -- `elf_dynamic` below reads `DT_NEEDED`
           and the `.gnu.version_r` symbol-version requirements, and the highest
           `GLIBC_x.y` in there *is* the manylinux floor (docs/LINUX.md §5.2).
           There is no Mach-O analogue for that, and no Android one either.

Everything returns `None` rather than raising when the bytes are not of that
format, so a caller can ask both questions and use whichever answered.
"""

from __future__ import annotations

import struct

# ------------------------------------------------------------------- Mach-O

_FAT_MAGICS = {0xCAFEBABE, 0xCAFEBABF}
_THIN_MAGICS = {0xFEEDFACE, 0xFEEDFACF}

# cputype values from <mach/machine.h>; the 0x01000000 bit is CPU_ARCH_ABI64.
CPU_NAMES = {0x0100000C: "arm64", 0x01000007: "x86_64", 0x00000007: "i386"}

LC_ID_DYLIB = 0x0D
LC_LOAD_DYLIB = 0x0C
LC_BUILD_VERSION = 0x32

# The pre-LC_BUILD_VERSION spelling, one command per platform. Still emitted:
# the iOS `_C.dylib` built here carries LC_VERSION_MIN_IPHONEOS rather than
# LC_BUILD_VERSION, because Rust's default deployment target for
# `aarch64-apple-ios` (10.0) predates the newer command. Reading only
# LC_BUILD_VERSION would report `platform: None` for it and quietly weaken every
# check downstream.
LC_VERSION_MIN = {
    0x24: "macos", 0x25: "ios", 0x2F: "tvos", 0x30: "watchos",
}
LC_VERSION_MIN_IPHONEOS = 0x25

# <mach-o/loader.h> PLATFORM_*. The pair that matters here is 2 vs 7: a wheel
# tagged `..._iphoneos` holding a `iossimulator` binary installs and then fails
# on a device only, which is the slowest possible place to find out.
MACHO_PLATFORMS = {
    1: "macos", 2: "ios", 3: "tvos", 4: "watchos", 5: "bridgeos",
    6: "maccatalyst", 7: "iossimulator", 8: "tvossimulator",
    9: "watchossimulator", 10: "driverkit",
}


def _ver(packed: int) -> tuple[int, int, int]:
    """Apple's xxxx.yy.zz packed into 32 bits."""
    return (packed >> 16, (packed >> 8) & 0xFF, packed & 0xFF)


def macho_arches(data: bytes) -> list[str]:
    """Architectures actually present in a Mach-O image (fat or thin)."""
    if len(data) < 8:
        return []
    magic_be = struct.unpack_from(">I", data)[0]
    if magic_be in _FAT_MAGICS:
        # `fat_arch` is 20 bytes, `fat_arch_64` (magic ...BF) is 32; both start
        # with cputype, which is the only field wanted here.
        stride = 20 if magic_be == 0xCAFEBABE else 32
        count = struct.unpack_from(">I", data, 4)[0]
        return [
            CPU_NAMES.get(c, f"cpu{c:#x}")
            for c in (
                struct.unpack_from(">I", data, 8 + i * stride)[0]
                for i in range(count)
            )
        ]
    for endian in ("<", ">"):
        magic = struct.unpack_from(endian + "I", data)[0]
        if magic in _THIN_MAGICS:
            cputype = struct.unpack_from(endian + "I", data, 4)[0]
            return [CPU_NAMES.get(cputype, f"cpu{cputype:#x}")]
    return []


def macho_info(data: bytes) -> dict | None:
    """Architecture, target platform, minimum OS and dylib list of a thin
    64-bit Mach-O. `None` if `data` is not one (fat images included -- nothing
    this project ships is fat, and pretending to summarise one would hide which
    slice the answer came from).
    """
    if len(data) < 32:
        return None
    magic = struct.unpack_from("<I", data)[0]
    if magic != 0xFEEDFACF:  # thin, 64-bit, little-endian
        return None
    cputype, _sub, _ftype, ncmds, _sizeofcmds, _flags, _res = struct.unpack_from(
        "<IIIIIII", data, 4
    )
    out: dict = {
        "format": "macho",
        "arch": CPU_NAMES.get(cputype, f"cpu{cputype:#x}"),
        "platform": None,
        "minos": None,
        "id": None,
        "dylibs": [],
    }
    off = 32
    for _ in range(ncmds):
        if off + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmdsize < 8 or off + cmdsize > len(data):
            break
        if cmd == LC_BUILD_VERSION and cmdsize >= 24:
            plat, minos, _sdk, _ntools = struct.unpack_from("<IIII", data, off + 8)
            out["platform"] = MACHO_PLATFORMS.get(plat, f"platform{plat}")
            out["minos"] = _ver(minos)
        elif cmd in LC_VERSION_MIN and cmdsize >= 16 and not out["platform"]:
            version, _sdk = struct.unpack_from("<II", data, off + 8)
            out["platform"] = LC_VERSION_MIN[cmd]
            out["minos"] = _ver(version)
        elif cmd in (LC_ID_DYLIB, LC_LOAD_DYLIB) and cmdsize >= 24:
            name_off = struct.unpack_from("<I", data, off + 8)[0]
            start = off + name_off
            end = data.find(b"\0", start, off + cmdsize)
            name = data[start: end if end >= 0 else off + cmdsize].decode(
                "utf-8", "replace"
            )
            if cmd == LC_ID_DYLIB:
                out["id"] = name
            else:
                out["dylibs"].append(name)
        off += cmdsize
    return out


# ---------------------------------------------------------------------- ELF

# e_machine values from <elf.h>.
ELF_MACHINES = {0xB7: "aarch64", 0x3E: "x86_64", 0x28: "arm", 0x03: "i386"}
ELF_TYPES = {1: "rel", 2: "exec", 3: "dyn", 4: "core"}


def elf_info(data: bytes) -> dict | None:
    """Class, endianness, machine and type of an ELF image. `None` otherwise."""
    if len(data) < 20 or data[:4] != b"\x7fELF":
        return None
    ei_class, ei_data = data[4], data[5]
    endian = "<" if ei_data == 1 else ">"
    e_type, e_machine = struct.unpack_from(endian + "HH", data, 16)
    return {
        "format": "elf",
        "bits": {1: 32, 2: 64}.get(ei_class, ei_class),
        "endian": {1: "little", 2: "big"}.get(ei_data, ei_data),
        "machine": ELF_MACHINES.get(e_machine, f"machine{e_machine:#x}"),
        "type": ELF_TYPES.get(e_type, f"type{e_type}"),
    }


# Section types from <elf.h>. `SHT_GNU_verneed` is the one with no Mach-O
# counterpart: it records, per needed library, which *symbol versions* the image
# requires. For a glibc target the highest `GLIBC_x.y` in it is what auditwheel
# calls the policy floor, and it is the only place that number exists -- CPython's
# `_sysconfigdata_*.py` has no field for it (docs/LINUX.md §3), unlike
# `ANDROID_API_LEVEL` and `IPHONEOS_DEPLOYMENT_TARGET`.
SHT_DYNSYM = 11
SHT_DYNAMIC = 6
SHT_GNU_VERNEED = 0x6FFFFFFE
SHT_GNU_VERSYM = 0x6FFFFFFF

DT_NULL, DT_NEEDED, DT_SONAME = 0, 1, 14

_EHDR64_SHOFF = 0x28
_EHDR64_SHENTSIZE = 0x3A
_SHDR64_SIZE = 64
_DYN64_SIZE = 16
_VERNEED64_SIZE = 16
_VERNAUX64_SIZE = 16


def _elf_sections(data: bytes):
    """`(endian, [section], cstr)` for a 64-bit ELF, or `None`.

    Section headers rather than `PT_DYNAMIC`, because they need no
    address-to-offset mapping and every `.so` a compiler emits has them. An image
    that has been stripped of them answers `None`, which callers report as "could
    not be read" -- never as "requires nothing", which is the direction that
    would turn a missing section into a passing check.
    """
    if len(data) < 0x40 or data[:4] != b"\x7fELF" or data[4] != 2:
        return None
    endian = "<" if data[5] == 1 else ">"
    try:
        shoff, = struct.unpack_from(endian + "Q", data, _EHDR64_SHOFF)
        shentsize, shnum, shstrndx = struct.unpack_from(
            endian + "HHH", data, _EHDR64_SHENTSIZE)
    except struct.error:
        return None
    if not shoff or not shnum or shentsize < _SHDR64_SIZE:
        return None
    if shoff + shnum * shentsize > len(data) or shstrndx >= shnum:
        return None

    sections = []
    for i in range(shnum):
        try:
            name, stype, _flags, addr, off, size, link, _info, _al, entsize = \
                struct.unpack_from(endian + "IIQQQQIIQQ", data,
                                   shoff + i * shentsize)
        except struct.error:
            return None
        sections.append({"name": name, "type": stype, "addr": addr, "off": off,
                         "size": size, "link": link, "entsize": entsize})

    def cstr(base: int, offset: int) -> str:
        start = base + offset
        if not 0 <= start < len(data):
            return ""
        end = data.find(b"\0", start)
        if end < 0:
            end = len(data)
        return data[start:end].decode("utf-8", "replace")

    shstr = sections[shstrndx]["off"]
    for section in sections:
        section["sname"] = cstr(shstr, section["name"])
    return endian, sections, cstr


def elf_dynamic(data: bytes) -> dict | None:
    """`DT_SONAME`, `DT_NEEDED` and the per-library version requirements.

        {"soname": "libpython3.13.so.1.0",
         "needed": ["libm.so.6", ..., "libc.so.6"],
         "versions": {"libc.so.6": {"GLIBC_2.2.5", ..., "GLIBC_2.17"}, ...}}

    `None` when the bytes are not a readable 64-bit ELF with section headers.
    An ELF that *is* readable but has no dynamic section answers with empty
    fields, which is a different thing and is reported differently by callers:
    the first is the check failing to run, the second is a finding.
    """
    parsed = _elf_sections(data)
    if parsed is None:
        return None
    endian, sections, cstr = parsed

    soname: str | None = None
    needed: list[str] = []
    versions: dict[str, set[str]] = {}
    by_index: dict[int, tuple[str, str]] = {}

    for section in sections:
        if section["type"] == SHT_DYNAMIC:
            if section["link"] >= len(sections):
                continue
            strtab = sections[section["link"]]["off"]
            for i in range(section["size"] // _DYN64_SIZE):
                try:
                    tag, val = struct.unpack_from(
                        endian + "qQ", data, section["off"] + i * _DYN64_SIZE)
                except struct.error:
                    break
                if tag == DT_NULL:
                    break
                if tag == DT_NEEDED:
                    needed.append(cstr(strtab, val))
                elif tag == DT_SONAME:
                    soname = cstr(strtab, val)

        elif section["type"] == SHT_GNU_VERNEED:
            if section["link"] >= len(sections):
                continue
            strtab = sections[section["link"]]["off"]
            off = section["off"]
            end = section["off"] + section["size"]
            seen = 0
            while off + _VERNEED64_SIZE <= end and seen < 4096:
                seen += 1
                try:
                    _v, cnt, vfile, vaux, vnext = struct.unpack_from(
                        endian + "HHIII", data, off)
                except struct.error:
                    break
                library = cstr(strtab, vfile)
                aux = off + vaux
                for _ in range(cnt):
                    if aux + _VERNAUX64_SIZE > end:
                        break
                    try:
                        _h, _fl, _o, name, anext = struct.unpack_from(
                            endian + "IHHII", data, aux)
                    except struct.error:
                        break
                    versions.setdefault(library, set()).add(cstr(strtab, name))
                    # `vna_other` is the index `.gnu.version` uses to point an
                    # undefined symbol at this exact (library, version) pair.
                    # It is the only thing in an ELF that binds a symbol to a
                    # library the way a Mach-O two-level namespace does.
                    by_index[_o & 0x7FFF] = (library, cstr(strtab, name))
                    if not anext:
                        break
                    aux += anext
                if not vnext:
                    break
                off += vnext

    return {"soname": soname, "needed": needed, "versions": versions,
            "version_index": by_index}


def elf_symbols(data: bytes) -> dict | None:
    """`.dynsym`, split into what the image defines and what it needs.

        {"defined":   {"PyList_New", ...},
         "undefined": [("memcpy", "libc.so.6", "GLIBC_2.14", False),
                       ("PyList_New", None, None, False),
                       ("__gmon_start__", None, None, True), ...]}

    The fourth element is `STB_WEAK`. It has to be carried, not filtered here,
    because a weak undefined symbol is *allowed* to stay unresolved -- every
    shared object gcc or clang emits carries `__gmon_start__`,
    `_ITM_registerTMCloneTable` and `_ITM_deregisterTMCloneTable`, none of which
    exists anywhere on an ordinary system. A resolver that counted those as
    failures would report three every time and teach its reader to ignore it.

    The second element of each undefined triple is **the library the symbol is
    bound to, when the ELF says so** -- and the whole difficulty of checking a
    Linux extension is that it usually does not.

    ELF resolves undefined symbols by a flat search across everything loaded, so
    unlike a two-level-namespace Mach-O (`tools/wheel/verify_ios_device.py`) an
    import carries no library name. The exception is symbol *versioning*:
    `.gnu.version` gives each `.dynsym` entry an index, and for an undefined
    symbol that index points into `.gnu.version_r`, which is grouped by library.
    glibc versions all of its exports, so every libc import does name its
    library; CPython versions none of its own, so no `Py*` import does.

    That asymmetry is the honest limit of ELF symbol checking, and it is why the
    Linux check is weaker than the iOS one. `None` means unversioned, not
    unbound.

    Returns `None` when the bytes are not a readable 64-bit ELF with section
    headers -- the same "could not run" answer `elf_dynamic` gives.
    """
    parsed = _elf_sections(data)
    if parsed is None:
        return None
    endian, sections, cstr = parsed

    dynamic = elf_dynamic(data) or {"version_index": {}}
    by_index = dynamic["version_index"]

    versym: list[int] = []
    for section in sections:
        if section["type"] == SHT_GNU_VERSYM and section["entsize"] == 2:
            count = section["size"] // 2
            versym = list(struct.unpack_from(
                endian + f"{count}H", data, section["off"]))
            break

    defined: set[str] = set()
    undefined: list[tuple[str, str | None, str | None, bool]] = []
    for section in sections:
        if section["type"] != SHT_DYNSYM or section["entsize"] != 24:
            continue
        if section["link"] >= len(sections):
            continue
        strtab = sections[section["link"]]["off"]
        for i in range(section["size"] // 24):
            try:
                name, info, _other, shndx, _value, _size = struct.unpack_from(
                    endian + "IBBHQQ", data, section["off"] + i * 24)
            except struct.error:
                break
            symbol = cstr(strtab, name)
            if not symbol:
                continue
            binding = info >> 4
            if shndx == 0:                        # SHN_UNDEF
                index = versym[i] & 0x7FFF if i < len(versym) else 1
                library, version = by_index.get(index, (None, None))
                undefined.append((symbol, library, version, binding == 2))
            elif binding in (1, 2):               # STB_GLOBAL, STB_WEAK
                defined.add(symbol)
    return {"defined": defined, "undefined": undefined}


def describe(data: bytes) -> str:
    """One line, for printing next to a filename."""
    macho = macho_info(data)
    if macho:
        bits = [f"Mach-O {macho['arch']}"]
        if macho["platform"]:
            minos = ".".join(str(n) for n in (macho["minos"] or ())[:2])
            bits.append(f"{macho['platform']} {minos}+")
        return " ".join(bits)
    elf = elf_info(data)
    if elf:
        return (f"ELF {elf['bits']}-bit {elf['endian']}-endian "
                f"{elf['machine']} {elf['type']}")
    if macho_arches(data):
        return "Mach-O fat: " + "+".join(macho_arches(data))
    return "not a recognised binary"
