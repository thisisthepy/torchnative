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
