#!/usr/bin/env python3
"""Inspect a cross-built wheel. This is not the same judgement as verify.py.

    python tools/wheel/verify_cross.py dist/torchnative-*-android_21_arm64_v8a.whl
    python tools/wheel/verify_cross.py dist/torchnative-*-ios_12_0_arm64_iphoneos.whl

`tools/wheel/verify.py` proves a host wheel by installing it into a clean venv
and watching `torch.__file__` come out of that venv. Nothing here can do that:
there is no interpreter on this machine that an Android or iOS wheel is for.

So this checks the two things that *are* decidable from the artefact, and says
plainly that they are not the same claim:

  the tag is one an installer will match      PEP 738 / PEP 730 spelling, run
                                              through `packaging.tags`, which is
                                              the code pip itself uses
  the contents are for that platform          every Mach-O and ELF member read
                                              directly -- architecture, and for
                                              Apple the `LC_BUILD_VERSION`
                                              platform, which is the only thing
                                              distinguishing an iphoneos build
                                              from an iphonesimulator one
  the pieces `import torch` reaches for        `torch/_C.abi3.so` under a suffix
                                              this target's interpreter actually
                                              searches, the global-deps library
                                              under the name
                                              `_load_global_deps` computes, and
                                              the wall-4 marker
  the archive is internally consistent        RECORD hashes every member, which
                                              is what pip verifies on install

What it does NOT establish: that the extension loads, that `import torch`
completes, or that any kernel computes. For Android that gap is closed
separately by `tools/wheel/verify_android.py`, which runs on a device. For iOS
it is open -- see docs/WHEEL.md §7.

Exit 0 means every check above passed. Any failure prints `FAIL:` lines and
exits 1.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binfmt import describe, elf_info, macho_info  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Where the cross-compiled CPython distributions live, so that the checks can be
# made against the interpreter the wheel is *for* rather than against a table
# written down here. Same default as tools/wheel/build.py.
import os  # noqa: E402

TARGET_PYTHON_ROOT = Path(os.environ.get(
    "TORCHNATIVE_TARGET_PYTHON", "/Volumes/macMini/caches/target-python"))


class Expectation:
    """What a given platform tag implies about the archive behind it."""

    def __init__(self, plat: str):
        self.plat = plat
        self.family: str
        self.interpreters: list[Path] = []

    # -- construction --------------------------------------------------------

    @staticmethod
    def parse(plat: str) -> "Expectation":
        if plat.startswith("android_"):
            return AndroidExpectation(plat)
        if plat.startswith("ios_"):
            return IOSExpectation(plat)
        raise SystemExit(
            f"{plat!r} is neither a PEP 738 android_* nor a PEP 730 ios_* tag. "
            "For a host wheel use tools/wheel/verify.py, which actually "
            "installs it."
        )

    # -- checks --------------------------------------------------------------

    def check_tag(self, problems: list[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        raise NotImplementedError

    def global_deps_name(self) -> str:
        # `_load_global_deps()` builds the filename with
        #     ".dylib" if platform.system() == "Darwin" else ".so"
        # and `platform.system()` is "Android" / "iOS" on these targets -- so
        # both want `.so`, including iOS where the file is a Mach-O dylib.
        return "libtorch_global_deps.so"


class AndroidExpectation(Expectation):
    family = "android"

    def __init__(self, plat: str):
        super().__init__(plat)
        m = re.match(r"^android_(\d+)_(.+)$", plat)
        if not m:
            raise SystemExit(f"malformed android tag {plat!r}")
        self.api = int(m.group(1))
        self.abi = m.group(2)
        self.interpreters = sorted(
            (TARGET_PYTHON_ROOT / "aarch64-linux-android" / "prefix" / "lib")
            .glob("libpython3.*.so"))

    def check_tag(self, problems: list[str]) -> None:
        _packaging_accepts(self, problems, api_level=self.api, abi=self.abi)
        if self.api < 16:
            problems.append(
                f"API level {self.api} is below 16, which packaging treats as "
                "the floor for a CPython-capable Android; no installer would "
                "generate a matching tag")

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        info = elf_info(data)
        if info is None:
            problems.append(f"{name} is not an ELF image: {describe(data)}")
            return
        want = {"arm64_v8a": "aarch64", "armeabi_v7a": "arm",
                "x86_64": "x86_64", "x86": "i386"}.get(self.abi)
        if want is None:
            problems.append(f"no known machine for Android ABI {self.abi!r}")
        elif info["machine"] != want:
            problems.append(
                f"{name} is {info['machine']}, but the tag says {self.abi} "
                f"(= {want})")
        if info["type"] != "dyn":
            problems.append(f"{name} is ELF type {info['type']}, expected dyn")


class IOSExpectation(Expectation):
    family = "ios"

    def __init__(self, plat: str):
        super().__init__(plat)
        m = re.match(r"^ios_(\d+)_(\d+)_(.+)$", plat)
        if not m:
            raise SystemExit(f"malformed ios tag {plat!r}")
        self.version = (int(m.group(1)), int(m.group(2)))
        self.multiarch = m.group(3)
        arch, _, sdk = self.multiarch.partition("_")
        self.arch = arch
        # The Mach-O platform id the SDK half of the multiarch implies. This is
        # the pair that nothing else distinguishes: an `iphoneos` and an
        # `iphonesimulator` build are the same architecture, the same size, and
        # the same symbols.
        self.platform_id = {"iphoneos": "ios",
                            "iphonesimulator": "iossimulator"}.get(sdk)
        if self.platform_id is None:
            raise SystemExit(f"unknown iOS sdk {sdk!r} in tag {plat!r}")
        subdir = {"ios": "arm64-iphoneos",
                  "iossimulator": "arm64-iphonesimulator"}[self.platform_id]
        framework = TARGET_PYTHON_ROOT / subdir / "Python.framework" / "Python"
        self.interpreters = [framework] if framework.exists() else []

    def check_tag(self, problems: list[str]) -> None:
        _packaging_accepts(self, problems, version=self.version,
                           multiarch=self.multiarch)

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        info = macho_info(data)
        if info is None:
            problems.append(
                f"{name} is not a thin 64-bit Mach-O: {describe(data)}")
            return
        if info["arch"] != self.arch:
            problems.append(
                f"{name} is {info['arch']}, but the tag says {self.arch}")
        if info["platform"] != self.platform_id:
            problems.append(
                f"{name} is built for {info['platform']!r}, but the tag says "
                f"{self.platform_id!r} -- these differ in nothing else, and the "
                "wrong one fails only on the real device")
        minos = (info["minos"] or (0, 0))[:2]
        if minos > self.version:
            problems.append(
                f"{name} needs iOS {minos[0]}.{minos[1]}, above the tag's "
                f"{self.version[0]}.{self.version[1]} -- the wheel claims an "
                "OS that cannot load it")
        if info["id"] and info["id"].startswith("/"):
            problems.append(
                f"{name} advertises the build machine's path as its install "
                f"name: {info['id']}")
        if name.endswith("_C.abi3.so") and self.platform_id == "ios":
            if not any("Python.framework" in d for d in info["dylibs"]):
                problems.append(
                    f"{name} does not link Python.framework; on a physical "
                    "device there is no libpython to resolve against "
                    f"(LC_LOAD_DYLIB: {info['dylibs']})")


def _packaging_accepts(exp: Expectation, problems: list[str], **kwargs) -> None:
    """Would pip's own tag generator produce this platform tag?

    `packaging.tags` is what pip uses to decide whether a wheel is for the
    machine it is running on, so asking it is the difference between checking
    the spelling against the specification and checking it against the
    implementation. It is not a substitute for the content checks: it looks at
    the string only.
    """
    try:
        from packaging import tags as ptags
    except ImportError:
        problems.append("packaging is not importable -- the tag was not checked "
                        "against the code pip uses")
        return
    generator = getattr(ptags, f"{exp.family}_platforms", None)
    if generator is None:
        problems.append(
            f"packaging in this environment has no {exp.family}_platforms; it "
            "is older than PEP 738/730 support and cannot check this tag")
        return
    accepted = list(generator(**kwargs))
    if exp.plat not in accepted:
        problems.append(
            f"packaging.tags.{exp.family}_platforms({kwargs}) does not yield "
            f"{exp.plat!r}; it starts {accepted[:3]}")
    else:
        print(f"  tag                 {exp.plat}  "
              f"(accepted by packaging.tags.{exp.family}_platforms)")


def _is_binary(data: bytes) -> bool:
    return macho_info(data) is not None or elf_info(data) is not None


def check_record(zf: zipfile.ZipFile, dist_info: str,
                 problems: list[str]) -> None:
    """Every member hashed, every hash right.

    pip verifies RECORD on install, so a repack that adds a file without
    updating it produces a wheel that fails at the moment it looks like it
    worked. Checking it here means that failure mode cannot reach a device.
    """
    record_name = f"{dist_info}/RECORD"
    if record_name not in zf.namelist():
        problems.append(f"no {record_name}")
        return
    listed: dict[str, str] = {}
    for row in csv.reader(io.StringIO(zf.read(record_name).decode())):
        if row:
            listed[row[0]] = row[1] if len(row) > 1 else ""
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if name not in listed:
            problems.append(f"{name} is in the archive but not in RECORD")
            continue
        if name == record_name:
            continue
        want = listed[name]
        got = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(zf.read(name)).digest()).rstrip(b"=").decode()
        if want != got:
            problems.append(f"{name}: RECORD says {want}, content is {got}")
    for name in listed:
        if name not in zf.namelist():
            problems.append(f"{name} is in RECORD but not in the archive")


def check_suffix_is_searched(exp: Expectation, problems: list[str]) -> None:
    """Will the target interpreter even look for `_C.abi3.so`?

    CPython's `_PyImport_DynLoadFiletab` (dynload_shlib.c) is
    `{SOABI suffix, ".abi3.so", SHLIB_SUFFIX}`, and those are string constants
    compiled into the interpreter. Reading them out of the target binary is the
    strongest form this question takes without a device: if `.abi3.so` were not
    in that table the extension would simply never be found, and the failure
    would be a `ModuleNotFoundError` with nothing pointing here.

    (For Android this is also directly measured -- `scripts/device_android.sh`
    records the list from the device as
    `['.cpython-313-aarch64-linux-android.so', '.abi3.so', '.so']`.)
    """
    if not exp.interpreters:
        problems.append(
            "no target CPython found to check the extension suffix against "
            f"(looked under {TARGET_PYTHON_ROOT}); set TORCHNATIVE_TARGET_PYTHON")
        return
    for path in exp.interpreters:
        blob = path.read_bytes()
        if b".abi3.so\x00" not in blob:
            problems.append(
                f"{path} does not contain the extension suffix '.abi3.so' -- "
                "this interpreter would never find torch/_C.abi3.so")
        else:
            print(f"  ext suffix          .abi3.so present in {path.name}")


# ------------------------------------------------------------------ self-test
#
# "실패할 수 없는 검증은 검증이 아니다" (CLAUDE.md §5.5). Everything above passes
# on the wheels this repository builds, which says nothing on its own -- an empty
# function passes too. So each check is given a wheel it must reject, built by
# damaging a good one in exactly the way that check exists to notice.
#
# The damage is done by patching header fields rather than by substituting some
# other artefact, so the self-test needs nothing on disk beyond the wheel itself:
# every fault mode runs every time, and none of them can be quietly skipped.


def _patch(data: bytes, offset: int, raw: bytes) -> bytes:
    return data[:offset] + raw + data[offset + len(raw):]


def _wrong_elf_machine(data: bytes) -> bytes:
    """e_machine := EM_X86_64. An aarch64 tag over an x86_64 ELF."""
    import struct as _s
    return _patch(data, 18, _s.pack("<H", 0x3E))


def _wrong_macho_platform(data: bytes) -> bytes:
    """Retarget the image at a platform the tag does not claim.

    With `LC_BUILD_VERSION`, flip its platform field between ios (2) and
    iossimulator (7) -- the one difference between the device and the simulator
    artefact that no size, architecture or symbol check would see.

    The iOS *device* build has no such command: Rust's default deployment
    target for `aarch64-apple-ios` is 10.0, which predates it, so the image
    carries `LC_VERSION_MIN_IPHONEOS` instead. There the equivalent damage is to
    make it `LC_VERSION_MIN_MACOSX`, i.e. a macOS dylib wearing an iOS tag.
    """
    import struct as _s
    from binfmt import LC_BUILD_VERSION
    ncmds = _s.unpack_from("<I", data, 16)[0]
    off = 32
    for _ in range(ncmds):
        cmd, cmdsize = _s.unpack_from("<II", data, off)
        if cmd == LC_BUILD_VERSION:
            plat = _s.unpack_from("<I", data, off + 8)[0]
            return _patch(data, off + 8, _s.pack("<I", 7 if plat == 2 else 2))
        if cmd == 0x25:  # LC_VERSION_MIN_IPHONEOS -> LC_VERSION_MIN_MACOSX
            return _patch(data, off, _s.pack("<I", 0x24))
        off += cmdsize
    raise SystemExit("no version-min load command to corrupt -- "
                     "self-test cannot run")


def _rewrite(src: Path, dst: Path, *, drop=(), replace=None, add=None,
             rename=None, wheel_tag=None) -> Path:
    """Copy a wheel, dropping/replacing/adding members. RECORD is deliberately
    *not* regenerated -- for the faults that change content this is itself one
    of the things being tested."""
    replace = replace or {}
    out = dst / (rename or src.name)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(
            out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in drop:
                continue
            data = zin.read(item.filename)
            if item.filename in replace:
                data = replace[item.filename](data)
            if wheel_tag and item.filename.endswith(".dist-info/WHEEL"):
                data = re.sub(rb"Tag: .*", b"Tag: " + wheel_tag.encode(), data)
            zout.writestr(item, data)
        for arcname, source in (add or {}).items():
            zout.writestr(arcname, zipfile.ZipFile(src).read(source))
    return out


def self_test(wheel: Path, reference: Path | None) -> int:
    import subprocess
    import tempfile

    with zipfile.ZipFile(wheel) as zf:
        ext_data = zf.read("torch/_C.abi3.so")
    is_macho = macho_info(ext_data) is not None
    stem = wheel.stem
    name, version, plat = stem.split("-")[0], stem.split("-")[1], stem.split("-")[-1]
    dist_info = f"{name}-{version}.dist-info"
    deps = "torch/lib/libtorch_global_deps.so"

    # Each entry: what is broken, how, and the words the report must contain.
    # Matching on the message rather than only on the exit code is what keeps a
    # fault from being "caught" by an unrelated check -- the file-list
    # comparison would otherwise absorb almost all of these.
    faults: list[tuple[str, dict, str]] = [
        ("extension built for the wrong platform",
         {"replace": {"torch/_C.abi3.so":
                      _wrong_macho_platform if is_macho else _wrong_elf_machine}},
         "built for" if is_macho else "but the tag says"),
        ("global-deps library missing",
         {"drop": (deps,)},
         "_load_global_deps"),
        ("global-deps library under the host's name",
         {"drop": (deps,),
          "add": {"torch/lib/libtorch_global_deps.dylib": deps}},
         "beside"),
        ("wall-4 marker missing",
         {"drop": ("torch/bin/torch_shm_manager",)},
         "torch_shm_manager"),
        ("a member edited without updating RECORD",
         {"replace": {"torch/version.py": lambda d: d + b"\n# tampered\n"}},
         "RECORD says"),
        ("part of the vendored tree dropped",
         {"drop": tuple(_some_tree_files(wheel))},
         "are missing here"),
        ("WHEEL Tag: out of step with the filename",
         {"wheel_tag": f"cp313-abi3-{plat}-BOGUS"},
         "declares Tag:"),
        ("platform tag no installer would generate",
         {"rename": stem.replace(plat, _impossible(plat)) + ".whl"},
         "does not yield"),
        ("abi tag downgraded from abi3",
         {"rename": stem.replace("-abi3-", "-cp313-") + ".whl"},
         "not 'abi3'"),
    ]

    print(f"SELF-TEST against {wheel.name}")
    print(f"  {len(faults)} fault modes; each must be reported, with the "
          "expected reason\n")
    caught = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for label, kwargs, expect in faults:
            broken = _rewrite(wheel, tmp, **kwargs)
            argv = [sys.executable, __file__, str(broken)]
            if reference:
                argv += ["--reference", str(reference)]
            proc = subprocess.run(argv, capture_output=True, text=True)
            report = proc.stdout + proc.stderr
            if proc.returncode == 0:
                print(f"  NOT CAUGHT  {label}")
            elif expect not in report:
                print(f"  WRONG REASON {label}")
                print(f"      wanted {expect!r}, got:")
                for line in report.splitlines():
                    if line.startswith("FAIL:"):
                        print(f"        {line}")
            else:
                caught += 1
                print(f"  caught      {label}")
            broken.unlink()

    print()
    if caught != len(faults):
        print(f"SELF-TEST: FAIL -- {len(faults) - caught} of {len(faults)} fault "
              "modes were not reported for the right reason", file=sys.stderr)
        return 1
    print(f"SELF-TEST: PASS -- {caught}/{len(faults)} fault modes rejected")
    return 0


def _some_tree_files(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return [n for n in zf.namelist()
                if n.startswith("torchgen/") and n.endswith(".py")][:20]


def _impossible(plat: str) -> str:
    """A same-shaped tag that `packaging.tags` will never yield.

    Below each family's own floor -- API 15 for Android, iOS 11 -- so the
    rejection comes from packaging's generator and not from a spelling rule
    invented here.
    """
    if plat.startswith("android_"):
        return re.sub(r"^android_\d+", "android_15", plat)
    return re.sub(r"^ios_\d+_\d+", "ios_11_0", plat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--reference", type=Path, default=None,
                    help="a wheel whose file list this one must match, to catch "
                         "a cross wheel that quietly lost part of the tree "
                         "(default: the newest macosx wheel beside it)")
    ap.add_argument("--self-test", action="store_true",
                    help="damage this wheel nine ways and check that each is "
                         "reported, with the right reason")
    args = ap.parse_args()

    if not args.wheel.exists():
        sys.exit(f"no such wheel: {args.wheel}")

    if args.self_test:
        reference = args.reference
        if reference is None:
            found = sorted(args.wheel.parent.glob("*macosx*.whl"))
            reference = found[-1] if found else None
        sys.exit(self_test(args.wheel, reference))

    parts = args.wheel.stem.split("-")
    if len(parts) < 5:
        sys.exit(f"{args.wheel.name} is not a PEP 427 wheel filename")
    name, version, pytag, abitag, plat = parts[0], parts[1], *parts[-3:]
    exp = Expectation.parse(plat)
    problems: list[str] = []

    print(f"{args.wheel.name}")
    print(f"  distribution        {name} {version}")
    print(f"  python / abi        {pytag} / {abitag}")
    if abitag != "abi3":
        problems.append(
            f"abi tag is {abitag!r}, not 'abi3' -- the wheel then serves one "
            "CPython version instead of 3.13 and every later one")
    exp.check_tag(problems)

    with zipfile.ZipFile(args.wheel) as zf:
        names = zf.namelist()
        dist_info = f"{name}-{version}.dist-info"

        # 1. WHEEL's Tag: and the filename have to agree. Installers read the
        #    filename; `wheel unpack`, auditors and `twine` read WHEEL.
        wheel_meta = zf.read(f"{dist_info}/WHEEL").decode()
        declared = [line.split(": ", 1)[1] for line in wheel_meta.splitlines()
                    if line.startswith("Tag: ")]
        want_tag = f"{pytag}-{abitag}-{plat}"
        if declared != [want_tag]:
            problems.append(
                f"{dist_info}/WHEEL declares Tag: {declared}, filename says "
                f"{want_tag}")
        else:
            print(f"  WHEEL Tag:          {want_tag}")
        if "Root-Is-Purelib: false" not in wheel_meta:
            problems.append(
                "WHEEL says Root-Is-Purelib: true -- a wheel carrying an "
                "extension unpacks into purelib and the extension lands in the "
                "wrong directory")

        # 2. Every binary in the archive is for the tagged platform. Walking all
        #    of them rather than a named list: the point of the check is to
        #    catch a member nobody thought about, and a list can only contain
        #    the ones somebody did.
        binaries = []
        for member in names:
            if member.endswith("/"):
                continue
            head = zf.read(member)
            if not _is_binary(head):
                continue
            binaries.append(member)
            exp.check_binary(member, head, problems)
        print(f"  binaries            {len(binaries)}")
        for member in binaries:
            print(f"                      {member}  "
                  f"{describe(zf.read(member))}")
        if not binaries:
            problems.append(
                "no binary members at all -- this is the py3-none-any shell "
                "again, in a platform-tagged wrapper")

        # 3. The three files `import torch` reaches for by name.
        ext = "torch/_C.abi3.so"
        if ext not in names:
            problems.append(f"no {ext}")
        deps = f"torch/lib/{exp.global_deps_name()}"
        if deps not in names:
            problems.append(
                f"no {deps} -- `_load_global_deps()` computes exactly this name "
                "on this platform, and without it `import torch` needs "
                "TORCH_USE_RTLD_GLOBAL=1 (docs/VENDOR.md wall 1)")
        strays = [n for n in names
                  if n.startswith("torch/lib/libtorch_global_deps") and n != deps]
        if strays:
            problems.append(f"{strays} beside {deps}; only one is looked for")
        if "torch/bin/torch_shm_manager" not in names:
            problems.append(
                "no torch/bin/torch_shm_manager -- `_manager_path()` checks it "
                "exists on every non-Windows platform (docs/VENDOR.md wall 4)")

        # 4. The archive agrees with itself.
        check_record(zf, dist_info, problems)

        # 5. The interpreter this is for would look for this filename.
        check_suffix_is_searched(exp, problems)

        # 6. Nothing was lost relative to a wheel that is known to work. The
        #    cross path swaps two members and adds one; anything else differing
        #    means the tree behind it changed between builds.
        reference = args.reference
        if reference is None:
            candidates = sorted(args.wheel.parent.glob(
                f"{name}-{version}-*macosx*.whl"))
            reference = candidates[-1] if candidates else None
        if reference is None:
            problems.append(
                "no reference wheel to compare the file list against; build the "
                "host wheel first or pass --reference")
        else:
            with zipfile.ZipFile(reference) as ref:
                ref_names = set(ref.namelist())
            here = set(names)
            # The global-deps filename differs by design (.dylib vs .so), so it
            # is excluded from the comparison on both sides rather than
            # special-cased into an allowance that would also hide a loss.
            def strip(s):
                return {n for n in s
                        if not n.startswith("torch/lib/libtorch_global_deps")
                        and not n.endswith("/RECORD")}
            lost = sorted(strip(ref_names) - strip(here))
            gained = sorted(strip(here) - strip(ref_names))
            if lost:
                problems.append(
                    f"{len(lost)} file(s) in {reference.name} are missing here, "
                    f"e.g. {lost[:5]}")
            if gained:
                problems.append(
                    f"{len(gained)} file(s) here are not in {reference.name}, "
                    f"e.g. {gained[:5]}")
            if not lost and not gained:
                print(f"  file list           identical to {reference.name} "
                      f"({len(strip(here)):,} entries)")

    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"PASS -- {args.wheel.name} is tagged for a platform an installer "
          "will match, and holds binaries for it")
    print("       NOT established here: that it loads, imports, or computes. "
          "See docs/WHEEL.md §7.")


if __name__ == "__main__":
    main()
