#!/usr/bin/env python3
"""Build a platform wheel that actually contains a torch you can import.

    python tools/wheel/build.py

The distribution on PyPI as `torchnative 0.0.1a0` is `py3-none-any` and holds
only the `torchnative/` skeleton: no `_C`, no vendored tree. `pip install
torchnative` gets you something where `import torch` raises ImportError. This
script exists so that the *default* way to build produces the other thing.

What it does beyond `pip wheel .`:

  preflight   Refuse to build unless `vendor/vendor_torch.sh` has laid down the
              upstream Python tree and `vendor/install_shim.sh` has put our `_C`
              in the hole. Both are gitignored, so a fresh clone has neither,
              and setuptools would happily emit the empty shell again -- which
              is how the PyPI one came to exist. The failure has to be loud.

  repack      Two fixes that have no setuptools spelling, applied to the
              finished archive (§ `_repack`):
                - `torch-<v>.dist-info/` from upstream, so that
                  `importlib.metadata.version("torch")` answers. transformers
                  gates the entire torch integration on that call
                  (docs/VENDOR.md), so it is load-bearing, not decoration.
                - the Mach-O install name of `_C.abi3.so`, which cargo sets to
                  the absolute path of the build machine's target directory.

  verify      Diff the archive against the source tree file-by-file and fail on
              anything missing. A wheel that is merely *smaller* than it should
              be installs fine and breaks later, at some import nobody ran.

Then check it for real -- building is not the proof:

    python tools/wheel/verify.py dist/<wheel>
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "torchnative" / "src" / "main"

# The two artefacts the vendor scripts produce. Their absence is the whole
# failure mode this script exists to catch, so they are named individually
# rather than checked as "does src/main look populated".
VENDORED_ROOT = SRC / "torch" / "__init__.py"
SHIM = SRC / "torch" / "_C.abi3.so"
STAMP = SRC / ".stamp"

SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_DIRS = {"__pycache__"}


def _fail(msg: str) -> None:
    sys.exit(f"tools/wheel/build.py: {msg}")


def preflight() -> dict[str, str]:
    """Refuse to build a wheel that would be missing its reason to exist."""
    if not VENDORED_ROOT.exists():
        _fail(
            f"no vendored torch tree at {VENDORED_ROOT.parent}\n"
            "  run: bash vendor/vendor_torch.sh\n"
            "  (the tree is gitignored -- a fresh clone never has it, and\n"
            "   building without it yields the py3-none-any shell on PyPI)"
        )
    if not SHIM.exists():
        _fail(
            f"no _C extension at {SHIM}\n"
            "  run: bash vendor/install_shim.sh\n"
            "  (without it the tree cannot import; that is by design --\n"
            "   vendor_torch.sh drops upstream's _C so the hole is visible)"
        )

    stamp: dict[str, str] = {}
    if STAMP.exists():
        for line in STAMP.read_text().splitlines():
            key, _, value = line.partition("=")
            stamp[key] = value

    # `native_left=0` is vendor_torch.sh's own measurement that it removed every
    # compiled artefact from the tree. If it is not 0, the tree still carries
    # upstream binaries and the wheel would ship somebody else's `_C` alongside
    # ours -- with upstream's `.cpython-313-darwin.so` suffix winning the
    # importlib race, so ours would be dead weight and nothing would say so.
    if stamp.get("native_left") not in (None, "0"):
        _fail(
            f"vendored tree still holds {stamp['native_left']} native artefact(s)"
            " -- re-run vendor/vendor_torch.sh"
        )
    return stamp


def tree_files(root: Path) -> set[str]:
    """Every file the wheel is expected to carry, as archive-relative paths."""
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(SKIP_SUFFIXES) or name == ".DS_Store":
                continue
            out.add(str((Path(dirpath) / name).relative_to(SRC)))
    return out


def run_pip_wheel(python: str, outdir: Path) -> Path:
    """Drive the PEP 517 build through pip.

    pip rather than `python -m build` because pip is in every virtualenv and
    `build` is not; `--no-build-isolation` because the build needs nothing that
    is not already installed, and isolation would download setuptools on every
    invocation. `--no-deps` because this distribution's dependency list is empty
    and resolving it would reach the network for no reason.
    """
    before = set(outdir.glob("*.whl")) if outdir.exists() else set()
    cmd = [
        python, "-m", "pip", "wheel", str(REPO),
        "--no-deps", "--no-build-isolation", "--wheel-dir", str(outdir),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    made = sorted(set(outdir.glob("*.whl")) - before)
    if len(made) != 1:
        # Rebuilding without a version bump overwrites in place, so an empty
        # diff is normal; more than one is not.
        made = sorted(outdir.glob("torchnative-*.whl"))
        if len(made) != 1:
            _fail(f"expected exactly one torchnative wheel in {outdir}, found {made}")
    return made[0]


def _record_line(arcname: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return arcname, "sha256=" + digest.decode("ascii"), str(len(data))


def _fix_install_name(data: bytes) -> bytes:
    """Strip the build machine's absolute path out of the Mach-O install name.

    cargo links a `cdylib` with `-install_name <CARGO_TARGET_DIR>/release/deps/
    lib_C.dylib`, so a wheel built here announces
    `/Volumes/macMini/caches/cargo-target-wheel/...` to anyone who runs `otool
    -L` on it. It is inert -- `dlopen()` of a module by path ignores
    `LC_ID_DYLIB`, which is why the extension imports anyway -- but a published
    artefact should not name a directory on the machine that built it.

    Doing it here rather than in `vendor/install_shim.sh` keeps the file that
    the test suites load byte-identical to the one cargo emitted, so a green
    suite is evidence about the artefact and not about this rewrite.
    """
    if sys.platform != "darwin" or not shutil.which("install_name_tool"):
        return data
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_C.abi3.so"
        path.write_bytes(data)
        subprocess.run(
            ["install_name_tool", "-id", "@rpath/_C.abi3.so", str(path)],
            check=True, capture_output=True,
        )
        # codesign: install_name_tool invalidates the ad-hoc signature that the
        # linker put on the arm64 slice, and macOS refuses to load an arm64
        # Mach-O with a *broken* signature (an unsigned one is fine, a stale one
        # is not). Re-sign ad-hoc. If codesign is missing, better to ship the
        # untouched original than an unloadable one.
        if shutil.which("codesign"):
            r = subprocess.run(
                ["codesign", "--force", "--sign", "-", str(path)],
                capture_output=True,
            )
            if r.returncode != 0:
                print("  ! codesign failed, keeping original install name",
                      r.stderr.decode(errors="replace").strip())
                return data
        else:
            print("  ! no codesign, keeping original install name")
            return data
        return path.read_bytes()


# Mach-O, enough of it to answer "which CPUs does this run on". cputype values
# from <mach/machine.h>; the 0x01000000 bit is CPU_ARCH_ABI64.
_FAT_MAGICS = {0xCAFEBABE, 0xCAFEBABF}
_THIN_MAGICS = {0xFEEDFACE, 0xFEEDFACF}
_CPU_NAMES = {0x0100000C: "arm64", 0x01000007: "x86_64", 0x00000007: "i386"}


def macho_arches(data: bytes) -> list[str]:
    """Architectures actually present in a Mach-O image."""
    if len(data) < 8:
        return []
    magic_be = struct.unpack_from(">I", data)[0]
    if magic_be in _FAT_MAGICS:
        # `fat_arch` is 20 bytes, `fat_arch_64` (magic ...BF) is 32; both start
        # with cputype, which is the only field wanted here.
        stride = 20 if magic_be == 0xCAFEBABE else 32
        count = struct.unpack_from(">I", data, 4)[0]
        out = []
        for i in range(count):
            cputype = struct.unpack_from(">I", data, 8 + i * stride)[0]
            out.append(_CPU_NAMES.get(cputype, f"cpu{cputype:#x}"))
        return out
    for endian in ("<", ">"):
        magic = struct.unpack_from(endian + "I", data)[0]
        if magic in _THIN_MAGICS:
            cputype = struct.unpack_from(endian + "I", data, 4)[0]
            return [_CPU_NAMES.get(cputype, f"cpu{cputype:#x}")]
    return []


def honest_macos_plat(plat: str, so_data: bytes) -> str:
    """Correct the architecture half of a macOS platform tag.

    setuptools derives the platform tag from `sysconfig.get_platform()` and then
    lets `calculate_macosx_platform_tag` walk the archive -- but that walk only
    ever raises the *deployment target*; it never touches the architecture. So a
    wheel built by the python.org universal2 interpreter is tagged
    `macosx_11_0_universal2` no matter what is inside it. Ours holds an arm64
    Mach-O and nothing else.

    The difference is not cosmetic. `universal2` tells pip on an Intel Mac that
    this wheel is for it; the install succeeds and `import torch` then dies in
    `dlopen` with a mach-o architecture mismatch, at which point the user's
    evidence points at their machine rather than at this tag. Narrowing it means
    those users get "no matching distribution", which is the truth.
    """
    m = re.match(r"^(macosx_\d+_\d+)_(.+)$", plat)
    if not m:
        return plat
    arches = set(macho_arches(so_data))
    if not arches:
        return plat
    if arches == {"arm64"}:
        arch = "arm64"
    elif arches == {"x86_64"}:
        arch = "x86_64"
    elif arches == {"arm64", "x86_64"}:
        arch = "universal2"
    else:
        return plat  # something unusual; leave setuptools' answer alone
    return f"{m.group(1)}_{arch}"


def _retag(name: str, plat: str) -> str:
    """Swap the platform component of a wheel filename."""
    stem, _, _ = name.rpartition(".whl")
    parts = stem.split("-")
    parts[-1] = plat
    return "-".join(parts) + ".whl"


def _repack(wheel: Path, extra: dict[str, bytes], dist_info: str) -> Path:
    """Rewrite the archive with `extra` added and RECORD regenerated.

    zipfile cannot delete or replace a member, so the whole archive is rebuilt.
    RECORD has to be regenerated anyway: it is a hash manifest, and pip verifies
    it on install, so appending files without touching it produces a wheel that
    fails at exactly the moment it looks like it worked.
    """
    tmp = wheel.with_suffix(".whl.tmp")
    record_name = f"{dist_info}/RECORD"
    wheel_name = f"{dist_info}/WHEEL"
    rows: list[tuple[str, str, str]] = []
    new_plat: str | None = None

    # The architecture correction has to be known before WHEEL is written, and
    # it comes out of the extension, so read that first.
    with zipfile.ZipFile(wheel) as src:
        so = next((n for n in src.namelist() if n.endswith("torch/_C.abi3.so")), None)
        if so is not None and sys.platform == "darwin":
            so_data = src.read(so)
            old_plat = wheel.stem.split("-")[-1]
            honest = honest_macos_plat(old_plat, so_data)
            if honest != old_plat:
                print(f"  retag: {old_plat} -> {honest} "
                      f"(extension is {'+'.join(macho_arches(so_data))})")
                new_plat = honest

    with zipfile.ZipFile(wheel) as src, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            if item.filename == record_name:
                continue  # regenerated at the end
            data = src.read(item.filename)
            if item.filename.endswith("torch/_C.abi3.so"):
                data = _fix_install_name(data)
                item = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                item.external_attr = 0o755 << 16
                item.compress_type = zipfile.ZIP_DEFLATED
            elif item.filename == wheel_name and new_plat:
                # `Tag:` in WHEEL and the filename have to agree; installers read
                # the filename, `wheel unpack` and auditors read this.
                data = b"".join(
                    (line.rsplit(b"-", 1)[0] + b"-" + new_plat.encode() + b"\n")
                    if line.startswith(b"Tag: ") else line + b"\n"
                    for line in data.splitlines()
                )
            dst.writestr(item, data)
            rows.append(_record_line(item.filename, data))

        for arcname, data in sorted(extra.items()):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, data)
            rows.append(_record_line(arcname, data))

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in sorted(rows):
            writer.writerow(row)
        writer.writerow([record_name, "", ""])
        dst.writestr(record_name, buf.getvalue())

    final = wheel if new_plat is None else wheel.with_name(
        _retag(wheel.name, new_plat)
    )
    tmp.replace(final)
    if final != wheel:
        wheel.unlink()
    return final


# A translation unit with one symbol, so that `nm` on the result says what it is
# instead of showing an empty library and leaving the next reader to guess.
_GLOBAL_DEPS_C = """
/* Generated by tools/wheel/build.py. See docs/WHEEL.md. */
const char torchnative_global_deps_note[] =
    "torchnative: this build has no global native dependencies; "
    "torch/__init__.py:_load_global_deps only needs this file to dlopen.";
"""


def global_deps_stub() -> dict[str, bytes]:
    """An empty `torch/lib/libtorch_global_deps` so `import torch` needs no env.

    Wall 1 in docs/VENDOR.md: `torch/__init__.py:_load_global_deps()` does an
    unconditional `ctypes.CDLL(torch/lib/libtorch_global_deps.<ext>,
    RTLD_GLOBAL)`, and `vendor_torch.sh` drops `torch/lib/` entirely. Every
    invocation in this repository routes around it with
    `TORCH_USE_RTLD_GLOBAL=1`, which upstream provides for exactly this case
    ("a build environment where libtorch_global_deps is not available").

    That is fine for a source tree and unacceptable for a wheel. `pip install
    torchnative && python -c "import torch"` has to work, and an environment
    variable the user has never heard of is not a thing they will set. Worse,
    the wall-1 branch is not free: it also does
    `sys.setdlopenflags(RTLD_GLOBAL | RTLD_LAZY)` for the whole `from torch._C
    import *`, whose device-side effect VENDOR.md §7 item 4 lists as unverified.
    Shipping the file takes the ordinary branch instead.

    Upstream's copy exists to force MKL/OpenMP into the global namespace ahead
    of libtorch. This build links neither -- `_C.abi3.so` is self-contained
    against Accelerate -- so *empty* is not a stub standing in for something
    real, it is the correct content. The file has to be a loadable image
    though, not the zero-byte marker that `install_shim.sh` uses for
    `torch/bin/torch_shm_manager`; `CDLL` actually dlopens this one.

    Built here rather than in `install_shim.sh` so the source tree keeps
    behaving exactly as every existing doc describes -- wall 1 still stands
    there, and the suites that set the variable are still measuring what they
    say they measure.
    """
    import tempfile

    ext = ".dylib" if sys.platform == "darwin" else ".so"
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        _fail(
            "no C compiler (set CC) -- cannot build the empty "
            f"torch/lib/libtorch_global_deps{ext}, without which the installed "
            "wheel cannot `import torch` unless the user sets "
            "TORCH_USE_RTLD_GLOBAL=1 (docs/VENDOR.md wall 1)"
        )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "global_deps.c"
        out = Path(tmp) / f"libtorch_global_deps{ext}"
        src.write_text(_GLOBAL_DEPS_C)
        r = subprocess.run([cc, "-shared", "-fPIC", "-o", str(out), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            _fail(f"{cc} could not build the global-deps stub:\n{r.stderr}")
        print(f"  + torch/lib/libtorch_global_deps{ext} "
              f"({out.stat().st_size:,} B, empty by design -- VENDOR.md wall 1)")
        return {f"torch/lib/libtorch_global_deps{ext}": out.read_bytes()}


def upstream_dist_info(version: str) -> dict[str, bytes]:
    """Upstream's `torch-<v>.dist-info`, addressed into the wheel's purelib.

    Why it cannot simply sit at the archive root: pip resolves a wheel's own
    metadata directory by scanning top-level names ending in `.dist-info` and
    raising `UnsupportedWheel: multiple .dist-info directories found` if there
    is more than one. Routing it through `torchnative-<v>.data/purelib/` puts it
    in site-packages at install time while leaving exactly one at the root.

    Upstream's RECORD is deliberately not copied (vendor_torch.sh drops it too).
    Ours lists these files, so `pip uninstall torchnative` removes them; a
    RECORD of its own would additionally invite `pip uninstall torch` to delete
    the tree out from under us.
    """
    found = sorted(SRC.glob("torch-*.dist-info"))
    if not found:
        print("  ! no upstream torch-*.dist-info in the vendored tree --")
        print("    importlib.metadata.version('torch') will raise, and")
        print("    transformers' is_torch_available() will answer False")
        return {}
    root = found[0]
    prefix = f"torchnative-{version}.data/purelib/{root.name}"
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "RECORD":
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out[f"{prefix}/{path.relative_to(root).as_posix()}"] = path.read_bytes()
    return out


def verify(wheel: Path, expected: set[str]) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    missing = sorted(n for n in expected if n not in names)
    if missing:
        head = "\n".join("    " + m for m in missing[:20])
        _fail(
            f"{len(missing)} file(s) present in {SRC} but absent from the wheel:\n"
            f"{head}\n"
            + ("    ...\n" if len(missing) > 20 else "")
            + "  a wheel that is merely smaller than the tree installs fine and\n"
            "  fails later, at whichever import first needs the missing file"
        )
    if "-none-any" in wheel.name:
        _fail(f"{wheel.name} is tagged py3-none-any -- setup.py's "
              "BinaryDistribution did not take effect")
    if "-abi3-" not in wheel.name:
        _fail(f"{wheel.name} is not abi3-tagged -- the bdist_wheel "
              "py_limited_api option did not take effect")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter whose pip/setuptools drive the build")
    ap.add_argument("--outdir", default=str(REPO / "dist"), type=Path)
    args = ap.parse_args()

    stamp = preflight()
    print(f"vendored torch {stamp.get('version', '?')} "
          f"({stamp.get('py_modules', '?')} modules) + _C.abi3.so "
          f"({SHIM.stat().st_size:,} B)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    wheel = run_pip_wheel(args.python, args.outdir)

    version = wheel.name.split("-")[1]
    extra = {**upstream_dist_info(version), **global_deps_stub()}
    wheel = _repack(wheel, extra, f"torchnative-{version}.dist-info")

    expected: set[str] = set()
    for pkg in ("torch", *stamp.get("packages", "").split(","), "torchnative"):
        if pkg and (SRC / pkg).is_dir():
            expected |= tree_files(SRC / pkg)
    verify(wheel, expected)

    with zipfile.ZipFile(wheel) as zf:
        entries = zf.namelist()
        uncompressed = sum(i.file_size for i in zf.infolist())
    print()
    print(f"{wheel}")
    print(f"  {len(entries):,} entries, "
          f"{wheel.stat().st_size / 1e6:.1f} MB compressed, "
          f"{uncompressed / 1e6:.1f} MB installed")
    print(f"  tag: {'-'.join(wheel.stem.split('-')[2:])}")
    print()
    print(f"next: python tools/wheel/verify.py {wheel}")


if __name__ == "__main__":
    main()
