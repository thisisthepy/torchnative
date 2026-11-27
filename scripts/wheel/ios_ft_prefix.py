#!/usr/bin/env python3
"""Compose the iOS-simulator CPython 3.15t prefix that `build.py --abi ft --target ios-arm64-sim` reads.

    python scripts/wheel/ios_ft_prefix.py <python-3.15.0t-iOS-XCframework.tar.gz> [dest]

The archive is python-multiplatform's (release `cpython-3.15.0t-mobile`), the same one its
bindings plugin embeds in the iOS app (`preparePythonXcframework`), so the wheel is built against
the interpreter that loads it (#437). It is CPython's own iOS layout, not a flat prefix: the
standard library is shared (`Python.xcframework/lib/python3.15t`) and each slice keeps its own
sysconfig module (`<slice>/lib-<arch>/python3.15t/_sysconfigdata_*.py`), its `Python.framework`
and its `bin/` compiler wrappers. `build.py` derives the tag and the extension name from
`<prefix>/lib/python3.*/_sysconfigdata_*.py`, so the prefix puts the arm64 simulator module
beside the stdlib:

    <dest>/lib/python3.15t/            the stdlib + _sysconfigdata_t_ios_arm64-iphonesimulator.py
    <dest>/Python.framework/           the simulator slice's
    <dest>/include/python3.15t/        its Headers
    <dest>/bin/                        its compiler wrappers

The archive's sha256 is checked before anything is written, and a refusal leaves `dest` absent.
"""

import hashlib
import pathlib
import shutil
import sys
import tarfile
import tempfile

ARCHIVE_NAME = "python-3.15.0t-iOS-XCframework.tar.gz"
#: python-multiplatform's SHA256SUMS for the release cpython-3.15.0t-mobile (checked 2026-11-27).
ARCHIVE_SHA256 = "b8c0a4b32d31738f6ead7ace28acee0cb2a590bd0ca7015e9c32af613445e6b5"
SLICE = "Python.xcframework/ios-arm64_x86_64-simulator"
SYSCONFIG = "_sysconfigdata_t_ios_arm64-iphonesimulator.py"

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO / ".caches" / "target-python" / "aarch64-apple-ios-sim-freethreaded"


class PrefixRefusal(RuntimeError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compose(archive, dest, sha256=ARCHIVE_SHA256):
    """Build `dest` from `archive`; refuse by name rather than compose from the wrong thing."""
    archive, dest = pathlib.Path(archive), pathlib.Path(dest)
    got = _sha256(archive)
    if got != sha256:
        raise PrefixRefusal(f"{archive} has sha256 {got}, not {sha256}; refusing to compose a prefix "
                            f"from an archive that is not the pinned {ARCHIVE_NAME}")
    with tempfile.TemporaryDirectory(dir=dest.parent if dest.parent.exists() else None) as tmp:
        tmp = pathlib.Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        stdlib = tmp / "Python.xcframework/lib/python3.15t"
        sim = tmp / SLICE
        sysconfig = sim / "lib-arm64/python3.15t" / SYSCONFIG
        for need in (stdlib, sim / "Python.framework", sim / "Python.framework/Headers", sim / "bin", sysconfig):
            if not need.exists():
                raise PrefixRefusal(f"{archive} has no {need.relative_to(tmp)}; it is not the XCframework "
                                    f"layout this prefix is composed from")
        staged = tmp / "prefix"
        (staged / "lib").mkdir(parents=True)
        (staged / "include").mkdir()
        shutil.copytree(stdlib, staged / "lib/python3.15t", symlinks=True)
        shutil.copy2(sysconfig, staged / "lib/python3.15t" / SYSCONFIG)
        shutil.copytree(sim / "Python.framework", staged / "Python.framework", symlinks=True)
        shutil.copytree(sim / "Python.framework/Headers", staged / "include/python3.15t", symlinks=True)
        shutil.copytree(sim / "bin", staged / "bin", symlinks=True)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(dest))
    return dest


def main(argv):
    if not 1 <= len(argv) <= 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    try:
        dest = compose(argv[0], argv[1] if len(argv) == 2 else DEFAULT_DEST)
    except PrefixRefusal as err:
        print(f"ios_ft_prefix: {err}", file=sys.stderr)
        return 1
    print(f"ios_ft_prefix: composed {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
