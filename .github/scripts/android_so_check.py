"""Checks on the Android `_C` cross-built with the ExecuTorch runtime (issue #350, step R3).

Run by `.github/workflows/test.yml`'s `android-build` job on the artefact
`cargo ndk -t arm64-v8a --platform 21 build --release --features executorch`
produced, with the NDK's own `llvm-readelf` and `llvm-nm`:

    python .github/scripts/android_so_check.py <tools-bin-dir> <lib_C.so>

It fails, naming each problem, when:

  * `libc++_shared.so` is in `NEEDED` (the C++ runtime must be linked
    statically: an Android Python process has no libc++_shared.so);
  * any `LOAD` segment is aligned below 0x4000 (16 KB page kernels);
  * a defined dynamic symbol starts with `xnn_`, `pthreadpool_` or
    `cpuinfo_`, or contains `executorch` (nothing of the runtime may be
    exported: only `PyInit__C` is meant to be).

Positive controls, so a parse that read nothing cannot pass: the dynamic
section must have at least one `NEEDED` entry, there must be at least one
`LOAD` segment, and `PyInit__C` must be among the exported symbols.

The decision is `problems()`, a function of the tools' text output, so
`tests/release/test_cigate.py` checks it on fixed text without an NDK.
What this cannot see: whether the library loads or computes on a device.
That is the device step (the R3 pull request's body).
"""

import subprocess
import sys
from pathlib import Path

FORBIDDEN_NEEDED = "libc++_shared.so"
MIN_LOAD_ALIGN = 0x4000
FORBIDDEN_PREFIXES = ("xnn_", "pthreadpool_", "cpuinfo_")
FORBIDDEN_SUBSTRING = "executorch"
REQUIRED_EXPORT = "PyInit__C"


def needed(dynamic_text):
    """`NEEDED` libraries from `llvm-readelf -d`."""
    out = []
    for line in dynamic_text.splitlines():
        if "(NEEDED)" in line and "[" in line:
            out.append(line.split("[", 1)[1].split("]", 1)[0])
    return out


def load_aligns(segments_text):
    """The `Align` column of every `LOAD` row of `llvm-readelf -lW`."""
    out = []
    for line in segments_text.splitlines():
        fields = line.split()
        if fields and fields[0] == "LOAD":
            out.append(int(fields[-1], 16))
    return out


def defined_dynamic(symbols_text):
    """Symbol names from `llvm-nm -D --defined-only`, version suffix dropped."""
    out = []
    for line in symbols_text.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            out.append(fields[-1].split("@", 1)[0])
    return out


def problems(dynamic_text, segments_text, symbols_text):
    found = []
    libs = needed(dynamic_text)
    if not libs:
        found.append("no NEEDED entry was read from llvm-readelf -d: the parse or the artefact is wrong")
    if FORBIDDEN_NEEDED in libs:
        found.append(f"{FORBIDDEN_NEEDED} is in NEEDED {libs}: the C++ runtime must be linked statically")
    aligns = load_aligns(segments_text)
    if not aligns:
        found.append("no LOAD segment was read from llvm-readelf -lW: the parse or the artefact is wrong")
    low = [hex(a) for a in aligns if a < MIN_LOAD_ALIGN]
    if low:
        found.append(f"LOAD segment alignment {low} below {hex(MIN_LOAD_ALIGN)}: does not load on a 16 KB page kernel")
    names = defined_dynamic(symbols_text)
    if REQUIRED_EXPORT not in names:
        found.append(f"{REQUIRED_EXPORT} is not exported: the parse or the artefact is wrong")
    leaked = sorted(n for n in names if n.startswith(FORBIDDEN_PREFIXES) or FORBIDDEN_SUBSTRING in n.lower())
    if leaked:
        found.append(f"{len(leaked)} runtime symbol(s) exported, first {leaked[:10]}")
    return found


def _run(tool, *args):
    proc = subprocess.run([str(tool), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{tool} {' '.join(args)} failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    tools, so = Path(argv[1]), argv[2]
    dynamic = _run(tools / "llvm-readelf", "-d", so)
    segments = _run(tools / "llvm-readelf", "-lW", so)
    symbols = _run(tools / "llvm-nm", "-D", "--defined-only", so)
    print(f"NEEDED: {needed(dynamic)}")
    print(f"LOAD alignments: {[hex(a) for a in load_aligns(segments)]}")
    print(f"defined dynamic symbols: {defined_dynamic(symbols)}")
    found = problems(dynamic, segments, symbols)
    for p in found:
        print(f"FAIL android_so_check: {p}")
    if found:
        return 1
    print("ok   android_so_check: no libc++_shared.so, every LOAD >= 0x4000, no runtime symbol exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
