# Vendored cpuinfo subset (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

cpuinfo's library sources for the two first targets, as its `CMakeLists.txt`
selects them: the common `src/{api,cache,init,log}.c`, `src/arm/{uarch,cache}.c`,
then `src/arm/mach/init.c` and `src/mach/topology.c` for macOS arm64, and
`src/arm/linux/*.c` (with `aarch64-isa.c`), `src/arm/android/properties.c` and
`src/linux/*.c` for Android arm64, plus the header closure. No x86, RISC-V,
Windows, FreeBSD or emscripten sources, no tests, tools, benchmarks or mocks
(ExecuTorch turns all of them off).

Upstream: https://github.com/pytorch/cpuinfo, commit `f9a03241f8c3d4ed0c9728f5d70bff873d43d4e0`, the commit ExecuTorch
v1.4.1's `backends/xnnpack/third-party/cpuinfo` submodule pins
(issue #319). Paths mirror the upstream checkout.

`BUILD_GROUPS.json` is the machine-readable build description for the later `cc`
build: for each group the files, the flags, the preprocessor definitions and
the include directories (relative to this directory; a sibling tree is
`../<tree>/`), per target where they differ. `--check` verifies that every
file it names is here and that every vendored source file is in exactly one
group or listed as deliberately unbuilt.

The sources are fetched once, at vendor time. A torchnative build never runs
the network, CMake or Python. Vendoring them: leader decision 2026-11-14.

## Regenerating

    git init cpuinfo && cd cpuinfo
    git fetch --depth 1 https://github.com/pytorch/cpuinfo f9a03241f8c3d4ed0c9728f5d70bff873d43d4e0
    git checkout FETCH_HEAD
    python scripts/vendor/vendor_executorch.py --cpuinfo-source cpuinfo

or, with all five checkouts side by side in DIR (the layout of ExecuTorch's
`backends/xnnpack/third-party/`), `--third-party DIR`. The script refuses a
checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --cpuinfo-source <checkout>`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's licence (BSD-2-Clause) and covers every file
here.
