# Vendored XNNPACK subset (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

XNNPACK pruned to what an aarch64 build with ExecuTorch v1.4.1's XNNPACK
options compiles: the production microkernels for scalar, NEON (and its
FP16, FMA, v8, dot-product, I8MM and SME variants) and the aarch64 assembly
microkernels, the tables, the operator, subgraph, config and runtime sources,
the header closure of all of them, and `generated/build_identifier.c`. No
x86, wasm, RISC-V, Hexagon, PowerPC or aarch32 microkernels, no non-production
microkernels (ExecuTorch sets `XNNPACK_BUILD_ALL_MICROKERNELS OFF`), no tests,
benchmarks or tools.

The configuration is ExecuTorch's (`backends/xnnpack/cmake/Dependencies.cmake`
and the root `CMakeLists.txt`) with two leader overrides of 2026-11-14:
KleidiAI off (`XNNPACK_ENABLE_KLEIDIAI OFF`; ExecuTorch's default is on, and
XNNPACK would then download KleidiAI at configure time) and SME2 off
(`XNNPACK_ENABLE_ARM_SME2 OFF`; ExecuTorch sets it on). SME (not SME2) stays at
XNNPACK's default, on; without KleidiAI every SME kernel selection in
`src/configs/` sits inside `#if XNN_ENABLE_KLEIDIAI`, so the SME files compile
to wrappers that are never selected.

`BUILD_GROUPS.json` was produced by evaluating upstream's `CMakeLists.txt`
(the copy in this directory) once per target with a small CMake evaluator in
`scripts/vendor/vendor_executorch.py`, not by running CMake. The microkernel
lists are upstream's checked-in `cmake/gen/*.cmake`, re-derived at vendor
time with upstream's `tools/update-microkernels.py` and compared;
`build_identifier.c` is upstream's `scripts/generate-build-identifier.py` run
over the production list, as the CMake custom command does.

Upstream: https://github.com/google/XNNPACK, commit `1adaa7c709d4839d29e1f219cb962b01c9e6a905`, the commit ExecuTorch
v1.4.1's `backends/xnnpack/third-party/XNNPACK` submodule pins
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

    git init XNNPACK && cd XNNPACK
    git fetch --depth 1 https://github.com/google/XNNPACK 1adaa7c709d4839d29e1f219cb962b01c9e6a905
    git checkout FETCH_HEAD
    python scripts/vendor/vendor_executorch.py --xnnpack-source XNNPACK

or, with all five checkouts side by side in DIR (the layout of ExecuTorch's
`backends/xnnpack/third-party/`), `--third-party DIR`. The script refuses a
checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --xnnpack-source <checkout>`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's licence (BSD-3-Clause) and covers every file
here.
