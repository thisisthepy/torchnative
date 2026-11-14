# Vendored FP16 subset (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

FP16's headers, `include/fp16.h` and `include/fp16/*.h` (the `.py` files there
are generators and are not vendored). Header only. At these commits no
vendored source includes them: XNNPACK carries its own `src/xnnpack/fp16.h`,
and ExecuTorch's CMake adds no FP16 target. They are here because the leader
decided (2026-11-14) to vendor all five submodules; `--includes` reports how
many vendored files include them.

Upstream: https://github.com/Maratyszcza/FP16, commit `4dfe081cf6bcd15db339cf2680b9281b8451eeb3`, the commit ExecuTorch
v1.4.1's `backends/xnnpack/third-party/FP16` submodule pins
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

    git init FP16 && cd FP16
    git fetch --depth 1 https://github.com/Maratyszcza/FP16 4dfe081cf6bcd15db339cf2680b9281b8451eeb3
    git checkout FETCH_HEAD
    python scripts/vendor/vendor_executorch.py --fp16-source FP16

or, with all five checkouts side by side in DIR (the layout of ExecuTorch's
`backends/xnnpack/third-party/`), `--third-party DIR`. The script refuses a
checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --fp16-source <checkout>`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's licence (MIT) and covers every file
here.
