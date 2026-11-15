# Vendored FXdiv subset (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

FXdiv's single header, `include/fxdiv.h`. Header only; XNNPACK and pthreadpool
include it (`FXDIV_USE_INLINE_ASSEMBLY=0`, the value its CMake sets when the
option is not given).

Upstream: https://github.com/Maratyszcza/FXdiv, commit `b408327ac2a15ec3e43352421954f5b1967701d1`, the commit ExecuTorch
v1.4.1's `backends/xnnpack/third-party/FXdiv` submodule pins
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

    git init FXdiv && cd FXdiv
    git fetch --depth 1 https://github.com/Maratyszcza/FXdiv b408327ac2a15ec3e43352421954f5b1967701d1
    git checkout FETCH_HEAD
    python scripts/vendor/vendor_executorch.py --fxdiv-source FXdiv

or, with all five checkouts side by side in DIR (the layout of ExecuTorch's
`backends/xnnpack/third-party/`), `--third-party DIR`. The script refuses a
checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --fxdiv-source <checkout>`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's licence (MIT) and covers every file
here.
