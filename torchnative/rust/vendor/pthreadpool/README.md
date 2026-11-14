# Vendored pthreadpool subset (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

pthreadpool's library for both first targets: `src/legacy-api.c` (ExecuTorch
sets `PTHREADPOOL_ALLOW_DEPRECATED_API ON`), `src/portable-api.c`,
`src/memory.c` and `src/pthreads.c`, plus the header closure. `src/fastpath.c`
is not here: upstream builds it only on x86 (`PTHREADPOOL_ENABLE_FASTPATH`).
On Apple, ExecuTorch forces `PTHREADPOOL_SYNC_PRIMITIVE condvar`, which defines
`PTHREADPOOL_USE_FUTEX=0` and `PTHREADPOOL_USE_GCD=0`; this commit has no GCD
implementation, so GCD is not used either way. ExecuTorch also compiles it with
`-fvisibility=hidden` on Apple.

Upstream: https://github.com/google/pthreadpool, commit `a56dcd79c699366e7ac6466792c3025883ff7704`, the commit ExecuTorch
v1.4.1's `backends/xnnpack/third-party/pthreadpool` submodule pins
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

    git init pthreadpool && cd pthreadpool
    git fetch --depth 1 https://github.com/google/pthreadpool a56dcd79c699366e7ac6466792c3025883ff7704
    git checkout FETCH_HEAD
    python scripts/vendor/vendor_executorch.py --pthreadpool-source pthreadpool

or, with all five checkouts side by side in DIR (the layout of ExecuTorch's
`backends/xnnpack/third-party/`), `--third-party DIR`. The script refuses a
checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --pthreadpool-source <checkout>`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's licence (BSD-2-Clause) and covers every file
here.
