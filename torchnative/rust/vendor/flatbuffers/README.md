# Vendored flatbuffers C++ headers (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

The C++ headers of flatbuffers: every file of upstream's `include/flatbuffers/`,
unmodified, plus upstream's `LICENSE`. Headers only: no sources, no flatc, no
tests, no language bindings. The ExecuTorch schema headers in
`../executorch/generated/include/executorch/schema/` include
`flatbuffers/flatbuffers.h` and `static_assert` exactly this version, because
flatc 24.3.25 generated them.

Upstream: https://github.com/google/flatbuffers, tag `v24.3.25`, commit
`595bf0007ab1929570c7671f091313c8fc20644e`.

The include root is `torchnative/rust/vendor/flatbuffers/include`, so with
`-I torchnative/rust/vendor/flatbuffers/include` an
`#include "flatbuffers/flatbuffers.h"` resolves.

The headers are fetched once, at vendor time. A torchnative build never runs
the network or flatc. Vendoring these headers: leader decision 2026-11-14.

## Regenerating

    git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers FB
    python scripts/vendor/vendor_executorch.py --flatbuffers-source FB

This writes only this directory and needs no ExecuTorch checkout. The script
refuses a checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --flatbuffers-source FB`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's Apache License 2.0 and covers every header here.
