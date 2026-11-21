# Vendored flatbuffers C++ headers and parser (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

The C++ headers of flatbuffers: every file of upstream's `include/flatbuffers/`,
unmodified, plus upstream's `LICENSE`. The ExecuTorch schema headers in
`../executorch/generated/include/executorch/schema/` include
`flatbuffers/flatbuffers.h` and `static_assert` exactly this version, because
flatc 24.3.25 generated them.

Upstream: https://github.com/google/flatbuffers, tag `v24.3.25`, commit
`595bf0007ab1929570c7671f091313c8fc20644e`.

The include root is `torchnative/rust/vendor/flatbuffers/include`, so with
`-I torchnative/rust/vendor/flatbuffers/include` an
`#include "flatbuffers/flatbuffers.h"` resolves.

The parser (manifest group `parser`, issue #357): the `.cpp` files of
upstream's `flatbuffers` library target, read from `FlatBuffers_Library_SRCS`
in upstream's `CMakeLists.txt` (kept here as group `inputs`), unmodified, plus
every file under `src/` their quoted includes reach.
`torchnative/rust/executorch_rt` compiles them so that exir's JSON-to-binary
step runs in process instead of exec'ing flatc (leader decision 2026-11-14 item
4). No flatc front end (`src/flatc*.cpp`), no other code generator, no tests.

Everything is fetched once, at vendor time. A torchnative build never runs
the network or flatc. Vendoring these headers: leader decision 2026-11-14.

## Regenerating

    git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers FB
    python scripts/vendor/vendor_executorch.py --flatbuffers-source FB

This writes only this directory and needs no ExecuTorch checkout. The script
refuses a checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --flatbuffers-source FB`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's Apache License 2.0 and covers every file here.
