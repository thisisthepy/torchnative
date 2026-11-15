# Vendored ExecuTorch runtime subset (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

The ATen-free ExecuTorch runtime: the core (program loader, method executor,
operator registry, PAL), the prim ops, and the whole portable kernel library,
with the header closure they include. It is the input of step R1 of issue
#312 (an `executorch_rt` crate built with `cc`, no CMake). Nothing here is
compiled yet.

Also the ExecuTorch XNNPACK backend runtime (`backends/xnnpack/runtime/`) and
`extension/threadpool/`, which upstream's `xnnpack_backend` links (issue #319,
step R2). They are here, not in a sibling directory, because their includes
spell `<executorch/backends/xnnpack/...>` and `<executorch/extension/...>`.
`BUILD_GROUPS.json` gives their build groups: the files, and the flags and
definitions upstream's CMake applies to them (read from
`backends/xnnpack/CMakeLists.txt`, `backends/xnnpack/cmake/Dependencies.cmake`,
`extension/threadpool/CMakeLists.txt` and the root `CMakeLists.txt`, not
measured by running CMake). The core, prim ops and portable kernels are not
described there yet: that is step R1.

Upstream: https://github.com/pytorch/executorch, tag `v1.4.1`, commit
`e4d02f41f7909e8ed5bf4a14ffc520d733453d9f`. Paths mirror the upstream checkout, so with
`-I torchnative/rust/vendor` an upstream `#include <executorch/...>` resolves.
Also add `-I` for `runtime/core/portable_type/c10`, for `generated/include`, and for
`torchnative/rust/vendor/flatbuffers/include` (the sibling flatbuffers tree).
The XNNPACK backend also needs `../xnnpack/include`, `../pthreadpool/include`
and `../cpuinfo/include`.

## Generated files

`generated/` holds what upstream produces at build time. Here it is produced
once, at vendor time, so a torchnative build needs neither flatc nor torchgen:

- `generated/include/executorch/schema/`: `flatc --cpp --cpp-std c++11 --gen-mutable --scoped-enums` on
  `schema/scalar_type.fbs` and `schema/program.fbs`, with flatc version 24.3.25.
- `generated/include/executorch/backends/xnnpack/serialization/schema_generated.h`: `flatc --cpp --cpp-std c++11 --scoped-enums` on
  `backends/xnnpack/serialization/runtime_schema.fbs`, then renamed from `runtime_schema_generated.h`, as
  upstream's `backends/xnnpack/CMakeLists.txt` does.
- `generated/portable_ops_lib/`: `python -m codegen.tools.gen_oplist` and
  `python -m codegen.gen` (upstream's own codegen, with torchgen from torch
  2.13.0, the version upstream's `torch_pin.py` names) on
  `kernels/portable/functions.yaml`.

## Siblings

flatbuffers' C++ headers (v24.3.25), which the generated schema headers
include, are vendored next to this directory in `../flatbuffers/` (leader
decision 2026-11-14). XNNPACK, cpuinfo, pthreadpool, FXdiv and FP16, at the
commits this tag's `backends/xnnpack/third-party/` submodules pin, are in
`../xnnpack/`, `../cpuinfo/`, `../pthreadpool/`, `../fxdiv/` and `../fp16/`
(leader decision 2026-11-14, issue #319).
`python scripts/vendor/vendor_executorch.py --includes` lists every include
that resolves in none of them.

## Regenerating

    git clone --depth 1 --branch v1.4.1 https://github.com/pytorch/executorch ET
    python scripts/vendor/vendor_executorch.py --source ET \
        --flatc PATH_TO_FLATC_24.3.25 --codegen-python PYTHON_WITH_TORCH_2.13.0

This writes only this directory; the siblings have their own checkouts
(`--flatbuffers-source`, `--third-party`, see their READMEs).

The script refuses a checkout at any other commit, a flatc of another
version and a torchgen of another torch version, each by name.

## Licence

`LICENSE` is upstream's BSD-style licence and covers every file here,
including the generated ones (their templates are upstream's).
