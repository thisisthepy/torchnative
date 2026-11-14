#!/usr/bin/env python3
"""Vendor the ATen-free ExecuTorch runtime subset, its XNNPACK backend and their third-party trees, or check them.

    python scripts/vendor/vendor_executorch.py --source ET_CHECKOUT
        write torchnative/rust/vendor/executorch/ from an ExecuTorch checkout
    python scripts/vendor/vendor_executorch.py --flatbuffers-source FB_CHECKOUT
        write torchnative/rust/vendor/flatbuffers/ from a flatbuffers checkout
    python scripts/vendor/vendor_executorch.py --third-party DIR
        write torchnative/rust/vendor/{xnnpack,cpuinfo,pthreadpool,fxdiv,fp16}/
        from DIR/{XNNPACK,cpuinfo,pthreadpool,FXdiv,FP16} (the layout of
        ExecuTorch's backends/xnnpack/third-party/); each tree also has its
        own --<tree>-source flag
        (any of these flags alone writes only its trees; several write several)
    python scripts/vendor/vendor_executorch.py --check
        verify every tree on disk against its MANIFEST.tsv and its
        BUILD_GROUPS.json (no checkout needed)
    python scripts/vendor/vendor_executorch.py --check [any source flag]
        the same, then regenerate each tree whose checkout is given into a
        scratch directory inside the repository and compare every file byte
        for byte
    python scripts/vendor/vendor_executorch.py --includes
        list every #include in the vendored trees that does not resolve inside
        them, split into C/C++ standard, platform and unresolved

`--source` may also come from TORCHNATIVE_ET_SRC, `--flatbuffers-source` from
TORCHNATIVE_FLATBUFFERS_SRC, `--third-party` from TORCHNATIVE_ET_THIRD_PARTY.
Issue #312, step R0 of the ExecuTorch runtime design (agent proposal, not
confirmed), and issue #319 for the XNNPACK parts. The vendoring itself, the
whole portable kernel library rather than an allow-list, and the rule that
nothing runs CMake, flatc or the network at torchnative BUILD time are the
leader's decisions of 2026-11-14. So are, for #319: vendoring XNNPACK, cpuinfo,
pthreadpool, FXdiv and FP16 through this script with the same sha256 manifest
and --check; cloning them at vendor time (never at build time); shipping their
licence files; KleidiAI and SME2 off; aarch64 only at first (macOS arm64,
Android arm64).

What is vendored in torchnative/rust/vendor/executorch/ (pinned: tag v1.4.1,
commit e4d02f41):

  core       EXECUTORCH_CORE_SRCS from upstream's build_variables.bzl, plus the
             three default PALs (posix, minimal, android)
  prim_ops   EXECUTORCH_SRCS (kernels/prim_ops)
  portable   every kernels/portable/cpu/**/*.cpp outside test/, which is the
             glob upstream's kernels/portable/CMakeLists.txt compiles. The bzl
             list PORTABLE_KERNELS_SRCS is NOT used: it lacks 7 op files and 2
             util files whose ops functions.yaml registers, so a build from it
             would fail to link the generated registrations.
  xnnpack_backend
             every backends/xnnpack/runtime/** source outside test/. The 9 in
             XNNPACK_BACKEND_SRCS (build_variables.bzl, which
             backends/xnnpack/CMakeLists.txt compiles into xnnpack_backend) are
             a build group; runtime/utils/utils.cpp is not compiled upstream and
             is listed as unbuilt.
  threadpool the sources extension/threadpool/CMakeLists.txt compiles into
             extension_threadpool, which xnnpack_backend links: the bzl
             THREADPOOL_SRCS plus cpuinfo_utils.cpp
  headers    the transitive #include closure of all of the above, resolved
             against the checkout with upstream's include roots
  generated  what upstream produces at build time, produced here once, at
             VENDOR time, never at torchnative build time:
               flatc 24.3.25 --cpp --cpp-std c++11 --gen-mutable --scoped-enums
                 on schema/{scalar_type,program}.fbs
               flatc 24.3.25 --cpp --cpp-std c++11 --scoped-enums on
                 backends/xnnpack/serialization/runtime_schema.fbs, renamed
                 runtime_schema_generated.h -> schema_generated.h as
                 backends/xnnpack/CMakeLists.txt's MV_COMMAND does
               python -m codegen.tools.gen_oplist and python -m codegen.gen with
                 torchgen from upstream torch 2.13.0 (torch_pin.py) on
                 kernels/portable/functions.yaml
  inputs     functions.yaml, the .fbs files the generated files came from, and
             the CMake files BUILD_GROUPS.json was read from
  build      BUILD_GROUPS.json: the XNNPACK backend and threadpool build groups
             with the flags and definitions upstream's CMake gives them
  license    upstream's LICENSE
  meta       README.md and .gitattributes written by this script

Why the XNNPACK backend goes into the executorch tree and not a sibling: its
sources include `<executorch/backends/xnnpack/...>` and
`<executorch/extension/threadpool/...>`, which resolve under
`-I torchnative/rust/vendor` only inside a directory named `executorch`; a
sibling would need a second include root that mirrors the name.

The flatbuffers tree (pinned: tag v24.3.25, commit 595bf000), a sibling
directory with its own MANIFEST.tsv (leader decision 2026-11-14):

  headers    every file of upstream's include/flatbuffers/ (all are .h), so
             `-I torchnative/rust/vendor/flatbuffers/include` resolves the
             generated schema headers' "flatbuffers/flatbuffers.h"
  license    upstream's LICENSE (Apache-2.0)
  meta       README.md and .gitattributes written by this script
  No sources, no flatc, no tests, no language bindings.

The five third-party trees (issue #319), siblings with their own MANIFEST.tsv
and BUILD_GROUPS.json, each pinned to the commit ExecuTorch v1.4.1's
backends/xnnpack/third-party/ gitlink names (verified against an ExecuTorch
checkout whenever one is given):

  xnnpack     XNNPACK 1adaa7c7, pruned to what an aarch64 build with
              ExecuTorch's XNNPACK options compiles. The source lists and flags
              come from evaluating XNNPACK's own CMakeLists.txt (a small
              evaluator below, for the subset of CMake that file uses) with
              the cache values ExecuTorch's CMake sets before add_subdirectory,
              per target (aarch64-apple-darwin, aarch64-linux-android). The
              microkernel lists are XNNPACK's checked-in cmake/gen/*.cmake,
              which the script first re-derives with upstream's own
              tools/update-microkernels.py and refuses to use if they differ.
              build_identifier.c is produced at vendor time by upstream's
              scripts/generate-build-identifier.py over the prod list.
  cpuinfo     cpuinfo f9a03241: src/ for arm64 + mach (macOS) and arm64 +
              linux + android (Android), read from its CMakeLists.txt
  pthreadpool pthreadpool a56dcd79: the pthreads implementation (no fastpath on
              arm64; this commit has no GCD implementation; ExecuTorch forces
              condvar on Apple)
  fxdiv       FXdiv b408327a: include/fxdiv.h (header only)
  fp16        FP16 4dfe081c: include/fp16.h and include/fp16/*.h (header only;
              no vendored source includes it, see its README)

Why Python and not sh like vendor_candle.sh: `--check` runs in the gate on
Linux and macOS and should be able to run on the Windows runner, which has no
POSIX shell for tests; and evaluating build_variables.bzl (upstream's CMake does
it with Python too), hashing, and the include closure are all native here.
scripts/vendor/vendor_executorch.sh is a thin wrapper for the shell habit.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
VENDOR = REPO / "torchnative" / "rust" / "vendor"
DEST = VENDOR / "executorch"
FB_DEST = VENDOR / "flatbuffers"

TAG = "v1.4.1"
COMMIT = "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f"
FLATC_VERSION = "flatc version 24.3.25"
TORCH_VERSION = "2.13.0"
MANIFEST = "MANIFEST.tsv"
BUILD_GROUPS = "BUILD_GROUPS.json"
SOURCE_EXT = (".c", ".cc", ".cpp", ".S")
SCAN_EXT = (".c", ".cc", ".cpp", ".S", ".h", ".inc")
TARGETS = ["aarch64-apple-darwin", "aarch64-linux-android"]

FB_TAG = "v24.3.25"
FB_COMMIT = "595bf0007ab1929570c7671f091313c8fc20644e"
FB_VERSION = (24, 3, 25)
FB_HEADERS = "include/flatbuffers/"
# Where `--includes` places the flatbuffers tree relative to the ExecuTorch
# one (they are siblings under torchnative/rust/vendor/).
FB_FROM_ET = "../flatbuffers"

PALS = ["runtime/platform/default/posix.cpp",
        "runtime/platform/default/minimal.cpp",
        "runtime/platform/default/android.cpp"]
HEADER_SEEDS = ["extension/data_loader/buffer_data_loader.h",
                "extension/memory_allocator/malloc_memory_allocator.h"]
INPUTS = ["kernels/portable/functions.yaml", "schema/scalar_type.fbs", "schema/program.fbs"]
XNN_RUNTIME = "backends/xnnpack/runtime/"
XNN_SCHEMA = "backends/xnnpack/serialization/runtime_schema.fbs"
XNN_INPUTS = [XNN_SCHEMA, "backends/xnnpack/CMakeLists.txt", "backends/xnnpack/cmake/Dependencies.cmake",
              "extension/threadpool/CMakeLists.txt"]
THREADPOOL_CMAKE = "extension/threadpool/CMakeLists.txt"

FLATC_ARGS = ["--cpp", "--cpp-std", "c++11", "--gen-mutable", "--scoped-enums"]
# backends/xnnpack/CMakeLists.txt: `flatc --cpp --cpp-std c++11 --scoped-enums`
# (no --gen-mutable), then MV_COMMAND renames runtime_<x>_generated.h to
# <x>_generated.h under ${_xnnpack_schema__include_dir}/executorch/.
XNN_FLATC_ARGS = ["--cpp", "--cpp-std", "c++11", "--scoped-enums"]
GEN_SCHEMA = "generated/include/executorch/schema"
GEN_XNN_SCHEMA = "generated/include/executorch/backends/xnnpack/serialization/schema_generated.h"
GEN_OPS = "generated/portable_ops_lib"
# The outputs upstream's gen_operators_lib compiles (tools/cmake/Codegen.cmake),
# plus the op list they were generated from. codegen.gen also writes a sharded
# RegisterCodegenUnboxedKernels_0.cpp (identical content) and RegisterKernels.h
# (for --manual-registration); upstream's CMake uses neither, so neither is kept.
GEN_OPS_FILES = ["RegisterCodegenUnboxedKernelsEverything.cpp", "Functions.h",
                 "NativeFunctions.h", "selected_operators.yaml"]

# Include roots, relative to the vendored directory, in the order a compiler
# given `-I torchnative/rust/vendor -I <tree>/runtime/core/portable_type/c10
# -I <tree>/generated/include` searches them (upstream: CMakeLists.txt
# `_common_include_directories` plus schema/CMakeLists.txt's include dir).
C10_ROOT = "runtime/core/portable_type/c10"
GEN_ROOT = "generated/include"
FB_ROOT = f"{FB_FROM_ET}/include"

# The five trees of ExecuTorch v1.4.1's backends/xnnpack/third-party/.
# `submodule` is the gitlink's directory name there (and under --third-party).
THIRD_PARTY = {
    "xnnpack": dict(label="XNNPACK", submodule="XNNPACK", url="https://github.com/google/XNNPACK",
                    commit="1adaa7c709d4839d29e1f219cb962b01c9e6a905", marker="include/xnnpack.h",
                    licence="BSD-3-Clause", date="2026-03-11"),
    "cpuinfo": dict(label="cpuinfo", submodule="cpuinfo", url="https://github.com/pytorch/cpuinfo",
                    commit="f9a03241f8c3d4ed0c9728f5d70bff873d43d4e0", marker="include/cpuinfo.h",
                    licence="BSD-2-Clause", date="2026-01-27"),
    "pthreadpool": dict(label="pthreadpool", submodule="pthreadpool", url="https://github.com/google/pthreadpool",
                        commit="a56dcd79c699366e7ac6466792c3025883ff7704", marker="include/pthreadpool.h",
                        licence="BSD-2-Clause", date="2026-05-02"),
    "fxdiv": dict(label="FXdiv", submodule="FXdiv", url="https://github.com/Maratyszcza/FXdiv",
                  commit="b408327ac2a15ec3e43352421954f5b1967701d1", marker="include/fxdiv.h",
                  licence="MIT", date="2020"),
    "fp16": dict(label="FP16", submodule="FP16", url="https://github.com/Maratyszcza/FP16",
                 commit="4dfe081cf6bcd15db339cf2680b9281b8451eeb3", marker="include/fp16.h",
                 licence="MIT", date="2020"),
}
# Per tree: include roots relative to the tree (`.` is the tree root), the
# order its CMake gives them; sibling trees appear as `../<tree>/...`.
TREE_ROOTS = {
    "executorch": [C10_ROOT, GEN_ROOT, FB_ROOT, "../xnnpack/include", "../pthreadpool/include",
                   "../cpuinfo/include"],
    "flatbuffers": ["include"],
    "xnnpack": [".", "include", "src", "../pthreadpool/include", "../cpuinfo/include", "../fxdiv/include"],
    "cpuinfo": ["include", "src"],
    "pthreadpool": ["include", "src", "../fxdiv/include"],
    "fxdiv": ["include"],
    "fp16": ["include"],
}

README = f"""# Vendored ExecuTorch runtime subset (torchnative)

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
`{BUILD_GROUPS}` gives their build groups: the files, and the flags and
definitions upstream's CMake applies to them (read from
`backends/xnnpack/CMakeLists.txt`, `backends/xnnpack/cmake/Dependencies.cmake`,
`extension/threadpool/CMakeLists.txt` and the root `CMakeLists.txt`, not
measured by running CMake). The core, prim ops and portable kernels are not
described there yet: that is step R1.

Upstream: https://github.com/pytorch/executorch, tag `{TAG}`, commit
`{COMMIT}`. Paths mirror the upstream checkout, so with
`-I torchnative/rust/vendor` an upstream `#include <executorch/...>` resolves.
Also add `-I` for `{C10_ROOT}`, for `{GEN_ROOT}`, and for
`torchnative/rust/vendor/flatbuffers/include` (the sibling flatbuffers tree).
The XNNPACK backend also needs `../xnnpack/include`, `../pthreadpool/include`
and `../cpuinfo/include`.

## Generated files

`generated/` holds what upstream produces at build time. Here it is produced
once, at vendor time, so a torchnative build needs neither flatc nor torchgen:

- `{GEN_SCHEMA}/`: `flatc {' '.join(FLATC_ARGS)}` on
  `schema/scalar_type.fbs` and `schema/program.fbs`, with {FLATC_VERSION}.
- `{GEN_XNN_SCHEMA}`: `flatc {' '.join(XNN_FLATC_ARGS)}` on
  `{XNN_SCHEMA}`, then renamed from `runtime_schema_generated.h`, as
  upstream's `backends/xnnpack/CMakeLists.txt` does.
- `{GEN_OPS}/`: `python -m codegen.tools.gen_oplist` and
  `python -m codegen.gen` (upstream's own codegen, with torchgen from torch
  {TORCH_VERSION}, the version upstream's `torch_pin.py` names) on
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

    git clone --depth 1 --branch {TAG} https://github.com/pytorch/executorch ET
    python scripts/vendor/vendor_executorch.py --source ET \\
        --flatc PATH_TO_FLATC_24.3.25 --codegen-python PYTHON_WITH_TORCH_{TORCH_VERSION}

This writes only this directory; the siblings have their own checkouts
(`--flatbuffers-source`, `--third-party`, see their READMEs).

The script refuses a checkout at any other commit, a flatc of another
version and a torchgen of another torch version, each by name.

## Licence

`LICENSE` is upstream's BSD-style licence and covers every file here,
including the generated ones (their templates are upstream's).
"""

GITATTRIBUTES = "# Byte-exact on every platform: MANIFEST.tsv pins sha256, so no EOL conversion.\n* -text\n"

FB_README = f"""# Vendored flatbuffers C++ headers (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

The C++ headers of flatbuffers: every file of upstream's `include/flatbuffers/`,
unmodified, plus upstream's `LICENSE`. Headers only: no sources, no flatc, no
tests, no language bindings. The ExecuTorch schema headers in
`../executorch/generated/include/executorch/schema/` include
`flatbuffers/flatbuffers.h` and `static_assert` exactly this version, because
flatc {'.'.join(map(str, FB_VERSION))} generated them.

Upstream: https://github.com/google/flatbuffers, tag `{FB_TAG}`, commit
`{FB_COMMIT}`.

The include root is `torchnative/rust/vendor/flatbuffers/include`, so with
`-I torchnative/rust/vendor/flatbuffers/include` an
`#include "flatbuffers/flatbuffers.h"` resolves.

The headers are fetched once, at vendor time. A torchnative build never runs
the network or flatc. Vendoring these headers: leader decision 2026-11-14.

## Regenerating

    git clone --depth 1 --branch {FB_TAG} https://github.com/google/flatbuffers FB
    python scripts/vendor/vendor_executorch.py --flatbuffers-source FB

This writes only this directory and needs no ExecuTorch checkout. The script
refuses a checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --flatbuffers-source FB`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's Apache License 2.0 and covers every header here.
"""

TP_WHAT = {
    "xnnpack": """\
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
""",
    "cpuinfo": """\
cpuinfo's library sources for the two first targets, as its `CMakeLists.txt`
selects them: the common `src/{api,cache,init,log}.c`, `src/arm/{uarch,cache}.c`,
then `src/arm/mach/init.c` and `src/mach/topology.c` for macOS arm64, and
`src/arm/linux/*.c` (with `aarch64-isa.c`), `src/arm/android/properties.c` and
`src/linux/*.c` for Android arm64, plus the header closure. No x86, RISC-V,
Windows, FreeBSD or emscripten sources, no tests, tools, benchmarks or mocks
(ExecuTorch turns all of them off).
""",
    "pthreadpool": """\
pthreadpool's library for both first targets: `src/legacy-api.c` (ExecuTorch
sets `PTHREADPOOL_ALLOW_DEPRECATED_API ON`), `src/portable-api.c`,
`src/memory.c` and `src/pthreads.c`, plus the header closure. `src/fastpath.c`
is not here: upstream builds it only on x86 (`PTHREADPOOL_ENABLE_FASTPATH`).
On Apple, ExecuTorch forces `PTHREADPOOL_SYNC_PRIMITIVE condvar`, which defines
`PTHREADPOOL_USE_FUTEX=0` and `PTHREADPOOL_USE_GCD=0`; this commit has no GCD
implementation, so GCD is not used either way. ExecuTorch also compiles it with
`-fvisibility=hidden` on Apple.
""",
    "fxdiv": """\
FXdiv's single header, `include/fxdiv.h`. Header only; XNNPACK and pthreadpool
include it (`FXDIV_USE_INLINE_ASSEMBLY=0`, the value its CMake sets when the
option is not given).
""",
    "fp16": """\
FP16's headers, `include/fp16.h` and `include/fp16/*.h` (the `.py` files there
are generators and are not vendored). Header only. At these commits no
vendored source includes them: XNNPACK carries its own `src/xnnpack/fp16.h`,
and ExecuTorch's CMake adds no FP16 target. They are here because the leader
decided (2026-11-14) to vendor all five submodules; `--includes` reports how
many vendored files include them.
""",
}


def tp_readme(name):
    spec = THIRD_PARTY[name]
    return f"""# Vendored {spec['label']} subset (torchnative)

This directory is generated. Never edit it by hand: every file is listed with
its sha256 in `MANIFEST.tsv`, and `tests/release/test_executorch_vendor.py`
runs `python scripts/vendor/vendor_executorch.py --check` in the gate.

## What it is

{TP_WHAT[name]}
Upstream: {spec['url']}, commit `{spec['commit']}`, the commit ExecuTorch
{TAG}'s `backends/xnnpack/third-party/{spec['submodule']}` submodule pins
(issue #319). Paths mirror the upstream checkout.

`{BUILD_GROUPS}` is the machine-readable build description for the later `cc`
build: for each group the files, the flags, the preprocessor definitions and
the include directories (relative to this directory; a sibling tree is
`../<tree>/`), per target where they differ. `--check` verifies that every
file it names is here and that every vendored source file is in exactly one
group or listed as deliberately unbuilt.

The sources are fetched once, at vendor time. A torchnative build never runs
the network, CMake or Python. Vendoring them: leader decision 2026-11-14.

## Regenerating

    git init {spec['submodule']} && cd {spec['submodule']}
    git fetch --depth 1 {spec['url']} {spec['commit']}
    git checkout FETCH_HEAD
    python scripts/vendor/vendor_executorch.py --{name}-source {spec['submodule']}

or, with all five checkouts side by side in DIR (the layout of ExecuTorch's
`backends/xnnpack/third-party/`), `--third-party DIR`. The script refuses a
checkout at any other commit, by name.
`python scripts/vendor/vendor_executorch.py --check --{name}-source <checkout>`
regenerates into scratch and compares byte for byte.

## Licence

`LICENSE` is upstream's licence ({spec['licence']}) and covers every file
here.
"""


STD_HEADERS = set("""
algorithm any array atomic bit bitset cassert cctype cerrno cfenv cfloat charconv chrono cinttypes
climits clocale cmath codecvt complex condition_variable csetjmp csignal cstdarg cstddef cstdint
cstdio cstdlib cstring ctime cuchar cwchar cwctype deque exception execution filesystem forward_list
fstream functional future initializer_list iomanip ios iosfwd iostream istream iterator limits list
locale map memory memory_resource mutex new numeric optional ostream queue random ratio regex
scoped_allocator set shared_mutex sstream stack stdexcept streambuf string string_view system_error
thread tuple type_traits typeindex typeinfo unordered_map unordered_set utility valarray variant vector
version span concepts numbers ranges source_location compare coroutine
assert.h complex.h ctype.h errno.h fenv.h float.h inttypes.h iso646.h limits.h locale.h math.h
setjmp.h signal.h stdalign.h stdarg.h stdatomic.h stdbool.h stddef.h stdint.h stdio.h stdlib.h
string.h tgmath.h threads.h time.h uchar.h wchar.h wctype.h
""".split())
# OS and compiler headers: present on the platforms whose #if selects them.
PLATFORM_PREFIXES = ("sys/", "android/", "mach/", "mach-o/", "zephyr/", "linux/", "asm/", "asm-generic/",
                     "libkern/", "emscripten/", "arm_", "sanitizer/")
PLATFORM_HEADERS = set("""
unistd.h pthread.h dlfcn.h fcntl.h sched.h malloc.h alloca.h execinfo.h cxxabi.h windows.h io.h
intrin.h immintrin.h arm_neon.h sal.h TargetConditionals.h emscripten.h process.h
arm_fp16.h arm_acle.h arm_sve.h arm_sme.h arm_bf16.h dispatch/dispatch.h errhandler.h
xmmintrin.h emmintrin.h tmmintrin.h smmintrin.h nmmintrin.h wasm_simd128.h riscv_vector.h
hexagon_types.h hexagon_protos.h hvx_hexagon_protos.h altivec.h spawn.h sysinfoapi.h elf.h
""".split())

INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\s*(.*)$")


class Refusal(Exception):
    """A named refusal: printed as-is, exit code 2."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(src, *args):
    done = subprocess.run(["git", "-C", str(src), *args], capture_output=True)
    if done.returncode != 0:
        raise Refusal(f"vendor_executorch: refusing: `git {' '.join(args)}` failed in {src}:\n"
                      f"{done.stderr.decode(errors='replace')}")
    return done.stdout.decode()


def verify_source(src):
    if not (src / "shim_et" / "xplat" / "executorch" / "build" / "build_variables.bzl").is_file():
        raise Refusal(f"vendor_executorch: refusing: {src} is not an ExecuTorch checkout "
                      f"(no shim_et/xplat/executorch/build/build_variables.bzl)")
    head = git(src, "rev-parse", "HEAD").strip()
    if head != COMMIT:
        raise Refusal(f"vendor_executorch: refusing: {src} is at commit {head}, not ExecuTorch "
                      f"{TAG} ({COMMIT}). Nothing was written.")
    dirty = git(src, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise Refusal(f"vendor_executorch: refusing: {src} has modified tracked files, so it is "
                      f"not {TAG} as published:\n{dirty}")
    pin = (src / "torch_pin.py").read_text()
    if f'TORCH_VERSION = "{TORCH_VERSION}"' not in pin:
        raise Refusal(f"vendor_executorch: refusing: {src}/torch_pin.py does not pin torch "
                      f"{TORCH_VERSION}; this script's codegen pin is stale")
    # The third-party pins in THIRD_PARTY are this tag's gitlinks; say so if not.
    links = git(src, "ls-tree", "HEAD", "backends/xnnpack/third-party/")
    for name, spec in THIRD_PARTY.items():
        want = f"160000 commit {spec['commit']}\tbackends/xnnpack/third-party/{spec['submodule']}"
        if want not in links.splitlines():
            raise Refusal(f"vendor_executorch: refusing: {src} at {TAG} does not pin "
                          f"backends/xnnpack/third-party/{spec['submodule']} to {spec['commit']}; "
                          f"this script's {name} pin is stale:\n{links}")
    return set(git(src, "ls-files", "-z").split("\0")) - {""}


def verify_fb_source(src):
    if not (src / FB_HEADERS / "base.h").is_file():
        raise Refusal(f"vendor_executorch: refusing: {src} is not a flatbuffers checkout "
                      f"(no {FB_HEADERS}base.h)")
    head = git(src, "rev-parse", "HEAD").strip()
    if head != FB_COMMIT:
        raise Refusal(f"vendor_executorch: refusing: {src} is at commit {head}, not flatbuffers "
                      f"{FB_TAG} ({FB_COMMIT}). Nothing was written.")
    dirty = git(src, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise Refusal(f"vendor_executorch: refusing: {src} has modified tracked files, so it is "
                      f"not flatbuffers {FB_TAG} as published:\n{dirty}")
    base = (src / FB_HEADERS / "base.h").read_text(encoding="utf-8")
    got = tuple(int(m.group(1)) if m else None for m in
                (re.search(rf"^#define FLATBUFFERS_VERSION_{p} (\d+)$", base, re.M)
                 for p in ("MAJOR", "MINOR", "REVISION")))
    if got != FB_VERSION:
        raise Refusal(f"vendor_executorch: refusing: {src}/{FB_HEADERS}base.h defines version {got}, "
                      f"not {FB_VERSION}")
    return set(git(src, "ls-files", "-z").split("\0")) - {""}


def verify_tp_source(name, src):
    spec = THIRD_PARTY[name]
    if not (src / spec["marker"]).is_file():
        raise Refusal(f"vendor_executorch: refusing: {src} is not a {spec['label']} checkout "
                      f"(no {spec['marker']})")
    head = git(src, "rev-parse", "HEAD").strip()
    if head != spec["commit"]:
        raise Refusal(f"vendor_executorch: refusing: {src} is at commit {head}, not {spec['label']} "
                      f"{spec['commit']} (the ExecuTorch {TAG} submodule pin). Nothing was written.")
    dirty = git(src, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise Refusal(f"vendor_executorch: refusing: {src} has modified tracked files, so it is "
                      f"not {spec['label']} {spec['commit']} as published:\n{dirty}")
    tracked = set(git(src, "ls-files", "-z").split("\0")) - {""}
    if "LICENSE" not in tracked:
        raise Refusal(f"vendor_executorch: refusing: {src} has no tracked LICENSE")
    return tracked


def select_fb(src, tracked):
    """Return {rel: (bytes, group, upstream)} for the flatbuffers tree."""
    headers = sorted(t for t in tracked if t.startswith(FB_HEADERS))
    not_h = [t for t in headers if not t.endswith(".h")]
    if not_h:
        raise Refusal(f"vendor_executorch: refusing: {FB_HEADERS} at {FB_TAG} holds non-header files "
                      f"{not_h}; this script vendors headers only")
    if "LICENSE" not in tracked:
        raise Refusal(f"vendor_executorch: refusing: {src} has no tracked LICENSE")
    tree = {rel: ((src / rel).read_bytes(), "headers", rel) for rel in headers}
    tree["LICENSE"] = ((src / "LICENSE").read_bytes(), "license", "LICENSE")
    tree["README.md"] = (FB_README.encode(), "meta", "torchnative")
    tree[".gitattributes"] = (GITATTRIBUTES.encode(), "meta", "torchnative")
    return tree


def build_variables(src):
    ns = {}
    path = src / "shim_et" / "xplat" / "executorch" / "build" / "build_variables.bzl"
    exec(compile(path.read_text(), str(path), "exec"), ns)  # what upstream's CMake does
    return ns


def find_flatc(explicit):
    cands = []
    if explicit:
        cands.append(pathlib.Path(explicit))
    else:
        for root in dict.fromkeys([REPO.resolve(), main_checkout()]):
            cands += sorted(root.glob(".caches/*/lib/python3*/site-packages/executorch/data/bin/flatc"))
        on_path = shutil.which("flatc")
        if on_path:
            cands.append(pathlib.Path(on_path))
    for c in cands:
        if not c.is_file():
            continue
        done = subprocess.run([str(c), "--version"], capture_output=True, text=True)
        if done.returncode == 0 and done.stdout.strip() == FLATC_VERSION:
            return c
        if explicit:
            raise Refusal(f"vendor_executorch: refusing: {c} says {done.stdout.strip()!r}, not "
                          f"{FLATC_VERSION!r}; the generated headers static_assert that version")
    raise Refusal(
        "vendor_executorch: refusing: no flatc 24.3.25 found (looked in --flatc / TORCHNATIVE_FLATC, "
        "the repository's .caches/*/.../executorch/data/bin/flatc, and PATH). Nothing is downloaded. "
        f"Missing would be: {GEN_SCHEMA}/scalar_type_generated.h and program_generated.h, made by "
        f"`flatc {' '.join(FLATC_ARGS)} -o {GEN_SCHEMA} scalar_type.fbs program.fbs` in schema/, and "
        f"{GEN_XNN_SCHEMA}.")


def codegen_python(explicit):
    py = explicit or sys.executable
    probe = ("import importlib.metadata as m, os, torchgen, yaml; "
             "print(m.version('torch')); print(os.path.dirname(torchgen.__file__))")
    done = subprocess.run([py, "-I", "-c", probe], capture_output=True, text=True)
    if done.returncode != 0:
        raise Refusal(f"vendor_executorch: refusing: {py} cannot import torchgen and yaml:\n{done.stderr}")
    version, torchgen_dir = done.stdout.split("\n")[:2]
    if version.split("+")[0] != TORCH_VERSION:
        raise Refusal(f"vendor_executorch: refusing: {py} has torch {version}, not {TORCH_VERSION} "
                      f"(upstream torch_pin.py); its torchgen would generate other registrations")
    return py, pathlib.Path(torchgen_dir)


def run(cmd, cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise Refusal(f"vendor_executorch: `{' '.join(map(str, cmd))}` failed (exit {done.returncode}):\n"
                      f"{done.stdout}\n{done.stderr}")
    return done


def generate(src, tracked, work, flatc, py, torchgen_dir):
    """Produce upstream's build-time files. Upstream code runs from a copy in
    `work`, so the checkout is never written (no __pycache__ either)."""
    for rel in sorted(t for t in tracked if t.startswith("codegen/") and "/test/" not in t):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, work / rel)
    for rel in INPUTS + [XNN_SCHEMA]:
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, work / rel)
    out = {}

    schema_out = work / "out-schema"
    schema_out.mkdir()
    run([flatc, *FLATC_ARGS, "-o", schema_out, "scalar_type.fbs", "program.fbs"], cwd=work / "schema")
    for name in ("scalar_type_generated.h", "program_generated.h"):
        out[f"{GEN_SCHEMA}/{name}"] = ((schema_out / name).read_bytes(),
                                       f"generated: {FLATC_VERSION} {' '.join(FLATC_ARGS)} schema/"
                                       + name.replace("_generated.h", ".fbs"))

    # backends/xnnpack/CMakeLists.txt: flatc runs in EXECUTORCH_ROOT on the
    # schema's repository-relative path, into <include>/executorch/backends/
    # xnnpack/serialization, and MV_COMMAND renames the output.
    xnn_out = work / "out-xnn-schema"
    xnn_out.mkdir()
    run([flatc, *XNN_FLATC_ARGS, "-o", xnn_out, XNN_SCHEMA], cwd=work)
    produced = sorted(p.name for p in xnn_out.iterdir())
    if produced != ["runtime_schema_generated.h"]:
        raise Refusal(f"vendor_executorch: flatc on {XNN_SCHEMA} produced {produced}, not the "
                      f"runtime_schema_generated.h that backends/xnnpack/CMakeLists.txt renames")
    out[GEN_XNN_SCHEMA] = ((xnn_out / "runtime_schema_generated.h").read_bytes(),
                           f"generated: {FLATC_VERSION} {' '.join(XNN_FLATC_ARGS)} {XNN_SCHEMA}, "
                           f"renamed runtime_schema_generated.h -> schema_generated.h")

    ops_out = work / "out-ops"
    ops_out.mkdir()
    oplist = ops_out / "selected_operators.yaml"
    yaml = "kernels/portable/functions.yaml"
    run([py, "-B", "-m", "codegen.tools.gen_oplist", f"--output_path={oplist}",
         f"--ops_schema_yaml_path={yaml}"], cwd=work)
    run([py, "-B", "-m", "codegen.gen", "--source-path=codegen", f"--install-dir={ops_out}",
         f"--tags-path={torchgen_dir / 'packaged/ATen/native/tags.yaml'}",
         f"--aten-yaml-path={torchgen_dir / 'packaged/ATen/native/native_functions.yaml'}",
         f"--op-selection-yaml-path={oplist}", f"--functions-yaml-path={yaml}"], cwd=work)
    for name in GEN_OPS_FILES:
        tool = "codegen.tools.gen_oplist" if name.endswith(".yaml") else "codegen.gen"
        out[f"{GEN_OPS}/{name}"] = ((ops_out / name).read_bytes(),
                                    f"generated: python -m {tool} (torchgen {TORCH_VERSION}) {yaml}")
    return out


def scan_includes(text):
    """(quoted, name, conditions) per #include; conditions are the enclosing
    #if lines, include guards dropped."""
    lines = text.splitlines()
    stack, found = [], []
    for i, line in enumerate(lines):
        m = INCLUDE.match(line)
        if m:
            found.append((m.group(1) == '"', m.group(2).strip(), [c for c in stack if c]))
            continue
        d = DIRECTIVE.match(line)
        if not d:
            continue
        kind, rest = d.group(1), d.group(2).strip()
        if kind in ("if", "ifdef", "ifndef"):
            guard = False
            if kind == "ifndef":
                for nxt in lines[i + 1:]:
                    if nxt.strip():
                        guard = re.match(r"^\s*#\s*define\s+" + re.escape(rest.split()[0]) + r"\b", nxt) is not None
                        break
            stack.append(None if guard else f"#{kind} {rest}")
        elif kind in ("elif", "else") and stack:
            top = stack[-1] or ""
            stack[-1] = f"{top} -> #{kind} {rest}".strip()
        elif kind == "endif" and stack:
            stack.pop()
    return found


def resolve(includer, quoted, name, exists):
    """The executorch tree's resolution (R0's, unchanged so the tree
    regenerates byte for byte)."""
    if quoted:
        rel = os.path.normpath(os.path.join(os.path.dirname(includer), name)).replace(os.sep, "/")
        if exists(rel):
            return rel
    if name.startswith("executorch/") and exists(name[len("executorch/"):]):
        return name[len("executorch/"):]
    for root in (C10_ROOT, GEN_ROOT, FB_ROOT):
        if exists(f"{root}/{name}"):
            return f"{root}/{name}"
    return None


def resolve_in(includer, quoted, name, exists, roots):
    """Resolution in a third-party tree: quoted includes relative to the
    includer first (as the compiler does), then each include root in order."""
    if quoted:
        rel = os.path.normpath(os.path.join(os.path.dirname(includer), name)).replace(os.sep, "/")
        if exists(rel):
            return rel
    for root in roots:
        rel = os.path.normpath(os.path.join(root, name)).replace(os.sep, "/")
        if exists(rel):
            return rel
    return None


def classify(name):
    if name in STD_HEADERS:
        return "std"
    if name in PLATFORM_HEADERS or name.startswith(PLATFORM_PREFIXES):
        return "platform"
    return "unresolved"


def closure(tree, src, tracked, roots, seeds=()):
    """Add to `tree` every tracked file the #include lines of its sources
    reach, resolved with `roots`; a file reached that way is a header unless
    it is a source file, which is returned as {path: includer} (it is
    compiled as part of its includer, not on its own)."""
    exists = lambda rel: rel in tree or rel in tracked  # noqa: E731
    queue = [r for r in tree if r.endswith(SCAN_EXT)] + [s for s in seeds if s not in tree]
    seen = set(queue)
    pulled = {}
    while queue:
        rel = queue.pop()
        if rel not in tree:
            tree[rel] = ((src / rel).read_bytes(), "headers", rel)
        for quoted, name, _ in scan_includes(tree[rel][0].decode("utf-8", errors="replace")):
            hit = resolve_in(rel, quoted, name, exists, roots)
            if hit and hit not in seen:
                seen.add(hit)
                queue.append(hit)
                if hit.endswith(SOURCE_EXT):
                    pulled[hit] = rel
    return pulled


def cmake_add_library_srcs(text, target):
    """The sources of `add_library(<target> ...)` in a simple CMakeLists.txt."""
    m = re.search(r"add_library\(\s*" + re.escape(target) + r"\s+([^)]*)\)", text, re.I)
    if not m:
        raise Refusal(f"vendor_executorch: no add_library({target} ...) to read sources from")
    return m.group(1).split()


# ExecuTorch's settings that reach the XNNPACK backend and extension_threadpool
# (root CMakeLists.txt, tools/cmake/preset/default.cmake, release build).
ET_DEFINES = ["ET_LOG_ENABLED=0", "ET_MIN_LOG_LEVEL=Info", "ET_ENABLE_PROGRAM_VERIFICATION=0",
              "ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0", "C10_USING_CUSTOM_GENERATED_MACROS"]
ET_INCLUDES = ["..", C10_ROOT, GEN_ROOT]
ET_OPTIONS = ["-Wno-deprecated-declarations", "-fPIC"]


def et_build_groups(ns, threadpool_srcs, xnn_srcs, unbuilt):
    threadpool_usage = ["ET_USE_THREADPOOL", "EXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES"]
    return {
        "schema": "torchnative-build-groups/1",
        "tree": "executorch",
        "upstream": {"repository": "https://github.com/pytorch/executorch", "tag": TAG, "commit": COMMIT},
        "generated_by": "scripts/vendor/vendor_executorch.py",
        "derived_from": ["CMakeLists.txt (root, not vendored)", "tools/cmake/preset/default.cmake (not vendored)",
                         "backends/xnnpack/CMakeLists.txt", "backends/xnnpack/cmake/Dependencies.cmake",
                         THREADPOOL_CMAKE, "shim_et/xplat/executorch/build/build_variables.bzl (not vendored)"],
        "evidence": "read from CMake by hand, not measured by running CMake",
        "targets": TARGETS,
        "build_type": {"name": "Release", "flags": ["-O3"], "defines": ["NDEBUG"],
                       "note": "CMake's default CMAKE_CXX_FLAGS_RELEASE for Clang, before target options"},
        "configurable": {
            "ET_LOG_ENABLED=0": "default.cmake turns EXECUTORCH_ENABLE_LOGGING on only for Debug; "
                                "tools/cmake/preset/android.cmake turns it on (then no ET_LOG_ENABLED=0)",
            "ET_MIN_LOG_LEVEL=Info": "EXECUTORCH_LOG_LEVEL, default Info",
            "ET_ENABLE_PROGRAM_VERIFICATION=0": "EXECUTORCH_ENABLE_PROGRAM_VERIFICATION, on only for Debug",
            "ENABLE_XNNPACK_SHARED_WORKSPACE": "EXECUTORCH_XNNPACK_SHARED_WORKSPACE, default ON",
            "ENABLE_XNNPACK_WEIGHTS_CACHE": "EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE: default.cmake OFF (not "
                                            "defined here); the macos.cmake (via apple_common.cmake) and "
                                            "android.cmake presets turn it ON",
            "ENABLE_XNNPACK_KLEIDI": "EXECUTORCH_XNNPACK_ENABLE_KLEIDI: default ON upstream, OFF here "
                                     "(leader decision 2026-11-14), so not defined",
            "EXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES": "extension/threadpool/CMakeLists.txt's default "
                                                           "when EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES is off",
        },
        "groups": [
            {"name": "xnnpack_backend", "cmake_target": "xnnpack_backend", "files": sorted(xnn_srcs),
             "std": {"c++": "c++17"}, "flags": ET_OPTIONS,
             "flags_by_target": {"aarch64-linux-android": ["-pthread"]},
             "defines": ET_DEFINES + ["ENABLE_XNNPACK_SHARED_WORKSPACE", *threadpool_usage,
                                      "XNN_LOG_LEVEL=0", "CPUINFO_SUPPORTED_PLATFORM=1"],
             "include_dirs": ET_INCLUDES + ["../flatbuffers/include", "../xnnpack/include",
                                            "../pthreadpool/include", "../cpuinfo/include"],
             "link": "whole-archive: upstream links it with executorch_target_link_options_shared_lib "
                     "(XNNPACKBackend.cpp registers the backend from a static initializer)"},
            {"name": "extension_threadpool", "cmake_target": "extension_threadpool",
             "files": [f"extension/threadpool/{s}" for s in threadpool_srcs],
             "std": {"c++": "c++17"}, "flags": ET_OPTIONS,
             "flags_by_target": {"aarch64-linux-android": ["-pthread"]},
             "defines": ET_DEFINES + threadpool_usage + ["CPUINFO_SUPPORTED_PLATFORM=1"],
             "include_dirs": ET_INCLUDES + ["../cpuinfo/include", "../pthreadpool/include"]},
        ],
        "interface": {"extension_threadpool": {"defines": threadpool_usage}},
        "unbuilt": unbuilt,
        "not_described": {"manifest_groups": ["core", "prim_ops", "portable", "generated"],
                          "reason": "the runtime core, prim ops and portable kernels are step R1 of #312; "
                                    "this file describes only what #319 added"},
    }


def build_groups_bytes(desc):
    return (json.dumps(desc, indent=1) + "\n").encode()


def select(src, tracked, generated):
    """Return {rel: (bytes, group, upstream)} for the whole tree."""
    ns = build_variables(src)
    xnn_built = sorted(ns["XNNPACK_BACKEND_SRCS"])
    xnn_all = sorted(t for t in tracked if t.startswith(XNN_RUNTIME) and t.endswith(SOURCE_EXT)
                     and "/test/" not in t)
    missing = set(xnn_built) - set(xnn_all)
    if missing:
        raise Refusal(f"vendor_executorch: XNNPACK_BACKEND_SRCS not under {XNN_RUNTIME}: {sorted(missing)}")
    tp_cmake = (src / THREADPOOL_CMAKE).read_text()
    tp_srcs = cmake_add_library_srcs(tp_cmake, "extension_threadpool")
    bzl_tp = [p.split("/")[-1] for p in ns["EXTENSION_THREADPOOL_SRCS"]]
    if set(bzl_tp) - set(tp_srcs):
        raise Refusal(f"vendor_executorch: {THREADPOOL_CMAKE} compiles {tp_srcs}, which lacks part of the "
                      f"bzl EXTENSION_THREADPOOL_SRCS {bzl_tp}")
    groups = {
        "core": sorted(ns["EXECUTORCH_CORE_SRCS"]) + PALS,
        "prim_ops": sorted(ns["EXECUTORCH_SRCS"]),
        "portable": sorted(t for t in tracked if t.startswith("kernels/portable/cpu/")
                           and t.endswith(".cpp") and "/test/" not in t),
        "xnnpack_backend": xnn_all,
        "threadpool": [f"extension/threadpool/{s}" for s in tp_srcs],
        "inputs": INPUTS + XNN_INPUTS,
        "license": ["LICENSE"],
    }
    missing_utils = set(ns["KERNELS_UTIL_ALL_DEPS_SRCS"]) - set(groups["portable"])
    if missing_utils:
        raise Refusal(f"vendor_executorch: KERNELS_UTIL_ALL_DEPS_SRCS not under the portable glob: "
                      f"{sorted(missing_utils)}")
    tree = {}
    for group, files in groups.items():
        for rel in files:
            if rel not in tracked:
                raise Refusal(f"vendor_executorch: {rel} ({group}) is not tracked at {TAG}")
            tree[rel] = ((src / rel).read_bytes(), group, rel)
    for rel, (data, upstream) in generated.items():
        tree[rel] = (data, "generated", upstream)

    seeds = HEADER_SEEDS + sorted(t for t in tracked if t.startswith(XNN_RUNTIME) and t.endswith(".h")
                                  and "/test/" not in t)
    exists = lambda rel: rel in tree or rel in tracked  # noqa: E731
    queue = [r for r in tree if r.endswith((".cpp", ".h"))] + seeds
    seen = set(queue)
    while queue:
        rel = queue.pop()
        if rel not in tree:
            tree[rel] = ((src / rel).read_bytes(), "headers", rel)
        for quoted, name, _ in scan_includes(tree[rel][0].decode("utf-8", errors="replace")):
            hit = resolve(rel, quoted, name, exists)
            if hit and hit not in seen:
                seen.add(hit)
                queue.append(hit)
    unbuilt = {rel: "not in XNNPACK_BACKEND_SRCS (build_variables.bzl), the list backends/xnnpack/"
                    "CMakeLists.txt compiles; vendored because all of backends/xnnpack/runtime/ is"
               for rel in xnn_all if rel not in xnn_built}
    desc = et_build_groups(ns, tp_srcs, xnn_built, unbuilt)
    tree[BUILD_GROUPS] = (build_groups_bytes(desc), "build", "torchnative")
    tree["README.md"] = (README.encode(), "meta", "torchnative")
    tree[".gitattributes"] = (GITATTRIBUTES.encode(), "meta", "torchnative")
    return tree


# ---------------------------------------------------------------------------
# A CMake evaluator for the subset XNNPACK's CMakeLists.txt uses.
#
# It is not CMake. It evaluates, in order: if/elseif/else/endif (NOT, AND, OR,
# MATCHES, STREQUAL, VERSION_LESS[_EQUAL], EQUAL, LESS[_EQUAL], GREATER,
# DEFINED, TARGET, POLICY, variables and constants), set (normal and CACHE),
# option, list(APPEND|SORT|LENGTH), include of cmake/gen files, the directory
# commands add_compile_definitions, add_compile_options, include_directories,
# the target commands add_library, target_link_libraries,
# target_compile_options, target_compile_definitions,
# target_include_directories, set_target_properties, and
# set_property(SOURCE ... COMPILE_FLAGS), with the generator expressions
# $<BOOL:>, $<CONFIG:>, $<NOT:>, $<0:>, $<1:>, $<BUILD_INTERFACE:> and
# $<INSTALL_INTERFACE:>. Any other command that is active in the branch the
# configuration selects refuses by name, so nothing it does not understand is
# skipped silently. Commands with no effect on what is compiled (message,
# project, install, find_program, string, ...) are ignored by name.
# ---------------------------------------------------------------------------

CMAKE_IGNORED = {"CMAKE_MINIMUM_REQUIRED", "CMAKE_POLICY", "PROJECT", "MESSAGE", "FIND_PROGRAM", "STRING",
                 "ENABLE_TESTING", "INSTALL", "FIND_PACKAGE", "CONFIGURE_FILE", "FILE", "ADD_CUSTOM_COMMAND"}
TRUE_CONST = {"1", "ON", "YES", "TRUE", "Y"}
FALSE_CONST = {"0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND", ""}
BINARY = "<binary>"


def cmake_parse(text, where):
    """[(NAME, [(quoted, text)])] for every command in a CMake file."""
    cmds, i, n = [], 0, len(text)
    ident = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        m = ident.match(text, i)
        if not m:
            raise Refusal(f"vendor_executorch: cannot parse {where} at offset {i}: {text[i:i + 40]!r}")
        name, i = m.group(0).upper(), m.end()
        while text[i] in " \t":
            i += 1
        if text[i] != "(":
            raise Refusal(f"vendor_executorch: cannot parse {where}: {name} without '('")
        i += 1
        args, depth = [], 0
        while True:
            c = text[i]
            if c.isspace():
                i += 1
            elif c == "#":
                i = text.find("\n", i)
            elif c == '"':
                j, buf = i + 1, []
                while text[j] != '"':
                    if text[j] == "\\" and text[j + 1] in '"\\':
                        buf.append(text[j + 1])
                        j += 2
                        continue
                    buf.append(text[j])
                    j += 1
                args.append((True, "".join(buf)))
                i = j + 1
            elif c == "(":
                depth += 1
                args.append((False, "("))
                i += 1
            elif c == ")":
                i += 1
                if depth == 0:
                    break
                depth -= 1
                args.append((False, ")"))
            else:
                j = i
                while text[j] not in ' \t\r\n()"#':
                    j += 1
                args.append((False, text[i:j]))
                i = j
        cmds.append((name, args))
    return cmds


def version_tuple(s):
    return tuple(int(x) for x in re.findall(r"\d+", s))


class CMakeEval:
    """One evaluation of XNNPACK's CMakeLists.txt for one target."""

    def __init__(self, src, gen_dir, variables, cache, external_targets, config="Release"):
        self.src, self.gen_dir, self.config = src, gen_dir, config
        self.vars = {k: list(v) for k, v in variables.items()}
        self.cache = {k: list(v) for k, v in cache.items()}
        self.external = external_targets
        self.undefined_read = set()
        self.dir_defines, self.dir_options, self.dir_includes = [], [], []
        self.targets = {}
        self.src_flags = {}
        self.link_libraries = []
        self.files_read = []

    # -- values -----------------------------------------------------------
    def defined(self, name):
        return name in self.vars or name in self.cache

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if name in self.cache:
            return self.cache[name]
        self.undefined_read.add(name)
        return []

    def expand(self, s):
        pat = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
        while True:
            m = pat.search(s)
            if not m:
                return s
            s = s[:m.start()] + ";".join(self.get(m.group(1))) + s[m.end():]

    def values(self, raw):
        out = []
        for quoted, a in raw:
            v = self.expand(a)
            if quoted:
                out.append(v)
            else:
                out.extend(x for x in v.split(";") if x != "")
        return out

    def genex(self, s):
        while "$<" in s:
            i = s.rfind("$<")
            j = s.find(">", i)
            body = s[i + 2:j]
            head, _, arg = body.partition(":")
            if head == "BOOL":
                val = "1" if self.truth_const(arg) else "0"
            elif head == "CONFIG":
                val = "1" if arg == self.config else "0"
            elif head == "NOT":
                val = "0" if arg == "1" else "1"
            elif head == "1":
                val = arg
            elif head == "0":
                val = ""
            elif head == "BUILD_INTERFACE":
                val = arg
            elif head == "INSTALL_INTERFACE":
                val = ""
            else:
                raise Refusal(f"vendor_executorch: the CMake evaluator has no generator expression $<{body}>")
            s = s[:i] + val + s[j + 1:]
        return s

    @staticmethod
    def truth_const(s):
        u = s.upper()
        if u in TRUE_CONST:
            return True
        if u in FALSE_CONST or u.endswith("-NOTFOUND"):
            return False
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s) != 0
        return None

    # -- conditions -------------------------------------------------------
    def operand(self, tok):
        quoted, text = tok
        text = self.expand(text)
        if not quoted and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text) and self.defined(text):
            return ";".join(self.get(text))
        return text

    def truth(self, tok):
        quoted, text = tok
        text = self.expand(text)
        const = self.truth_const(text)
        if const is not None:
            return const
        if quoted:
            return False
        if self.defined(text):
            return bool(self.truth_const(";".join(self.get(text))) is not False and self.get(text))
        self.undefined_read.add(text)
        return False

    def condition(self, toks):
        pos = [0]
        binops = {"MATCHES", "STREQUAL", "VERSION_LESS", "VERSION_LESS_EQUAL", "VERSION_GREATER",
                  "VERSION_GREATER_EQUAL", "EQUAL", "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL"}

        def peek():
            return toks[pos[0]] if pos[0] < len(toks) else None

        def take():
            pos[0] += 1
            return toks[pos[0] - 1]

        def is_kw(tok, *kw):
            return tok is not None and not tok[0] and tok[1] in kw

        def primary():
            tok = take()
            if is_kw(tok, "("):
                v = disj()
                if not is_kw(take(), ")"):
                    raise Refusal(f"vendor_executorch: unbalanced condition {toks}")
                return v
            if is_kw(tok, "DEFINED"):
                return self.defined(take()[1])
            if is_kw(tok, "TARGET"):
                name = self.expand(take()[1])
                return name in self.targets or name in self.external
            if is_kw(tok, "POLICY"):
                take()
                return True
            if is_kw(tok, "EXISTS", "COMMAND", "IS_DIRECTORY"):
                raise Refusal(f"vendor_executorch: the CMake evaluator has no condition {tok[1]}")
            nxt = peek()
            if is_kw(nxt, *binops):
                op = take()[1]
                left, right = self.operand(tok), self.operand(take())
                if op == "MATCHES":
                    return re.search(right, left) is not None
                if op == "STREQUAL":
                    return left == right
                if op.startswith("VERSION_"):
                    a, b = version_tuple(left), version_tuple(right)
                    return {"VERSION_LESS": a < b, "VERSION_LESS_EQUAL": a <= b,
                            "VERSION_GREATER": a > b, "VERSION_GREATER_EQUAL": a >= b}[op]
                a, b = float(left or "nan"), float(right or "nan")
                return {"EQUAL": a == b, "LESS": a < b, "LESS_EQUAL": a <= b, "GREATER": a > b,
                        "GREATER_EQUAL": a >= b}[op]
            return self.truth(tok)

        def negation():
            if is_kw(peek(), "NOT"):
                take()
                return not negation()
            return primary()

        def conj():
            v = negation()
            while is_kw(peek(), "AND"):
                take()
                v = negation() and v
            return v

        def disj():
            v = conj()
            while is_kw(peek(), "OR"):
                take()
                v = conj() or v
            return v

        v = disj()
        if pos[0] != len(toks):
            raise Refusal(f"vendor_executorch: the CMake evaluator could not read condition {toks}")
        return v

    # -- commands ---------------------------------------------------------
    def run_file(self, rel):
        path = self.gen_dir / rel[len(BINARY) + 1:] if rel.startswith(BINARY) else self.src / rel
        self.files_read.append(rel)
        self.run(cmake_parse(path.read_text(encoding="utf-8"), rel), rel)

    def run(self, cmds, where):
        stack = []  # (parent_active, taken, active)
        active = True
        for name, raw in cmds:
            if name == "IF":
                taken = active and self.condition(raw)
                stack.append((active, taken))
                active = taken
                continue
            if name == "ELSEIF":
                parent, taken = stack[-1]
                now = parent and not taken and self.condition(raw)
                stack[-1] = (parent, taken or now)
                active = now
                continue
            if name == "ELSE":
                parent, taken = stack[-1]
                active = parent and not taken
                stack[-1] = (parent, True)
                continue
            if name == "ENDIF":
                active = stack.pop()[0]
                continue
            if active:
                self.command(name, raw, where)
        if stack:
            raise Refusal(f"vendor_executorch: unbalanced if/endif in {where}")

    def target(self, name):
        if name not in self.targets:
            raise Refusal(f"vendor_executorch: CMake target {name} is used before add_library")
        return self.targets[name]

    def scoped(self, args, start):
        """[(scope, value)] for `target_*(<t> [BEFORE] SCOPE v... SCOPE v...)`."""
        out, scope = [], None
        for a in args[start:]:
            if a in ("PRIVATE", "PUBLIC", "INTERFACE"):
                scope = a
            elif a == "BEFORE":
                continue
            else:
                out.append((scope or "PUBLIC", a))
        return out

    def command(self, name, raw, where):
        args = self.values(raw)
        if name == "MESSAGE" and args and args[0] == "FATAL_ERROR":
            raise Refusal(f"vendor_executorch: {where} stops this configuration: {' '.join(args[1:])}")
        if name in CMAKE_IGNORED:
            return
        if name == "SET":
            var, rest = args[0], args[1:]
            if "PARENT_SCOPE" in rest:
                raise Refusal(f"vendor_executorch: set(... PARENT_SCOPE) in {where}")
            if "CACHE" in rest:
                k = rest.index("CACHE")
                if var not in self.cache or "FORCE" in rest:
                    self.cache[var] = rest[:k]
            elif rest:
                self.vars[var] = rest
            else:
                self.vars.pop(var, None)
        elif name == "OPTION":
            if not self.defined(args[0]):
                self.cache[args[0]] = [args[2] if len(args) > 2 else "OFF"]
        elif name == "SET_PROPERTY" and args[0] == "CACHE":
            return
        elif name == "LIST":
            op, var = args[0], args[1]
            if op == "APPEND":
                self.vars[var] = self.get(var) + args[2:]
            elif op == "SORT":
                self.vars[var] = sorted(self.get(var))
            elif op == "LENGTH":
                self.vars[args[2]] = [str(len(self.get(var)))]
            else:
                raise Refusal(f"vendor_executorch: the CMake evaluator has no list({op})")
        elif name == "INCLUDE":
            if args[0].endswith(".cmake"):
                self.run_file(args[0])
            elif args[0] != "GNUInstallDirs":
                raise Refusal(f"vendor_executorch: include({args[0]}) in {where}")
        elif name == "EXECUTE_PROCESS":
            if not any("update-microkernels.py" in a for a in args):
                raise Refusal(f"vendor_executorch: an active execute_process in {where}: {args}")
            # The script ran it at vendor time (see xnnpack_microkernel_check).
            self.vars[args[args.index("RESULT_VARIABLE") + 1]] = ["0"]
        elif name == "FIND_LIBRARY":
            self.vars[args[0]] = [f"lib{args[1]}"]
        elif name in ("ADD_COMPILE_DEFINITIONS", "ADD_COMPILE_OPTIONS"):
            if self.targets:
                raise Refusal(f"vendor_executorch: {name} after a target was created in {where}; the "
                              f"evaluator applies directory properties to every target")
            dest = self.dir_defines if name == "ADD_COMPILE_DEFINITIONS" else self.dir_options
            dest += [self.genex(a) for a in args]
        elif name == "INCLUDE_DIRECTORIES":
            self.dir_includes += [a for a in args if a != "SYSTEM"]
        elif name == "ADD_LIBRARY":
            tname, rest = args[0], args[1:]
            kind = rest[0] if rest and rest[0] in ("STATIC", "SHARED", "OBJECT", "INTERFACE", "MODULE") else "DEFAULT"
            if "IMPORTED" in rest or "ALIAS" in rest:
                raise Refusal(f"vendor_executorch: an active imported or alias add_library({tname}) in {where}")
            srcs = rest[1:] if kind != "DEFAULT" else rest
            self.targets[tname] = dict(kind=kind, srcs=srcs, links=[], options=[], defines=[], includes=[],
                                       props={}, c_standard=self.get("CMAKE_C_STANDARD"),
                                       c_extensions=self.get("CMAKE_C_EXTENSIONS"),
                                       cxx_standard=self.get("CMAKE_CXX_STANDARD"),
                                       cxx_extensions=self.get("CMAKE_CXX_EXTENSIONS"))
        elif name == "TARGET_LINK_LIBRARIES":
            t = args[0]
            for scope, lib in self.scoped(args, 1):
                if lib.startswith("lib"):
                    self.link_libraries.append(lib[3:])
                    continue
                if lib in self.targets or lib in self.external:
                    if t in self.targets:
                        self.target(t)["links"].append((scope, lib))
                    elif t in self.external:
                        raise Refusal(f"vendor_executorch: {where} links into the external target {t}")
                else:
                    raise Refusal(f"vendor_executorch: {where}: {t} links {lib}, which is no target")
        elif name in ("TARGET_COMPILE_OPTIONS", "TARGET_COMPILE_DEFINITIONS", "TARGET_INCLUDE_DIRECTORIES"):
            key = {"TARGET_COMPILE_OPTIONS": "options", "TARGET_COMPILE_DEFINITIONS": "defines",
                   "TARGET_INCLUDE_DIRECTORIES": "includes"}[name]
            for scope, v in self.scoped(args, 1):
                v = self.genex(v)
                if v:
                    self.target(args[0])[key].append((scope, v))
        elif name == "SET_TARGET_PROPERTIES":
            k = args.index("PROPERTIES")
            for t in args[:k]:
                props = args[k + 1:]
                for p, v in zip(props[::2], props[1::2]):
                    self.target(t)["props"][p] = v
        elif name == "SET_PROPERTY" and args[0] == "SOURCE":
            k = args.index("PROPERTY")
            mode = args[k - 1] if args[k - 1] in ("APPEND_STRING", "APPEND") else "SET"
            srcs = [s for s in args[1:k - (mode != "SET")] if s != "+"]
            prop, value = args[k + 1], " ".join(args[k + 2:])
            if prop != "COMPILE_FLAGS":
                raise Refusal(f"vendor_executorch: set_property(SOURCE ... {prop}) in {where}")
            for s in srcs:
                self.src_flags[s] = (self.src_flags.get(s, "") + value) if mode == "APPEND_STRING" else value
        elif name == "SET_PROPERTY" and args[0] == "TARGET":
            if args[args.index("PROPERTY") + 1] != "PUBLIC_HEADER":
                raise Refusal(f"vendor_executorch: set_property(TARGET ...) in {where}: {args}")
        else:
            raise Refusal(f"vendor_executorch: the CMake evaluator meets an active {name}({' '.join(args)}) "
                          f"in {where} and does not know what it does; refusing rather than skipping it")

    # -- usage requirements ----------------------------------------------
    def usage(self, name, seen=None):
        """(defines, options, includes) target `name` gives its consumers."""
        seen = seen or set()
        if name in seen:
            return [], [], []
        seen.add(name)
        if name in self.external:
            e = self.external[name]
            return list(e.get("defines", [])), list(e.get("options", [])), list(e.get("includes", []))
        t = self.targets[name]
        d = [v for s, v in t["defines"] if s != "PRIVATE"]
        o = [v for s, v in t["options"] if s != "PRIVATE"]
        i = [v for s, v in t["includes"] if s != "PRIVATE"]
        for scope, dep in t["links"]:
            if scope != "PRIVATE":
                dd, oo, ii = self.usage(dep, seen)
                d, o, i = d + dd, o + oo, i + ii
        return d, o, i

    def compile_settings(self, name):
        t = self.targets[name]
        d = list(self.dir_defines) + [v for s, v in t["defines"] if s != "INTERFACE"]
        o = list(self.dir_options) + [v for s, v in t["options"] if s != "INTERFACE"]
        i = list(self.dir_includes) + [v for s, v in t["includes"] if s != "INTERFACE"]
        for scope, dep in t["links"]:
            if scope != "INTERFACE":
                dd, oo, ii = self.usage(dep)
                d, o, i = d + dd, o + oo, i + ii
        dedupe = lambda xs: list(dict.fromkeys(xs))  # noqa: E731  (CMake de-duplicates, first wins)
        c_ext = t["props"].get("C_EXTENSIONS", (t["c_extensions"] or ["OFF"])[0])
        std = {}
        if t["c_standard"]:
            std["c"] = ("gnu" if self.truth_const(c_ext) else "c") + t["c_standard"][0]
        if t["cxx_standard"]:
            cxx_ext = t["props"].get("CXX_EXTENSIONS", (t["cxx_extensions"] or ["OFF"])[0])
            std["c++"] = ("gnu++" if self.truth_const(cxx_ext) else "c++") + t["cxx_standard"][0]
        return dedupe(d), dedupe(o), dedupe(i), std


def parse_cmake_lists(text, where):
    """{VAR: [items]} for every SET(VAR items...) in a cmake/gen file."""
    out = {}
    for name, raw in cmake_parse(text, where):
        if name == "SET":
            out[raw[0][1]] = [t for _, t in raw[1:]]
    return out


# What ExecuTorch's CMake sets before add_subdirectory(XNNPACK), and the
# leader's overrides. Citations are upstream files at v1.4.1.
XNN_SEED_CACHE = {
    "XNNPACK_LIBRARY_TYPE": (["static"], "backends/xnnpack/cmake/Dependencies.cmake"),
    "XNNPACK_BUILD_BENCHMARKS": (["OFF"], "backends/xnnpack/cmake/Dependencies.cmake"),
    "XNNPACK_BUILD_TESTS": (["OFF"], "backends/xnnpack/cmake/Dependencies.cmake"),
    "XNNPACK_ENABLE_AVXVNNI": (["OFF"], "backends/xnnpack/cmake/Dependencies.cmake"),
    "XNNPACK_ENABLE_AVX512VNNIGFNI": (["OFF"], "backends/xnnpack/cmake/Dependencies.cmake"),
    "XNNPACK_ENABLE_ARM_SME2": (["OFF"], "leader decision 2026-11-14 (ExecuTorch's Dependencies.cmake sets ON)"),
    "XNNPACK_ENABLE_KLEIDIAI": (["OFF"], "leader decision 2026-11-14 (EXECUTORCH_XNNPACK_ENABLE_KLEIDI OFF; "
                                         "ExecuTorch's default.cmake has it ON)"),
    "XNNPACK_BUILD_ALL_MICROKERNELS": (["OFF"], "backends/xnnpack/cmake/Dependencies.cmake"),
}
XNN_SEED_VARS = {
    "CMAKE_POSITION_INDEPENDENT_CODE": (["ON"], "backends/xnnpack/cmake/Dependencies.cmake"),
    "CPUINFO_SOURCE_DIR": (["../cpuinfo"], "root CMakeLists.txt (EXECUTORCH_BUILD_CPUINFO)"),
    "FXDIV_SOURCE_DIR": (["../fxdiv"], "root CMakeLists.txt"),
    "PTHREADPOOL_SOURCE_DIR": (["../pthreadpool"], "root CMakeLists.txt (EXECUTORCH_BUILD_PTHREADPOOL)"),
    "PROJECT_BINARY_DIR": ([BINARY], "the CMake binary directory: build_identifier.c and microkernels.cmake"),
    "PROJECT_SOURCE_DIR": (["."], "XNNPACK's source directory"),
    "CMAKE_SIZEOF_VOID_P": (["8"], "aarch64"),
    "CMAKE_GENERATOR": (["Ninja"], "ExecuTorch's documented generator"),
    "Python_EXECUTABLE": (["python3"], "any Python; only update-microkernels.py and "
                                       "generate-build-identifier.py use it, both run at vendor time"),
}
XNN_TARGET_VARS = {
    "aarch64-apple-darwin": {"CMAKE_SYSTEM_NAME": ["Darwin"], "CMAKE_SYSTEM_PROCESSOR": ["arm64"],
                             "APPLE": ["1"], "CMAKE_C_COMPILER_ID": ["AppleClang"],
                             "CMAKE_CXX_COMPILER_ID": ["AppleClang"], "CMAKE_C_COMPILER_VERSION": ["17.0.0"]},
    "aarch64-linux-android": {"CMAKE_SYSTEM_NAME": ["Android"], "CMAKE_SYSTEM_PROCESSOR": ["aarch64"],
                              "ANDROID": ["1"], "ANDROID_NDK_MAJOR": ["27"], "CMAKE_C_COMPILER_ID": ["Clang"],
                              "CMAKE_CXX_COMPILER_ID": ["Clang"], "CMAKE_C_COMPILER_VERSION": ["18.0.2"]},
}
# Usage requirements of the targets ExecuTorch adds before XNNPACK (read from
# their CMakeLists.txt at these pins; see cpuinfo_desc and pthreadpool_desc).
XNN_EXTERNAL = {
    "aarch64-apple-darwin": {
        "cpuinfo": {"includes": ["../cpuinfo/include"], "defines": ["CPUINFO_SUPPORTED_PLATFORM=1"]},
        "pthreadpool": {"includes": ["../pthreadpool/include"]},
        "fxdiv": {"includes": ["../fxdiv/include"], "defines": ["FXDIV_USE_INLINE_ASSEMBLY=0"]},
    },
    "aarch64-linux-android": {
        "cpuinfo": {"includes": ["../cpuinfo/include"], "defines": ["CPUINFO_SUPPORTED_PLATFORM=1"]},
        "pthreadpool": {"includes": ["../pthreadpool/include"], "options": ["-pthread"]},
        "fxdiv": {"includes": ["../fxdiv/include"], "defines": ["FXDIV_USE_INLINE_ASSEMBLY=0"]},
    },
}
XNN_LANG_FLAGS = ["CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_ASM_FLAGS", "CMAKE_SHARED_LINKER_FLAGS",
                  "CMAKE_EXE_LINKER_FLAGS"]
XNN_ISA_PREFIX = "PROD_"
XNN_ISA_SUFFIX = "_MICROKERNEL_SRCS"
GEN_BUILD_ID = "generated/build_identifier.c"


def xnnpack_microkernel_check(src, work):
    """Re-derive cmake/gen/*.cmake with upstream's tools/update-microkernels.py
    (what XNNPACK's CMake runs at configure time) and refuse if any generated
    file differs from the checked-in one. Returns the generated directory."""
    out = work / "update-microkernels"
    out.mkdir()
    run([sys.executable, "-B", src / "tools" / "update-microkernels.py", "--root_dir", src, "--output", out],
        cwd=work)
    gen = out / "cmake" / "gen"
    for p in sorted(gen.iterdir()):
        mine = src / "cmake" / "gen" / p.name
        if not mine.is_file() or mine.read_bytes() != p.read_bytes():
            raise Refusal(f"vendor_executorch: refusing: upstream's tools/update-microkernels.py produces a "
                          f"cmake/gen/{p.name} that differs from the checked-in one; the lists this script "
                          f"reads would not be the ones XNNPACK's CMake uses")
    return out


def xnnpack_evaluate(src, gen_out, triple):
    variables = {k: v for k, (v, _) in XNN_SEED_VARS.items()}
    variables.update(XNN_TARGET_VARS[triple])
    for k in XNN_LANG_FLAGS:
        variables[k] = []
    cache = {k: v for k, (v, _) in XNN_SEED_CACHE.items()}
    ev = CMakeEval(src, gen_out, variables, cache, XNN_EXTERNAL[triple])
    ev.run_file("CMakeLists.txt")
    return ev


def xnnpack_groups(ev, src):
    """Build groups for one evaluated target: every object that lands in the
    two archives ExecuTorch links (XNNPACK and xnnpack-microkernels-prod)."""
    archives = {}
    for lib in ("XNNPACK", "xnnpack-microkernels-prod"):
        t = ev.targets[lib]
        members = [lib] + [dep for _, dep in t["links"]
                           if dep in ev.targets and ev.targets[dep]["kind"] == "OBJECT"]
        archives[lib] = list(dict.fromkeys(members))
    isa_lists = {}
    for var, items in ev.vars.items():
        if var.startswith(XNN_ISA_PREFIX) and var.endswith(XNN_ISA_SUFFIX) and var != "PROD_MICROKERNEL_SRCS":
            for f in items:
                isa_lists.setdefault(f, []).append(var[len(XNN_ISA_PREFIX):-len(XNN_ISA_SUFFIX)].lower())
    prod = ev.get("PROD_MICROKERNEL_SRCS")
    groups, seen = [], {}
    for archive, members in archives.items():
        for tname in members:
            defines, options, includes, std = ev.compile_settings(tname)
            srcs = [GEN_BUILD_ID if s == f"{BINARY}/build_identifier.c" else s for s in ev.targets[tname]["srcs"]]
            buckets = {}
            for s in srcs:
                if s in prod:
                    isas = isa_lists.get(s, [])
                    if len(isas) != 1:
                        raise Refusal(f"vendor_executorch: prod microkernel {s} is in {len(isas)} ISA lists {isas}")
                    key = f"{tname}/{isas[0]}"
                elif tname == "xnnpack-microkernels-prod":
                    key = f"{tname}/tables"
                else:
                    key = tname
                flags = tuple(ev.src_flags.get(s, "").split())
                prev = buckets.setdefault(key, (flags, []))
                if prev[0] != flags:
                    raise Refusal(f"vendor_executorch: {s} has COMPILE_FLAGS {flags}, unlike the rest of {key} "
                                  f"{prev[0]}")
                prev[1].append(s)
            for key, (flags, files) in buckets.items():
                for f in files:
                    if f in seen:
                        raise Refusal(f"vendor_executorch: {f} is compiled by both {seen[f]} and {key}")
                    seen[f] = key
                langs = {"c" if f.endswith(".c") else "c++" if f.endswith(".cc") else "asm" for f in files}
                groups.append({"name": key.replace("xnnpack-", "", 1) if key.startswith("xnnpack-") else key,
                               "cmake_target": tname, "archive": archive, "files": sorted(files),
                               "std": {k: v for k, v in std.items() if k in langs},
                               "flags": list(options) + list(flags), "defines": defines,
                               "include_dirs": includes})
    return groups, archives


def merge_by_target(per_target):
    """One list of groups from {triple: [groups]}: a list field that differs
    between targets keeps what all share in `field` and the rest in
    `field_by_target`."""
    triples = list(per_target)
    names = [g["name"] for g in per_target[triples[0]]]
    for t in triples[1:]:
        if [g["name"] for g in per_target[t]] != names:
            raise Refusal(f"vendor_executorch: the targets {triples} produce different group lists")
    merged = []
    for k, name in enumerate(names):
        gs = {t: per_target[t][k] for t in triples}
        out = {}
        for field, val in gs[triples[0]].items():
            vals = {t: gs[t][field] for t in triples}
            if all(v == val for v in vals.values()):
                out[field] = val
            elif isinstance(val, list) and field != "files":
                common = [x for x in val if all(x in v for v in vals.values())]
                out[field] = common
                out[f"{field}_by_target"] = {t: [x for x in v if x not in common] for t, v in vals.items()
                                             if [x for x in v if x not in common]}
            else:
                raise Refusal(f"vendor_executorch: group {name} differs between targets in {field}: {vals}")
        merged.append(out)
    return merged


def select_xnnpack(src, tracked, work):
    gen_out = xnnpack_microkernel_check(src, work)
    evs = {t: xnnpack_evaluate(src, gen_out, t) for t in TARGETS}
    per_target, archives = {}, {}
    for t, ev in evs.items():
        per_target[t], archives[t] = xnnpack_groups(ev, src)
    groups = merge_by_target(per_target)
    if len({json.dumps(a) for a in archives.values()}) != 1:
        raise Refusal(f"vendor_executorch: XNNPACK archive members differ between targets: {archives}")

    # build_identifier.c: what the CMake custom command generates from the
    # sorted prod list (FILE(GENERATE) writes it without a final newline).
    ev0 = evs[TARGETS[0]]
    prod_lists = {t: sorted(ev.get("PROD_MICROKERNEL_SRCS")) for t, ev in evs.items()}
    if len({tuple(v) for v in prod_lists.values()}) != 1:
        raise Refusal("vendor_executorch: the prod microkernel list differs between targets")
    prod = prod_lists[TARGETS[0]]
    listing = work / "prod_microkernel_srcs.txt"
    listing.write_text("\n".join(prod))
    run([sys.executable, "-B", "scripts/generate-build-identifier.py", "--output", work / "build_identifier.c",
         "--input_file_list", listing], cwd=src)

    tree = {}
    for g in groups:
        for rel in g["files"]:
            if rel == GEN_BUILD_ID:
                continue
            if rel not in tracked:
                raise Refusal(f"vendor_executorch: {rel} (group {g['name']}) is not tracked in XNNPACK")
            kind = "microkernels" if (rel in prod or g["name"] == "microkernels-prod/tables") else "sources"
            tree[rel] = ((src / rel).read_bytes(), kind, rel)
    tree[GEN_BUILD_ID] = ((work / "build_identifier.c").read_bytes(), "generated",
                          "generated: python scripts/generate-build-identifier.py over the sorted "
                          "PROD_MICROKERNEL_SRCS (CMakeLists.txt's custom command)")
    inputs = ["CMakeLists.txt", "tools/update-microkernels.py", "scripts/generate-build-identifier.py"]
    inputs += sorted(f for f in ev0.files_read if not f.startswith(BINARY) and f != "CMakeLists.txt")
    for rel in inputs:
        tree[rel] = ((src / rel).read_bytes(), "inputs", rel)
    tree["LICENSE"] = ((src / "LICENSE").read_bytes(), "license", "LICENSE")
    pulled = closure(tree, src, tracked, TREE_ROOTS["xnnpack"])
    for rel in pulled:
        tree[rel] = (tree[rel][0], "headers", rel)
    unbuilt = {rel: f"not compiled on its own: #include'd by {who}" for rel, who in sorted(pulled.items())}

    built_targets = [t for a in archives[TARGETS[0]].values() for t in a]
    other = sorted(t for t in ev0.targets if t not in built_targets and ev0.targets[t]["kind"] != "INTERFACE")
    lang = {t: {k[len("CMAKE_"):-len("_FLAGS")].lower().replace("cxx", "c++"): ev.get(k)[0].split()
                for k in XNN_LANG_FLAGS if ev.get(k) and ev.get(k)[0].strip()} for t, ev in evs.items()}
    desc = {
        "schema": "torchnative-build-groups/1",
        "tree": "xnnpack",
        "upstream": {"repository": THIRD_PARTY["xnnpack"]["url"], "commit": THIRD_PARTY["xnnpack"]["commit"],
                     "pinned_by": f"ExecuTorch {TAG} backends/xnnpack/third-party/XNNPACK"},
        "generated_by": "scripts/vendor/vendor_executorch.py (its CMake evaluator over CMakeLists.txt)",
        "evidence": "evaluated from upstream CMake by a subset evaluator, not measured by running CMake",
        "targets": TARGETS,
        "seed": {"cache": {k: {"value": v, "from": w} for k, (v, w) in XNN_SEED_CACHE.items()},
                 "variables": {k: {"value": v, "from": w} for k, (v, w) in XNN_SEED_VARS.items()},
                 "by_target": XNN_TARGET_VARS, "external_targets": XNN_EXTERNAL},
        "configuration": {t: {k: ev.get(k) for k in sorted(set(ev.cache) | set(ev.vars))
                              if k.startswith("XNNPACK_ENABLE_") or k in ("XNNPACK_TARGET_PROCESSOR",
                                                                          "XNNPACK_BUILD_ALL_MICROKERNELS",
                                                                          "XNNPACK_LIBRARY_TYPE")}
                          for t, ev in evs.items()},
        "undefined_variables_read": {t: sorted(ev.undefined_read) for t, ev in evs.items()},
        "build_type": {"name": "Release", "flags": ["-O3"], "defines": ["NDEBUG"],
                       "note": "CMake's default CMAKE_<LANG>_FLAGS_RELEASE for Clang, placed before the "
                               "target options; a group's own -Os or -O2 comes before the -O2 every group "
                               "inherits from xnnpack-base, and clang applies the last one"},
        "common_flags": ["-fPIC"],
        "target_flags": lang,
        "link_libraries": sorted(set(ev0.link_libraries)),
        "archives": archives[TARGETS[0]],
        "groups": groups,
        "not_built_targets": {t: "not linked into XNNPACK or xnnpack-microkernels-prod (object files of an "
                                 "OBJECT library land only in targets that link it directly)"
                              for t in other},
        "unbuilt": unbuilt,
    }
    tree[BUILD_GROUPS] = (build_groups_bytes(desc), "build", "torchnative")
    tree["README.md"] = (tp_readme("xnnpack").encode(), "meta", "torchnative")
    tree[".gitattributes"] = (GITATTRIBUTES.encode(), "meta", "torchnative")
    return tree


# cpuinfo's CMakeLists.txt at f9a03241, read by hand (it uses macros the
# evaluator does not model). Each path is checked to occur in that file.
CPUINFO_COMMON = ["src/api.c", "src/cache.c", "src/init.c", "src/log.c", "src/arm/uarch.c", "src/arm/cache.c"]
CPUINFO_BY_TARGET = {
    "aarch64-apple-darwin": ["src/arm/mach/init.c", "src/mach/topology.c"],
    "aarch64-linux-android": ["src/arm/linux/init.c", "src/arm/linux/cpuinfo.c", "src/arm/linux/clusters.c",
                              "src/arm/linux/chipset.c", "src/arm/linux/midr.c", "src/arm/linux/hwcap.c",
                              "src/arm/linux/aarch64-isa.c", "src/arm/android/properties.c",
                              "src/linux/smallfile.c", "src/linux/multiline.c", "src/linux/cpulist.c",
                              "src/linux/processors.c"],
}
PTHREADPOOL_SRCS = ["src/legacy-api.c", "src/portable-api.c", "src/memory.c", "src/pthreads.c"]


def check_paths_in(text, paths, where):
    for p in paths:
        if not re.search(r"(?<![\w/.-])" + re.escape(p) + r"(?![\w/.-])", text):
            raise Refusal(f"vendor_executorch: {p} does not occur in {where}; the hand-read source list is stale")


def tp_tree_base(name, src, tracked, files, group):
    tree = {}
    for rel in files:
        if rel not in tracked:
            raise Refusal(f"vendor_executorch: {rel} is not tracked in {THIRD_PARTY[name]['label']}")
        tree[rel] = ((src / rel).read_bytes(), group, rel)
    tree["LICENSE"] = ((src / "LICENSE").read_bytes(), "license", "LICENSE")
    return tree


def tp_desc(name, extra):
    spec = THIRD_PARTY[name]
    desc = {"schema": "torchnative-build-groups/1", "tree": name,
            "upstream": {"repository": spec["url"], "commit": spec["commit"],
                         "pinned_by": f"ExecuTorch {TAG} backends/xnnpack/third-party/{spec['submodule']}"},
            "generated_by": "scripts/vendor/vendor_executorch.py",
            "evidence": "read from upstream CMake by hand, not measured by running CMake",
            "targets": TARGETS, "build_type": {"name": "Release", "flags": ["-O3"], "defines": ["NDEBUG"]},
            "common_flags": ["-fPIC"]}
    desc.update(extra)
    return desc


def tp_finish(name, tree, src, tracked, desc):
    pulled = closure(tree, src, tracked, TREE_ROOTS[name])
    desc.setdefault("unbuilt", {})
    for rel, who in pulled.items():
        tree[rel] = (tree[rel][0], "headers", rel)
        desc["unbuilt"][rel] = f"not compiled on its own: #include'd by {who}"
    tree[BUILD_GROUPS] = (build_groups_bytes(desc), "build", "torchnative")
    tree["README.md"] = (tp_readme(name).encode(), "meta", "torchnative")
    tree[".gitattributes"] = (GITATTRIBUTES.encode(), "meta", "torchnative")
    return tree


def select_cpuinfo(src, tracked, work):
    cm = (src / "CMakeLists.txt").read_text()
    files = CPUINFO_COMMON + [f for fs in CPUINFO_BY_TARGET.values() for f in fs]
    check_paths_in(cm, files, "cpuinfo/CMakeLists.txt")
    tree = tp_tree_base("cpuinfo", src, tracked, files, "sources")
    tree["CMakeLists.txt"] = (cm.encode(), "inputs", "CMakeLists.txt")
    base = {"std": {"c": "c99"}, "flags": [], "include_dirs": ["include", "src"],
            "defines": ["CPUINFO_LOG_LEVEL=2"],
            "defines_by_target": {"aarch64-apple-darwin": ["CPUINFO_LOG_TO_STDIO=1"],
                                  "aarch64-linux-android": ["CPUINFO_LOG_TO_STDIO=0", "_GNU_SOURCE=1"]}}
    groups = [dict(name="cpuinfo/common", cmake_target="cpuinfo", files=CPUINFO_COMMON, **base)]
    for t, fs in CPUINFO_BY_TARGET.items():
        g = dict(name=f"cpuinfo/{t}", cmake_target="cpuinfo", files=fs, only_targets=[t], **base)
        g["defines_by_target"] = {t: base["defines_by_target"][t]}
        groups.append(g)
    desc = tp_desc("cpuinfo", {
        "derived_from": ["CMakeLists.txt", "root CMakeLists.txt of ExecuTorch (CPUINFO_* cache values)"],
        "settings": {"CPUINFO_LIBRARY_TYPE": "static (ExecuTorch)", "CPUINFO_LOG_LEVEL": "error -> "
                     "CPUINFO_LOG_LEVEL=2 (ExecuTorch)", "CPUINFO_LOG_TO_STDIO": "OFF on Android, ON elsewhere "
                     "(cpuinfo's default; ExecuTorch does not set it)", "CPUINFO_BUILD_TOOLS/UNIT_TESTS/MOCK_TESTS/"
                     "BENCHMARKS": "OFF (ExecuTorch)", "C_STANDARD": "99, C_EXTENSIONS NO "
                     "(CPUINFO_TARGET_ENABLE_C99)", "POSITION_INDEPENDENT_CODE": "ON (ExecuTorch)"},
        "link_libraries_by_target": {"aarch64-linux-android": ["log", "pthread"]},
        "interface": {"defines": ["CPUINFO_SUPPORTED_PLATFORM=1"], "include_dirs": ["include"]},
        "groups": groups})
    return tp_finish("cpuinfo", tree, src, tracked, desc)


def select_pthreadpool(src, tracked, work):
    cm = (src / "CMakeLists.txt").read_text()
    check_paths_in(cm, PTHREADPOOL_SRCS + ["src/fastpath.c"], "pthreadpool/CMakeLists.txt")
    tree = tp_tree_base("pthreadpool", src, tracked, PTHREADPOOL_SRCS, "sources")
    tree["CMakeLists.txt"] = (cm.encode(), "inputs", "CMakeLists.txt")
    groups = [{"name": "pthreadpool", "cmake_target": "pthreadpool", "files": PTHREADPOOL_SRCS,
               "std": {"c": "c11"}, "flags": [],
               "flags_by_target": {"aarch64-apple-darwin": ["-fvisibility=hidden"],
                                   "aarch64-linux-android": ["-pthread"]},
               "defines": ["PTHREADPOOL_USE_FASTPATH=0", "FXDIV_USE_INLINE_ASSEMBLY=0"],
               "defines_by_target": {"aarch64-apple-darwin": ["PTHREADPOOL_USE_FUTEX=0", "PTHREADPOOL_USE_GCD=0"]},
               "include_dirs": ["include", "src", "../fxdiv/include"]}]
    desc = tp_desc("pthreadpool", {
        "derived_from": ["CMakeLists.txt", "root CMakeLists.txt of ExecuTorch (PTHREADPOOL_* cache values, "
                                           "-fvisibility=hidden on Apple)"],
        "settings": {"PTHREADPOOL_LIBRARY_TYPE": "static (ExecuTorch)",
                     "PTHREADPOOL_ALLOW_DEPRECATED_API": "ON (ExecuTorch), so src/legacy-api.c and no "
                                                         "PTHREADPOOL_NO_DEPRECATED_API",
                     "PTHREADPOOL_SYNC_PRIMITIVE": "condvar on Apple (ExecuTorch); default elsewhere, where "
                                                   "threadpool-common.h picks futex on __linux__",
                     "PTHREADPOOL_ENABLE_FASTPATH": "OFF: upstream's default off the x86 processors",
                     "GCD": "PTHREADPOOL_USE_GCD=0 is defined on Apple, and this commit has no GCD code path",
                     "C_STANDARD": "11, C_EXTENSIONS NO"},
        "link_libraries_by_target": {"aarch64-linux-android": ["pthread"]},
        "interface": {"include_dirs": ["include"], "flags_by_target": {"aarch64-linux-android": ["-pthread"]}},
        "groups": groups,
        "unbuilt": {}})
    return tp_finish("pthreadpool", tree, src, tracked, desc)


def select_headers_only(name, files):
    def select_fn(src, tracked, work):
        missing = [f for f in files if f not in tracked]
        if missing:
            raise Refusal(f"vendor_executorch: {THIRD_PARTY[name]['label']} has no {missing}")
        tree = tp_tree_base(name, src, tracked, files, "headers")
        interface = {"include_dirs": ["include"]}
        if name == "fxdiv":
            interface["defines"] = ["FXDIV_USE_INLINE_ASSEMBLY=0"]
        desc = tp_desc(name, {"derived_from": ["CMakeLists.txt (an INTERFACE library: headers only)"],
                              "interface": interface, "groups": [], "unbuilt": {}})
        return tp_finish(name, tree, src, tracked, desc)
    return select_fn


def fp16_headers(tracked):
    return sorted(t for t in tracked if t.startswith("include/") and t.endswith(".h"))


SELECT_TP = {
    "xnnpack": select_xnnpack,
    "cpuinfo": select_cpuinfo,
    "pthreadpool": select_pthreadpool,
    "fxdiv": select_headers_only("fxdiv", ["include/fxdiv.h"]),
    "fp16": None,  # its header list depends on the checkout; see regenerate_tp
}


ET_TITLE = f"# ExecuTorch {TAG} ({COMMIT}) subset for torchnative, issue #312."
FB_TITLE = f"# flatbuffers {FB_TAG} ({FB_COMMIT}) C++ headers for torchnative, issue #312."


def tp_title(name):
    spec = THIRD_PARTY[name]
    return f"# {spec['label']} {spec['commit']} (ExecuTorch {TAG} submodule pin) subset for torchnative, issue #319."


def manifest_text(tree, title=ET_TITLE):
    rows = [title,
            "# Written by scripts/vendor/vendor_executorch.py; never edit by hand.",
            "# group\tsha256\tbytes\tpath\tupstream"]
    for rel in sorted(tree):
        data, group, upstream = tree[rel]
        rows.append(f"{group}\t{sha256(data)}\t{len(data)}\t{rel}\t{upstream}")
    return "\n".join(rows) + "\n"


def write_tree(tree, root, title=ET_TITLE):
    for rel, (data, _, _) in tree.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (root / MANIFEST).write_bytes(manifest_text(tree, title).encode())


def read_manifest(root, commit=COMMIT):
    path = root / MANIFEST
    if not path.is_file():
        return None, [f"missing: {MANIFEST} (no manifest at {root})"]
    entries, problems = {}, []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not any(commit in ln for ln in lines if ln.startswith("#")):
        problems.append(f"manifest: {MANIFEST} does not name commit {commit}")
    for n, ln in enumerate(lines, 1):
        if not ln or ln.startswith("#"):
            continue
        cols = ln.split("\t")
        if len(cols) != 5:
            problems.append(f"manifest: line {n} has {len(cols)} fields, not 5")
            continue
        group, digest, size, rel, upstream = cols
        entries[rel] = (group, digest, int(size), upstream)
    return entries, problems


def files_on_disk(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {MANIFEST}


def check_against_manifest(root, commit=COMMIT):
    if not root.is_dir():
        return [f"missing: {root} (the tree does not exist)"], {}
    entries, problems = read_manifest(root, commit)
    if entries is None:
        return problems, {}
    on_disk = files_on_disk(root)
    for rel in sorted(set(entries) - on_disk):
        problems.append(f"missing: {rel}")
    for rel in sorted(on_disk - set(entries)):
        problems.append(f"extra: {rel}")
    for rel in sorted(set(entries) & on_disk):
        data = (root / rel).read_bytes()
        _, digest, size, _ = entries[rel]
        got = sha256(data)
        if got != digest or len(data) != size:
            problems.append(f"modified: {rel} (sha256 expected {digest}, got {got})")
    return problems, entries


def check_build_groups(root, entries):
    """BUILD_GROUPS.json names only files of the tree, and every source file
    of the tree is in exactly one group or listed as unbuilt (or belongs to a
    manifest group the description says it does not describe)."""
    path = root / BUILD_GROUPS
    if not path.is_file():
        return [f"build: no {BUILD_GROUPS}"]
    try:
        desc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"build: {BUILD_GROUPS} is not JSON: {e}"]
    problems, named = [], {}
    for g in desc.get("groups", []):
        for rel in g.get("files", []):
            named.setdefault(rel, []).append(g["name"])
            if rel not in entries or not (root / rel).is_file():
                problems.append(f"build: group {g['name']} names {rel} which is not in the tree")
        if not g.get("files"):
            problems.append(f"build: group {g['name']} names no file")
    for rel in desc.get("unbuilt", {}):
        named.setdefault(rel, []).append("unbuilt")
        if rel not in entries or not (root / rel).is_file():
            problems.append(f"build: unbuilt names {rel} which is not in the tree")
    for rel, where in sorted(named.items()):
        if len(where) > 1:
            problems.append(f"build: {rel} is named {len(where)} times ({', '.join(where)})")
    exempt = set(desc.get("not_described", {}).get("manifest_groups", []))
    for rel, (group, _, _, _) in sorted(entries.items()):
        if rel.endswith(SOURCE_EXT) and rel not in named and group not in exempt:
            problems.append(f"build: {rel} is a source file in no group and not listed as unbuilt")
    return problems


def compare_trees(expected, actual):
    a, b = files_on_disk(expected) | {MANIFEST}, files_on_disk(actual) | {MANIFEST}
    problems = [f"missing (regeneration has it): {r}" for r in sorted(a - b)]
    problems += [f"extra (regeneration lacks it): {r}" for r in sorted(b - a)]
    for rel in sorted(a & b):
        pa, pb = expected / rel, actual / rel
        if not pb.is_file() or not pa.is_file() or pa.read_bytes() != pb.read_bytes():
            problems.append(f"differs from regeneration: {rel}")
    return problems


def regenerate(src, scratch, flatc_arg, py_arg):
    tracked = verify_source(src)
    flatc = find_flatc(flatc_arg)
    py, torchgen_dir = codegen_python(py_arg)
    work = pathlib.Path(tempfile.mkdtemp(prefix="gen-", dir=scratch))
    try:
        generated = generate(src, tracked, work, flatc, py, torchgen_dir)
        tree = select(src, tracked, generated)
        stage = work / "stage"
        stage.mkdir()
        write_tree(tree, stage)
    except BaseException:
        shutil.rmtree(work)
        raise
    return work, stage, tree


def regenerate_fb(src, scratch):
    tracked = verify_fb_source(src)
    tree = select_fb(src, tracked)
    work = pathlib.Path(tempfile.mkdtemp(prefix="gen-fb-", dir=scratch))
    try:
        stage = work / "stage"
        stage.mkdir()
        write_tree(tree, stage, FB_TITLE)
    except BaseException:
        shutil.rmtree(work)
        raise
    return work, stage, tree


def regenerate_tp(name, src, scratch):
    src = pathlib.Path(src).resolve()
    tracked = verify_tp_source(name, src)
    work = pathlib.Path(tempfile.mkdtemp(prefix=f"gen-{name}-", dir=scratch))
    try:
        fn = SELECT_TP[name] or select_headers_only(name, fp16_headers(tracked))
        tree = fn(src, tracked, work)
        stage = work / "stage"
        stage.mkdir()
        write_tree(tree, stage, tp_title(name))
    except BaseException:
        shutil.rmtree(work)
        raise
    return work, stage, tree


def group_sizes(entries):
    out = {}
    for group, _, size, _ in entries.values():
        n, b = out.get(group, (0, 0))
        out[group] = (n + 1, b + size)
    return out


def unresolved_includes(roots):
    """Every tree, keyed by `<tree>/<path>` so a sibling include root such as
    `../cpuinfo/include` resolves across trees as the compiler would."""
    files = {}
    for name, root in roots.items():
        if root.is_dir():
            files.update({f"{name}/{rel}": root / rel for rel in files_on_disk(root)})
    exists = lambda rel: rel in files  # noqa: E731
    report = {"std": {}, "platform": {}, "unresolved": {}}
    included_by = {name: set() for name in roots}
    for key in sorted(files):
        if not key.endswith(SCAN_EXT):
            continue
        tree = key.split("/", 1)[0]
        text = files[key].read_text(encoding="utf-8", errors="replace")
        # The vendor directory itself is ExecuTorch's first include root
        # (`-I torchnative/rust/vendor`, so `<executorch/...>` resolves).
        roots = ([""] if tree == "executorch" else []) + [
            os.path.normpath(f"{tree}/{r}").replace(os.sep, "/") for r in TREE_ROOTS[tree]]
        for quoted, name, conds in scan_includes(text):
            hit = resolve_in(key, quoted, name, exists, roots)
            if hit:
                other = hit.split("/", 1)[0]
                if other != tree and other in included_by:
                    included_by[other].add(key)
                continue
            report[classify(name)].setdefault(name, []).append((key, conds))
    return report, included_by


def main_checkout():
    """The main checkout when REPO is a worktree (AGENTS.md §19), else REPO."""
    done = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute",
                           "--git-common-dir"], capture_output=True, text=True)
    if done.returncode == 0:
        return pathlib.Path(done.stdout.strip()).parent.resolve()
    return REPO


def inside_repo(path):
    """Inside this checkout, or inside the main checkout it is a worktree of
    (whose .scratch/ is where AGENTS.md §22 puts backups and scratch)."""
    for root in {REPO.resolve(), main_checkout()}:
        try:
            path.resolve().relative_to(root)
            return True
        except ValueError:
            continue
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify instead of write")
    ap.add_argument("--includes", action="store_true", help="list includes the trees do not resolve")
    ap.add_argument("--source", default=os.environ.get("TORCHNATIVE_ET_SRC"),
                    help=f"ExecuTorch checkout at {TAG} (env TORCHNATIVE_ET_SRC)")
    ap.add_argument("--dest", default=os.environ.get("TORCHNATIVE_ET_VENDOR_DIR", str(DEST)))
    ap.add_argument("--flatbuffers-source", default=os.environ.get("TORCHNATIVE_FLATBUFFERS_SRC"),
                    help=f"flatbuffers checkout at {FB_TAG} (env TORCHNATIVE_FLATBUFFERS_SRC)")
    ap.add_argument("--flatbuffers-dest", default=os.environ.get("TORCHNATIVE_FB_VENDOR_DIR", str(FB_DEST)))
    ap.add_argument("--third-party", default=os.environ.get("TORCHNATIVE_ET_THIRD_PARTY"),
                    help="directory holding XNNPACK/, cpuinfo/, pthreadpool/, FXdiv/ and FP16/ checkouts "
                         "(env TORCHNATIVE_ET_THIRD_PARTY)")
    for name, spec in THIRD_PARTY.items():
        ap.add_argument(f"--{name}-source", help=f"{spec['label']} checkout at {spec['commit'][:8]}")
        ap.add_argument(f"--{name}-dest", default=str(VENDOR / name))
    ap.add_argument("--flatc", default=os.environ.get("TORCHNATIVE_FLATC"))
    ap.add_argument("--codegen-python", default=os.environ.get("TORCHNATIVE_ET_CODEGEN_PYTHON"))
    ap.add_argument("--scratch", default=str(REPO / ".scratch"),
                    help="where regeneration stages (must be inside the repository)")
    args = ap.parse_args(argv)
    dest = pathlib.Path(args.dest)
    fb_dest = pathlib.Path(args.flatbuffers_dest)
    scratch = pathlib.Path(args.scratch)
    fb_src = pathlib.Path(args.flatbuffers_source) if args.flatbuffers_source else None
    tp_src, tp_dest = {}, {}
    for name, spec in THIRD_PARTY.items():
        tp_dest[name] = pathlib.Path(getattr(args, f"{name}_dest"))
        given = getattr(args, f"{name}_source")
        if given:
            tp_src[name] = pathlib.Path(given)
        elif args.third_party:
            tp_src[name] = pathlib.Path(args.third_party) / spec["submodule"]

    trees = [("executorch", dest, COMMIT, args.source and pathlib.Path(args.source)),
             ("flatbuffers", fb_dest, FB_COMMIT, fb_src)]
    trees += [(name, tp_dest[name], spec["commit"], tp_src.get(name)) for name, spec in THIRD_PARTY.items()]

    def regen(name, src):
        if name == "executorch":
            return regenerate(src, scratch, args.flatc, args.codegen_python)
        if name == "flatbuffers":
            return regenerate_fb(src, scratch)
        return regenerate_tp(name, src, scratch)

    def label(name):
        if name == "executorch":
            return f"ExecuTorch {TAG}"
        if name == "flatbuffers":
            return f"flatbuffers {FB_TAG}"
        return f"{THIRD_PARTY[name]['label']} {THIRD_PARTY[name]['commit'][:8]}"

    try:
        if args.includes:
            report, included_by = unresolved_includes({name: root for name, root, _, _ in trees})
            for kind in ("unresolved", "platform", "std"):
                print(f"== {kind}: {len(report[kind])} distinct")
                for name in sorted(report[kind]):
                    sites = report[kind][name]
                    print(f"  <{name}>  ({len(sites)} site(s))")
                    if kind != "std":
                        for rel, conds in sites:
                            print(f"      {rel}" + (f"   under {' && '.join(conds)}" if conds else "   UNCONDITIONAL"))
            print("== included from other trees")
            for name in included_by:
                print(f"  {name}: {len(included_by[name])} file(s) of other trees include it")
            return 0

        if args.check:
            failed = False
            for name, root, commit, src in trees:
                problems, entries = check_against_manifest(root, commit)
                if not problems and name != "flatbuffers":
                    problems += check_build_groups(root, entries)
                if not problems:
                    total = sum(e[2] for e in entries.values())
                    print(f"{name} {root}: {len(entries)} files, {total} bytes match {MANIFEST}")
                    for g, (n, b) in sorted(group_sizes(entries).items()):
                        print(f"  {g:15s} {n:5d} files {b:9d} bytes")
                if src and not problems:
                    if not inside_repo(scratch):
                        raise Refusal(f"vendor_executorch: refusing: scratch {scratch} is outside {REPO}")
                    scratch.mkdir(parents=True, exist_ok=True)
                    work, stage, _ = regen(name, src)
                    try:
                        problems += compare_trees(stage, root)
                    finally:
                        shutil.rmtree(work)
                    if not problems:
                        print(f"{root} is {label(name)} regenerated from {src}, byte for byte")
                if problems:
                    failed = True
                    print(f"vendor_executorch --check: {name} tree {root} does NOT match "
                          f"({len(problems)} difference(s)):", file=sys.stderr)
                    for p in problems:
                        print(f"  {name}: {p}", file=sys.stderr)
            if failed:
                print("Regenerate with `python scripts/vendor/vendor_executorch.py --source <checkout>`, "
                      "`--flatbuffers-source <checkout>` or `--third-party <dir>`; never edit the trees by hand.",
                      file=sys.stderr)
                return 1
            return 0

        wanted = [(name, root, src) for name, root, _, src in trees if src]
        if not wanted:
            raise Refusal("vendor_executorch: refusing: no --source (or TORCHNATIVE_ET_SRC), no "
                          "--flatbuffers-source (or TORCHNATIVE_FLATBUFFERS_SRC) and no --third-party or "
                          f"--<tree>-source given; writing needs an ExecuTorch checkout at {TAG} ({COMMIT}), "
                          f"a flatbuffers checkout at {FB_TAG} ({FB_COMMIT}), third-party checkouts at "
                          "their pins, or any of them")
        if not all(inside_repo(root) for _, root, _ in wanted) or not inside_repo(scratch):
            raise Refusal(f"vendor_executorch: refusing: every destination and --scratch must be inside {REPO}")
        scratch.mkdir(parents=True, exist_ok=True)
        # Verify and stage every requested tree before writing any of them.
        staged = []
        try:
            for name, root, src in wanted:
                work, stage, tree = regen(name, src)
                staged.append((work, stage, tree, root))
            for _, stage, tree, root in staged:
                if root.exists():
                    shutil.rmtree(root)
                shutil.copytree(stage, root)
                print(f"wrote {root}: {len(tree)} files + {MANIFEST}")
        finally:
            for work, _, _, _ in staged:
                shutil.rmtree(work)
        return 0
    except Refusal as r:
        print(str(r), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
