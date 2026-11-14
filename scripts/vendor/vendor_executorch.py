#!/usr/bin/env python3
"""Vendor the ATen-free ExecuTorch runtime subset and the flatbuffers C++ headers, or check them.

    python scripts/vendor/vendor_executorch.py --source ET_CHECKOUT
        write torchnative/rust/vendor/executorch/ from an ExecuTorch checkout
    python scripts/vendor/vendor_executorch.py --flatbuffers-source FB_CHECKOUT
        write torchnative/rust/vendor/flatbuffers/ from a flatbuffers checkout
        (either flag alone writes only its tree; both write both)
    python scripts/vendor/vendor_executorch.py --check
        verify both trees on disk against their MANIFEST.tsv (no checkout needed)
    python scripts/vendor/vendor_executorch.py --check [--source ET] [--flatbuffers-source FB]
        the same, then regenerate each tree whose checkout is given into a
        scratch directory inside the repository and compare every file byte
        for byte
    python scripts/vendor/vendor_executorch.py --includes
        list every #include in the vendored trees that does not resolve inside
        them, split into C/C++ standard, platform and unresolved

`--source` may also come from TORCHNATIVE_ET_SRC, `--flatbuffers-source` from
TORCHNATIVE_FLATBUFFERS_SRC. Issue #312, step R0 of the
ExecuTorch runtime design (agent proposal, not confirmed). The vendoring itself,
the whole portable kernel library rather than an allow-list, and the rule that
nothing runs CMake, flatc or the network at torchnative BUILD time are the
leader's decisions of 2026-11-14.

What is vendored (pinned: tag v1.4.1, commit e4d02f41):

  core       EXECUTORCH_CORE_SRCS from upstream's build_variables.bzl, plus the
             three default PALs (posix, minimal, android)
  prim_ops   EXECUTORCH_SRCS (kernels/prim_ops)
  portable   every kernels/portable/cpu/**/*.cpp outside test/, which is the
             glob upstream's kernels/portable/CMakeLists.txt compiles. The bzl
             list PORTABLE_KERNELS_SRCS is NOT used: it lacks 7 op files and 2
             util files whose ops functions.yaml registers, so a build from it
             would fail to link the generated registrations.
  headers    the transitive #include closure of all of the above, resolved
             against the checkout with upstream's include roots
  generated  what upstream produces at build time, produced here once, at
             VENDOR time, never at torchnative build time:
               flatc 24.3.25 --cpp --cpp-std c++11 --gen-mutable --scoped-enums
                 on schema/{scalar_type,program}.fbs
               python -m codegen.tools.gen_oplist and python -m codegen.gen with
                 torchgen from upstream torch 2.13.0 (torch_pin.py) on
                 kernels/portable/functions.yaml
  inputs     functions.yaml and the two .fbs files the generated files came from
  license    upstream's LICENSE
  meta       README.md and .gitattributes written by this script

The flatbuffers tree (pinned: tag v24.3.25, commit 595bf000), a sibling
directory with its own MANIFEST.tsv (leader decision 2026-11-14):

  headers    every file of upstream's include/flatbuffers/ (all are .h), so
             `-I torchnative/rust/vendor/flatbuffers/include` resolves the
             generated schema headers' "flatbuffers/flatbuffers.h"
  license    upstream's LICENSE (Apache-2.0)
  meta       README.md and .gitattributes written by this script
  No sources, no flatc, no tests, no language bindings.

Why two manifests and not one: each tree stays self-describing (its manifest
names its own upstream commit), each is written from its own checkout without
the other's, and --check is the same function run twice.

Not vendored in this step: XNNPACK, cpuinfo, pthreadpool, FXdiv, FP16,
extension/threadpool. `--includes` names what that leaves open.

Why Python and not sh like vendor_candle.sh: `--check` runs in the gate on
Linux and macOS and should be able to run on the Windows runner, which has no
POSIX shell for tests; and evaluating build_variables.bzl (upstream's CMake does
it with Python too), hashing, and the include closure are all native here.
scripts/vendor/vendor_executorch.sh is a thin wrapper for the shell habit.
"""

import argparse
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
DEST = REPO / "torchnative" / "rust" / "vendor" / "executorch"
FB_DEST = REPO / "torchnative" / "rust" / "vendor" / "flatbuffers"

TAG = "v1.4.1"
COMMIT = "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f"
FLATC_VERSION = "flatc version 24.3.25"
TORCH_VERSION = "2.13.0"
MANIFEST = "MANIFEST.tsv"

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

FLATC_ARGS = ["--cpp", "--cpp-std", "c++11", "--gen-mutable", "--scoped-enums"]
GEN_SCHEMA = "generated/include/executorch/schema"
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

Upstream: https://github.com/pytorch/executorch, tag `{TAG}`, commit
`{COMMIT}`. Paths mirror the upstream checkout, so with
`-I torchnative/rust/vendor` an upstream `#include <executorch/...>` resolves.
Also add `-I` for `{C10_ROOT}`, for `{GEN_ROOT}`, and for
`torchnative/rust/vendor/flatbuffers/include` (the sibling flatbuffers tree).

## Generated files

`generated/` holds what upstream produces at build time. Here it is produced
once, at vendor time, so a torchnative build needs neither flatc nor torchgen:

- `{GEN_SCHEMA}/`: `flatc {' '.join(FLATC_ARGS)}` on
  `schema/scalar_type.fbs` and `schema/program.fbs`, with {FLATC_VERSION}.
- `{GEN_OPS}/`: `python -m codegen.tools.gen_oplist` and
  `python -m codegen.gen` (upstream's own codegen, with torchgen from torch
  {TORCH_VERSION}, the version upstream's `torch_pin.py` names) on
  `kernels/portable/functions.yaml`.

## Not here yet

The XNNPACK backend with its third-party trees. flatbuffers' C++ headers
(v24.3.25), which the generated schema headers include, are vendored next to
this directory in `../flatbuffers/` (leader decision 2026-11-14).
`python scripts/vendor/vendor_executorch.py --includes` lists every include
that resolves in neither directory.

## Regenerating

    git clone --depth 1 --branch {TAG} https://github.com/pytorch/executorch ET
    python scripts/vendor/vendor_executorch.py --source ET \\
        --flatc PATH_TO_FLATC_24.3.25 --codegen-python PYTHON_WITH_TORCH_{TORCH_VERSION}

This writes only this directory; `../flatbuffers/` has its own checkout
(`--flatbuffers-source`, see its README).

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
PLATFORM_PREFIXES = ("sys/", "android/", "mach/", "mach-o/", "zephyr/")
PLATFORM_HEADERS = set("""
unistd.h pthread.h dlfcn.h fcntl.h sched.h malloc.h alloca.h execinfo.h cxxabi.h windows.h io.h
intrin.h immintrin.h arm_neon.h sal.h TargetConditionals.h emscripten.h process.h
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
        f"`flatc {' '.join(FLATC_ARGS)} -o {GEN_SCHEMA} scalar_type.fbs program.fbs` in schema/.")


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


def generate(src, tracked, work, flatc, py, torchgen_dir):
    """Produce upstream's build-time files. Upstream code runs from a copy in
    `work`, so the checkout is never written (no __pycache__ either)."""
    for rel in sorted(t for t in tracked if t.startswith("codegen/") and "/test/" not in t):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, work / rel)
    for rel in INPUTS:
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


def classify(name):
    if name in STD_HEADERS:
        return "std"
    if name in PLATFORM_HEADERS or name.startswith(PLATFORM_PREFIXES):
        return "platform"
    return "unresolved"


def select(src, tracked, generated):
    """Return {rel: (bytes, group, upstream)} for the whole tree."""
    ns = build_variables(src)
    groups = {
        "core": sorted(ns["EXECUTORCH_CORE_SRCS"]) + PALS,
        "prim_ops": sorted(ns["EXECUTORCH_SRCS"]),
        "portable": sorted(t for t in tracked if t.startswith("kernels/portable/cpu/")
                           and t.endswith(".cpp") and "/test/" not in t),
        "inputs": INPUTS,
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

    exists = lambda rel: rel in tree or rel in tracked  # noqa: E731
    queue = [r for r in tree if r.endswith((".cpp", ".h"))] + HEADER_SEEDS
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
    tree["README.md"] = (README.encode(), "meta", "torchnative")
    tree[".gitattributes"] = (GITATTRIBUTES.encode(), "meta", "torchnative")
    return tree


ET_TITLE = f"# ExecuTorch {TAG} ({COMMIT}) subset for torchnative, issue #312."
FB_TITLE = f"# flatbuffers {FB_TAG} ({FB_COMMIT}) C++ headers for torchnative, issue #312."


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


def group_sizes(entries):
    out = {}
    for group, _, size, _ in entries.values():
        n, b = out.get(group, (0, 0))
        out[group] = (n + 1, b + size)
    return out


def unresolved_includes(root, fb_root=FB_DEST):
    """Both trees, the flatbuffers one seen from the ExecuTorch one as
    FB_FROM_ET (its sibling), so its headers resolve on FB_ROOT."""
    files = {rel: root / rel for rel in files_on_disk(root)}
    if fb_root.is_dir():
        files.update({f"{FB_FROM_ET}/{rel}": fb_root / rel for rel in files_on_disk(fb_root)})
    exists = lambda rel: rel in files  # noqa: E731
    report = {"std": {}, "platform": {}, "unresolved": {}}
    for rel in sorted(files):
        if not rel.endswith((".cpp", ".h")):
            continue
        for quoted, name, conds in scan_includes(files[rel].read_text(encoding="utf-8", errors="replace")):
            if resolve(rel, quoted, name, exists):
                continue
            report[classify(name)].setdefault(name, []).append((rel, conds))
    return report


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
    ap.add_argument("--includes", action="store_true", help="list includes the tree does not resolve")
    ap.add_argument("--source", default=os.environ.get("TORCHNATIVE_ET_SRC"),
                    help=f"ExecuTorch checkout at {TAG} (env TORCHNATIVE_ET_SRC)")
    ap.add_argument("--dest", default=os.environ.get("TORCHNATIVE_ET_VENDOR_DIR", str(DEST)))
    ap.add_argument("--flatbuffers-source", default=os.environ.get("TORCHNATIVE_FLATBUFFERS_SRC"),
                    help=f"flatbuffers checkout at {FB_TAG} (env TORCHNATIVE_FLATBUFFERS_SRC)")
    ap.add_argument("--flatbuffers-dest", default=os.environ.get("TORCHNATIVE_FB_VENDOR_DIR", str(FB_DEST)))
    ap.add_argument("--flatc", default=os.environ.get("TORCHNATIVE_FLATC"))
    ap.add_argument("--codegen-python", default=os.environ.get("TORCHNATIVE_ET_CODEGEN_PYTHON"))
    ap.add_argument("--scratch", default=str(REPO / ".scratch"),
                    help="where regeneration stages (must be inside the repository)")
    args = ap.parse_args(argv)
    dest = pathlib.Path(args.dest)
    fb_dest = pathlib.Path(args.flatbuffers_dest)
    scratch = pathlib.Path(args.scratch)
    fb_src = pathlib.Path(args.flatbuffers_source) if args.flatbuffers_source else None

    try:
        if args.includes:
            report = unresolved_includes(dest, fb_dest)
            for kind in ("unresolved", "platform", "std"):
                print(f"== {kind}: {len(report[kind])} distinct")
                for name in sorted(report[kind]):
                    sites = report[kind][name]
                    print(f"  <{name}>  ({len(sites)} site(s))")
                    if kind != "std":
                        for rel, conds in sites:
                            print(f"      {rel}" + (f"   under {' && '.join(conds)}" if conds else "   UNCONDITIONAL"))
            return 0

        if args.check:
            failed = False
            for name, root, commit, src in (("executorch", dest, COMMIT, args.source),
                                            ("flatbuffers", fb_dest, FB_COMMIT, fb_src)):
                problems, entries = check_against_manifest(root, commit)
                if not problems:
                    total = sum(e[2] for e in entries.values())
                    print(f"{name} {root}: {len(entries)} files, {total} bytes match {MANIFEST}")
                    for g, (n, b) in sorted(group_sizes(entries).items()):
                        print(f"  {g:10s} {n:5d} files {b:9d} bytes")
                if src and not problems:
                    if not inside_repo(scratch):
                        raise Refusal(f"vendor_executorch: refusing: scratch {scratch} is outside {REPO}")
                    scratch.mkdir(parents=True, exist_ok=True)
                    if name == "executorch":
                        work, stage, _ = regenerate(pathlib.Path(src), scratch, args.flatc,
                                                    args.codegen_python)
                    else:
                        work, stage, _ = regenerate_fb(src, scratch)
                    try:
                        problems += compare_trees(stage, root)
                    finally:
                        shutil.rmtree(work)
                    if not problems:
                        label = f"ExecuTorch {TAG}" if name == "executorch" else f"flatbuffers {FB_TAG}"
                        print(f"{root} is {label} regenerated from {src}, byte for byte")
                if problems:
                    failed = True
                    print(f"vendor_executorch --check: {name} tree {root} does NOT match "
                          f"({len(problems)} difference(s)):", file=sys.stderr)
                    for p in problems:
                        print(f"  {name}: {p}", file=sys.stderr)
            if failed:
                print("Regenerate with `python scripts/vendor/vendor_executorch.py --source <checkout>` "
                      "or `--flatbuffers-source <checkout>`; never edit the trees by hand.", file=sys.stderr)
                return 1
            return 0

        if not args.source and not fb_src:
            raise Refusal("vendor_executorch: refusing: no --source (or TORCHNATIVE_ET_SRC) and no "
                          "--flatbuffers-source (or TORCHNATIVE_FLATBUFFERS_SRC) given; writing needs an "
                          f"ExecuTorch checkout at {TAG} ({COMMIT}), a flatbuffers checkout at {FB_TAG} "
                          f"({FB_COMMIT}), or both")
        if not inside_repo(dest) or not inside_repo(fb_dest) or not inside_repo(scratch):
            raise Refusal(f"vendor_executorch: refusing: --dest, --flatbuffers-dest and --scratch must be "
                          f"inside {REPO}")
        scratch.mkdir(parents=True, exist_ok=True)
        # Verify and stage every requested tree before writing any of them.
        staged = []
        try:
            if fb_src:
                work, stage, tree = regenerate_fb(fb_src, scratch)
                staged.append((work, stage, tree, fb_dest))
            if args.source:
                work, stage, tree = regenerate(pathlib.Path(args.source), scratch, args.flatc,
                                               args.codegen_python)
                staged.append((work, stage, tree, dest))
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
