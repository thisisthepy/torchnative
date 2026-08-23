#!/bin/sh
# Build the host artefact, rename it to `_C.so`, and run the smoke tests
# against it. Renaming is not incidental: cargo emits `lib_C.dylib`, and Python
# will only load a file named `_C.so` (RUST_CROSSBUILD.md §2).
set -eu

crate_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
target_dir=${CARGO_TARGET_DIR:-$crate_dir/target}
stage=${TORCH_C_STAGE:-${TMPDIR:-/tmp}/torch-c-stage}

# `cd` rather than `--manifest-path`: cargo discovers `.cargo/config.toml` from
# the *working directory*, not from the manifest. Building this crate from
# elsewhere silently drops `-undefined dynamic_lookup` and the link fails with
# a wall of undefined `_Py*` symbols. Same trap as the hardcoded iOS `-F` path,
# from the other side -- see build.rs.
cd -- "$crate_dir"
cargo build --release

mkdir -p "$stage"
if [ -f "$target_dir/release/lib_C.dylib" ]; then
    cp "$target_dir/release/lib_C.dylib" "$stage/_C.so"
elif [ -f "$target_dir/release/lib_C.so" ]; then
    cp "$target_dir/release/lib_C.so" "$stage/_C.so"
else
    echo "no host artefact under $target_dir/release" >&2
    exit 1
fi

PYTHONPATH="$stage" exec "${PYTHON:-python3}" "$crate_dir/pytests/test_shim.py"
