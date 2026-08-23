#!/bin/sh
# Build the host artefact, rename it to `_C.abi3.so`, and run the smoke tests
# against it. Renaming is not incidental: cargo emits `lib_C.dylib`, and Python
# only loads a file whose name ends in one of importlib's extension suffixes
# (RUST_CROSSBUILD.md §2).
#
# The suffix is `.abi3.so`, not a bare `.so`: ABI3.md §7 item 2. An untagged
# `_C.so` loads into *any* interpreter, which is precisely the silent failure
# the abi3 build exists to remove -- so the filename should say what it is.
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
rm -f "$stage/_C.so"
if [ -f "$target_dir/release/lib_C.dylib" ]; then
    cp "$target_dir/release/lib_C.dylib" "$stage/_C.abi3.so"
elif [ -f "$target_dir/release/lib_C.so" ]; then
    cp "$target_dir/release/lib_C.so" "$stage/_C.abi3.so"
else
    echo "no host artefact under $target_dir/release" >&2
    exit 1
fi

PYTHONPATH="$stage" exec "${PYTHON:-python3}" "$crate_dir/pytests/test_shim.py"
