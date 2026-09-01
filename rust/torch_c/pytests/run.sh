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
repo_root=$(CDPATH='' cd -- "$crate_dir/../.." && pwd)
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

# Four tests in test_shim.py (capture/checkpoint/device/meta) do not import
# the artefact staged above -- they shell out to a subprocess that puts the
# vendored tree on PYTHONPATH, and that subprocess loads
# `$vendor_dir/torch/_C.abi3.so`. `vendor/install_shim.sh` is the only thing
# that writes that file; this script never has. So a source change that only
# this script rebuilds does not reach those four tests -- they would keep
# testing whatever `install_shim.sh` last installed, silently, with a green
# result (docs/CAPTURE.md §8).
#
# Refuse by name rather than install it here: installing would make this
# script a build step for a *different* artefact (the one that ships inside
# the vendored tree) with its own failure modes (torch/bin/torch_shm_manager,
# TORCHNATIVE_VENDOR_DIR) that have nothing to do with "run the smoke tests".
# Comparing bytes rather than mtimes because this crate rebuilds
# byte-identical output when nothing relevant changed (measured: a `touch` +
# rebuild of dtype.rs reproduced the previous dylib exactly on this
# toolchain), so a byte compare does not nag on a no-op rebuild the way an
# mtime check would.
#
# `cmp` distinguishes three outcomes and this has to distinguish them too: 0 is
# same, 1 is differ, anything above 1 is "the comparison itself failed". Reading
# every non-zero as "stale" conflates the check with its own failure -- which
# happened: under memory pressure from a concurrent build the kernel killed
# `cmp` with SIGKILL, the guard read exit 137 as a difference, and the suite
# refused with a message telling the reader to reinstall a shim that was already
# current. That is the repeated defect of this repository wearing a new hat, so
# the two are separated and only one of them is a staleness claim.
vendor_dir=${TORCHNATIVE_VENDOR_DIR:-$repo_root/torchnative/src/main}
vendor_shim="$vendor_dir/torch/_C.abi3.so"
if [ -f "$vendor_shim" ]; then
    cmp -s "$stage/_C.abi3.so" "$vendor_shim" && cmp_status=0 || cmp_status=$?
else
    cmp_status=0
fi
if [ "$cmp_status" -gt 1 ]; then
    cat >&2 <<EOF
run.sh: refusing to run -- could not compare against $vendor_shim.

\`cmp\` exited $cmp_status, which is neither "same" (0) nor "different" (1), so
whether the vendored shim is current is unknown. This is not a staleness
report. A SIGKILL here has meant memory pressure from a concurrent build;
re-running once the machine is quieter has been enough.
EOF
    exit 1
fi
if [ "$cmp_status" -eq 1 ]; then
    cat >&2 <<EOF
run.sh: refusing to run -- $vendor_shim is stale.

It does not match what was just built from rust/torch_c/src. The capture,
checkpoint, device, and meta tests read that file (not this script's
staged artefact) through a vendored-tree subprocess, so running them now
would silently re-test the old build instead of catching a change here.

Fix: run vendor/install_shim.sh, then re-run this script.
EOF
    exit 1
fi

PYTHONPATH="$stage" "${PYTHON:-python3}" "$crate_dir/pytests/test_shim.py" || exit $?

# The golden harness has its own self-test -- it injects a fault shaped like a
# plausible misimplementation at each comparator and checks the comparator
# rejects it. Nothing invoked it, so the gate existed without ever being pulled;
# it caught that the previous injection reached exactly one case out of 1781.
# It builds nothing, so it costs a few seconds here.
#
# TORCH_C_ARTEFACT points it at the artefact this script just staged, since
# tools/golden/loader.py otherwise falls back to a fixed cache path that may
# hold an entirely different build.
TORCH_C_ARTEFACT="$stage/_C.abi3.so" \
    "${PYTHON:-python3}" "$repo_root/tools/golden/compare.py" --self-test || exit $?

# The documentation checker, for the same reason the golden self-test is here:
# a gate nobody pulls is not a gate. An audit found false claims spread across
# six of eleven load-bearing documents, and every one arrived the same way --
# a later commit closed a gap and nobody returned to the document that had
# named it. The markers assert only what has a single ground truth (an op in
# `_aten_implemented()`, a key in a table, a count read from a suite's own
# summary line), so this cannot cry wolf on prose; docs/DOCWATCH.md says what
# it structurally cannot see.
TORCH_C_ARTEFACT="$stage/_C.abi3.so" \
    exec "${PYTHON:-python3}" "$repo_root/tools/docwatch/check_docs.py"
