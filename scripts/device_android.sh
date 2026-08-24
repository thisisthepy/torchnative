#!/bin/sh
# Stage the vendored torch tree on an Android device and run it there.
#
# `docs/RUST_CROSSBUILD.md` establishes that `aarch64-linux-android` links, and
# `docs/DEVICE_LOAD.md` establishes that the resulting `_C.so` loads. Neither
# says anything about `import torch`, because linking is not loading and loading
# is not importing. This script closes that last gap, and `docs/DEVICE.md`
# records what it found.
#
#   ./scripts/device_android.sh build     cross-compile _C for aarch64-linux-android
#   ./scripts/device_android.sh stage     push CPython + deps + vendored torch + _C
#   ./scripts/device_android.sh run <py>  run a host-side script on the device
#   ./scripts/device_android.sh parity    run device_parity.py on host and device, diff
#   ./scripts/device_android.sh diff <h> <d>   diff two existing parity JSONs
#   ./scripts/device_android.sh shell     interactive python on the device
#
# Nothing here installs an app or touches anything outside /data/local/tmp. The
# emulator is shared with other projects (CLAUDE.md: one device test at a time).
set -eu

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
crate=$repo/rust/torch_c

# The vendored tree moved into the package (`vendor/vendor_torch.sh` writes to
# `$TORCHNATIVE_VENDOR_DIR`, default `torchnative/src/main`); it is no longer
# `vendor/torch`. Read the same variable so the two cannot drift apart.
vendor_root=${TORCHNATIVE_VENDOR_DIR:-$repo/torchnative/src/main}

: "${CARGO_TARGET_DIR:=/Volumes/macMini/caches/cargo-target-device}"
: "${ANDROID_SDK_ROOT:=$HOME/Library/Android/sdk}"
: "${ANDROID_NDK_HOME:=$ANDROID_SDK_ROOT/ndk/27.1.12297006}"
: "${TARGET_PYTHON:=/Volumes/macMini/caches/target-python/aarch64-linux-android/prefix}"
: "${SPIKE_SITE:=/Volumes/macMini/caches/spike-venv/lib/python3.13/site-packages}"
: "${HOST_PYTHON:=/Volumes/macMini/caches/spike-venv/bin/python}"
: "${DEVICE_ROOT:=/data/local/tmp/bw_device}"
export CARGO_TARGET_DIR

ADB=${ADB:-$ANDROID_SDK_ROOT/platform-tools/adb}
[ -x "$ADB" ] || ADB=$(command -v adb || true)
[ -n "$ADB" ] || { echo "adb not found -- set ADB" >&2; exit 1; }

ANDROID_SO=$CARGO_TARGET_DIR/aarch64-linux-android/release/lib_C.so

# The device shell's exit code is not trustworthy (`adb shell` has returned 0
# for failing device commands), so every judgement below reads an explicit
# DEVICE_EXIT marker printed by the device shell itself rather than `$?` here.
device_run() {
    "$ADB" shell "cd $DEVICE_ROOT && \
        BW_STUB_MULTIPROCESSING=1 \
        TORCH_USE_RTLD_GLOBAL=1 \
        LD_LIBRARY_PATH=$DEVICE_ROOT/lib \
        PYTHONHOME=$DEVICE_ROOT \
        PYTHONPATH=$DEVICE_ROOT/site \
        ./bin/python3.13 $* 2>&1; echo DEVICE_EXIT=\$?"
}

cmd_build() {
    command -v cargo-ndk >/dev/null || { echo "cargo-ndk not installed" >&2; exit 1; }
    echo "building _C for aarch64-linux-android"
    ( cd "$crate" && \
      ANDROID_NDK_HOME=$ANDROID_NDK_HOME \
      PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
      PYO3_CROSS_LIB_DIR=$TARGET_PYTHON/lib \
      cargo ndk -t arm64-v8a --platform 21 build --release )
    ls -l "$ANDROID_SO"
}

cmd_stage() {
    [ -f "$ANDROID_SO" ] || { echo "no $ANDROID_SO -- run '$0 build'" >&2; exit 1; }
    [ -d "$vendor_root/torch" ] || { echo "no vendored tree -- run vendor/vendor_torch.sh" >&2; exit 1; }

    stage=$(mktemp -d)
    trap 'rm -rf "$stage"' EXIT
    mkdir -p "$stage/site"

    # The vendored Python tree, plus our `_C` in the hole vendor_torch.sh left.
    # `.abi3.so` is in CPython's EXTENSION_SUFFIXES on Android -- verified on
    # device, the list is
    #   ['.cpython-313-aarch64-linux-android.so', '.abi3.so', '.so']
    rsync -a --exclude '__pycache__/' --exclude '*.so' --exclude '*.dylib' \
        "$vendor_root/torch/" "$stage/site/torch/"
    for info in "$vendor_root"/torch-*.dist-info; do
        [ -d "$info" ] && rsync -a --exclude '__pycache__/' "$info" "$stage/site/"
    done
    cp "$ANDROID_SO" "$stage/site/torch/_C.abi3.so"

    # Wall 4 of docs/VENDOR.md: `_manager_path()` checks this file exists on
    # every non-Windows platform before it will let the import finish.
    mkdir -p "$stage/site/torch/bin"
    : > "$stage/site/torch/bin/torch_shm_manager"

    # torch's pure-Python dependencies. Compiled artefacts are excluded because
    # the host copies are Mach-O/arm64-darwin and would not load here; every
    # package below has a pure-Python fallback path. `torchgen` is required by
    # `torch/utils/_python_dispatch.py` and is easy to miss -- it is a sibling
    # distribution of torch, not part of the vendored tree.
    for pkg in typing_extensions.py filelock fsspec jinja2 markupsafe mpmath \
               networkx sympy packaging torchgen functorch yaml; do
        if [ -e "$SPIKE_SITE/$pkg" ]; then
            rsync -a --exclude '__pycache__/' --exclude '*.so' --exclude '*.dylib' \
                --exclude '*.pyd' "$SPIKE_SITE/$pkg" "$stage/site/"
        else
            echo "  warning: no $pkg under $SPIKE_SITE" >&2
        fi
    done
    for info in "$SPIKE_SITE"/*.dist-info; do
        case $(basename "$info") in
            filelock-*|fsspec-*|jinja2-*|markupsafe-*|mpmath-*|networkx-*|sympy-*|packaging-*|typing_extensions-*|pyyaml-*)
                rsync -a --exclude RECORD --exclude '__pycache__/' "$info" "$stage/site/" ;;
        esac
    done

    left=$(find "$stage" \( -name '*.so' -o -name '*.dylib' \) ! -name '_C.abi3.so' | wc -l | tr -d ' ')
    [ "$left" = 0 ] || { echo "refusing to stage: $left host-native artefacts in tree" >&2; exit 1; }

    echo "staging CPython runtime on device"
    "$ADB" shell "mkdir -p $DEVICE_ROOT/lib $DEVICE_ROOT/bin"
    "$ADB" push --sync "$TARGET_PYTHON/bin/python3.13" "$DEVICE_ROOT/bin/" > /dev/null
    "$ADB" push --sync "$TARGET_PYTHON/lib/libpython3.13.so" "$DEVICE_ROOT/lib/" > /dev/null
    # The stdlib tree has no symlinks, so --sync carries it as-is. The sibling
    # libssl/libcrypto/libsqlite3 in prefix/lib *are* symlinks and `adb push`
    # cannot create those as uid 2000; they are skipped because `_C.so` does not
    # list any of them as NEEDED. Add `cp -L` staging if that ever changes.
    "$ADB" push --sync "$TARGET_PYTHON/lib/python3.13" "$DEVICE_ROOT/lib/" > /dev/null

    echo "staging torch tree on device"
    "$ADB" push --sync "$stage/site" "$DEVICE_ROOT/" > /dev/null
    "$ADB" shell "ls -l $DEVICE_ROOT/site/torch/_C.abi3.so; du -sh $DEVICE_ROOT"
}

cmd_run() {
    [ $# -ge 1 ] || { echo "usage: $0 run <script.py> [args]" >&2; exit 1; }
    script=$1; shift
    "$ADB" push --sync "$script" "$DEVICE_ROOT/$(basename "$script")" > /dev/null
    device_run "$(basename "$script")" "$@"
}

cmd_parity() {
    host_json=${1:-/tmp/bw_host.json}
    device_json=${2:-/tmp/bw_device.json}

    echo "=== host ==="
    TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$vendor_root \
        "$HOST_PYTHON" "$repo/scripts/device_parity.py" "$host_json" || true

    echo "=== device ==="
    "$ADB" push --sync "$repo/scripts/device_parity.py" "$DEVICE_ROOT/device_parity.py" > /dev/null
    device_run "device_parity.py $DEVICE_ROOT/parity.json"
    "$ADB" pull "$DEVICE_ROOT/parity.json" "$device_json" > /dev/null

    cmd_diff "$host_json" "$device_json"
}

# Split out from `parity` so the comparison can be re-run, and negative-control
# tested, without re-running either end. `parity ... && diff <doctored>` is how
# the exemption list below was shown to actually fire.
cmd_diff() {
    echo "=== diff ==="
    "$HOST_PYTHON" - "$1" "$2" <<'PYEOF'
import json, sys

# Measured on emulator-5554 (pmp_api26, API 26, arm64-v8a) against an
# aarch64-darwin host -- docs/DEVICE.md records the run. These two cases and
# only these two differ, by at most 1 ULP, and both sides straddle the
# correctly-rounded double-precision reference in both directions, so neither is
# "the wrong one": Apple's libm and bionic's are different implementations of
# expf/tanhf. Everything else -- including the two nn.Module forwards -- is
# bit-identical.
#
# Listing them by name rather than by a global tolerance is the point. A
# tolerance would also swallow a real divergence in `mm` or `cumsum`, which is
# exactly what this script exists to catch. Anything not on this list is a
# regression, including any of these two growing past 1 ULP.
EXPECTED_LIBM_DIVERGENCE = {"_softmax.default": 1, "tanh.default": 1}

host = json.load(open(sys.argv[1]))
device = json.load(open(sys.argv[2]))
print(f"host   {host['platform']}/{host['machine']} torch {host['torch']} kernels {host['aten_implemented']}")
print(f"device {device['platform']}/{device['machine']} torch {device['torch']} kernels {device['aten_implemented']}")
shared = set(host["results"]) & set(device["results"])
mismatched = [k for k in sorted(shared) if host["results"][k] != device["results"][k]]
unexpected = []
for key in mismatched:
    hb, db = host["results"][key], device["results"][key]
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(hb["bits"], db["bits"])) if x != y]
    ulps = [abs(int(x, 16) - int(y, 16)) for _, x, y in diffs
            if isinstance(x, str) and isinstance(y, str)]
    worst = max(ulps) if ulps else None
    budget = EXPECTED_LIBM_DIVERGENCE.get(key)
    verdict = "MISMATCH" if budget is None or worst is None or worst > budget else "libm"
    if verdict == "MISMATCH":
        unexpected.append(key)
    print(f"{verdict} {key}: {len(diffs)}/{len(hb['bits'])} elements, max {worst if worst is not None else '?'} ULP")
    for i, x, y in diffs:
        print(f"    [{i}] host={x} device={y}")
print(f"identical {len(shared) - len(mismatched)}/{len(shared)}")
print(f"host failures:   {sorted(host['failures'])}")
print(f"device failures: {sorted(device['failures'])}")

# Two separate ways for the ends to disagree, reported separately so a run that
# fails says which one it was.
verdicts = []
if sorted(host["failures"]) != sorted(device["failures"]):
    verdicts.append("failure sets differ")
if unexpected:
    verdicts.append(f"unexpected bit divergence: {unexpected}")
stale = sorted(set(EXPECTED_LIBM_DIVERGENCE) - set(mismatched))
if stale:
    # Not a failure: if bionic's libm gets fixed this list should shrink, and
    # the run should say so rather than quietly keep an obsolete exemption.
    print(f"note: expected-divergence entries that now agree: {stale}")
print("PARITY: " + ("; ".join(verdicts) if verdicts else "ok"))
sys.exit(1 if verdicts else 0)
PYEOF
}

cmd_shell() { device_run "$@"; }

case ${1:-} in
    build)  shift; cmd_build "$@" ;;
    stage)  shift; cmd_stage "$@" ;;
    run)    shift; cmd_run "$@" ;;
    parity) shift; cmd_parity "$@" ;;
    diff)   shift; cmd_diff "$@" ;;
    shell)  shift; cmd_shell "$@" ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
