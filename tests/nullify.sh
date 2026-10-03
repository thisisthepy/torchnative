#!/bin/sh
# Rebuild, restage, and run test_voice4.py. Prints ok/FAIL lines only.
set -e
W=${W:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$W/rust/torch_c"
PATH="$HOME/.cargo/bin:$PATH" cargo build --release > /tmp/null_build.log 2>&1 || { echo "BUILD FAILED"; grep '^error' -A4 /tmp/null_build.log | head -20; exit 2; }
STAGE=${TMPDIR:-/tmp}/torch-c-stage-$(printf '%s' "$W" | cksum | cut -d' ' -f1)
cp "$W/rust/torch_c/target/release/lib_C.dylib" "$STAGE/_C.abi3.so"
cp "$W/rust/torch_c/target/release/lib_C.dylib" /Volumes/macMini/voice4-work/vendor/torch/_C.abi3.so
cd "$W"
TORCH_C_VENDOR_DIR=/Volumes/macMini/voice4-work/vendor \
TORCH_C_VOICE4_ASSETS=/Volumes/macMini/voice4-work \
HF_HOME=/Volumes/macMini/voice4-work/hf \
PYTHONPATH="$STAGE:$W/rust/torch_c/pytests" \
/Volumes/macMini/thisisthepy/torchnative/.caches/spike-venv/bin/python rust/torch_c/pytests/test_voice4.py 2>&1 | grep -E "^(ok|FAIL)" || true
