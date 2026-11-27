#!/usr/bin/env bash
# Xcode Run Script phase for torchnative's iOS example (#437), from toolchain's usage-example
# (toolchain#22): put this app's python/ payload, CPython's standard library and Python.framework
# into the .app.
#
# The payload comes from this build: stagePythonBundleIosForXcode prints PYTHON_PAYLOAD_DIR=, which is
# passed to python-multiplatform's install-python.sh as PYTHON_PAYLOAD_DIR. install-python.sh is the
# ONLY phase that copies <app>/python/.
#
# Python.xcframework comes from python-multiplatform's published plugin (preparePythonXcframework,
# python-multiplatform#90), so no python-multiplatform checkout is needed, unlike toolchain's
# usage-example, which predates that task.
set -euo pipefail

# 1. This app's payload, from examples/ios's own build (its own wrapper, two levels up). Captured,
#    parsed, then checked, so a failing Gradle can never hand rsync an empty source.
cd "$SRCROOT/../.."
out=$(./gradlew -q stagePythonBundleIosForXcode)
payload=$(printf '%s\n' "$out" | sed -n 's/^PYTHON_PAYLOAD_DIR=//p')
[ -n "$payload" ] && [ -d "$payload" ] || { echo "error: no Python payload dir from stagePythonBundleIosForXcode" >&2; exit 1; }
export PYTHON_PAYLOAD_DIR="$payload"

# 2. Python.framework (libpython) into <app>/Frameworks, for the simulator or the device slice.
./gradlew -q preparePythonXcframework
xcframework="$PWD/build/xcode-frameworks/Python.xcframework"
case "${EFFECTIVE_PLATFORM_NAME:-}" in
    -iphonesimulator) slice="ios-arm64_x86_64-simulator" ;;
    -iphoneos) slice="ios-arm64" ;;
    *) echo "error: unknown EFFECTIVE_PLATFORM_NAME '${EFFECTIVE_PLATFORM_NAME:-}'" >&2; exit 1 ;;
esac
[ -d "$xcframework/$slice/Python.framework" ] || { echo "error: no $xcframework/$slice/Python.framework after preparePythonXcframework" >&2; exit 1; }
: "${TARGET_BUILD_DIR:?Xcode must set TARGET_BUILD_DIR}" "${FRAMEWORKS_FOLDER_PATH:?Xcode must set FRAMEWORKS_FOLDER_PATH}"
mkdir -p "$TARGET_BUILD_DIR/$FRAMEWORKS_FOLDER_PATH"
rsync -a --delete "$xcframework/$slice/Python.framework/" "$TARGET_BUILD_DIR/$FRAMEWORKS_FOLDER_PATH/Python.framework/"
if [ -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" ]; then
    codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" --timestamp=none "$TARGET_BUILD_DIR/$FRAMEWORKS_FOLDER_PATH/Python.framework"
fi

# 3. The stdlib prefix and the payload, by python-multiplatform's phase script, as the copy
#    installPythonXcodeScript writes to build/python-multiplatform/xcode/ (python-multiplatform#320).
./gradlew -q installPythonXcodeScript
GRADLEW="$PWD/gradlew" /bin/bash build/python-multiplatform/xcode/install-python.sh
