#!/usr/bin/env bash
# examples/ios in the macOS job's simulator (#437), the way toolchain's ios-e2e runs its usage-example:
# build the app with the toolchain Gradle plugin and xcodebuild, install it on a simulator, launch it
# with --common-task cpu, and read the JSON line it writes to Documents/common-task.txt.
#
# Called from .github/workflows/test.yml's macos job, shard 3 only, after the gate, on the host shim
# that job installed. Its exit code is the step's: 0 only for a COMMON_TASK line whose ids agree with
# upstream's reference.json.
#
# The toolchain plugin (1.0.0-alpha) and python-multiplatform (3.15.0-alpha03) are not on Maven Central
# yet, so pypackpack, toolchain and python-multiplatform are checked out at develop under .ci/ and
# published to mavenLocal first (leader, 2026-11-27), and their SHAs are printed. The .ci/ is under the
# runner's temp, not the repository root, which takes no new top-level entry (AGENTS.md section 2).
# This switches to their Maven Central coordinates, and the checkouts go away, once they are published.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
temp=${RUNNER_TEMP:-$repo/.scratch}
export JAVA_HOME=${JAVA_HOME_21_arm64:-${JAVA_HOME:?a JDK 21 is needed}}

# uv comes first: python-multiplatform-ksp's typedpythonGateBundle runs `uv build` during its
# publish (run 45990525704 stopped there on FileNotFoundError), and pypackpack installs the
# app's dependencies with it.
echo "== uv"
python -m pip install --quiet uv
uv --version

echo "== the three unpublished builds, at develop"
ci="$temp/.ci"
mkdir -p "$ci"
for name in pypackpack toolchain python-multiplatform; do
    rm -rf "${ci:?}/$name"
    git clone --quiet --depth 1 --branch develop "https://github.com/thisisthepy/$name" "$ci/$name"
    echo "$name develop $(git -C "$ci/$name" rev-parse HEAD)"
done
(cd "$ci/pypackpack" && ./gradlew :packpack:publishToMavenLocal --console=plain)
(cd "$ci/toolchain" && ./gradlew :toolchain:publishToMavenLocal --console=plain)
(cd "$ci/python-multiplatform" && ./gradlew \
    :python-multiplatform:publishIosSimulatorArm64PublicationToMavenLocal \
    :python-multiplatform:publishKotlinMultiplatformPublicationToMavenLocal \
    :python-multiplatform-ksp:publishToMavenLocal --console=plain)
(cd "$ci/python-multiplatform" && ./gradlew -p python-multiplatform-gradle-plugin publishToMavenLocal --console=plain)

echo "== torchnative's cp315t simulator wheel"
archive="$temp/python-3.15.0t-iOS-XCframework.tar.gz"
curl --fail --location --silent --show-error --retry 5 -o "$archive" \
    https://github.com/thisisthepy/python-multiplatform/releases/download/cpython-3.15.0t-mobile/python-3.15.0t-iOS-XCframework.tar.gz
python scripts/wheel/ios_ft_prefix.py "$archive"
prefix="$repo/.caches/target-python/aarch64-apple-ios-sim-freethreaded"
cat > "$temp/pyo3-ios-sim-ft.cfg" <<'EOF'
implementation=CPython
version=3.15
shared=true
abi3=false
lib_name=Python
pointer_width=64
build_flags=Py_GIL_DISABLED
suppress_build_script_link_lines=true
EOF
rustup target add aarch64-apple-ios-sim
export TORCHNATIVE_FT_CARGO_TARGET_DIR="$temp/target-ft"
(cd torchnative/rust/torch_c && CARGO_TARGET_DIR="$TORCHNATIVE_FT_CARGO_TARGET_DIR" PYO3_CROSS=1 \
    PYO3_CROSS_PYTHON_VERSION=3.15 PYO3_CROSS_LIB_DIR="$prefix/lib" \
    PYO3_CONFIG_FILE="$temp/pyo3-ios-sim-ft.cfg" IPHONEOS_DEPLOYMENT_TARGET=13.0 \
    cargo build --target aarch64-apple-ios-sim --release --no-default-features)
rm -rf "$temp/ios-wheel" && mkdir -p "$temp/ios-wheel"
python scripts/wheel/build.py --abi ft --target ios-arm64-sim --outdir "$temp/ios-wheel"
rm -rf examples/ios/build/wheels && mkdir -p examples/ios/build/wheels
cp "$temp"/ios-wheel/*-cp315t-ios_*_arm64_iphonesimulator.whl examples/ios/build/wheels/
ls -l examples/ios/build/wheels

echo "== the app, for the arm64 simulator"
(cd examples/ios && ./gradlew --console=plain preparePythonXcframework)
slice="$repo/examples/ios/build/xcode-frameworks/Python.xcframework/ios-arm64_x86_64-simulator"
xcodebuild -project examples/ios/src/iosMain/app.xcodeproj -target iosApp \
    -sdk iphonesimulator -configuration Debug \
    ARCHS=arm64 ONLY_ACTIVE_ARCH=YES CODE_SIGNING_ALLOWED=NO \
    SYMROOT="$temp/build-ios" \
    FRAMEWORK_SEARCH_PATHS="\$(inherited) $slice" \
    OTHER_LDFLAGS="\$(inherited) -framework Python" \
    build
app=$(find "$temp/build-ios/Debug-iphonesimulator" -maxdepth 1 -name '*.app' | head -n 1)
[ -n "$app" ] || { echo "no .app built"; exit 1; }
test -f "$app/python/tn_example/common_task.py" || { echo "no python/tn_example in the app"; exit 1; }
test -d "$app/python/torch" || { echo "no python/torch in the app"; exit 1; }
test -d "$app/Frameworks/Python.framework" || { echo "no Python.framework in the app"; exit 1; }

echo "== the common task in the simulator"
udid=$(xcrun simctl list devices available -j | python3 -c 'import json,sys; d=json.load(sys.stdin)["devices"]; print(next(x["udid"] for k,v in d.items() if "iOS" in k for x in v if x["name"].startswith("iPhone")))')
xcrun simctl boot "$udid" || true
xcrun simctl bootstatus "$udid" -b
for i in $(seq 1 60); do
    xcrun simctl spawn "$udid" launchctl list 2>/dev/null | grep -q com.apple.SpringBoard && break
    sleep 5
done
xcrun simctl install "$udid" "$app"
bundle=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Info.plist")
# The first launch downloads the model (about 540 MB), so the wait is longer than ios-e2e's.
report=""
for attempt in 1 2; do
    data=$(xcrun simctl get_app_container "$udid" "$bundle" data)
    rm -f "$data/Documents/common-task.txt"
    xcrun simctl launch --terminate-running-process "$udid" "$bundle" --common-task cpu || echo "launch returned non-zero"
    for i in $(seq 1 120); do
        [ -s "$data/Documents/common-task.txt" ] && break
        sleep 10
    done
    if [ -s "$data/Documents/common-task.txt" ]; then
        report=$(cat "$data/Documents/common-task.txt")
        break
    fi
    echo "attempt $attempt: no report after 1200 s"
done
xcrun simctl shutdown "$udid" || true
echo "$report"
printf '%s\n' "$report" | grep -q '^COMMON_TASK {' || { echo "no COMMON_TASK line"; exit 1; }
printf '%s\n' "$report" | grep -q '"agrees_with_upstream": true' || { echo "the ids differ from upstream's"; exit 1; }
echo "examples/ios: the common task agrees with upstream in the simulator"
