# torchnative on iOS

The common task of [`examples/`](../README.md) in an iOS app on the arm64 simulator: SmolLM2-135M in
float32, greedy, 32 new tokens after "On-device inference is", on the simulator's CPU. The token ids
are checked against upstream torch's (`src/commonMain/python/src/main/tn_example/reference.json`, the
same ids as every other example).

Status: `partial`. The app, its build, the cp315t simulator wheel and its CI step exist; the app has
not run yet. pypackpack builds all six dependencies for the iOS simulator since pypackpack#185 (the
maturin three, `safetensors`, `tokenizers` and `hf-xet`, through pypackpack#182). Not started: a
physical iPhone (the device wheel is not wired, `scripts/wheel/ftabi.py`) and CoreML on a device.

## Hardware

- The iOS simulator, arm64, on a Mac (CI's macOS runner, like toolchain's ios-e2e).
- A physical iPhone: not yet.

Where it runs: `.github/workflows/test.yml`'s macOS job runs `.github/scripts/examples_ios.sh` after
the gate, on shard 3 only. The script builds the wheel and the app and runs the common task in a
simulator; its exit code is the step's. The toolchain plugin and python-multiplatform are not on
Maven Central yet, so the script checks out pypackpack, toolchain and python-multiplatform at
`develop` (under the runner's temp) and publishes them to mavenLocal first, printing their SHAs. It
switches to their Maven Central coordinates once they are published.

## How it is built

An app built by the toolchain Gradle plugin, the shape of the toolchain repository's `usage-example`
and of `examples/android`:

- `build.gradle.kts`: a Kotlin Multiplatform `iosSimulatorArm64` framework (`Project`, which the
  Xcode project imports) and the `python { }` block. `iosSimulatorArm64()` there registers the
  dependency install for `arm64-apple-ios-simulator`.
- The app's Python is CPython 3.15.0t (free-threaded), python-multiplatform's iOS XCframework
  (`python-3.15.0t-iOS-XCframework.tar.gz`, sha256 `b8c0a4b3...e6b5`), which the bindings plugin
  downloads and stages itself (`preparePythonXcframework`). PyPI's `cp313-abi3` iOS wheel cannot load
  there, so torchnative's `cp315-cp315t-ios_14_0_arm64_iphonesimulator` wheel is built from this
  repository first (below) and read from `build/wheels/`.
- From PyPI: `transformers`. From `pip { jit }`: `numpy`, `safetensors`, `tokenizers`, `regex`, and
  with them `hf-xet` and `pyyaml`, cross-built from source with the recipes in `recipes/`. PyPI has no
  iOS wheel for any of them (checked 2026-11-27).
- `src/iosMain/app.xcodeproj` (from usage-example) builds the app. Its two phases run this build's
  `embedAndSignAppleFrameworkForXcode` and `install-python-phase.sh`, which puts Python.framework, the
  standard library and the app's Python payload into the `.app`.

First the torchnative wheel, from the repository root. It needs the simulator slice of the same
XCframework composed into `.caches/target-python/aarch64-apple-ios-sim-freethreaded`
(`scripts/wheel/ftabi.py` names the layout when it is missing):

```sh
(cd torchnative/rust/torch_c && CARGO_TARGET_DIR=$PWD/target-ft PYO3_CROSS=1 \
  PYO3_CROSS_PYTHON_VERSION=3.15 \
  PYO3_CROSS_LIB_DIR=$PWD/../../../.caches/target-python/aarch64-apple-ios-sim-freethreaded/lib \
  PYO3_CONFIG_FILE=<the 3.15t config ftabi.py names> IPHONEOS_DEPLOYMENT_TARGET=13.0 \
  cargo build --target aarch64-apple-ios-sim --release --no-default-features)
uv run python scripts/wheel/build.py --abi ft --target ios-arm64-sim
mkdir -p examples/ios/build/wheels && cp dist/*-cp315t-ios_*_arm64_iphonesimulator.whl examples/ios/build/wheels/
```

Then the app, with a JDK 17 or 21 and Xcode:

```sh
cd examples/ios
xcodebuild -project src/iosMain/app.xcodeproj -target iosApp -sdk iphonesimulator -arch arm64 \
  -configuration Debug CODE_SIGNING_ALLOWED=NO build
```

## How to run it

```sh
xcrun simctl install <device> build/Debug-iphonesimulator/TorchnativeExample.app
xcrun simctl launch <device> org.thisisthepy.torchnative.example --common-task cpu
cat "$(xcrun simctl get_app_container <device> org.thisisthepy.torchnative.example data)/Documents/common-task.txt"
```

The app runs the task and exits before any UI (`src/iosMain/kotlin/.../CommonTask.ios.kt`). The
first run downloads the model (about 540 MB) into the app's Caches directory.

## Expected output

`common-task.txt` holds one line, `COMMON_TASK {...}`, with `device`, `text`, `tokens_per_second`,
`ids` and `agrees_with_upstream`, which must be `true`; or `COMMON_TASK_FAILED: <error>`.

## Last recorded result

Not run yet.
