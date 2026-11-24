# torchnative on Android

The common task of [`examples/`](../README.md) on an Android phone: SmolLM2-135M in float32, greedy,
32 new tokens after "On-device inference is", streamed, on the phone's CPU. The token ids are
checked against upstream torch's, recorded on a host in
`src/commonMain/python/src/main/tn_example/reference.json`. Then the same model with
`.to(device.npu)` through ExecuTorch (#423).

Status: `partial`. The app and the common task are written, and the task agrees with upstream on a
host (below). The APK is not built yet: `pip { jit }` is not wired in the toolchain plugin, and
the four recipes in `recipes/` come from the toolchain team (#437).

## Hardware

- Galaxy Z Flip4 (Snapdragon 8+ Gen 1), arm64-v8a.
- Galaxy Tab S9 Ultra (Snapdragon 8 Gen 2), when it is attached.

torchnative's Android wheel is arm64-v8a only, so the app is built for that ABI alone.

## How it is built

A Gradle build of its own, using the toolchain plugin and python-multiplatform (the shape of the
toolchain repository's `usage-example`). The plugin installs the Python dependencies per ABI:

- From PyPI: `torchnative` (an `android_24_arm64_v8a` abi3 wheel) and `transformers`.
- From `pip { jit }`: `numpy`, `safetensors`, `tokenizers` and `regex`. PyPI has no Android
  wheels for them (checked 2026-11-25), so they are cross-built from source with the recipes in
  `recipes/` (one `<name>.toml` per package, keys as in `[tool.ppp.recipes.<name>]`). Using
  `pip { jit }` here is the owner's proposal of 2026-11-25.

```sh
./gradlew assembleDebug
```

The toolchain plugin and python-multiplatform resolve from mavenLocal until they are published.

## How to run it

```sh
export ANDROID_SERIAL=<the phone's serial>
adb install -r build/outputs/apk/debug/*.apk
adb shell am start -n org.thisisthepy.torchnative.example/.MainActivity --es device cpu --ez headless true
adb logcat -s TorchnativeExample
```

The first run downloads the model (about 540 MB) into the app's files directory. `--es device npu`
runs `.to(device.npu)`. A device that cannot run says so and the run fails; it never falls back to
the CPU.

## Expected output

The last logcat line is `COMMON_TASK {...}` with `device`, `text`, `tokens_per_second`, `ids`, and
`agrees_with_upstream`, which must be `true`. Upstream's text for this task is:

> a very powerful technique for learning from data. It is a powerful technique for learning from
> data because it is a very fast and efficient way to learn from data.

## Recorded results

| Date | Commit | Device | Result |
|---|---|---|---|
| 2026-11-25 | this branch | Mac mini M1 host, torchnative shim (not the phone) | `cpu`: 32 of 32 ids equal upstream's, 36.25 tokens/s. Checks the task's code, not the phone |

`reference.json` was written by `reference.py` with upstream torch 2.13.0 and transformers 5.15.1.
