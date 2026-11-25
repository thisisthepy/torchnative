# torchnative examples

Runnable samples, one folder per platform. Each one runs the **same task** so results can be
compared across devices, then adds what only that device can show.

Requested by the owner on 2026-11-25 ("examples에 넣고, 샘플 구성을 해당 폴더 안에 README.md 새로
만들어서 기록해줘. 각 디바이스별로 뭘 예시로 보는건지 기획 잘 해서"). The Android build through
pypackpack's `pip { jit }` is also the owner's proposal of the same day ("빌드한다고 하면 아예 예제
폴더에 pypackpack으로 pip jit으로 하는 기능 사용해서 돌리는거 코드 작성해두면 어떨까?"). The
folder list and the per-device content below were proposed by the leader agent and approved by the
owner on 2026-11-25 ("그래 그렇게 해").

## The common task

Every example runs this first, so a number from one device means the same thing on another:

- Model: `HuggingFaceTB/SmolLM2-135M`, float32, loaded with `transformers`' `from_pretrained`.
- Prompt: `"On-device inference is"`, greedy, 32 new tokens, streamed.
- Output: the device name, the generated text, tokens per second, and whether the token ids equal
  the `cpu` run's ids (the agreement check).

An example that cannot run its device (no hardware, never built) says so and exits non-zero. It
never falls back to `cpu` silently.

## Layout

| Folder | Hardware we test on | What it shows beyond the common task | Status today |
|---|---|---|---|
| `macos/` | Mac mini M1 | `cpu`, `mps` (Metal) and `.to(device.npu)` (Apple Neural Engine via CoreML) on one model, side by side; the `mps` refusal list (`_C._shim_mps_host_readback_ops`) printed when an op is refused | cpu, mps, ANE run (README Status) |
| `android/` | Galaxy Z Flip4 (Snapdragon 8+ Gen 1); Galaxy Tab S9 Ultra (Snapdragon 8 Gen 2) | A Gradle app built with the toolchain plugin. Its `pip { jit { localRecipes { ... } } }` block cross-builds numpy, safetensors, tokenizers and regex, which have no Android wheels on PyPI (checked 2026-11-25). Then `cpu` generate on the phone, and `.to(device.npu)` through ExecuTorch (#423, XNNPACK first, QNN after) | cpu runs on device; the four wheels and #423 not yet |
| `ios/` | iOS simulator arm64; an iPhone when one is attached | `cpu` generate in the simulator; CoreML / Neural Engine on a device | simulator partial; device never run |
| `linux/` | x86_64 CI runner; aarch64 | `cpu` generate; `vulkan` (thirty-one ops, BERT forward); `cuda` on an NVIDIA machine | cpu partial; vulkan and cuda never run on Linux |
| `windows/` | x86_64; arm64 | `cpu` generate; `.to(device.npu)` on an Intel Core Ultra NPU through OpenVINO | cpu partial; Intel NPU never run on hardware |
| `web/` | a desktop browser | tensor basics on the WASM `cpu` build (no LLM: the model download and memory do not fit the target) | cpu runs |

## What each folder contains

- `README.md`: what the example shows, the hardware, how to run it, the expected output, and the
  last recorded result with its date.
- The run files. Desktop examples are one script plus a `pyproject.toml`, run with `uv run` or
  `ppp`. Android and iOS are Gradle projects that use the toolchain plugin.
- `android/recipes/`: one `<name>.toml` per cross-built package, the recipe-directory format
  toolchain uses for `localRecipes` (keys as in `[tool.ppp.recipes.<name>]`).

## Order of work

1. `android/`: unblocks #423 on the Flip4. Depends on toolchain wiring `localRecipes` and on a
   maturin cross build for safetensors and tokenizers.
2. `macos/`: everything already runs; the example fixes the numbers in one place.
3. `linux/`, `windows/`: CPU first; CUDA, Vulkan and the Intel NPU when hardware is available.
4. `ios/`, `web/`.

## Rules for examples

- Install and run with `uv` or `ppp`. Never `pip install`.
- Build output, downloaded models and cross-built wheels go under the example's git-ignored
  `build/` or the repository's existing caches, never next to the sources.
- A recorded result names the commit, the device and the date.
