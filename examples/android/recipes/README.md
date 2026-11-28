# Recipes for `pip { jit }`

The toolchain plugin hands this directory to pypackpack's cross build (`localRecipes` in
`../build.gradle.kts`, toolchain#152, pypackpack#168). With `jit` declared, the plugin pins the
example's requirements for `aarch64-linux-android`. It builds every pin that has no Android wheel on
PyPI from its sdist, in pypackpack's cross build environment (NDK clang for API 24, CPython 3.15t).
A package's recipe is `<name>.toml` here, with the keys of pypackpack's `[tool.ppp.recipes.<name>]`:

- `patches`: paths relative to this directory, applied with `patch -p1` to the unpacked sdist.
- `env`: extra environment variables for the build.

A package that builds as it is needs no file. The format was approved by the leader on 2026-11-25
(Decision 16 in the toolchain report).

## What is built, and what needs a recipe

The four requirements (`numpy`, `safetensors`, `tokenizers`, `regex`) resolve to 19 pins. Six of
them have no Android wheel on PyPI: the four, plus `hf-xet` and `pyyaml`, which `tokenizers` pulls in
through `huggingface-hub`. Checked 2026-11-25 with uv 0.12.3.

| Package | Build backend | Recipe |
|---|---|---|
| numpy 2.5.5 | meson-python | `numpy.toml`: a cross build cannot run numpy's long double probe, so `longdouble_format` defaults to `IEEE_QUAD_LE` (Android's long double on both ABIs) |
| safetensors 0.8.0 | maturin (PyO3 0.28) | none |
| tokenizers 0.23.3 | maturin (PyO3 0.29) | none |
| regex 2026.11.23 | setuptools | none |
| hf-xet 1.8.0 | maturin | `hf-xet.toml`: TLS roots from the device's system CA store (same rule as python-multiplatform #549), because rustls-platform-verifier needs a JNI init nothing in the app performs and every download panicked |
| pyyaml 6.0.4 | setuptools | none (the pure-Python fallback; no libyaml) |

## Last recorded result

2026-11-25, pypackpack `feat/cross-recipes` 52c4cfd (pypackpack#169), macOS arm64 host, NDK 27.1.
All six built as `cp315-cp315t-android_24_arm64_v8a` wheels with this directory as the recipe
directory, and all 23 extension modules in them are aarch64 ELF. Not yet run on a device: importing
them on the Galaxy Z Flip4 is this example's own check.

2026-11-27, hf-xet with `hf-xet.toml` (toolchain teammate, team lead's Decision (a)), toolchain develop's
usage-example with `hf-xet` and this directory as `localRecipes`, on an arm64 API 36 emulator. Without the
recipe, `hf_xet.download_files` of a public Xet file (`hf-internal-testing/tiny-random-gpt2`'s
`model.safetensors`) fails with `Task panic: "Expect rustls-platform-verifier to be initialized"`. With it, the
same download completes and its sha256 matches. hf_xet's cache defaults under `HOME`, which an Android app cannot
write, so set `HF_HOME` (or `HF_XET_CACHE`) to a writable directory such as the app's cache directory. A
recipe change for a version already installed is not picked up without removing the installed copy and the
cached wheel (see the toolchain report).
