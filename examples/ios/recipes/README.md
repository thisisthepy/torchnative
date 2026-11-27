# Recipes for `pip { jit }`, iOS simulator

The same mechanism as `../../android/recipes/` (toolchain#152, pypackpack#168): the toolchain plugin
hands this directory to pypackpack's cross build, which builds every pin with no iOS wheel on PyPI
from its sdist, here for `arm64-apple-ios-simulator` (xcrun clang, iOS 13.0, CPython 3.15t). A
package's recipe is `<name>.toml`, with the keys of pypackpack's `[tool.ppp.recipes.<name>]`.

It is a directory of its own because recipes match by package name, not by target: numpy's long double
is a double on Apple arm64 and quad precision on Android.

| Package | Build backend | iOS simulator (toolchain, 2026-11-27) | Recipe |
|---|---|---|---|
| numpy 2.5.5 | meson-python | wired, never built for iOS | `numpy.toml`: `longdouble_format` `IEEE_DOUBLE_LE` (the patch applies to the 2.5.5 sdist) |
| regex | setuptools | supported | none |
| pyyaml 6.0.4 | setuptools | supported (the pure-Python fallback) | none |
| safetensors | maturin | refused: pypackpack#182 | none |
| tokenizers | maturin | refused: pypackpack#182 | none |
| hf-xet 1.8.0 | maturin | refused: pypackpack#182 (no rustls patch needed on iOS: Security.framework) | none |

## Last recorded result

Not built yet.
