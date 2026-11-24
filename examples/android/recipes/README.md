# Recipes for `pip { jit }`

One `<name>.toml` per package that PyPI has no Android wheel for (`numpy`, `safetensors`,
`tokenizers`, `regex`), with the keys of `[tool.ppp.recipes.<name>]` (`patches`, `env`). The
toolchain plugin hands this directory to pypackpack's cross build (`localRecipes` in
`../build.gradle.kts`). The recipes come from the toolchain team (#437).
