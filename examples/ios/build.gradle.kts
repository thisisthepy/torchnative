// torchnative's iOS example (examples/README.md, issue #437): the common task, SmolLM2-135M greedy,
// 32 new tokens, in an iOS app on the arm64 simulator, checked against upstream's token ids.

group = "org.thisisthepy.torchnative"
version = "0.1.0"

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.toolchain)
    alias(libs.plugins.python.multiplatform.bindings)
}

python {
    compileSdk = PY3_15_0
    // The Python half: `tn_example` (the common task, the same file as examples/android's).
    localLibraryPath = "src/commonMain/python"
    // The simulator's variant, which registers its dependency install and the jit cross build for
    // arm64-apple-ios-simulator.
    iosSimulatorArm64()
    sourceSets {
        val commonMain by getting {
            dependencies {
                // The app's runtime is CPython 3.15.0t, so this must be the cp315t simulator wheel, which
                // is built locally (README, "How it is built") and read from `local` below. PyPI's
                // cp313-abi3 iOS wheel cannot load on a free-threaded interpreter.
                implementation("torchnative")
                implementation("transformers")
                // No iOS wheels on PyPI (checked 2026-11-27), so these come from `jit` below. The
                // maturin three (safetensors, tokenizers, and hf-xet through huggingface-hub) wait for
                // pypackpack#182, which brings maturin cross builds to iOS.
                implementation("numpy")
                implementation("safetensors")
                implementation("tokenizers")
                implementation("regex")
            }
        }
    }
    defaultConfig {
        versionCode = 1
        versionName = "0.1.0"
        pip {
            autoUpdate = false
            repositories {
                central {
                    setUrl("https://pypi.org/simple", "https://pypi.org/simple")
                }
                // torchnative's cp315t simulator wheel, from scripts/wheel/build.py --abi ft.
                local {
                    url = layout.buildDirectory.dir("wheels").get().asFile.toURI().toString()
                }
                // This directory's own recipes, not examples/android's: recipes match by package
                // name, and numpy's long double differs between the two (recipes/numpy.toml).
                jit {
                    localRecipes {
                        add("recipes")
                    }
                }
            }
        }
    }
    packaging {
        fileName = "torchnative-example"
    }
}

kotlin {
    // The simulator only: the device wheel is not built yet (scripts/wheel/ftabi.py).
    iosSimulatorArm64().binaries.framework {
        // `Project` is what src/iosMain/app.xcodeproj imports (it comes from toolchain's usage-example).
        baseName = "Project"
        isStatic = true
    }
    sourceSets {
        iosMain.dependencies {
            implementation(libs.python.multiplatform)
        }
    }
}
