import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// torchnative's Android example (examples/README.md, issue #437): the common task, SmolLM2-135M
// greedy, 32 new tokens, on the phone's CPU, checked against upstream's token ids; then
// `.to(device.npu)` (#423).

group = "org.thisisthepy.torchnative"
version = "0.1.0"

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.application)
    alias(libs.plugins.toolchain)
    alias(libs.plugins.python.multiplatform.bindings)
}

python {
    compileSdk = PY3_15_0
    // The Python half: `tn_example` (the common task) and its dependencies.
    localLibraryPath = "src/commonMain/python"
    sourceSets {
        val commonMain by getting {
            dependencies {
                // The app's runtime is CPython 3.15.0t, so this must be the cp315t wheel, which
                // is built locally (README, "How it is built") and read from `local` below. PyPI's
                // cp313-abi3 Android wheel cannot load on a free-threaded interpreter.
                implementation("torchnative")
                implementation("transformers")
                // No Android wheels on PyPI (checked 2026-11-25), so these four come from `jit`
                // below: cross-built from source with the recipes in `recipes/`.
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
                // torchnative's cp315t Android wheel, from scripts/wheel/build.py --abi ft.
                local {
                    url = layout.buildDirectory.dir("wheels").get().asFile.toURI().toString()
                }
                // The owner's proposal (2026-11-25): build what PyPI lacks through pypackpack's
                // `pip { jit }`. One `<name>.toml` per package, keys as `[tool.ppp.recipes.<name>]`.
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
    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }
    sourceSets {
        androidMain.dependencies {
            implementation(libs.python.multiplatform)
        }
    }
}

android {
    namespace = "$group.example"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "$group.example"
        minSdk = libs.versions.android.minSdk.get().toInt()
        targetSdk = libs.versions.android.targetSdk.get().toInt()
        versionCode = 1
        versionName = version as String
        // torchnative's Android wheel is arm64-v8a only.
        ndk { abiFilters += "arm64-v8a" }
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}
