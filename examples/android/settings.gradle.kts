// examples/android is its own Gradle build (torchnative #437), run with its own wrapper:
// ./gradlew assembleDebug. It is a consumer of the toolchain plugin and python-multiplatform,
// shaped like toolchain's usage-example. Both resolve from Maven Central once published and fall
// back to mavenLocal while unreleased (`./gradlew :toolchain:publishToMavenLocal` in the toolchain
// repository).
pluginManagement {
    repositories {
        google {
            mavenContent {
                includeGroupAndSubgroups("androidx")
                includeGroupAndSubgroups("com.android")
                includeGroupAndSubgroups("com.google")
            }
        }
        mavenCentral()
        gradlePluginPortal()
        mavenLocal {
            content { includeGroupAndSubgroups("org.thisisthepy") }
        }
    }
}

dependencyResolutionManagement {
    repositories {
        google {
            mavenContent {
                includeGroupAndSubgroups("androidx")
                includeGroupAndSubgroups("com.android")
                includeGroupAndSubgroups("com.google")
            }
        }
        mavenCentral()
        mavenLocal {
            content { includeGroupAndSubgroups("org.thisisthepy") }
        }
    }
}

rootProject.name = "torchnative-example-android"
