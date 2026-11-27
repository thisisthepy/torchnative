// examples/ios is its own Gradle build (torchnative #437), run with its own wrapper. Xcode drives it
// through src/iosMain/app.xcodeproj (embedAndSignAppleFrameworkForXcode, install-python-phase.sh).
// It is a consumer of the toolchain plugin and python-multiplatform, shaped like toolchain's
// usage-example and like examples/android. Both resolve from Maven Central once published and fall
// back to mavenLocal while unreleased.
pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
        mavenLocal {
            content { includeGroupAndSubgroups("org.thisisthepy") }
        }
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
        mavenLocal {
            content { includeGroupAndSubgroups("org.thisisthepy") }
        }
    }
}

rootProject.name = "torchnative-example-ios"
