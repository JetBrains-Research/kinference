rootProject.name = "kinference"

include(":ndarray")
include(":serialization")
include(":inference:inference-api")
include(":inference:inference-ir")
include(":inference:inference-core")
include(":inference:inference-ort")
include(":inference:inference-ort-gpu")
include(":inference:inference-tfjs")

include(":utils:test-utils")
include(":utils:logger")
include(":utils:model-profiler")
include(":utils:common-utils")

include(":adapters:multik:adapter-multik-core")
include(":adapters:multik:adapter-multik-ort")

include(":adapters:kmath:adapter-kmath-core")
include(":adapters:kmath:adapter-kmath-ort")


pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
    }

    resolutionStrategy {
        eachPlugin {
            if (requested.id.id == "io.kinference.primitives") {
                useModule("io.kinference.primitives:gradle-plugin-jvm:${requested.version}")
            }
        }
    }
}
