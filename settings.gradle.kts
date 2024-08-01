rootProject.name = "kinference"

include(":ndarray:ndarray-api")
include(":ndarray:ndarray-core")
include(":ndarray:ndarray-tfjs")

include(":serialization:serializer-protobuf")
include(":serialization:serializer-tiled")

include(":inference:inference-api")
include(":inference:inference-ir")
include(":inference:inference-ir-trees")
include(":inference:inference-core")
include(":inference:inference-ort")
include(":inference:inference-ort-gpu")
include(":inference:inference-tfjs")

include(":utils:utils-testing")
include(":utils:utils-logger")
include(":utils:utils-profiling")
include(":utils:utils-common")

include(":adapters:multik:adapter-multik-core")
include(":adapters:multik:adapter-multik-ort")
include(":adapters:multik:adapter-multik-ort-gpu")

include(":adapters:kmath:adapter-kmath-core")
include(":adapters:kmath:adapter-kmath-ort")
include(":adapters:kmath:adapter-kmath-ort-gpu")


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
