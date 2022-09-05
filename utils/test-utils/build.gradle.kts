import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")

                api(project(":ndarray:ndarray-api"))
               // api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":serialization:serializer-tiled"))
                api(project(":utils:logger"))
                api(project(":utils:model-profiler"))
                api(project(":utils:common-utils"))

                api(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
            }
        }

    }
}
