import io.kinference.gradle.Versions
import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests
import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        browser()

        configureTests()
        configureBenchmarkTests()
        configureHeavyTests()
    }

    sourceSets {
        val jsMain by getting() {
            dependencies {
                implementation(project(":serialization"))
                api(project(":inference:inference-ir"))

                implementation(npm("@tensorflow/tfjs-core", Versions.TFJS))
                implementation(npm("@tensorflow/tfjs-backend-webgl", Versions.TFJS))

                implementation(project(":inference:inference-api"))

                api(project(":utils:logger"))
                api(project(":utils:model-profiler"))

                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }

        val jsTest by getting() {
            dependencies {
                implementation(kotlin("test-js"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:test-utils"))
            }
        }
    }
}
