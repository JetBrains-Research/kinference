import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests
import io.kinference.gradle.configureTests
import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        browser()

        configureTests()
        configureHeavyTests()
        configureBenchmarkTests()
    }

    jvm {
        configureTests()
        configureHeavyTests()
        configureBenchmarkTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
                implementation("com.squareup.okio:okio:3.0.0")

                api(project(":inference:inference-api"))
                api(project(":inference:inference-ir"))
                api(project(":ndarray"))
                api(project(":utils:logger"))
                api(project(":utils:model-profiler"))
                api(project(":utils:common-utils"))

                api(project(":serialization"))
            }
        }


        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:test-utils"))
            }
        }

        val jvmTest by getting {
            dependencies {
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
            }
        }
    }
}
