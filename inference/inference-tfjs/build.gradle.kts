import io.kinference.gradle.Versions
import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests
import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

kotlin {
    js(IR) {
        browser()

        configureTests()
        configureBenchmarkTests()
        configureHeavyTests()
    }

    sourceSets {
        val jsMain by getting {
            dependencies {
                api(project(":serialization:serializer-protobuf"))

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-tfjs"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-ir"))

                api(project(":utils:utils-logger"))
                api(project(":utils:utils-profiling"))
                api(project(":utils:utils-common"))

                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
            }
        }

        val jsTest by getting {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
