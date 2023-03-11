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
        configureHeavyTests()
        configureBenchmarkTests()
    }

    jvm {
        configureTests()
//        configureHeavyTests()
//        configureBenchmarkTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
                implementation("com.squareup.okio:okio:3.0.0")

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-ir"))

                api(project(":utils:utils-logger"))
                api(project(":utils:utils-profiling"))
                api(project(":utils:utils-common"))

                api(project(":serialization:serializer-tiled"))
            }
        }


        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:utils-testing"))
            }
        }

        val jvmTest by getting {
            dependencies {
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
            }
        }
    }
}
