import io.kinference.gradle.*

group = rootProject.group
version = rootProject.version

plugins {
    alias(libs.plugins.kotlin.atomicfu)
}

kotlin {
    jvm {
        configureTests()
        configureHeavyTests()
        configureBenchmarkTests()
    }

    sourceSets {
        jvmMain {
            dependencies {
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.atomicfu)
                implementation(libs.okio)

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-ir"))
                api(project(":inference:inference-ir-trees"))

                api(project(":utils:utils-logger"))
                api(project(":utils:utils-profiling"))
                api(project(":utils:utils-common"))

                api(project(":serialization:serializer-tiled"))
            }
        }


        jvmTest {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
