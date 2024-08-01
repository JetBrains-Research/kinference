import io.kinference.gradle.*

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
        jsMain {
            dependencies {
                api(project(":serialization:serializer-protobuf"))

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-tfjs"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-ir"))
                api(project(":inference:inference-ir-trees"))

                api(project(":utils:utils-logger"))
                api(project(":utils:utils-profiling"))
                api(project(":utils:utils-common"))

                api(libs.kotlinx.coroutines.core)
            }
        }

        jsTest {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
