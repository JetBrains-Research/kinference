import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureHeavyTests()
        configureBenchmarkTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":inference:inference-api"))
                api(project(":serialization:serializer-protobuf"))
                api(project(":utils:utils-logger"))
                api(project(":utils:utils-common"))
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }

        val jvmMain by getting {
            dependencies {
                api(libs.onnxruntime.cpu)
            }
        }
    }
}
