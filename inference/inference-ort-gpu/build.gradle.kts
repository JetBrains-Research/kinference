import io.kinference.gradle.Versions
import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureGpuTests

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureGpuTests()
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
                api("com.microsoft.onnxruntime:onnxruntime_gpu:${Versions.ort}")
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit5"))
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
            }
        }
    }
}
