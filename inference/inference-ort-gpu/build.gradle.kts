import io.kinference.gradle.*

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureGpuTests()
        configureBenchmarkTests()
    }

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":inference:inference-api"))
                api(project(":serialization:serializer-protobuf"))
                api(project(":utils:utils-logger"))
                api(project(":utils:utils-common"))
                api(libs.onnxruntime.gpu)
            }
        }

        jvmTest {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
