import io.kinference.gradle.*

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureHeavyTests()
        configureBenchmarkTests()
    }

    sourceSets {
        jvmMain {
            dependencies {
                api(libs.onnxruntime.cpu)
                api(project(":inference:inference-api"))
                api(project(":serialization:serializer-protobuf"))
                api(project(":utils:utils-logger"))
                api(project(":utils:utils-common"))
            }
        }

        jvmTest {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
