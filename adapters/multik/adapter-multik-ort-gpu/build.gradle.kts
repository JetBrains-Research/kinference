import io.kinference.gradle.configureGpuTests

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureGpuTests()
    }

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":inference:inference-ort-gpu"))
                api(project(":inference:inference-api"))
                api(libs.multik.core)
            }
        }

        jvmTest {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
