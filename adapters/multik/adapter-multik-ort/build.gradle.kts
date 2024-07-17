import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureTests()
    }

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":inference:inference-ort"))
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
