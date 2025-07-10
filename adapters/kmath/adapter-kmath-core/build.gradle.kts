import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.testplugin")
}

kotlin {
    jvm {
        configureTests()
    }

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-core"))

                api(libs.kmath.core)
            }
        }

        jvmTest {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
