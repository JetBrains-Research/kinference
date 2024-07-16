import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":inference:inference-api"))
                api(libs.multik.core)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }

        val jvmMain by getting {
            dependencies {
                api(project(":inference:inference-ort"))
            }
        }
    }
}
