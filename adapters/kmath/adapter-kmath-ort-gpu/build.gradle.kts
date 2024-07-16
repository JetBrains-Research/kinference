import io.kinference.gradle.configureGpuTests

group = rootProject.group
version = rootProject.version

repositories {
    maven("https://repo.kotlin.link")
}

kotlin {
    jvm {
        configureGpuTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":inference:inference-api"))
                api(libs.kmath.core)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }

        val jvmMain by getting {
            dependencies {
                api(project(":inference:inference-ort-gpu"))
            }
        }
    }
}
