import io.kinference.gradle.Versions
import io.kinference.gradle.configureGpuTests

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureGpuTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":inference:inference-api"))
                api("org.jetbrains.kotlinx:multik-core:${Versions.multik}")
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
