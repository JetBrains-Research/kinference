import io.kinference.gradle.configureTests
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
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:test-utils"))
            }
        }

        val jvmMain by getting {
            dependencies {
                api(project(":inference:inference-ort-gpu"))
                api("org.jetbrains.kotlinx:multik-api:${Versions.multik}")
                api("org.jetbrains.kotlinx:multik-default:${Versions.multik}")
            }
        }

        val jvmTest by getting {
            dependencies {
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
                implementation(kotlin("test-junit5"))
            }
        }
    }
}
