import io.kinference.gradle.configureTests
import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        testRuns["test"].executionTask {
            configureTests()

            enabled = !project.hasProperty("disable-tests")
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":inference:inference-api"))
                api(project(":ndarray"))
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
                api(project(":inference:inference-core"))
                api("org.jetbrains.kotlinx:multik-api:${Versions.multik}")
                api("org.jetbrains.kotlinx:multik-default:${Versions.multik}")
            }
        }

        val jvmTest by getting {
            dependencies {
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
            }
        }
    }
}
