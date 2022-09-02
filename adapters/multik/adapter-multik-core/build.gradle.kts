import io.kinference.gradle.configureTests
import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        configureTests()
    }

    js(IR) {
        browser()
        configureTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-core"))

                api("org.jetbrains.kotlinx:multik-core:${Versions.multik}")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:test-utils"))
            }
        }

        val jvmTest by getting {
            dependencies {
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
            }
        }
    }
}
