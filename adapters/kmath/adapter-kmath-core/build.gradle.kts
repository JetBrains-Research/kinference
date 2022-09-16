import io.kinference.gradle.configureTests
import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

repositories {
    maven("https://repo.kotlin.link")
}

kotlin {
    jvm {
        configureTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-core"))

                api("space.kscience:kmath-core:${Versions.kmath}")
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
