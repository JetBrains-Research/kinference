import io.kinference.gradle.configureGpuTests
import io.kinference.gradle.Versions

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

        val jvmMain by getting {
            dependencies {
                api(project(":inference:inference-ort-gpu"))
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit5"))
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
            }
        }
    }
}
