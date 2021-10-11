import io.kinference.gradle.configureTests

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
        val jvmMain by getting {
            dependencies {
                implementation(project(":inference-api"))
                implementation(project(":inference"))
                implementation(project(":ndarray"))
                implementation("org.jetbrains.kotlinx:multik-api:0.0.1")
                implementation("org.jetbrains.kotlinx:multik-default:0.0.1")
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-common"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":test-runner"))
                implementation(kotlin("test-junit5"))

                runtimeOnly("org.junit.jupiter:junit-jupiter-engine:5.6.2")
            }
        }
    }
}
