import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests
import io.kinference.gradle.configureTests
import io.kinference.gradle.s3.S3Dependency

plugins {
    kotlin("kapt") apply true
}

group = rootProject.group
version = rootProject.version

kotlin {
    js {
        testRuns["test"].configureAllExecutions {
            filter {
                excludeTestsMatching("*.heavy_*")
                excludeTestsMatching("*.benchmark_*")
            }

            executionTask.get().enabled = !project.hasProperty("disable-tests")
        }

        testRuns.create("heavy").configureAllExecutions {
            filter {
                includeTestsMatching("*.heavy_*")
            }

            executionTask.get().enabled = !project.hasProperty("disable-tests")
            executionTask.get().doFirst {
                S3Dependency.withDefaultS3Dependencies(this)
            }
        }

        testRuns.create("benchmark").configureAllExecutions {
            filter {
                includeTestsMatching("*.benchmark_*")
            }

            executionTask.get().enabled = !project.hasProperty("disable-tests")
            executionTask.get().doFirst {
                io.kinference.gradle.s3.S3Dependency.withDefaultS3Dependencies(this)
            }
        }

        browser {
            testTask {
                useKarma {
                    useChrome()
                }
            }
        }
    }

    jvm {
        testRuns["test"].executionTask {
            configureTests()

            enabled = !project.hasProperty("disable-tests")
        }

        testRuns.create("heavy").executionTask {
            configureHeavyTests()

            enabled = !project.hasProperty("disable-tests")
        }

        testRuns.create("benchmark").executionTask {
            configureBenchmarkTests()

            enabled = !project.hasProperty("disable-tests")
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))

                implementation(project(":inference:inference-ir"))
                implementation(project(":utils:webgpu-utils:webgpu-compute"))
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
                implementation("org.openjdk.jmh:jmh-core:1.25.1")
                api("org.slf4j:slf4j-simple:1.7.30")
                implementation(kotlin("test-junit5"))

                runtimeOnly("org.junit.jupiter:junit-jupiter-engine:5.6.2")

                configurations["kapt"].dependencies.add(implementation("org.openjdk.jmh:jmh-generator-annprocess:1.25.1"))
            }
        }
    }
}
