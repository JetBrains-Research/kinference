import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests
import io.kinference.gradle.configureTests
import io.kinference.gradle.s3.S3Dependency

group = rootProject.group
version = rootProject.version

plugins {
    kotlin("kapt") apply true
}

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
                S3Dependency.withDefaultS3Dependencies(this)
            }
        }

        browser {
            testTask {
                useKarma {
                    useChromeHeadless()
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
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.5.2")

                api(project(":inference:inference-api"))
                api(project(":inference:inference-ir"))
                api(project(":ndarray"))
                api(project(":utils:logger"))
                api(project(":utils:model-profiler"))

                api(project(":serialization"))
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

                implementation("com.microsoft.onnxruntime:onnxruntime:1.9.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.0.1")

                configurations["kapt"].dependencies.add(implementation("org.openjdk.jmh:jmh-generator-annprocess:1.25.1"))
            }
        }
    }
}

idea {
    module.generatedSourceDirs.plusAssign(files("src/commonMain/kotlin-gen"))
}
