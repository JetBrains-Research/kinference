import io.kinference.gradle.s3.S3Dependency
import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
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

            executionTask.get().useKarma {
                useChrome()
            }
        }

        testRuns.create("benchmark").configureAllExecutions {
            filter {
                includeTestsMatching("*.benchmark_*")
            }

            executionTask.get().enabled = !project.hasProperty("disable-tests")

            executionTask.get().useKarma {
                useChrome()
            }
            executionTask.get().doFirst {
                S3Dependency.withDefaultS3Dependencies(this)
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

    sourceSets {
        val jsMain by getting() {
            dependencies {
                implementation(project(":serialization"))
                api(project(":inference:inference-ir"))

                implementation(npm("@tensorflow/tfjs-core", Versions.TFJS))
                implementation(npm("@tensorflow/tfjs-backend-webgl", Versions.TFJS))

                implementation(project(":inference:inference-api"))

                api(project(":utils:logger"))
                api(project(":utils:model-profiler"))

                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }

        val jsTest by getting() {
            dependencies {
                implementation(kotlin("test-js"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:test-utils"))
            }
        }
    }
}
