import io.kinference.gradle.s3.S3Dependency

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

        useCommonJs()
    }

    sourceSets {
        val jsMain by getting() {
            dependencies {
                implementation("io.github.microutils:kotlin-logging:2.0.4")
                implementation(project(":serialization"))
                implementation(npm("@tensorflow/tfjs-core", "3.9.0"))
                implementation(npm("@tensorflow/tfjs-backend-cpu", "3.9.0"))
                implementation(npm("@tensorflow/tfjs-backend-webgl", "3.9.0"))
                implementation(project(":inference-api"))

                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2")
            }
        }

        val jsTest by getting() {
            dependencies {
                implementation(kotlin("test-js"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":test-runner"))
            }
        }
    }
}
