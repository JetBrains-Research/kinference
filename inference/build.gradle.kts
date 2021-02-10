import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests
import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

plugins {
    id("com.squareup.wire") version "3.6.0" apply true
    kotlin("kapt") apply true
}

wire {
    sourcePath {
        srcDir("src/commonMain/proto")
    }

    kotlin {
        out = "src/commonMain/kotlin-gen"
    }
}

kotlin {
    js {
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
        }

        testRuns.create("heavy").executionTask {
            configureHeavyTests {
                s3Test("bert:standard:en:v1")
                s3Test("bert:gec:en:standard:v2")
                s3Test("gpt2:flcc-py-completion:quantized:v2")
                s3Test("gpt2:grazie:distilled:quantized:v6")
                s3Test("gpt2:r-completion:standard:v1")
                s3Test("gpt2:r-completion:quantized:v1")
                s3Test("catboost:ij-completion-ranker:v1")
                s3Test("catboost:license-detector:v1")
            }
        }

        testRuns.create("benchmark").executionTask {
            configureBenchmarkTests {
                s3Test("bert:standard:en:v1")
                s3Test("bert:gec:en:standard:v2")
                s3Test("gpt2:flcc-py-completion:quantized:v2")
                s3Test("gpt2:grazie:distilled:quantized:v6")
                s3Test("gpt2:r-completion:standard:v1")
                s3Test("gpt2:r-completion:quantized:v1")
                s3Test("catboost:ij-completion-ranker:v1")
                s3Test("catboost:license-detector:v1")
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))
                api("com.squareup.wire:wire-runtime-multiplatform:3.6.0")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2")
                api(project(":ndarray"))
            }
        }


        val commonTest by getting {
            dependencies {
                implementation(kotlin("test-common"))
                implementation(kotlin("test-annotations-common"))
            }
        }

        val jvmMain by getting {
            dependencies {
                api("ch.qos.logback:logback-classic:1.2.3")
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation("org.openjdk.jmh:jmh-core:1.25.1")

                implementation(kotlin("test-junit5"))

                runtimeOnly("org.junit.jupiter:junit-jupiter-engine:5.6.2")

                implementation("com.microsoft.onnxruntime:onnxruntime:1.4.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.0.1")

                configurations["kapt"].dependencies.add(implementation("org.openjdk.jmh:jmh-generator-annprocess:1.25.1"))
            }
        }

        val jsTest by getting {
            dependencies {
                implementation(kotlin("test-js"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:1.4.2")
            }
        }
    }
}

idea {
    module.generatedSourceDirs.plusAssign(files("src/commonMain/kotlin-gen"))
}
