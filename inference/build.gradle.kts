import io.kinference.gradle.useBenchmarkTests
import io.kinference.gradle.useHeavyTests
import tanvd.kosogor.proxy.publishJar

group = rootProject.group
version = rootProject.version

plugins {
    id("com.squareup.wire") version "3.5.0" apply true
    kotlin("kapt") apply true
}

//useHeavyTests()
//useBenchmarkTests()

wire {
    sourcePath {
        srcDir("src/commonMain/proto")
    }

    kotlin {
        out = "src/commonMain/kotlin-gen"
    }
}

kotlin {
    jvm {
        testRuns["test"].executionTask {
            useJUnitPlatform {
                excludeTags("heavy")
                excludeTags("benchmark")
            }
            maxHeapSize = "20m"

            testLogging {
                events("passed", "skipped", "failed")
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))
                api("com.squareup.wire:wire-runtime-multiplatform:3.5.0")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.3.8")
            }
        }

        val jvmMain by getting {
            dependencies {
                api(project(":ndarray"))
                api("ch.qos.logback:logback-classic:1.2.3")
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation("org.openjdk.jmh:jmh-core:1.25.1")

                implementation("org.junit.jupiter:junit-jupiter:5.6.2")
                implementation("com.microsoft.onnxruntime:onnxruntime:1.4.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.0.1")

                configurations["kapt"].dependencies.add(implementation("org.openjdk.jmh:jmh-generator-annprocess:1.25.1"))
                implementation(project(":loaders"))
            }
        }
    }
}

idea {
    module.generatedSourceDirs.plusAssign(files("src/commonMain/kotlin-gen"))
}

//publishJar {
//    bintray {
//        username = "tanvd"
//        repository = "io.kinference"
//        info {
//            description = "KInference inference module"
//            vcsUrl = "https://github.com/JetBrains-Research/kinference"
//            githubRepo = "https://github.com/JetBrains-Research/kinference"
//            labels.addAll(listOf("kotlin", "inference", "ml"))
//        }
//    }
//}
//
