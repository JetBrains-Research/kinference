group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") version "0.1.12" apply true
}

kotlin {
    jvm {
        testRuns["test"].executionTask {
            useJUnitPlatform()
        }
    }

    js {
        browser {
            testTask {
                useKarma {
                    useChromeHeadless()
                }
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            repositories {
                mavenCentral()
                maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
            }

            dependencies {
                api("io.kinference.primitives:primitives-annotations:0.1.12")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation("io.kotest:kotest-framework-engine:4.4.1")
                implementation("io.kotest:kotest-assertions-core:4.4.1")
            }
        }

        val jvmTest by getting {
            dependsOn(commonTest)
            dependencies {
                implementation("io.kotest:kotest-runner-junit5:4.4.1")
            }
        }
    }
}

