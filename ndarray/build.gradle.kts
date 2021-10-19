group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") version "0.1.16" apply true
}

kotlin {
    jvm()

    js {
        browser {
            testTask {
                useKarma {
                    useChromeHeadless()
                }
            }
        }

        useCommonJs()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))

                api("io.kinference.primitives:primitives-annotations:0.1.16")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.5.2")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
            }
        }

        val jsMain by getting {
            dependencies {
                implementation(npm("regl", "2.0.1"))
            }
        }
    }
}
