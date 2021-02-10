group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") version "0.1.10" apply true
}

kotlin {
    jvm {

    }

    js(IR) {
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
                api(kotlin("stdlib"))
                api(kotlin("stdlib-common"))
                api("io.kinference.primitives:primitives-annotations:0.1.11")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test-common"))
                implementation(kotlin("test-annotations-common"))
            }
        }

        val jvmTest by getting {
            dependsOn(commonTest)
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-junit"))
            }
        }

        val jsMain by getting {
            dependencies {
                api(kotlin("stdlib-js"))
            }
        }

        val jsTest by getting {
            dependencies {
                implementation(kotlin("test-js"))
            }
        }
    }
}

