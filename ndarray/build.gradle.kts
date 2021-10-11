group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") version "0.1.14" apply true
}

kotlin {
    jvm {

    }

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
            repositories {
                mavenCentral()
                maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
            }

            dependencies {
                api(kotlin("stdlib"))
                api("io.kinference.primitives:primitives-annotations:0.1.14")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2")
                implementation("io.github.microutils:kotlin-logging:2.0.4")
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
            dependsOn(commonTest)
            dependencies {
                implementation(kotlin("test-junit"))
            }
        }

        val jsMain by getting {
            dependencies {
                implementation(npm("regl", "2.0.1"))
            }
        }

        val jsTest by getting {
            dependsOn(commonTest)
            dependencies {
                implementation(kotlin("test-js"))
            }
        }
    }
}
