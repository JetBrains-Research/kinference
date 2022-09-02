import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    jvm()

    js(BOTH) {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api("io.ktor:ktor-io:2.1.0")
                api("io.kinference.primitives:primitives-annotations:${Versions.kinferencePrimitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }

        val jsMain by getting {
            dependencies {
                api("io.ktor:ktor-io-js:2.1.0")
            }
        }

        val jvmMain by getting {
            dependencies {
                api("io.ktor:ktor-io-jvm:2.1.0")
            }
        }
    }
}
