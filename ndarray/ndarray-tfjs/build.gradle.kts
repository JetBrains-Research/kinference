import io.kinference.gradle.Versions
import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        configureTests()
        browser()
    }

    sourceSets {
        val jsMain by getting {
            dependencies {
                implementation(npm("@tensorflow/tfjs-core", Versions.TFJS))
                implementation(npm("@tensorflow/tfjs-backend-webgl", Versions.TFJS))

                api(project(":ndarray:ndarray-api"))
                api("io.kinference.primitives:primitives-annotations:${Versions.kinferencePrimitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }

        val jsTest by getting {
            dependencies {
                implementation(kotlin("test-js"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:test-utils"))
            }
        }
    }
}
