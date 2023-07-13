import io.kinference.gradle.Versions
import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

kotlin {
    js(IR) {
        browser()
        configureTests()
    }

    sourceSets {
        val jsMain by getting {
            dependencies {
                implementation(npm("@tensorflow/tfjs-core", Versions.tfjs))
                implementation(npm("@tensorflow/tfjs-backend-webgl", Versions.tfjs))

                api(project(":ndarray:ndarray-api"))
                api(project(":utils:utils-common"))
                api("io.kinference.primitives:primitives-annotations:${Versions.primitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
            }
        }

        val jsTest by getting {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
