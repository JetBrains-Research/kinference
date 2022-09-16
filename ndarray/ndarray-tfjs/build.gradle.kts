import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        browser()
    }

    sourceSets {
        val jsMain by getting {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api("io.kinference.primitives:primitives-annotations:${Versions.kinferencePrimitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }
    }
}
