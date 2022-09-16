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
                api(project(":utils:common-utils"))

                api("io.kinference.primitives:primitives-annotations:${Versions.kinferencePrimitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }
    }
}
