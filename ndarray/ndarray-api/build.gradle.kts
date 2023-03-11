import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    jvm()

    js(IR) {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":utils:utils-common"))

                api("io.kinference.primitives:primitives-annotations:${Versions.primitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
            }
        }
    }
}
