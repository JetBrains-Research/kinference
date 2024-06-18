import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") apply true
}

kotlin {
    js(IR) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
                api("io.kinference.primitives:primitives-annotations:${Versions.primitives}")
                implementation("com.squareup.okio:okio:${Versions.okio}")
            }
        }
    }
}
