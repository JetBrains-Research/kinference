import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") version "0.1.25" apply true
    id("org.jetbrains.kotlin.plugin.atomicfu") version "2.0.0-Beta3"
}

kotlin {
    jvm()

    js(IR) {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api("io.kinference.primitives:primitives-annotations:${Versions.primitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
                implementation("org.jetbrains.kotlinx:atomicfu:0.23.2")
            }
        }

        val jvmMain by getting {
            dependencies {
                api("org.apache.commons:commons-math4-core:4.0-beta1")
            }
        }
    }
}
