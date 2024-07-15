import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

plugins {
    alias(libs.plugins.primitives) apply true
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
                api(libs.primitives.annotations)
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
                implementation("org.jetbrains.kotlinx:atomicfu:${Versions.atomicfu}")
            }
        }

        val jvmMain by getting {
            dependencies {
                api("org.apache.commons:commons-math4-core:4.0-beta1")
            }
        }
    }
}
