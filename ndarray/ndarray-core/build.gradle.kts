group = rootProject.group
version = rootProject.version

plugins {
    alias(libs.plugins.kinference.primitives) apply true
    alias(libs.plugins.kotlin.atomicfu)
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
                api(libs.kinference.primitives.annotations)
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.atomicfu)
            }
        }

        val jvmMain by getting {
            dependencies {
                api(libs.apache.commons.math4.core)
            }
        }
    }
}
