group = rootProject.group
version = rootProject.version

plugins {
    alias(libs.plugins.kinference.primitives) apply true
    alias(libs.plugins.kotlin.atomicfu)
}

kotlin {
    jvm()

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api(libs.kinference.primitives.annotations)
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.atomicfu)
                api(libs.apache.commons.math4.core)
            }
        }
    }
}
