import io.kinference.gradle.*

group = rootProject.group
version = rootProject.version

plugins {
    alias(libs.plugins.kinference.primitives) apply true
}

kotlin {
    js(IR) {
        browser()
    }

    jvm()

    sourceSets {
        commonMain {
            dependencies {
                api(libs.kinference.primitives.annotations)
                api(libs.kotlinx.coroutines.core)
                api(libs.okio)
            }
        }
    }
}
