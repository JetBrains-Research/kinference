import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

plugins {
    alias(libs.plugins.primitives) apply true
}

kotlin {
    js(IR) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(libs.primitives.annotations)
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
                implementation("com.squareup.okio:okio:${Versions.okio}")
            }
        }
    }
}
