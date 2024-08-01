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
                api(libs.kotlinx.coroutines.core)
                implementation(libs.okio)

                api(project(":inference:inference-api"))

                api(project(":utils:utils-logger"))
                api(project(":utils:utils-profiling"))
                api(project(":utils:utils-common"))

                api(kotlin("test-common"))
                api(kotlin("test-annotations-common"))

                api(libs.kinference.primitives.annotations)
            }
        }

        jvmMain {
            dependencies {
                api(libs.slf4j.simple)
                api(kotlin("test-junit5"))
            }
        }

        jsMain {
            dependencies {
                api(kotlin("test-js"))
            }
        }
    }
}
