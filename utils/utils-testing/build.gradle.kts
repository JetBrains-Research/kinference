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
        val commonMain by getting {
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

        val jvmMain by getting {
            dependencies {
                api(libs.slf4j.simple)
                api(kotlin("test-junit5"))
            }
        }

        val jsMain by getting {
            dependencies {
                api(kotlin("test-js"))
            }
        }
    }
}
