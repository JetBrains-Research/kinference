import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    js(IR) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.coroutines}")
                implementation("com.squareup.okio:okio:${Versions.okio}")

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))

                api(project(":utils:utils-logger"))
                api(project(":utils:utils-profiling"))
                api(project(":utils:utils-common"))

                api(kotlin("test"))
                api(kotlin("test-annotations-common"))
            }
        }

        val jvmMain by getting {
            dependencies {
                runtimeOnly("org.junit.jupiter:junit-jupiter-engine:5.8.2")
            }
        }
    }
}