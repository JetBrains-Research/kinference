import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    js {
        browser {
            testTask {
                useKarma {
                    useChromeHeadless()
                }
            }
        }
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))

                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")

                api(project(":ndarray"))
                api(project(":inference:inference-api"))
                api(project(":serialization"))
                api(project(":utils:logger"))
                api(project(":utils:model-profiler"))

                api(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
            }
        }

    }
}

idea {
    module.generatedSourceDirs.plusAssign(files("src/commonMain/kotlin-gen"))
}
