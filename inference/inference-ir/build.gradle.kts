group = rootProject.group
version = rootProject.version

kotlin {
    js{
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
                api(project(":inference:inference-api"))
                api(project(":utils:logger"))
                api(project(":serialization"))
                api(project(":utils:model-profiler"))
            }
        }
    }
}
