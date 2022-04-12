group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":inference:inference-api"))
                api(project(":utils:logger"))
                api(project(":serialization"))
                api(project(":utils:model-profiler"))
            }
        }
    }
}
