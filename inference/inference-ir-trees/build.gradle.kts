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
                api(project(":utils:utils-logger"))
                api(project(":ndarray:ndarray-api"))
            }
        }
    }
}
