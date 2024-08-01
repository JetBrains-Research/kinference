import io.kinference.gradle.*

group = rootProject.group
version = rootProject.version

kotlin {
    js(IR) {
        browser()
    }

    jvm()

    sourceSets {
        commonMain {
            dependencies {
                api(project(":utils:utils-logger"))
                api(project(":ndarray:ndarray-api"))
            }
        }
    }
}
