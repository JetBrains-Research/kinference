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
                api(project(":utils:utils-common"))
                api(libs.kinference.primitives.annotations)
                api(libs.wire.runtime)
            }
        }
    }
}
