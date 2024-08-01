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
                implementation(libs.okio)
                api(project(":utils:utils-common"))
            }
        }
    }
}
