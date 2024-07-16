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
                api(project(":utils:utils-common"))
                api(libs.kinference.primitives.annotations)
                api(libs.wire.runtime)
            }
        }
    }
}
