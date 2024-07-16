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
                implementation(libs.okio)
                api(project(":utils:utils-common"))
            }
        }
    }
}
