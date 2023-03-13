group = rootProject.group
version = rootProject.version


kotlin {
    jvm()

    js(IR) {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":utils:utils-common"))
            }
        }
    }
}
