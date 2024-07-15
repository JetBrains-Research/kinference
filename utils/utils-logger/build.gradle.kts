group = rootProject.group
version = rootProject.version


kotlin {
    jvm()

    js(IR) {
        browser()
    }

    sourceSets {
        val jvmMain by getting {
            dependencies {
                api(libs.slf4j.api)
            }
        }

        val jsMain by getting {
            dependencies {
                api(npm("loglevel", libs.versions.loglevel.get()))
            }
        }
    }
}
