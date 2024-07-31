group = rootProject.group
version = rootProject.version


kotlin {
    jvm()

    js(IR) {
        browser()
    }

    sourceSets {
        jvmMain {
            dependencies {
                api(libs.slf4j.api)
            }
        }

        jsMain {
            dependencies {
                api(npm("loglevel", libs.versions.loglevel.get()))
            }
        }
    }
}
