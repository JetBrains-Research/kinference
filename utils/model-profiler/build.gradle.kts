group = rootProject.group
version = rootProject.version


kotlin {
    jvm()

    js {
        browser {
            testTask {
                useKarma {
                    useChromeHeadless()
                }
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))
            }
        }
    }
}
