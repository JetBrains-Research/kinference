group = rootProject.group
version = rootProject.version

kotlin {
    js{
        browser {
            testTask {
                useKarma {
                    useChromeHeadless()
                }
            }
        }
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))
                api(project(":ndarray"))

                api("com.squareup.wire:wire-runtime-multiplatform:3.6.0")
            }
        }
    }
}
