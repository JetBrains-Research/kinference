group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("com.squareup.okio:okio:3.0.0")
                api(project(":utils:common-utils"))
            }
        }
    }
}
