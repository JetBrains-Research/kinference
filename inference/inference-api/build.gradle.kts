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
                api("io.ktor:ktor-io:2.1.0")
                implementation("com.squareup.okio:okio:3.0.0")
            }
        }
    }
}
