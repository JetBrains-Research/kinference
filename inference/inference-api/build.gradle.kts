import io.kinference.gradle.Versions

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
                implementation("com.squareup.okio:okio:${Versions.okio}")
                api(project(":utils:utils-common"))
            }
        }
    }
}
