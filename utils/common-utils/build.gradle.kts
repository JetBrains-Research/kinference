import io.kinference.gradle.Versions

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
            }
        }

        val jsMain by getting {
            dependencies {
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }
    }
}
