import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version


kotlin {
    jvm()

    js(BOTH) {
        browser()
    }

    sourceSets {
        val jvmMain by getting {
            dependencies {
                api("org.slf4j:slf4j-api:${Versions.slf4j}")
            }
        }

        val jsMain by getting {
            dependencies {
                api(npm("loglevel", Versions.loglevel))
            }
        }
    }
}
