group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") version "0.1.10" apply true
}

kotlin {
    jvm()
    js {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            repositories {
                mavenCentral()
                maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
            }

            dependencies {
                api(kotlin("stdlib"))
                api("io.kinference.primitives:primitives-annotations:0.1.10")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.4.2")
            }
        }
    }
}

