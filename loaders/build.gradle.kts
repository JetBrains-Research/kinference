group = rootProject.group
version = rootProject.version

kotlin {
    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))
                api(project(":ndarray"))
            }
        }

        val jvmMain by getting {
            dependencies {
                api("com.amazonaws:aws-java-sdk-s3:1.11.896")
            }
        }
    }
}
