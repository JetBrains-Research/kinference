import tanvd.kosogor.proxy.publishJar

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

//publishJar {
//    bintray {
//        username = "tanvd"
//        repository = "io.kinference"
//        info {
//            description = "KInference loaders module"
//            vcsUrl = "https://github.com/JetBrains-Research/kinference"
//            githubRepo = "https://github.com/JetBrains-Research/kinference"
//            labels.addAll(listOf("kotlin", "inference", "ml", "loader"))
//        }
//    }
//}
