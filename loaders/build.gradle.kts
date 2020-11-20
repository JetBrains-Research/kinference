import tanvd.kosogor.proxy.publishJar

group = rootProject.group
version = rootProject.version

dependencies {
    api(kotlin("stdlib"))

    api(project(":ndarray"))

    api("com.amazonaws", "aws-java-sdk-s3", "1.11.896")
}


publishJar {
    bintray {
        username = "tanvd"
        repository = "io.kinference"
        info {
            description = "KInference loaders module"
            vcsUrl = "https://github.com/JetBrains-Research/kinference"
            githubRepo = "https://github.com/JetBrains-Research/kinference"
            labels.addAll(listOf("kotlin", "inference", "ml", "loader"))
        }
    }
}
