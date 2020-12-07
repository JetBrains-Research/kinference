import tanvd.kosogor.proxy.publishJar

group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") version "0.1.7" apply true
}

dependencies {
    api(kotlin("stdlib"))

    api("org.slf4j", "slf4j-api", "1.7.30")

    api("io.kinference.primitives", "primitives-annotations", "0.1.7")

    implementation("org.jetbrains.kotlinx", "kotlinx-coroutines-core", "1.4.2")
}


publishJar {
    bintray {
        username = "tanvd"
        repository = "io.kinference"
        info {
            description = "KInference NDArray module"
            vcsUrl = "https://github.com/JetBrains-Research/kinference"
            githubRepo = "https://github.com/JetBrains-Research/kinference"
            labels.addAll(listOf("kotlin", "inference", "ml", "array"))
        }
    }
}
