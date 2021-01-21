import io.kinference.gradle.useHeavyTests
import tanvd.kosogor.proxy.publishJar

group = rootProject.group
version = rootProject.version

plugins {
    kotlin("plugin.serialization") version "1.4.20" apply true
}

dependencies {
    api(project(":inference"))

    implementation("com.github.ben-manes.caffeine", "caffeine", "2.8.5")
    implementation("info.debatty", "java-string-similarity", "1.2.1")
    implementation("org.jetbrains.kotlinx", "kotlinx-serialization-json", "1.0.1")

    testImplementation("org.junit.jupiter", "junit-jupiter", "5.6.2")
    testImplementation(project(":loaders"))
}

publishJar {
    bintray {
        username = "tanvd"
        repository = "io.kinference"
        info {
            description = "KInference algorithms module"
            vcsUrl = "https://github.com/JetBrains-Research/kinference"
            githubRepo = "https://github.com/JetBrains-Research/kinference"
            labels.addAll(listOf("kotlin", "inference", "ml"))
        }
    }
}


tasks.test {
    useJUnitPlatform {
        excludeTags("heavy")
        excludeTags("benchmark")
    }
    maxHeapSize = "20m"

    testLogging {
        events("passed", "skipped", "failed")
    }
}

useHeavyTests()
