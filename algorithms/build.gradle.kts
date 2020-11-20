import io.kinference.gradle.useHeavyTests
import tanvd.kosogor.proxy.publishJar

group = rootProject.group
version = rootProject.version

useHeavyTests()

dependencies {
    api(project(":inference"))

    implementation("com.fasterxml.jackson.core", "jackson-databind", "2.11.3")

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
