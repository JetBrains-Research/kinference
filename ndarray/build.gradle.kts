import io.kinference.primitives.primitives
import io.kinference.gradle.generatedDir

group = rootProject.group
version = rootProject.version

primitives {
    generationPath = file(generatedDir)
}

tasks.compileKotlin {
    dependsOn("generateSources")
}

dependencies {
    api(kotlin("stdlib"))
    api("io.kinference.primitives","primitives-annotations","0.1.2")
    api("org.slf4j","slf4j-api","1.7.30")
    api("ch.qos.logback","logback-classic","1.2.3")
}
