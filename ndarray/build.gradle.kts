import io.kinference.gradle.generatedDir
import io.kinference.primitives.primitives
import tanvd.kosogor.proxy.publishJar

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

    api("org.slf4j", "slf4j-api", "1.7.30")

    api("io.kinference.primitives", "primitives-annotations", "0.1.2")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.3.8")
}

publishJar {}
