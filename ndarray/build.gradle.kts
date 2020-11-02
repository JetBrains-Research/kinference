import io.kinference.primitives.primitives
import io.kinference.gradle.generatedDir
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
    api("io.kinference.primitives","primitives-annotations","0.1.1")
}

publishJar {}
