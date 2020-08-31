import io.kinference.primitives.primitives
import io.kinference.gradle.generatedDir
import io.kinference.gradle.kotlin

group = rootProject.group
version = rootProject.version

plugins {
    idea
    kotlin("jvm") apply true
    id("io.kinference.primitives")
}

primitives {
    generationPath = file(generatedDir)
}

sourceSets {
    main {
        kotlin.srcDirs(generatedDir)
    }
}

tasks.compileKotlin {
    dependsOn("generateSources")
}

idea {
    module.generatedSourceDirs.plusAssign(files(generatedDir))
}

dependencies {
    implementation(kotlin("stdlib"))
    api("io.kinference.primitives","primitives-annotations","0.1.1")
}
