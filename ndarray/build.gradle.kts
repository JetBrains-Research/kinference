import io.kinference.primitives.primitives
import org.jetbrains.research.kotlin.inference.generatedDir
import org.jetbrains.research.kotlin.inference.kotlin

group = "org.jetbrains.research.kotlin.inference"
version = "0.1.0"

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
