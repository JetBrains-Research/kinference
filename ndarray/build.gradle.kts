import org.jetbrains.research.kotlin.inference.generatedDir
import org.jetbrains.research.kotlin.inference.kotlin

group = "org.jetbrains.research.kotlin.inference"
version = "0.1.0"

plugins {
    idea
    kotlin("jvm") apply true
    id("org.jetbrains.research.kotlin.inference.primitives-generator")
}

primitives {
    generationPath = generatedDir
}

sourceSets {
    main {
        kotlin.srcDirs(generatedDir)
    }
}

idea {
    module.generatedSourceDirs.plusAssign(files(generatedDir))
}

dependencies {
    implementation(kotlin("stdlib"))
    api(project(":annotations"))
}
