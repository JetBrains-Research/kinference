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
    api("io.kinference.primitives","primitives-annotations","0.1.1")

    implementation("com.beust", "klaxon", "5.0.1")
}
