group = rootProject.group
version = rootProject.version

plugins {
    antlr
}

dependencies {
    antlr("org.antlr:antlr4:4.9.2")

    implementation(project(":utils:webgpu-utils:wgpu:jnr-internal-api-jvm"))
    implementation("com.squareup:kotlinpoet:1.10.1")
}

tasks.generateGrammarSource {
    arguments = arguments + listOf("-package", "io.kinference.utils.wgpu.generation.grammar", "-visitor")
}

tasks.compileKotlin {
    dependsOn(tasks.generateGrammarSource)
}
