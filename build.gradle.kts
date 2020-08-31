import io.kinference.gradle.generatedDir
import io.kinference.gradle.kotlin
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile

group = "io.kinference"
version = "0.1.0"

plugins {
    id("tanvd.kosogor") version "1.0.9" apply true

    kotlin("jvm") version "1.3.72" apply true
    kotlin("kapt") version "1.3.72" apply false

    id("io.kinference.primitives") version ("0.1.1") apply false

    id("io.gitlab.arturbosch.detekt") version ("1.11.0") apply true
    idea apply true
}

allprojects {
    repositories {
        jcenter()
        gradlePluginPortal()
    }
}

subprojects {
    apply {
        plugin("kotlin")
        plugin("idea")
        plugin("io.kinference.primitives")
        plugin("tanvd.kosogor")
        plugin("io.gitlab.arturbosch.detekt")
    }

    tasks.withType<KotlinJvmCompile> {
        kotlinOptions {
            jvmTarget = "1.8"
            languageVersion = "1.3"
            apiVersion = "1.3"
        }
    }

    detekt {
        parallel = true

        config = rootProject.files("detekt.yml")

        reports {
            xml {
                enabled = false
            }
            html {
                enabled = false
            }
        }
    }

    sourceSets {
        main {
            kotlin.srcDirs(generatedDir)
        }
    }

    idea {
        module.generatedSourceDirs.plusAssign(files(generatedDir))
    }

    tasks.compileTestKotlin {
        doFirst {
            source = source.filter { generatedDir !in it.path }.asFileTree
        }
    }
}
