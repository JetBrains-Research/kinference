import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile

group = "io.kinference"
version = "0.1.2"

plugins {
    kotlin("multiplatform") version "1.4.30" apply false
    idea apply true
    id("io.gitlab.arturbosch.detekt") version ("1.11.0") apply true
}

allprojects {
    repositories {
        maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
        jcenter()
        gradlePluginPortal()
    }
}

subprojects {
    apply {
        plugin("org.jetbrains.kotlin.multiplatform")

        plugin("idea")

        plugin("io.gitlab.arturbosch.detekt")
    }

    tasks.withType<KotlinJvmCompile> {
        kotlinOptions {
            jvmTarget = "11"
            languageVersion = "1.4"
            apiVersion = "1.4"
            freeCompilerArgs = freeCompilerArgs + listOf("-Xopt-in=kotlin.RequiresOptIn", "-Xopt-in=kotlin.ExperimentalUnsignedTypes")
        }
    }

    detekt {
        parallel = true

        config = rootProject.files("detekt.yml")

        reports {
            xml.enabled = false
            html.enabled = false
        }
    }
}
