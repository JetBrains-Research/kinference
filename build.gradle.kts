import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.dsl.KotlinCompile

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
        mavenCentral()
    }
}

subprojects {
    apply {
        plugin("org.jetbrains.kotlin.multiplatform")

        plugin("idea")

        plugin("io.gitlab.arturbosch.detekt")
    }

    tasks.withType<KotlinCompile<*>> {
        kotlinOptions {
            freeCompilerArgs = freeCompilerArgs + listOf(
                    "-Xopt-in=kotlin.RequiresOptIn",
                    "-Xopt-in=kotlin.ExperimentalUnsignedTypes",
                    "-Xopt-in=kotlin.time.ExperimentalTime"
            )
        }
    }

    tasks.withType<KotlinJvmCompile> {
        kotlinOptions {
            jvmTarget = "11"
            languageVersion = "1.4"
            apiVersion = "1.4"
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
