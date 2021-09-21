import org.jetbrains.kotlin.gradle.dsl.KotlinCompile
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile

group = "io.kinference"
version = "0.1.4"

plugins {
    kotlin("multiplatform") version "1.5.30" apply false
    idea apply true
    id("io.gitlab.arturbosch.detekt") version ("1.18.1") apply true
    `maven-publish`
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
        plugin("maven-publish")

        plugin("idea")

        plugin("io.gitlab.arturbosch.detekt")
    }


    publishing {
        repositories {
            maven("https://packages.jetbrains.team/maven/p/ki/maven") {
                name = "SpacePackages"

                credentials {
                    username = System.getenv("JB_SPACE_CLIENT_ID")
                    password = System.getenv("JB_SPACE_CLIENT_SECRET")
                }
            }
        }
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
