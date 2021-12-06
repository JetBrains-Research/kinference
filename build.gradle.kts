import org.jetbrains.kotlin.gradle.dsl.KotlinCompile
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension

group = "io.kinference"
version = "0.1.7"

plugins {
    kotlin("multiplatform") version "1.5.31" apply false
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
    if (this.subprojects.isNotEmpty()) return@subprojects

    apply {
        plugin("org.jetbrains.kotlin.multiplatform")

        plugin("maven-publish")
        plugin("idea")
        plugin("io.gitlab.arturbosch.detekt")
    }


    publishing {
        repositories {
            maven {
                name = "SpacePackages"
                url = uri("https://packages.jetbrains.team/maven/p/ki/maven")

                credentials {
                    username = System.getenv("JB_SPACE_CLIENT_ID")
                    password = System.getenv("JB_SPACE_CLIENT_SECRET")
                }
            }
        }
    }

    extensions.getByType(KotlinMultiplatformExtension::class.java).apply {
        sourceSets.all {
            languageSettings {
                optIn("kotlin.RequiresOptIn")
                optIn("kotlin.time.ExperimentalTime")
                optIn("kotlin.ExperimentalUnsignedTypes")
                optIn("kotlinx.serialization.ExperimentalSerializationApi")
            }

            languageSettings {
                apiVersion = "1.5"
                languageVersion = "1.5"
            }
        }

        tasks.withType<KotlinJvmCompile> {
            kotlinOptions {
                jvmTarget = "11"
            }
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
