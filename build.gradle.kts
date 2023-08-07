import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnLockMismatchReport
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnPlugin
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootExtension
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

group = "io.kinference"
version = "0.2.14"

plugins {
    kotlin("multiplatform") apply false
    idea apply true
    `maven-publish`
}

allprojects {
    repositories {
        mavenCentral()
        maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
        maven(url = "https://packages.jetbrains.team/maven/p/grazi/grazie-platform-public")
    }

    plugins.withType<YarnPlugin>() {
        the<YarnRootExtension>().yarnLockMismatchReport = YarnLockMismatchReport.WARNING
    }
}

subprojects {
    if (this.subprojects.isNotEmpty()) return@subprojects

    apply {
        plugin("org.jetbrains.kotlin.multiplatform")

        plugin("maven-publish")
        plugin("idea")
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
                optIn("kotlin.ExperimentalUnsignedTypes")
            }

            languageSettings {
                apiVersion = "1.9"
                languageVersion = "1.9"
            }
        }

        tasks.withType<KotlinJvmCompile> {
            kotlinOptions {
                jvmTarget = "17"
            }
        }
    }
}
