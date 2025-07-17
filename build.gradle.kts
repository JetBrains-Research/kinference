import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompilerOptions
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.dsl.KotlinVersion
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnLockMismatchReport
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnPlugin
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootExtension
import org.jetbrains.kotlin.gradle.tasks.KotlinCompilationTask
import org.jetbrains.kotlin.utils.addToStdlib.applyIf

group = "io.kinference"
version = "0.2.27"

plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.kinference.primitives) apply false
    `maven-publish`
    idea apply true
}

allprojects {
    repositories {
        mavenCentral()
        maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
        maven(url = "https://packages.jetbrains.team/maven/p/grazi/grazie-platform-public")
        maven("https://repo.kotlin.link")
    }

    plugins.withType<YarnPlugin>() {
        the<YarnRootExtension>().yarnLockMismatchReport = YarnLockMismatchReport.WARNING
    }
}

subprojects {
    if (this.subprojects.isNotEmpty()) return@subprojects

    apply {
        plugin("org.jetbrains.kotlin.multiplatform")
        plugin("idea")
    }


    applyIf(path != ":examples") {
        apply(plugin = "maven-publish")

        publishing {
            publications {
                all {
                    if (this !is MavenPublication) return@all

                    pom {
                        name = "KInference"
                        description =
                            "KInference is a library that simplifies the execution of complex ONNX machine learning models in Kotlin. " +
                            "It ensures efficient inference of these models on various platforms and is designed for both server-side and local usage."

                        licenses {
                            license {
                                name = "Apache License, Version 2.0"
                                url = "https://www.apache.org/licenses/LICENSE-2.0"
                            }
                        }

                        scm {
                            url = "https://github.com/JetBrains-Research/kinference"
                        }
                    }
                }
            }

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
    }

    extensions.getByType(KotlinMultiplatformExtension::class.java).apply {
        sourceSets.all {
            languageSettings {
                optIn("kotlin.RequiresOptIn")
                optIn("kotlin.ExperimentalUnsignedTypes")
            }
        }
    }

    val kotlinVersion = KotlinVersion.KOTLIN_2_0
    val jvmTargetVersion = JvmTarget.JVM_17

    tasks.withType(KotlinCompilationTask::class.java) {
        compilerOptions {
            apiVersion.set(kotlinVersion)
            languageVersion.set(kotlinVersion)

            if (this is KotlinJvmCompilerOptions) {
                jvmTarget.set(jvmTargetVersion)
            }
        }
    }

    tasks.withType(JavaCompile::class.java) {
        sourceCompatibility = jvmTargetVersion.toString()
        targetCompatibility = jvmTargetVersion.toString()
    }
}
