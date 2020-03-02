import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile

group = "org.jetbrains.research.kotlin.mpp.inference"
version = "0.0.1"

plugins {
    kotlin("jvm") version "1.3.61" apply true
    id("org.jetbrains.kotlin.plugin.serialization") version "1.3.61" apply true
    id("io.gitlab.arturbosch.detekt") version ("1.1.1") apply true
}

repositories {
    jcenter()
    maven("https://dl.bintray.com/mipt-npm/scientifik")
}

detekt {
    parallel = true
    failFast = false
    config = files(File(rootProject.projectDir, "buildScripts/detekt/detekt.yml"))
    reports {
        xml {
            enabled = false
        }
        html {
            enabled = false
        }
    }
}

tasks.withType<KotlinJvmCompile> {
    kotlinOptions {
        jvmTarget = "1.8"
        languageVersion = "1.3"
        apiVersion = "1.3"
    }
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
    implementation("io.jhdf", "jhdf", "0.5.3")
    api("scientifik", "kmath-core-jvm", "0.1.3") {
        exclude("org.jetbrains.kotlin")
    }
    implementation("org.jetbrains.kotlinx", "kotlinx-serialization-runtime", "0.14.0") {
        exclude("org.jetbrains.kotlin")
    }
}
