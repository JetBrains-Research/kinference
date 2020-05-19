import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile
import org.jetbrains.research.kotlin.mpp.inference.kotlin

group = "org.jetbrains.research.kotlin.mpp.inference"
version = "0.1.0"

plugins {
    idea
    kotlin("jvm") version "1.3.72" apply true
    id("com.squareup.wire") version "3.1.0" apply true
    id("io.gitlab.arturbosch.detekt") version ("1.6.0") apply true
}

repositories {
    jcenter()
    maven("https://dl.bintray.com/mipt-npm/scientifik")
}

val generatedDir = "src/main/kotlin-gen"

wire {
    protoPath("src/main/proto")

    kotlin {
        out = generatedDir
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

detekt {
    parallel = true
    failFast = false
    config = files(File(rootProject.projectDir, "buildScripts/detekt/detekt.yml"))
    reports {
        xml.enabled = false
        html.enabled = false
    }
}

tasks.withType<KotlinJvmCompile> {
    kotlinOptions {
        jvmTarget = "1.8"
        languageVersion = "1.3"
        apiVersion = "1.3"
    }
}

tasks.test {
    useJUnitPlatform()

    testLogging {
        events("passed", "skipped", "failed")
    }
}

dependencies {
    implementation(kotlin("stdlib"))
    api("com.squareup.wire", "wire-runtime", "3.1.0")
    api("scientifik", "kmath-core-jvm", "0.1.3")
    testImplementation("org.junit.jupiter", "junit-jupiter", "5.6.2")
}
