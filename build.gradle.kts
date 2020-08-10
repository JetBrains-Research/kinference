import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile
import org.jetbrains.research.kotlin.inference.kotlin
import tanvd.kosogor.proxy.publishJar

group = "org.jetbrains.research.kotlin.inference"
version = "0.1.0"

plugins {
    idea
    id("tanvd.kosogor") version "1.0.7" apply true
    kotlin("jvm") version "1.4.0-rc" apply true
    id("com.squareup.wire") version "3.2.2" apply true
    id("io.gitlab.arturbosch.detekt") version ("1.6.0") apply true
}

repositories {
    jcenter()
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

tasks.compileTestKotlin {
    doFirst() {
        source = source.filter { "kotlin-gen" !in it.path }.asFileTree
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
        languageVersion = "1.4"
        apiVersion = "1.3"
    }
}

tasks.test {
    useJUnitPlatform {
        excludeTags("heavy")
    }
    maxHeapSize = "20m"

    testLogging {
        events("passed", "skipped", "failed")
    }
}

tasks.create("testHeavy", Test::class.java) {
    group = "verification"

    useJUnitPlatform {
        includeTags("heavy")
    }

    maxHeapSize = "4096m"

    testLogging {
        events("passed", "skipped", "failed")
    }
}

publishJar {
    publication {
        artifactId = "kotlin-inference"
    }
}

dependencies {
    implementation(kotlin("stdlib"))
    api("com.squareup.wire", "wire-runtime", "3.2.2")
    testImplementation("org.junit.jupiter", "junit-jupiter", "5.6.2")
}
