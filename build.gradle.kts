import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile
import org.jetbrains.research.kotlin.inference.kotlin
import tanvd.kosogor.proxy.publishJar

group = "org.jetbrains.research.kotlin.inference"
version = "0.1.0"

plugins {
    idea
    id("tanvd.kosogor") version "1.0.9" apply true
    kotlin("jvm") version "1.4.0" apply true
    id("com.squareup.wire") version "3.2.2" apply true
    id("io.gitlab.arturbosch.detekt") version ("1.11.0") apply true
    kotlin("kapt") version "1.4.0"
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
    doFirst {
        source = source.filter { generatedDir !in it.path }.asFileTree
    }
}

idea {
    module.generatedSourceDirs.plusAssign(files(generatedDir))
}

detekt {
    config = files(file("detekt.yml"))
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
        excludeTags("benchmark")
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
        excludeTags("benchmark")
    }

    maxHeapSize = "8192m"

    testLogging {
        events("passed", "skipped", "failed")
    }
}

tasks.create("testPerfomance", Test::class.java) {
    group = "verification"

    useJUnitPlatform {
        excludeTags("heavy")
        includeTags("benchmark")
    }

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

    testImplementation("org.openjdk.jmh:jmh-core:1.25.1")
    testImplementation("org.openjdk.jmh:jmh-generator-annprocess:1.25.1")
    kaptTest("org.openjdk.jmh:jmh-generator-annprocess:1.25.1")

    testImplementation("org.junit.jupiter", "junit-jupiter", "5.6.2")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.4.0")
}
