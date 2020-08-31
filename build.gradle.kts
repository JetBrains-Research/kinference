import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import tanvd.kosogor.proxy.publishJar
import io.kinference.gradle.generatedDir
import io.kinference.gradle.kotlin

group = "io.kinference"
version = "0.1.0"

plugins {
    id("tanvd.kosogor") version "1.0.9" apply true
    kotlin("jvm") version "1.3.72" apply true

    id("com.squareup.wire") version "3.2.2" apply true
    id("io.gitlab.arturbosch.detekt") version ("1.11.0") apply true
    id("io.kinference.primitives") version ("0.1.1") apply false

    kotlin("kapt") version "1.3.72"
}

allprojects {
    repositories {
        jcenter()
    }

    tasks.withType<KotlinCompile> {
        kotlinOptions {
            jvmTarget = "1.8"
            languageVersion = "1.3"
            apiVersion = "1.3"
        }
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


wire {
    protoPath("src/main/proto")

    kotlin {
        out = generatedDir
    }
}

tasks.compileTestKotlin {
    doFirst {
        source = source.filter { generatedDir !in it.path }.asFileTree
    }
}


detekt {
    config = files(file("detekt.yml"))
    reports {
        xml.enabled = false
        html.enabled = false
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
    implementation(project(":ndarray"))
    api("com.squareup.wire", "wire-runtime", "3.2.2")

    testImplementation("org.openjdk.jmh:jmh-core:1.25.1")
    testImplementation("org.openjdk.jmh:jmh-generator-annprocess:1.25.1")
    kaptTest("org.openjdk.jmh:jmh-generator-annprocess:1.25.1")

    testImplementation("org.junit.jupiter", "junit-jupiter", "5.6.2")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.4.0")
}
