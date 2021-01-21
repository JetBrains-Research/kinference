import io.kinference.gradle.useBenchmarkTests
import io.kinference.gradle.useHeavyTests
import tanvd.kosogor.proxy.publishJar

group = rootProject.group
version = rootProject.version

plugins {
    id("com.squareup.wire") version "3.2.2" apply true
    kotlin("kapt") apply true
}

wire {
    protoPath("src/main/proto")

    kotlin {
        out = "src/main/kotlin-gen"
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


useHeavyTests()
useBenchmarkTests()

sourceSets {
    main {
        allJava.srcDir(file("src/main/kotlin-gen"))
    }
}

idea {
    module.generatedSourceDirs.plusAssign(files("src/main/kotlin-gen"))
}

dependencies {
    api(project(":ndarray"))

    api("ch.qos.logback", "logback-classic", "1.2.3")

    api("com.squareup.wire", "wire-runtime", "3.2.2")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.3.8")

    testImplementation("org.openjdk.jmh", "jmh-core", "1.25.1")
    testImplementation("org.openjdk.jmh", "jmh-generator-annprocess", "1.25.1")
    kaptTest("org.openjdk.jmh", "jmh-generator-annprocess", "1.25.1")

    testImplementation("org.junit.jupiter", "junit-jupiter", "5.6.2")
    testImplementation("com.microsoft.onnxruntime", "onnxruntime", "1.4.0")

    testImplementation(project(":loaders"))
}


publishJar {
    bintray {
        username = "tanvd"
        repository = "io.kinference"
        info {
            description = "KInference inference module"
            vcsUrl = "https://github.com/JetBrains-Research/kinference"
            githubRepo = "https://github.com/JetBrains-Research/kinference"
            labels.addAll(listOf("kotlin", "inference", "ml"))
        }
    }
}

