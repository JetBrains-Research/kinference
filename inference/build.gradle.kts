import io.kinference.gradle.*
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
        out = generatedDir
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

publishJar {
    publication {
        artifactId = "kotlin-inference"
    }
}

dependencies {
    api(project(":ndarray"))

    implementation("org.slf4j", "slf4j-api", "1.7.30")

    api("com.squareup.wire", "wire-runtime", "3.2.2")

    testImplementation("org.openjdk.jmh", "jmh-core", "1.25.1")
    testImplementation("org.openjdk.jmh", "jmh-generator-annprocess", "1.25.1")
    kaptTest("org.openjdk.jmh", "jmh-generator-annprocess", "1.25.1")

    testImplementation("org.junit.jupiter", "junit-jupiter", "5.6.2")
    testImplementation("com.microsoft.onnxruntime", "onnxruntime", "1.4.0")
}
