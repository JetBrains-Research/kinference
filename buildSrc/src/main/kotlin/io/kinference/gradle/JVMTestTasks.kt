package io.kinference.gradle

import io.kinference.gradle.s3.S3Dependency
import org.gradle.api.tasks.testing.logging.TestLogEvent
import org.gradle.kotlin.dsl.get
import org.gradle.kotlin.dsl.invoke
import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget

fun KotlinJvmTarget.configureTests() {
    testRuns["test"].executionTask {
        maxHeapSize = "400m"

        useJUnitPlatform()

        filter {
            excludeTestsMatching("*.heavy_*")
            excludeTestsMatching("*.benchmark_*")
            excludeTestsMatching("*.gpu_*")
        }

        testLogging {
            events(TestLogEvent.STANDARD_ERROR, TestLogEvent.STARTED, TestLogEvent.PASSED, TestLogEvent.FAILED, TestLogEvent.SKIPPED)
        }

        enabled = !project.hasProperty("disable-tests")
    }
}

fun KotlinJvmTarget.configureHeavyTests() {
    testRuns.create("heavy").executionTask {
        group = "verification"

        useJUnitPlatform()

        maxHeapSize = "4G"


        filter {
            includeTestsMatching("*.heavy_*")
        }

        testLogging {
            events(TestLogEvent.STANDARD_ERROR, TestLogEvent.STARTED, TestLogEvent.PASSED, TestLogEvent.FAILED, TestLogEvent.SKIPPED)
        }

        doFirst {
            S3Dependency.withDefaultS3Dependencies(this)
        }

        enabled = !project.hasProperty("disable-tests")
    }
}

fun KotlinJvmTarget.configureBenchmarkTests() {
    testRuns.create("benchmark").executionTask {
        group = "verification"

        maxHeapSize = "4G"

        useJUnitPlatform()

        filter {
            includeTestsMatching("*.benchmark_*")
        }

        testLogging {
            events(TestLogEvent.STANDARD_ERROR, TestLogEvent.STARTED, TestLogEvent.PASSED, TestLogEvent.FAILED, TestLogEvent.SKIPPED)
        }

        doFirst {
            S3Dependency.withDefaultS3Dependencies(this)
        }

        enabled = !project.hasProperty("disable-tests")
    }
}

fun KotlinJvmTarget.configureGpuTests() {
    testRuns.create("gpu").executionTask {
        group = "verification"

        maxHeapSize = "4G"

        useJUnitPlatform()

        filter {
            includeTestsMatching("*.gpu_*")
        }

        testLogging {
            events(TestLogEvent.STANDARD_ERROR, TestLogEvent.STARTED, TestLogEvent.PASSED, TestLogEvent.FAILED, TestLogEvent.SKIPPED)
        }

        doFirst {
            S3Dependency.withDefaultS3Dependencies(this)
        }

        enabled = !project.hasProperty("disable-tests")
    }
}
