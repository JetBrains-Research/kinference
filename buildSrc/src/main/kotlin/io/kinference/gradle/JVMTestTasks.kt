package io.kinference.gradle

import io.kinference.gradle.s3.S3Dependency
import org.gradle.api.tasks.testing.Test
import org.gradle.api.tasks.testing.logging.TestLogEvent

fun Test.configureTests() {
    maxHeapSize = "400m"

    useJUnitPlatform()

    filter {
        excludeTestsMatching("*.heavy_*")
        excludeTestsMatching("*.benchmark_*")
    }

    testLogging {
        events(TestLogEvent.STANDARD_ERROR, TestLogEvent.STARTED, TestLogEvent.PASSED, TestLogEvent.FAILED, TestLogEvent.SKIPPED)
    }
}

fun Test.configureHeavyTests() {
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
}

fun Test.configureBenchmarkTests() {
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
}
