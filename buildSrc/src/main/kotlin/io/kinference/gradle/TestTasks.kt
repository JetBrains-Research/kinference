package io.kinference.gradle

import io.kinference.gradle.s3.S3Dependency
import org.gradle.api.tasks.testing.Test
import org.gradle.api.tasks.testing.logging.TestLogEvent

fun Test.configureTests() {
    useJUnitPlatform {
        excludeTags("heavy")
        excludeTags("benchmark")
    }
    maxHeapSize = "400m"

    testLogging {
        events(TestLogEvent.STANDARD_ERROR, TestLogEvent.STARTED, TestLogEvent.PASSED, TestLogEvent.FAILED, TestLogEvent.SKIPPED)
    }
}

fun Test.configureHeavyTests(dependencies: S3Dependency.Context.() -> Unit = {}) {
    val context = S3Dependency.Context(project).also { it.dependencies() }

    group = "verification"

    useJUnitPlatform {
        includeTags("heavy")
        excludeTags("benchmark")
    }

    maxHeapSize = "4G"

    testLogging {
        events(TestLogEvent.STANDARD_ERROR, TestLogEvent.STARTED, TestLogEvent.PASSED, TestLogEvent.FAILED, TestLogEvent.SKIPPED)
    }

    doFirst {
        context.dependencies.forEach { it.resolve() }
    }
}

fun Test.configureBenchmarkTests(dependencies: S3Dependency.Context.() -> Unit = {}) {
    val context = S3Dependency.Context(project).also { it.dependencies() }

    group = "verification"

    useJUnitPlatform {
        excludeTags("heavy")
        includeTags("benchmark")
    }

    testLogging {
        events(TestLogEvent.STANDARD_ERROR, TestLogEvent.STARTED, TestLogEvent.PASSED, TestLogEvent.FAILED, TestLogEvent.SKIPPED)
    }

    doFirst {
        context.dependencies.forEach { it.resolve() }
    }
}
