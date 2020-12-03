package io.kinference.gradle

import org.gradle.api.*
import org.gradle.api.tasks.testing.Test

inline fun <reified T : Task> Project.getOrCreateTask(name: String, type: Class<T>, configuration: Action<T>): Task {
    return tasks.findByName(name) ?: tasks.create(name, type, configuration)
}

fun Project.useHeavyTests() {
    getOrCreateTask("testHeavy", Test::class.java) {
        group = "verification"

        useJUnitPlatform {
            includeTags("heavy")
            excludeTags("benchmark")
        }

        maxHeapSize = "2G"

        testLogging {
            events("passed", "skipped", "failed")
        }
    }
}

fun Project.useBenchmarkTests() {
    getOrCreateTask("testPerformance", Test::class.java) {
        group = "verification"

        useJUnitPlatform {
            excludeTags("heavy")
            includeTags("benchmark")
        }

        testLogging {
            events("passed", "skipped", "failed")
        }
    }
}
