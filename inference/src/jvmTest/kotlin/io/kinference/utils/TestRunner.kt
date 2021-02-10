package io.kinference.utils

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.runBlocking

actual object TestRunner {
    actual enum class Platform {
        JS, JVM
    }

    actual val platform: Platform = Platform.JVM

    actual fun runTest(block: suspend CoroutineScope.() -> Unit) {
        runBlocking { block() }
    }
}
