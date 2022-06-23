package io.kinference.utils

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.runBlocking

actual object TestRunner {
    actual fun runTest(block: suspend CoroutineScope.() -> Unit) {
        runBlocking { block() }
    }
}

actual object PlatformUtils {
    actual val platform: Platform = Platform.JVM
}
