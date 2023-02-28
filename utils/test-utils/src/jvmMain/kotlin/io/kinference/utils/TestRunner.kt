package io.kinference.utils

import kotlinx.coroutines.*
import kotlinx.coroutines.runBlocking

actual object TestRunner {
    actual fun runTest(platform: Platform?, block: suspend CoroutineScope.() -> Unit) {
        if (platform == null || platform == Platform.JVM) {
            runBlocking(Dispatchers.Default) { block() }
        }
    }
}
