package io.kinference.utils

import kotlinx.coroutines.*

actual object TestRunner {
    actual fun runTest(platform: Platform?, block: suspend CoroutineScope.() -> Unit) {
        if (platform == null || platform == Platform.JS) {
            GlobalScope.promise {
                block()
            }
        }
    }
}
