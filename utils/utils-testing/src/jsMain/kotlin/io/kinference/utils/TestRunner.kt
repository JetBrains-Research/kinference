package io.kinference.utils

import kotlinx.coroutines.*

actual object TestRunner {
    actual fun runTest(platform: Platform?, block: suspend CoroutineScope.() -> Unit): dynamic {
        if (platform == null || platform == Platform.JS) {
            return GlobalScope.promise {
                block()
            }
        }

        return Unit
    }
}
