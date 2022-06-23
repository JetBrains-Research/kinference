package io.kinference.utils

import kotlinx.coroutines.*

actual object TestRunner {
    actual fun runTest(block: suspend CoroutineScope.() -> Unit): dynamic {
        return GlobalScope.promise {
            block()
        }
    }
}

actual object PlatformUtils {
    actual val platform: Platform = Platform.JS
}
