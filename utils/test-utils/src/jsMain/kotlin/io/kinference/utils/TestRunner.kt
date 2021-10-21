package io.kinference.utils

import kotlinx.coroutines.*

actual object TestRunner {
    actual enum class Platform {
        JS, JVM
    }

    actual val platform: Platform = Platform.JS

    actual fun runTest(block: suspend CoroutineScope.() -> Unit): dynamic {
        return GlobalScope.promise {
            block()
        }
    }
}
