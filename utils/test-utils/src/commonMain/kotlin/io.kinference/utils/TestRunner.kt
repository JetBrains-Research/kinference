package io.kinference.utils

import kotlinx.coroutines.CoroutineScope

expect object TestRunner {
    fun runTest(platform: Platform? = null, block: suspend CoroutineScope.() -> Unit)
}
