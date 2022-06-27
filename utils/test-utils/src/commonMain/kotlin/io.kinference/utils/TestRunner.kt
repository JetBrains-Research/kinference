package io.kinference.utils

import kotlinx.coroutines.CoroutineScope

expect object TestRunner {
    fun runTest(block: suspend CoroutineScope.() -> Unit)
}
