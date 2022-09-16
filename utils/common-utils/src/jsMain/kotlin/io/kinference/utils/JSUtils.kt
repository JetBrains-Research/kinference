package io.kinference.utils

import kotlinx.coroutines.*
import kotlin.coroutines.CoroutineContext

actual fun runBlocking(context: CoroutineContext, block: suspend CoroutineScope.() -> Unit): dynamic =
    CoroutineScope(Dispatchers.Unconfined).launch { block() }
