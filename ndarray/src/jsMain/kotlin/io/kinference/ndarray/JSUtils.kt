package io.kinference.ndarray

import kotlinx.coroutines.*
import kotlin.coroutines.CoroutineContext

actual fun runBlocking(context: CoroutineContext, block: suspend CoroutineScope.() -> Unit): dynamic =
    GlobalScope.promise(context = context) { block() }
