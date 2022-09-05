package io.kinference.utils

import kotlinx.coroutines.CoroutineScope
import kotlin.coroutines.CoroutineContext

actual fun runBlocking(context: CoroutineContext, block: suspend CoroutineScope.() -> Unit) =
    kotlinx.coroutines.runBlocking(context, block)
