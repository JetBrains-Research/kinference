package io.kinference.ndarray

import kotlinx.coroutines.CoroutineScope
import mu.KLogger
import mu.KotlinLogging
import kotlin.coroutines.CoroutineContext

actual fun runBlocking(context: CoroutineContext, block: suspend CoroutineScope.() -> Unit) =
    kotlinx.coroutines.runBlocking(context, block) //{ block() }

actual fun logger(name: String) = KotlinLogging.logger(name)
actual fun logger(func: () -> Unit) = KotlinLogging.logger(func)
