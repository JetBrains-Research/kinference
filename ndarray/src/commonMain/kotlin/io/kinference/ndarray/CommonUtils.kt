package io.kinference.ndarray

import kotlinx.coroutines.CoroutineScope
import mu.KLogger
import mu.KotlinLogging
import kotlin.coroutines.CoroutineContext

expect fun runBlocking(context: CoroutineContext, block: suspend CoroutineScope.() -> Unit)

expect fun logger(name: String): KLogger
expect fun logger(func: () -> Unit): KLogger
