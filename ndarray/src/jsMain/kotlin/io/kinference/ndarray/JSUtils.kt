package io.kinference.ndarray

import kotlinx.coroutines.*
import mu.*
import kotlin.coroutines.CoroutineContext

actual fun runBlocking(context: CoroutineContext, block: suspend CoroutineScope.() -> Unit): dynamic =
    CoroutineScope(Dispatchers.Unconfined).launch { block() }

private object JSLogAppender : Appender {
    override fun trace(message: Any?) {
        console.log(message)
        console.log("\n")
    }

    override fun debug(message: Any?){
        console.log(message)
        console.log("\n")
    }

    override fun info(message: Any?){
        console.info(message)
        console.info("\n")
    }

    override fun warn(message: Any?){
        console.warn(message)
        console.warn("\n")
    }

    override fun error(message: Any?){
        console.error(message)
        console.error("\n")
    }
}

actual fun logger(name: String): KLogger {
    KotlinLoggingConfiguration.APPENDER = JSLogAppender
    return KotlinLogging.logger(name)
}

actual fun logger(func: () -> Unit): KLogger {
    KotlinLoggingConfiguration.APPENDER = JSLogAppender
    return KotlinLogging.logger(func)
}
