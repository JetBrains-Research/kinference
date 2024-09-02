package io.kinference.utils

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

object ResourcesDispatcher {
    private val tokenChannel = Channel<Unit>(capacity = PlatformUtils.cores)

    suspend fun reserveCore() {
        tokenChannel.send(Unit)
    }

    suspend fun releaseCore() {
        tokenChannel.receive()
    }
}

interface PredictionKey<T : PredictionContext> : CoroutineContext.Key<T>

sealed class PredictionContext(
    val dispatcher: CoroutineDispatcher
) : AbstractCoroutineContextElement(PredictionContext) {
    companion object Key : PredictionKey<PredictionContext>
}

interface ArrayStorage {
    fun resetState()
}

abstract class AllocatorContext<T : ArrayStorage>(
    dispatcher: CoroutineDispatcher,
    val storage: T
) : PredictionContext(dispatcher) {

    fun finalizeContext() {
        storage.resetState()
    }
}

class NoAllocatorContext(dispatcher: CoroutineDispatcher) : PredictionContext(dispatcher)

fun CoroutineScope.launchWithLimitOrDefault(block: suspend CoroutineScope.() -> Unit) {
    this.launch(coroutineContext[PredictionContext]?.dispatcher ?: Dispatchers.Default, block = block)
}
