@file:OptIn(ExperimentalStdlibApi::class)
package io.kinference.utils

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlin.coroutines.*

object ResourcesDispatcher {
    private val tokenChannel = Channel<Unit>(capacity = PlatformUtils.cores)

    suspend fun reserveCore() {
        tokenChannel.send(Unit)
    }

    suspend fun releaseCore() {
        tokenChannel.receive()
    }
}

sealed class PredictionContext(
    val dispatcher: CoroutineDispatcher
) : AbstractCoroutineContextElement(PredictionContext) {
    companion object Key : CoroutineContext.Key<PredictionContext>

    override val key
        get() = Key

    override fun <E : CoroutineContext.Element> get(key: CoroutineContext.Key<E>): E? = getPolymorphicElement(key)

    override fun minusKey(key: CoroutineContext.Key<*>): CoroutineContext = minusPolymorphicKey(key)
}

interface ArrayStorage {
    fun resetState()
}

abstract class AllocatorContext<T : ArrayStorage>(
    dispatcher: CoroutineDispatcher,
    val storage: T
) : PredictionContext(dispatcher) {
    companion object Key : AbstractCoroutineContextKey<PredictionContext, AllocatorContext<*>>(
        PredictionContext.Key,
        { it as? AllocatorContext<*> }
    )

    fun finalizeContext() {
        storage.resetState()
    }
}

class NoAllocatorContext(dispatcher: CoroutineDispatcher) : PredictionContext(dispatcher) {
    companion object Key : AbstractCoroutineContextKey<PredictionContext, NoAllocatorContext>(
        PredictionContext.Key,
        { it as? NoAllocatorContext }
    )
}

fun CoroutineScope.launchWithLimitOrDefault(block: suspend CoroutineScope.() -> Unit) {
    this.launch(coroutineContext[PredictionContext]?.dispatcher ?: Dispatchers.Default, block = block)
}
