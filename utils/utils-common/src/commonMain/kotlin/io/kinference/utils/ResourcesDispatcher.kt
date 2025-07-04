@file:OptIn(ExperimentalStdlibApi::class)
package io.kinference.utils

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.selects.select
import kotlin.coroutines.*

fun interface SuspendCloseable {
    suspend fun close()
}

suspend inline fun <T : SuspendCloseable, R> T.suspendUse(block: (T) -> R): R {
    try {
        return block(this)
    } finally {
        this.close()
    }
}

sealed class ResourceController(
    protected val tokenChannel: Channel<Unit>,
    private val onLeaseCreated: (suspend () -> Unit) = {  },
    private val onLeaseClosed: (suspend () -> Unit) = {  }
) {

    suspend fun reserveCore(): CoreLease {
        var lease: CoreLease? = null

        return runCatching {
            select {
                tokenChannel.onSend(Unit) {
                    CoreLease().also {
                        lease = it
                        onLeaseCreated()
                    }
                }
            }
        }.getOrElse { ex ->
            if (ex is CancellationException) {
                lease?.close() // manually release
                onLeaseClosed()
            }
            throw ex
        }
    }

    inner class CoreLease : SuspendCloseable {
        override suspend fun close() {
            withContext(NonCancellable) {
                tokenChannel.receive()
                onLeaseClosed()
            }
        }
    }
}

object ResourcesDispatcher: ResourceController(Channel(PlatformUtils.cores))

// Internal implementation for tests
internal class TestResourcesDispatcher private constructor(
    onLeaseCreated: (suspend () -> Unit) = {  },
    onLeaseClosed: (suspend () -> Unit) = {  }
) : ResourceController(Channel(capacity = 1), onLeaseCreated, onLeaseClosed) {
    val testChannel get() = tokenChannel

    companion object {
        fun createTestResourceDispatcher(onLeaseCreated: (suspend () -> Unit) = {  }, onLeaseClosed: (suspend () -> Unit) = {  }): TestResourcesDispatcher {
            return TestResourcesDispatcher(onLeaseCreated, onLeaseClosed)
        }
    }
}

sealed class PredictionContext(
    val dispatcher: CoroutineDispatcher
) : AbstractCoroutineContextElement(PredictionContext) {
    companion object Key : CoroutineContext.Key<PredictionContext>

    override val key
        get() = Key

    @OptIn(ExperimentalStdlibApi::class)
    override fun <E : CoroutineContext.Element> get(key: CoroutineContext.Key<E>): E? = getPolymorphicElement(key)

    @OptIn(ExperimentalStdlibApi::class)
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
