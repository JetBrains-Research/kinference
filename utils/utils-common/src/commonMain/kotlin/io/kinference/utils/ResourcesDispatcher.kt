package io.kinference.utils

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
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

class ParallelismLimiterContext(val dispatcher: CoroutineDispatcher) : CoroutineContext.Element {
    companion object Key : CoroutineContext.Key<ParallelismLimiterContext>
    override val key: CoroutineContext.Key<*> get() = Key
}

fun CoroutineScope.launchWithLimitOrDefault(block: suspend CoroutineScope.() -> Unit) {
    this.launch(coroutineContext[ParallelismLimiterContext.Key]?.dispatcher ?: Dispatchers.Default) {
        block()
    }
}
