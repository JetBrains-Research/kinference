package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.memory.contexts.*
import io.kinference.ndarray.arrays.memory.storage.*
import io.kinference.utils.*
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.coroutines.ContinuationInterceptor
import kotlin.coroutines.coroutineContext

class PredictionContextDispatcher(private val predictionConfig: PredictionConfig) : Closeable {
    private val limiter: MemoryManager = MemoryManager(
        memoryLimit = predictionConfig.memoryThreshold,
        cacheClearingInterval = predictionConfig.memoryClearingInterval,
        onCacheClear = ::clearCache)

    private val contextQueue: ConcurrentLinkedQueue<PredictionContext> = ConcurrentLinkedQueue()
    val allocationMode
        get() = predictionConfig.allocationMode

    suspend fun getPredictionContext(): PredictionContext {
        val allocatorContext = when (predictionConfig.allocationMode) {
            AllocationMode.NoAllocation -> getNoAllocatorContext()
            AllocationMode.Manual -> getManualAllocatorContext()
            AllocationMode.Auto -> getAutoAllocatorContext()
        }
        return allocatorContext
    }

    private suspend fun getNoAllocatorContext(): PredictionContext {
        return contextQueue.poll() ?: (NoAllocatorContext(getDispatcherWithLimit()))
    }

    private suspend fun getAutoAllocatorContext(): PredictionContext {
        limiter.updateLastAccessTime()
        return contextQueue.poll() ?: (AutoAllocatorContext(getDispatcherWithLimit(), AutoArrayHandlingStorage(limiter)))
    }

    private suspend fun getManualAllocatorContext(): PredictionContext {
        limiter.updateLastAccessTime()
        return contextQueue.poll() ?: (ManualAllocatorContext(getDispatcherWithLimit(), ManualArrayHandlingStorage(limiter)))
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    suspend fun getDispatcherWithLimit(): CoroutineDispatcher {
        val currentDispatcher = coroutineContext[ContinuationInterceptor] as? CoroutineDispatcher
            ?: Dispatchers.Default

        return if (currentDispatcher.isLikelyLimited()) {
            currentDispatcher
        } else {
            currentDispatcher.limitedParallelism(predictionConfig.parallelismLimit)
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    fun CoroutineDispatcher.isLikelyLimited(): Boolean {
        val className = this.javaClass.name
        return "LimitedDispatcher" in className || "Limited" in this::class.simpleName.orEmpty()
    }

    fun clearCache() {
        limiter.stopMonitoring()
        contextQueue.clear()
        limiter.resetLimit()
    }

    override fun close() {
        clearCache()
    }

    fun returnStorage(context: PredictionContext) {
        if (context is AllocatorContext<*>) {
            context.finalizeContext()
        }
        contextQueue.offer(context)
    }
}
