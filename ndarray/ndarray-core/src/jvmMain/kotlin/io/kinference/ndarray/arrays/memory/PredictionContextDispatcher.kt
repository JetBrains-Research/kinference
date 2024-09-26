package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.memory.contexts.*
import io.kinference.ndarray.arrays.memory.storage.*
import io.kinference.utils.*
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentLinkedQueue

class PredictionContextDispatcher(private val predictionConfig: PredictionConfig) : Closeable {
    private val limiter: MemoryManager = MemoryManager(
        memoryLimit = predictionConfig.memoryThreshold,
        cacheClearingInterval = predictionConfig.memoryClearingInterval,
        onCacheClear = ::clearCache)

    private val contextQueue: ConcurrentLinkedQueue<PredictionContext> = ConcurrentLinkedQueue()
    val allocationMode
        get() = predictionConfig.allocationMode

    fun getPredictionContext(): PredictionContext {
        val allocatorContext = when (predictionConfig.allocationMode) {
            AllocationMode.NoAllocation -> getNoAllocatorContext()
            AllocationMode.Manual -> getManualAllocatorContext()
            AllocationMode.Auto -> getAutoAllocatorContext()
        }
        return allocatorContext
    }

    private fun getNoAllocatorContext(): PredictionContext {
        return contextQueue.poll() ?: (NoAllocatorContext(getDispatcher()))
    }

    private fun getAutoAllocatorContext(): PredictionContext {
        limiter.updateLastAccessTime()
        return contextQueue.poll() ?: (AutoAllocatorContext(getDispatcher(), AutoArrayHandlingStorage(limiter)))
    }

    private fun getManualAllocatorContext(): PredictionContext {
        limiter.updateLastAccessTime()
        return contextQueue.poll() ?: (ManualAllocatorContext(getDispatcher(), ManualArrayHandlingStorage(limiter)))
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun getDispatcher(): CoroutineDispatcher {
        return Dispatchers.Default.limitedParallelism(predictionConfig.parallelismLimit)
    }

    fun clearCache() {
        limiter.stopMonitoring()
        contextQueue.clear()
        limiter.resetLimit()
    }

    override suspend fun close() {
        clearCache()
    }

    fun returnStorage(context: PredictionContext) {
        if (context is AllocatorContext<*>) {
            context.finalizeContext()
        }
        contextQueue.offer(context)
    }
}
