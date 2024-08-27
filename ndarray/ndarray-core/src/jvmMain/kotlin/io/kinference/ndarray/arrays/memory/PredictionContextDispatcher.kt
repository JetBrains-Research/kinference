package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.memory.contexts.*
import io.kinference.ndarray.arrays.memory.storage.*
import io.kinference.utils.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.coroutines.CoroutineContext

interface ArrayStorage {
    fun resetState()
}

class PredictionContextDispatcher(private val predictionConfig: PredictionConfig) : Closeable {
    private val limiter: MemoryManager = MemoryManager(
        memoryLimit = predictionConfig.memoryThreshold,
        cacheClearingInterval = predictionConfig.memoryClearingInterval,
        onCacheClear = ::clearCache)

    private val contextQueue: ConcurrentLinkedQueue<CoroutineContext> = ConcurrentLinkedQueue()
    val allocationMode
        get() = predictionConfig.allocationMode

    fun getPredictionContext(): CoroutineContext {
        val allocatorContext = when (predictionConfig.allocationMode) {
            AllocationMode.NoAllocation -> getNoAllocatorContext()
            AllocationMode.Manual -> getManualAllocatorContext()
            AllocationMode.Auto -> getAutoAllocatorContext()
        }
        return allocatorContext
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun getNoAllocatorContext(): CoroutineContext {
        return contextQueue.poll() ?: (NoAllocatorContext() + ParallelismLimiterContext(Dispatchers.Default.limitedParallelism(predictionConfig.parallelismLimit)))
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun getAutoAllocatorContext(): CoroutineContext {
        limiter.updateLastAccessTime()
        return contextQueue.poll() ?: (AutoAllocatorContext(AutoArrayHandlingStorage(limiter)) + ParallelismLimiterContext(Dispatchers.Default.limitedParallelism(predictionConfig.parallelismLimit)))
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun getManualAllocatorContext(): CoroutineContext {
        limiter.updateLastAccessTime()
        return contextQueue.poll() ?: (ManualAllocatorContext(ManualArrayHandlingStorage(limiter)) + ParallelismLimiterContext(Dispatchers.Default.limitedParallelism(predictionConfig.parallelismLimit)))
    }

    fun clearCache() {
        limiter.stopMonitoring()
        contextQueue.clear()
        limiter.resetLimit()
    }

    override suspend fun close() {
        clearCache()
    }

    fun returnStorage(context: CoroutineContext) {
        contextQueue.offer(context)
    }
}
