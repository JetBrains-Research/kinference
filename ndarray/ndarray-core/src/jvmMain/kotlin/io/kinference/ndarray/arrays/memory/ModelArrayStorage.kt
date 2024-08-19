package io.kinference.ndarray.arrays.memory

import io.kinference.primitives.types.DataType
import io.kinference.utils.Closeable
import java.util.concurrent.ConcurrentLinkedQueue

class ModelArrayStorage(private val limiter: MemoryLimiter = MemoryLimiters.NoAllocator) : Closeable {
    private val autoStorageQueue: ConcurrentLinkedQueue<ArrayStorage> = ConcurrentLinkedQueue()

    companion object {
        private const val INIT_SIZE_VALUE: Int = 2
        private val typeSize: Int = DataType.entries.size
    }

    fun createAutoAllocatorContext(): AutoAllocatorContext {
        return AutoAllocatorContext(getStorage(autoStorageQueue), ::returnStorage)
    }

    fun createManualAllocatorContext(): ManualAllocatorContext {
        limiter.resetLimit()
        return ManualAllocatorContext(SingleArrayStorage(typeSize, INIT_SIZE_VALUE, limiter))
    }

    fun clearCache() {
        autoStorageQueue.clear()
    }

    override suspend fun close() {
        clearCache()
    }

    private fun getStorage(queue: ConcurrentLinkedQueue<ArrayStorage>): ArrayStorage {
        return queue.poll() ?: ArrayStorage(typeSize, INIT_SIZE_VALUE, limiter)
    }

    private fun returnStorage(storage: ArrayStorage) {
        autoStorageQueue.offer(storage)
    }
}
