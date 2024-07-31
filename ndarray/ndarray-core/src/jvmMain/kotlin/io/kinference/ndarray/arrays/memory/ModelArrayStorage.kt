package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.ArrayTypes
import io.kinference.utils.Closeable
import java.util.concurrent.ConcurrentLinkedQueue

class ModelArrayStorage(private val limiter: MemoryLimiter = MemoryLimiters.NoAllocator) : Closeable {
    private val unusedArrays: ConcurrentLinkedQueue<ArrayStorage> = ConcurrentLinkedQueue()

    companion object {
        private const val INIT_SIZE_VALUE: Int = 2
        private val typeSize: Int = ArrayTypes.values().size
    }

    fun createAllocatorContext(): AllocatorContext {
        return AllocatorContext(getStorage(), limiter, ::returnStorage)
    }

    fun clearCache() {
        unusedArrays.clear()
    }

    override suspend fun close() {
        clearCache()
    }

    private fun getStorage(): ArrayStorage {
        return unusedArrays.poll() ?: ArrayStorage(typeSize, INIT_SIZE_VALUE, limiter)
    }

    private fun returnStorage(storage: ArrayStorage) {
        unusedArrays.offer(storage)
    }
}
