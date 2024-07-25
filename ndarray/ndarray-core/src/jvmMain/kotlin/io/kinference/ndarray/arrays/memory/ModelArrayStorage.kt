package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.ArrayTypes
import io.kinference.utils.Closeable

import io.kinference.utils.ConcurrentQueue

class ModelArrayStorage(private val limiter: MemoryLimiter = MemoryLimiters.Default) : Closeable {
    private val unusedArrays: ConcurrentQueue<ArrayStorage> = ConcurrentQueue()

    companion object {
        private const val INIT_SIZE_VALUE: Int = 2
        private val typeSize: Int = ArrayTypes.entries.size
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
        return unusedArrays.removeFirstOrNull() ?: ArrayStorage(typeSize, INIT_SIZE_VALUE)
    }

    private fun returnStorage(storage: ArrayStorage) {
        unusedArrays.addLast(storage)
    }
}
