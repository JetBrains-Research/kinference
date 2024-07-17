package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.ArrayTypes
import io.kinference.utils.ConcurrentQueue

object ArrayDispatcher {
    private const val INIT_SIZE_VALUE: Int = 2
    private val typeSize: Int = ArrayTypes.entries.size

    private val unusedArrays: ConcurrentQueue<ArrayStorage> = ConcurrentQueue()

    internal fun getStorage(): ArrayStorage {
        return unusedArrays.removeFirstOrNull() ?: ArrayStorage(typeSize, INIT_SIZE_VALUE)
    }

    internal fun returnStorage(storage: ArrayStorage) {
        unusedArrays.addLast(storage)
    }

    fun clearCache() {
        unusedArrays.clear()
    }
}
