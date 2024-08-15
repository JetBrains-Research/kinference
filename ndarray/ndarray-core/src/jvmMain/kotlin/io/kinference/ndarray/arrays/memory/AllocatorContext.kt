package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import kotlin.coroutines.CoroutineContext

data class AllocatorContext internal constructor(
    private val unusedContainers: ArrayStorage,
    private val limiter: MemoryLimiter,
    private val returnStorageFn: (ArrayStorage) -> Unit
) : CoroutineContext.Element {

    companion object Key : CoroutineContext.Key<AllocatorContext>
    override val key: CoroutineContext.Key<*> get() = Key

    internal fun getArrayContainers(type: ArrayTypes, size: Int, count: Int): Array<Any> {
        return Array(count) { unusedContainers.getArrayContainer(type, size) }
    }

    fun closeAllocated() {
        unusedContainers.moveUsedArrays()
        returnStorageFn(unusedContainers)
    }
}
