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
        return if (limiter !is NoAllocatorMemoryLimiter) {
            Array(count) { unusedContainers.getArrayContainer(type, size) }
        } else {
            Array(count) { unusedContainers.create(type, size) }
        }
    }

    fun closeOperator() {
        unusedContainers.moveUsedArrays()
    }

    fun closeAllocated() {
        unusedContainers.moveUsedArrays()
        returnStorageFn(unusedContainers)
    }
}
