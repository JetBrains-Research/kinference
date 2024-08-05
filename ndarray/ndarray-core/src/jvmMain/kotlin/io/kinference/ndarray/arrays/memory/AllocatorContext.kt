package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import kotlin.coroutines.CoroutineContext

data class AllocatorContext internal constructor(
    private val unusedContainers: ArrayStorage,
    private val limiter: MemoryLimiter,
    private val returnStorageFn: (ArrayStorage) -> Unit
) : CoroutineContext.Element {
    private val usedContainers: ArrayDeque<ArrayContainer> = ArrayDeque()

    companion object Key : CoroutineContext.Key<AllocatorContext>
    override val key: CoroutineContext.Key<*> get() = Key

    internal fun getArrayContainers(type: ArrayTypes, size: Int, count: Int): Array<ArrayContainer> {
        return if (limiter !is NoAllocatorMemoryLimiter) {
            val result = Array(count) { unusedContainers.getArrayContainer(type, size) }
            usedContainers.addAll(result)
            result
        } else {
            Array(count) { ArrayContainer(type, size) }
        }
    }

    fun closeAllocated() {
        usedContainers.forEach {
            if (limiter.checkMemoryLimitAndAdd(it.sizeBytes.toLong())) {
                unusedContainers[it.arrayTypeIndex, it.arraySizeIndex].addLast(it)
            }
        }
        usedContainers.clear()
        returnStorageFn(unusedContainers)
    }
}
