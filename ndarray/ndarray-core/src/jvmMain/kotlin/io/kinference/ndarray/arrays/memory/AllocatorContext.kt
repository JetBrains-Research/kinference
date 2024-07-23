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
        val arrayContainers = arrayOfNulls<ArrayContainer>(count)
        for (i in 0 until count) {
            val container = unusedContainers.getArrayContainer(type, size)
            if (!container.isNewlyCreated)
                limiter.freeMemory(container.size.toLong())
            arrayContainers[i] = container
            usedContainers.add(container)
        }
        return arrayContainers as Array<ArrayContainer>
    }

    fun closeAllocated() {
        usedContainers.forEach {
            if (!it.isOutput && limiter.checkMemoryLimitAndAdd(it.size.toLong())) {
                unusedContainers[it.arrayTypeIndex, it.arraySizeIndex].addLast(it)
            }
        }
        usedContainers.clear()
        returnStorageFn(unusedContainers)
    }
}
