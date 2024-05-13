package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
import kotlin.coroutines.CoroutineContext

data class AllocatorContext(val modelName: String, val cycleId: Long) : CoroutineContext.Element {
    private val usedContainers: ArrayDeque<ArrayContainer> = ArrayDeque()
    private val unusedContainers: ArrayStorage = ArrayDispatcher.getStorage()

    companion object Key : CoroutineContext.Key<AllocatorContext>
    override val key: CoroutineContext.Key<*> get() = Key

    internal fun getArrayContainers(type: ArrayTypes, size: Int, count: Int): Array<ArrayContainer> {
        val arrayContainers = Array(count) { unusedContainers.getArrayContainer(type, size) }
        usedContainers.addAll(arrayContainers)
        return arrayContainers
    }


    fun closeAllocated() {
        usedContainers.forEach {
            if (it.marker != ArrayUsageMarker.Output) {
                it.marker = ArrayUsageMarker.Unused
                unusedContainers[it.arrayTypeIndex, it.arraySizeIndex].addLast(it)
            }
        }
        ArrayDispatcher.returnStorage(unusedContainers)
    }
}
