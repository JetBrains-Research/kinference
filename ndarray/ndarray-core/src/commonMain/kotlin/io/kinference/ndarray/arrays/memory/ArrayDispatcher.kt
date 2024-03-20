package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.ArrayTypes
import io.kinference.ndarray.arrays.ArrayUsageMarker
import kotlinx.atomicfu.atomic

object ArrayDispatcher {
    private val modelDispatchers = mutableMapOf<String, ModelArrayDispatcher>()

    fun addModelContext(modelContext: String) {
        modelDispatchers[modelContext] = ModelArrayDispatcher()
    }

    fun removeModelContext(modelContext: String) {
        modelDispatchers.remove(modelContext)?.close()
    }

    fun beginOperatorMode(modelContext: String) {
        modelDispatchers[modelContext]!!.beginOperatorMode()
    }

    fun endOperatorMode(modelContext: String) {
        modelDispatchers[modelContext]!!.endOperatorMode()
    }

    fun getArraysAndMarkers(modelContext: String, type: ArrayTypes, size: Int, count: Int): Array<ArrayContainer> {
        if (modelContext == "NoContext")
            return Array(count) { ArrayContainer(type, size) }

        val dispatcher = modelDispatchers[modelContext]!!
        return dispatcher.getArraysAndMarkers(type, size, count)
    }

    fun releaseAllOutputArrays(modelContext: String) {
        modelDispatchers[modelContext]!!.releaseAllOutputArrays()
    }
}

class ModelArrayDispatcher {
    companion object {
        private const val INIT_SIZE_VALUE: Int = 2
    }

    private val typeSize: Int = ArrayTypes.entries.size

    private var operatorMode: Boolean = false

    private var contextUsedArrays: ArrayStorage = ArrayStorage(typeSize, INIT_SIZE_VALUE)
    private var contextUnusedArrays: ArrayStorage = ArrayStorage(typeSize, INIT_SIZE_VALUE)

    private var sizeIndices: IntArray = IntArray(typeSize)
    private var sizes: Array<IntArray> = Array(typeSize) { IntArray(INIT_SIZE_VALUE) }

    class LockFreeArrayContainerQueue {
        // Initialize the head with the emptyContainer sentinel node
        private val head = atomic(ArrayContainer.emptyContainer())
        private val tail = atomic(head.value)
        private val isClosed = atomic(false)

        fun addLast(container: ArrayContainer) {
            // Check if the queue is closed
            if (isClosed.value) {
                throw IllegalStateException("Cannot add to a closed queue.")
            }

            while (true) {
                val curTail = tail.value
                val tailNext = curTail.next.value

                if (curTail == tail.value) { // Ensure the tail hasn't been moved
                    if (tailNext == null) { // Tail is at the end, attempt to append the new container
                        if (curTail.next.compareAndSet(null, container)) {
                            // Successfully added, try to move the tail to the new container
                            tail.compareAndSet(curTail, container)
                            return
                        }
                    } else {
                        // Tail not pointing to the last node, try to advance the tail
                        tail.compareAndSet(curTail, tailNext)
                    }
                }
            }
        }

        fun removeFirstOrNull(): ArrayContainer? {
            // Check if the queue is closed
            if (isClosed.value) {
                throw IllegalStateException("Cannot add to a closed queue.")
            }

            while (true) {
                val curHead = head.value
                val headNext = curHead.next.value

                if (headNext != null) {
                    // Attempt to update the sentinel's next pointer to skip the first data node, effectively dequeuing it
                    if (curHead.next.compareAndSet(headNext, headNext.next.value)) {
                        // Nullify the removed node's next pointer to avoid dangling references
                        headNext.next.value = null

                        // Check if we've just removed the last element, making the queue empty (besides the sentinel)
                        if (headNext == tail.value) {
                            // Attempt to reset the tail to the sentinel node, since the queue is now empty
                            tail.compareAndSet(headNext, curHead)
                        }

                        return headNext  // Return the dequeued container
                    }
                } else {
                    // The queue is empty (besides the sentinel)
                    return null
                }
            }
        }

        fun close() {
            isClosed.value = true

            while (true) {
                val currentHead = head.value
                val currentTail = tail.value

                // If the queue is already empty, return.
                if (currentHead.next.value == null) {
                    return
                }

                // Attempt to update the head to point to the tail, effectively clearing the queue.
                if (head.compareAndSet(currentHead, currentTail)) {
                    // Clear the 'next' references from the removed elements.
                    var container = currentHead
                    while (container != currentTail) {
                        val nextContainer = container.next.value
                        container.next.value = null  // Nullify the 'next' reference.
                        container = nextContainer ?: break
                    }

                    // Check if the tail has moved due to concurrent additions.
                    if (tail.value == currentTail) {
                        return  // The queue is successfully cleared.
                    }
                }
            }
        }
    }

    private class ArrayStorage(typeLength: Int, sizeLength: Int) {
        /**
         * Structure is as follows:
         * 1. Array by predefined types (all types are known compiled time)
         * 2. Array by size. Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
         * 3. Queue of array containers (used as FIFO)
         */
        private var storage: Array<Array<LockFreeArrayContainerQueue>> =
            Array(typeLength) { Array(sizeLength) { LockFreeArrayContainerQueue() } }

        operator fun get(typeIndex: Int, sizeIndex: Int): LockFreeArrayContainerQueue {
            return storage[typeIndex][sizeIndex]
        }

        fun getSizeLength(typeIndex: Int): Int {
            return storage[typeIndex].size
        }

        fun grow(typeIndex: Int, newSize: Int) {
            // Create a new array of the new size
            val newStorage: Array<LockFreeArrayContainerQueue> = Array(newSize) { LockFreeArrayContainerQueue() }

            // Transfer the old data into the new arrays
            for (i in storage[typeIndex].indices) {
                newStorage[i] = storage[typeIndex][i]
            }

            // Assign the new arrays back to the contextUsedArrays and contextUnusedArrays
            storage[typeIndex] = newStorage
        }
    }

    fun beginOperatorMode() {
        operatorMode = true
    }

    fun endOperatorMode() {
        operatorMode = false
    }

    fun getArraysAndMarkers(type: ArrayTypes, size: Int, count: Int): Array<ArrayContainer> {
//        return Array(count) { ArrayContainer(type, size) }
        return Array(count) { getArray(type, size) }
    }

    fun releaseAllOutputArrays() {
        // Iterate through all contexts, types and sizes
        for (i in 0 until typeSize) {
            for (j in 0 until sizes[i].size) {
                val usedArrays = contextUsedArrays[i, j]
                val unusedArrays = contextUnusedArrays[i, j]

                // Move all context outputs into unused and permanently remove all global outputs
                var isProcessed = false
                while (!isProcessed) {
                    val container = usedArrays.removeFirstOrNull()
                    if (container != null) {
                        if (container.marker != ArrayUsageMarker.GlobalOutput) {
                            container.marker = ArrayUsageMarker.Unused
                            unusedArrays.addLast(container)
                        }
                    } else {
                        isProcessed = true
                    }
                }
            }
        }
    }

    fun close() {
        // Iterate through all contexts, types and sizes
        for (i in 0 until typeSize) {
            for (j in 0 until sizes[i].size) {
                val usedArrays = contextUsedArrays[i, j]
                val unusedArrays = contextUnusedArrays[i, j]

                usedArrays.close()
                unusedArrays.close()
            }
        }

        contextUsedArrays = ArrayStorage(typeSize, INIT_SIZE_VALUE)
        contextUnusedArrays = ArrayStorage(typeSize, INIT_SIZE_VALUE)
        sizeIndices = IntArray(typeSize)
        sizes = Array(typeSize) { IntArray(INIT_SIZE_VALUE) }
    }

    private fun getArray(type: ArrayTypes, size: Int): ArrayContainer {
        // If we are not in the operator mode, then we don't store a created array
        if (!operatorMode) {
            return ArrayContainer(type, size)
        }

        val tIndex = type.index
        val sIndex = sizes[tIndex].indexOf(size)

        // Checking that we have this array size in our storage for this context and type
        if (sIndex != -1) {
            val array = contextUnusedArrays[tIndex, sIndex].removeFirstOrNull()
            array?.let {
                it.marker = ArrayUsageMarker.Used
                ArrayContainer.resetArray(it)
                contextUsedArrays[tIndex, sIndex].addLast(it)
                return it
            }
        }

        val newArray = ArrayContainer(type, size)
        putArray(tIndex, sIndex, size, newArray)
        return newArray
    }

    private fun putArray(typeIndex: Int, sizeIndex: Int, size: Int, array: ArrayContainer) {
        // Checking that we have the corresponding size-representing index and enough space if the size is new.
        // If not, then we grow the corresponding array and add a new index.
        val idx = if (sizeIndex != -1) {
            sizeIndex
        } else {
            if (sizeIndices[typeIndex] >= contextUnusedArrays.getSizeLength(typeIndex))
                grow(typeIndex)

            val idx = sizeIndices[typeIndex]++
            sizes[typeIndex][idx] = size
            idx
        }
        contextUsedArrays[typeIndex, idx].addLast(array)
    }

    private fun grow(typeIndex: Int) {
        // Determine the new size, typically double the current size
        val newSize = sizes[typeIndex].size * 2

        // Actual grow
        contextUsedArrays.grow(typeIndex, newSize)
        contextUnusedArrays.grow(typeIndex, newSize)

        // Resize the sizes array
        sizes[typeIndex] = sizes[typeIndex].copyOf(newSize)
    }
}
