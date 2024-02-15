package io.kinference.ndarray.arrays

import kotlinx.atomicfu.AtomicRef
import kotlinx.atomicfu.atomic

sealed class ArrayContainer(var marker: ArrayUsageMarker = ArrayUsageMarker.Used) {
    val markAsOutput: StateMarker = {
        marker = it
    }

    // Atomic reference to the next node, initialized to a special instance of empty container
    val next: AtomicRef<ArrayContainer?> = atomic(null)

    private class EmptyArrayContainer: ArrayContainer()

    companion object {
        fun emptyContainer(): ArrayContainer = EmptyArrayContainer()

        operator fun invoke(type: ArrayTypes, size: Int): ArrayContainer {
            return when (type) {
                ArrayTypes.ByteArray -> ByteArrayContainer(ByteArray(size))         // 8-bit signed
                ArrayTypes.UByteArray -> UByteArrayContainer(UByteArray(size))      // 8-bit unsigned
                ArrayTypes.ShortArray -> ShortArrayContainer(ShortArray(size))      // 16-bit signed
                ArrayTypes.UShortArray -> UShortArrayContainer(UShortArray(size))   // 16-bit unsigned
                ArrayTypes.IntArray -> IntArrayContainer(IntArray(size))            // 32-bit signed
                ArrayTypes.UIntArray -> UIntArrayContainer(UIntArray(size))         // 32-bit unsigned
                ArrayTypes.LongArray -> LongArrayContainer(LongArray(size))         // 64-bit signed
                ArrayTypes.ULongArray -> ULongArrayContainer(ULongArray(size))      // 64-bit unsigned
                ArrayTypes.FloatArray -> FloatArrayContainer(FloatArray(size))
                ArrayTypes.DoubleArray -> DoubleArrayContainer(DoubleArray(size))
                ArrayTypes.BooleanArray -> BooleanArrayContainer(BooleanArray(size))
                else -> throw IllegalArgumentException("Unsupported array type")
            }
        }

        fun resetArray(arrayContainer: ArrayContainer) {
            when (arrayContainer) {
                is ByteArrayContainer -> arrayContainer.array.fill(0)       // 8-bit signed
                is UByteArrayContainer -> arrayContainer.array.fill(0u)     // 8-bit unsigned
                is ShortArrayContainer -> arrayContainer.array.fill(0)      // 16-bit signed
                is UShortArrayContainer -> arrayContainer.array.fill(0u)    // 16-bit unsigned
                is IntArrayContainer -> arrayContainer.array.fill(0)        // 32-bit signed
                is UIntArrayContainer -> arrayContainer.array.fill(0u)      // 32-bit unsigned
                is LongArrayContainer -> arrayContainer.array.fill(0L)      // 64-bit signed
                is ULongArrayContainer -> arrayContainer.array.fill(0U)     // 64-bit unsigned
                is FloatArrayContainer -> arrayContainer.array.fill(0.0f)
                is DoubleArrayContainer -> arrayContainer.array.fill(0.0)
                is BooleanArrayContainer -> arrayContainer.array.fill(false)
                else -> throw IllegalArgumentException("Unsupported array type")
            }
        }
    }
}

object ArrayDispatcher {
    private const val INIT_SIZE_VALUE: Int = 2
    private val typeSize: Int = ArrayTypes.entries.size

    private var operatorMode: Boolean = false
    private var operatorsCount: Int = 0
    private var currentOperatorContextIndex: Int = -1

    // We need stack here because our computational graph can be divided into subgraphs.
    private val contextStack: ArrayDeque<Int> = ArrayDeque()

    private var contextUsedArrays: ArrayStorage = ArrayStorage(0, 0, 0)
    private var contextUnusedArrays: ArrayStorage = ArrayStorage(0, 0, 0)
    private var contextOutputArrays: ArrayStorage = ArrayStorage(0, 0, 0)

    private var sizeIndices: Array<IntArray> = arrayOf()
    private var sizes: Array<Array<IntArray>> = arrayOf()

    class LockFreeArrayContainerQueue {
        // Initialize the head with the emptyContainer sentinel node
        private val head = atomic(ArrayContainer.emptyContainer())
        private val tail = atomic(head.value)

        fun addLast(container: ArrayContainer) {
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
    }

    private class ArrayStorage(contextLength: Int, typeLength: Int, sizeLength: Int) {
        /**
         * Structure is as follows:
         * 1. Array by operators
         * 2. Array by predefined types (all types are known compiled time)
         * 3. Array by size. Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
         * 4. Queue of array containers (used as FIFO)
         */
        private var storage: Array<Array<Array<LockFreeArrayContainerQueue>>> =
            Array(contextLength) { Array(typeLength) { Array(sizeLength) { LockFreeArrayContainerQueue() } } }

        operator fun get(contextIndex: Int, typeIndex: Int, sizeIndex: Int): LockFreeArrayContainerQueue {
            return storage[contextIndex][typeIndex][sizeIndex]
        }

        fun getSizeLength(contextIndex: Int, typeIndex: Int): Int {
            return storage[contextIndex][typeIndex].size
        }

        fun grow(contextIndex: Int, typeIndex: Int, newSize: Int) {
            // Create a new array of the new size
            val newStorage: Array<LockFreeArrayContainerQueue> =
                Array(newSize) { LockFreeArrayContainerQueue() }

            // Transfer the old data into the new arrays
            for (i in storage[contextIndex][typeIndex].indices) {
                newStorage[i] = storage[contextIndex][typeIndex][i]
            }

            // Assign the new arrays back to the contextUsedArrays and contextUnusedArrays
            storage[contextIndex][typeIndex] = newStorage
        }
    }

    fun initStorage(operatorsCount: Int) {
        this.operatorsCount = operatorsCount
        contextUsedArrays = ArrayStorage(operatorsCount, typeSize, INIT_SIZE_VALUE)
        contextUnusedArrays = ArrayStorage(operatorsCount, typeSize, INIT_SIZE_VALUE)
        contextOutputArrays = ArrayStorage(operatorsCount, typeSize, INIT_SIZE_VALUE)
        sizeIndices = Array(operatorsCount) { IntArray(typeSize) }
        sizes = Array(operatorsCount) { Array(typeSize) { IntArray(INIT_SIZE_VALUE) } }
    }

    fun setOperatorContext(contextIndex: Int) {
        pushOperatorContext(contextIndex)
    }

    fun getArraysAndMarkers(type: ArrayTypes, size: Int, count: Int): Array<ArrayContainer> {
        return Array(count) { getArray(type, size) }
    }

    fun releaseUsedInContext() {
        // When the context is released, move used arrays to unused struct and others into outputs
        for (i in 0 until typeSize) {
            for (j in 0 until sizeIndices[currentOperatorContextIndex][i]) {
                val usedArrays = contextUsedArrays[currentOperatorContextIndex, i, j]
                val unusedArrays = contextUnusedArrays[currentOperatorContextIndex, i, j]
                val outputArrays = contextOutputArrays[currentOperatorContextIndex, i, j]

                var isProcessed = false
                while (!isProcessed) {
                    val container = usedArrays.removeFirstOrNull()
                    if (container != null) {
                        if (container.marker != ArrayUsageMarker.ContextOutput) {
                            container.marker = ArrayUsageMarker.Unused
                            unusedArrays.addLast(container)
                        } else {
                            outputArrays.addLast(container)
                        }
                    } else {
                        isProcessed = true
                    }
                }
            }
        }

        popOperatorContext()
    }

    fun releaseAllOutputArrays() {
        // Iterate through all contexts, types and sizes
        for (i in 0 until operatorsCount) {
            for (j in 0 until typeSize) {
                for (k in 0 until sizes[i][j].size) {
                    val outputArrays = contextOutputArrays[i, j, k]
                    val unusedArrays = contextUnusedArrays[i, j, k]

                    // Move all context outputs into unused and permanently remove all global outputs
                    var isProcessed = false
                    while (!isProcessed) {
                        val container = outputArrays.removeFirstOrNull()
                        if (container != null) {
                            if (container.marker == ArrayUsageMarker.ContextOutput) {
                                unusedArrays.addLast(container)
                            }
                            container.marker = ArrayUsageMarker.Unused
                        } else {
                            isProcessed = true
                        }
                    }
                }
            }
        }
    }

    private fun getArray(type: ArrayTypes, size: Int): ArrayContainer {
        // If we are not in the operator mode, then we don't store a created array
        if (!operatorMode) {
            return ArrayContainer(type, size)
        }

        val tIndex = type.index
        val sIndex = sizes[currentOperatorContextIndex][tIndex].indexOf(size)

        // Checking that we have this array size in our storage for this context and type
        if (sIndex != -1) {
            val array = contextUnusedArrays[currentOperatorContextIndex, tIndex, sIndex].removeFirstOrNull()
            array?.let {
                it.marker = ArrayUsageMarker.Used
                ArrayContainer.resetArray(it)
                contextUsedArrays[currentOperatorContextIndex, tIndex, sIndex].addLast(it)
                return it
            }
        }

        val newArray = ArrayContainer(type, size)
        putArray(tIndex, sIndex, size, newArray)
        return newArray
    }

    private fun putArray(typeIndex: Int, sizeIndex: Int, size: Int, array: ArrayContainer) {
        // Checking that we have corresponding size-representing index and enough space if the size is new.
        // If not, then we grow the corresponding array and add a new index.
        val idx = if (sizeIndex != -1) {
            sizeIndex
        } else {
            if (sizeIndices[currentOperatorContextIndex][typeIndex] >= contextUnusedArrays.getSizeLength(currentOperatorContextIndex, typeIndex))
                grow(typeIndex)

            val idx = sizeIndices[currentOperatorContextIndex][typeIndex]++
            sizes[currentOperatorContextIndex][typeIndex][idx] = size
            idx
        }
        contextUsedArrays[currentOperatorContextIndex, typeIndex, idx].addLast(array)
    }

    private fun grow(typeIndex: Int) {
        // Determine the new size, typically double the current size
        val newSize = sizes[currentOperatorContextIndex][typeIndex].size * 2

        // Actual grow
        contextUsedArrays.grow(currentOperatorContextIndex, typeIndex, newSize)
        contextUnusedArrays.grow(currentOperatorContextIndex, typeIndex, newSize)
        contextOutputArrays.grow(currentOperatorContextIndex, typeIndex, newSize)

        // Resize the sizes array
        sizes[currentOperatorContextIndex][typeIndex] = sizes[currentOperatorContextIndex][typeIndex].copyOf(newSize)
    }

    private fun pushOperatorContext(newContextIndex: Int) {
        contextStack.addFirst(newContextIndex)  // Save the current context
        currentOperatorContextIndex = newContextIndex
        operatorMode = true
    }

    private fun popOperatorContext() {
        currentOperatorContextIndex = if (contextStack.isNotEmpty()) {
            contextStack.removeFirst()
        } else {
            operatorMode = false
            -1
        }
    }
}
