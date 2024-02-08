package io.kinference.ndarray.arrays

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

    private class ArrayStorage(contextLength: Int, typeLength: Int, sizeLength: Int) {
        /**
         * Structure is as follows:
         * 1. Array by operators
         * 2. Array by predefined types (all types are known compiled time)
         * 3. Array by size. Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
         * 4. Queue of array containers (used as FIFO)
         */
        private var storage: Array<Array<Array<ArrayDeque<ArrayContainer<*>>>>> =
            Array(contextLength) { Array(typeLength) { Array(sizeLength) { ArrayDeque() } } }

        operator fun get(contextIndex: Int, typeIndex: Int, sizeIndex: Int): ArrayDeque<ArrayContainer<*>> {
            return storage[contextIndex][typeIndex][sizeIndex]
        }

        fun getSizeLength(contextIndex: Int, typeIndex: Int): Int {
            return storage[contextIndex][typeIndex].size
        }

        fun grow(contextIndex: Int, typeIndex: Int, newSize: Int) {
            // Create a new array of the new size
            val newStorage: Array<ArrayDeque<ArrayContainer<*>>> =
                Array(newSize) { ArrayDeque() }

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

    inline fun <reified T> getArrays(type: ArrayTypes, size: Int, count: Int): Array<T> {
        return Array(count) { (getArray(type, size)).array as T }
    }

    inline fun <reified T> getArraysAndMarkers(type: ArrayTypes, size: Int, count: Int): Array<ArrayContainer<T>> {
        return Array(count) { getArray(type, size) as ArrayContainer<T> }
    }

    fun getArray(type: ArrayTypes, size: Int): ArrayContainer<*> {
        // If we are not in the operator mode, then we don't store a created array
        if (!operatorMode) {
            return type.createArray(size)
        }

        val tIndex = type.index
        val sIndex = sizes[currentOperatorContextIndex][tIndex].indexOf(size)

        // Checking that we have this array size in our storage for this context and type
        if (sIndex != -1) {
            val array = contextUnusedArrays[currentOperatorContextIndex, tIndex, sIndex].removeFirstOrNull()
            array?.let {
                it.marker = ArrayUsageMarker.Used
                resetPrimitiveArray(it)
                contextUsedArrays[currentOperatorContextIndex, tIndex, sIndex].addLast(it)
                return it
            }
        }

        val newArray = type.createArray(size)
        putArray(tIndex, sIndex, size, newArray)
        return newArray
    }

    fun releaseUsedInContext() {
        // When the context is released, move used arrays to unused struct and others into outputs
        for (i in 0 until typeSize) {
            for (j in 0 until sizeIndices[currentOperatorContextIndex][i]) {
                val usedArrays = contextUsedArrays[currentOperatorContextIndex, i, j]
                val unusedArrays = contextUnusedArrays[currentOperatorContextIndex, i, j]
                val outputArrays = contextOutputArrays[currentOperatorContextIndex, i, j]

                while (usedArrays.isNotEmpty()) {
                    val container = usedArrays.removeFirst()
                    if (container.marker != ArrayUsageMarker.ContextOutput) {
                        container.marker = ArrayUsageMarker.Unused
                        unusedArrays.addLast(container)
                    } else {
                        outputArrays.addLast(container)
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
                    while (outputArrays.isNotEmpty()) {
                        val container = outputArrays.removeFirst()
                        if (container.marker == ArrayUsageMarker.ContextOutput) {
                            unusedArrays.addLast(container)
                        }

                        container.marker = ArrayUsageMarker.Unused
                    }
                }
            }
        }
    }

    private fun putArray(typeIndex: Int, sizeIndex: Int, size: Int, array: ArrayContainer<*>) {
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
