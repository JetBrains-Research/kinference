package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.ArrayTypes

internal class ArrayStorage(typeLength: Int, sizeLength: Int, private val limiter: MemoryLimiter) {
    /**
     * Structure is as follows:
     * 1. Array by predefined types (all types are known compiled time)
     * 2. Array by size. Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
     * 3. Queue of array containers (used as FIFO)
     */
    private var storage: Array<Array<ArrayDeque<ArrayContainer>>> =
        Array(typeLength) { Array(sizeLength) { ArrayDeque() } }

    private var sizeIndices: IntArray = IntArray(typeLength)
    private var sizes: Array<IntArray> = Array(typeLength) { IntArray(sizeLength) }


    operator fun get(typeIndex: Int, sizeIndex: Int): ArrayDeque<ArrayContainer> {
        return storage[typeIndex][sizeIndex]
    }

    fun getArrayContainer(type: ArrayTypes, size: Int): ArrayContainer {
        val tIndex = type.index
        val sIndex = sizes[tIndex].indexOf(size)

        // Checking that we have this array size in our storage for this type
        val idx = if (sIndex != -1) {
            val array = storage[tIndex][sIndex].removeFirstOrNull()
            array?.let {
                ArrayContainer.resetArray(it)
                limiter.deductMemory(it.sizeBytes.toLong())
                return it
            }
            sIndex
        } else {
            if (sizeIndices[tIndex] >= storage[tIndex].size)
                grow(tIndex)

            val idx = sizeIndices[tIndex]++
            sizes[tIndex][idx] = size
            idx
        }

        return ArrayContainer(type, size, idx)
    }

    private fun grow(typeIndex: Int) {
        val newSize = sizes[typeIndex].size * 2
        val newStorage: Array<ArrayDeque<ArrayContainer>> = Array(newSize) { ArrayDeque() }

        for (i in storage[typeIndex].indices) {
            newStorage[i] = storage[typeIndex][i]
        }

        storage[typeIndex] = newStorage
        sizes[typeIndex] = sizes[typeIndex].copyOf(newSize)
    }
}
