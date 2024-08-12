package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.ArrayTypes

internal class ArrayStorage(typeLength: Int, sizeLength: Int, private val limiter: MemoryLimiter) {
    /**
     * Structure is as follows:
     * 1. Array by predefined types (all types are known compiled time)
     * 2. Array by size. Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
     * 3. Queue of array containers (used as FIFO)
     */
    private var storageUnused: Array<Array<ArrayDeque<Any>>> =
        Array(typeLength) { Array(sizeLength) { ArrayDeque() } }

    private var storageUsed: Array<Array<ArrayDeque<Any>>> =
        Array(typeLength) { Array(sizeLength) { ArrayDeque() } }

    private var sizeIndices: IntArray = IntArray(typeLength)
    private var sizes: Array<IntArray> = Array(typeLength) { IntArray(sizeLength) }


    operator fun get(typeIndex: Int, sizeIndex: Int): ArrayDeque<Any> {
        return storageUnused[typeIndex][sizeIndex]
    }

    fun getArrayContainer(type: ArrayTypes, size: Int): Any {
        val tIndex = type.index
        val sIndex = sizes[tIndex].indexOf(size)

        // Checking that we have this array size in our storage for this type
        val idx = if (sIndex != -1) {
            val array = storageUnused[tIndex][sIndex].removeFirstOrNull()
            array?.let {
                resetArray(it)
                limiter.deductMemory((type.size * size).toLong())
                storageUsed[tIndex][sIndex].addLast(it)
                return it
            }
            sIndex
        } else {
            if (sizeIndices[tIndex] >= storageUnused[tIndex].size)
                grow(tIndex)

            val idx = sizeIndices[tIndex]++
            sizes[tIndex][idx] = size
            idx
        }

        val array = create(type, size)
        storageUsed[tIndex][idx].addLast(array)

        return array
    }

    fun moveUsedArrays() {
        storageUsed.forEachIndexed { typeIndex, arraysByType ->
            arraysByType.forEachIndexed { sizeIndex, arrayDeque ->
                arrayDeque.forEach {
                    storageUnused[typeIndex][sizeIndex].addLast(it)
                }
                arrayDeque.clear()
            }
        }
    }

    private fun grow(typeIndex: Int) {
        val newSize = sizes[typeIndex].size * 2
        val newStorageUnused: Array<ArrayDeque<Any>> = Array(newSize) { ArrayDeque() }
        val newStorageUsed: Array<ArrayDeque<Any>> = Array(newSize) { ArrayDeque() }

        for (i in storageUnused[typeIndex].indices) {
            newStorageUnused[i] = storageUnused[typeIndex][i]
            newStorageUsed[i] = storageUsed[typeIndex][i]
        }

        storageUnused[typeIndex] = newStorageUnused
        storageUsed[typeIndex] = newStorageUsed
        sizes[typeIndex] = sizes[typeIndex].copyOf(newSize)
    }

    fun create(type: ArrayTypes, size: Int): Any {
        return when (type) {
            ArrayTypes.ByteArray -> ByteArray(size)         // 8-bit signed
            ArrayTypes.UByteArray -> UByteArray(size)       // 8-bit unsigned
            ArrayTypes.ShortArray -> ShortArray(size)       // 16-bit signed
            ArrayTypes.UShortArray -> UShortArray(size)     // 16-bit unsigned
            ArrayTypes.IntArray -> IntArray(size)           // 32-bit signed
            ArrayTypes.UIntArray -> UIntArray(size)         // 32-bit unsigned
            ArrayTypes.LongArray -> LongArray(size)         // 64-bit signed
            ArrayTypes.ULongArray -> ULongArray(size)       // 64-bit unsigned
            ArrayTypes.FloatArray -> FloatArray(size)
            ArrayTypes.DoubleArray -> DoubleArray(size)
            ArrayTypes.BooleanArray -> BooleanArray(size)
            else -> throw IllegalArgumentException("Unsupported array type")
        }
    }

    private fun resetArray(array: Any) {
        when (array) {
            is ByteArray -> array.fill(0)       // 8-bit signed
            is UByteArray -> array.fill(0u)     // 8-bit unsigned
            is ShortArray -> array.fill(0)      // 16-bit signed
            is UShortArray -> array.fill(0u)    // 16-bit unsigned
            is IntArray -> array.fill(0)        // 32-bit signed
            is UIntArray -> array.fill(0u)      // 32-bit unsigned
            is LongArray -> array.fill(0L)      // 64-bit signed
            is ULongArray -> array.fill(0U)     // 64-bit unsigned
            is FloatArray -> array.fill(0.0f)
            is DoubleArray -> array.fill(0.0)
            is BooleanArray -> array.fill(false)
            else -> throw IllegalArgumentException("Unsupported array type")
        }
    }
}
