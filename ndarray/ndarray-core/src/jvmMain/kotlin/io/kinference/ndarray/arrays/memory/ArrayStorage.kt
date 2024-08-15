package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.ArrayTypes

internal class ArrayStorage(typeLength: Int, sizeLength: Int, private val limiter: MemoryLimiter) {
    /**
     * This is a storage for arrays which are available for retrieving
     *
     * Structure is as follows:
     * 1. Array by predefined types (all types are known compiled time)
     * 2. Array by size. Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
     * 3. Queue of array containers (used as FIFO)
     */
    private var storageUnused: Array<Array<ArrayDeque<Any>>> =
        Array(typeLength) { Array(sizeLength) { ArrayDeque() } }

    /**
     * This is a storage for arrays which are currently in use.
     * They should be moved back into unused storage when there is no need for them.
     *
     * Structure is as follows:
     * 1. Array by predefined types (all types are known compiled time)
     * 2. Array by size.
     * Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
     * 3. Queue of array containers (used as FIFO)
     */
    private var storageUsed: Array<Array<ArrayDeque<Any>>> =
        Array(typeLength) { Array(sizeLength) { ArrayDeque() } }

    private var sizeIndices: IntArray = IntArray(typeLength)
    private var sizes: Array<IntArray> = Array(typeLength) { IntArray(sizeLength) }

    internal fun getArrayContainer(type: ArrayTypes, size: Int): Any {
        return if (limiter.checkMemoryLimitAndAdd(ArrayTypes.sizeInBytes(type.index, size))) {
            val tIndex = type.index
            val sIndex = getSizeIndex(tIndex, size)
            val array = storageUnused[tIndex][sIndex].removeFirstOrNull()?.also(::resetArray)
                ?: create(type, size)

            storageUsed[tIndex][sIndex].addLast(array)
            array
        } else {
            create(type, size)
        }
    }

    internal fun moveUsedArrays() {
        storageUsed.forEachIndexed { typeIndex, arraysByType ->
            arraysByType.forEachIndexed { sizeIndex, arrayDeque ->
                arrayDeque.forEach {
                    storageUnused[typeIndex][sizeIndex].addLast(it)
                }
                arrayDeque.clear()
            }
        }
        limiter.resetLimit()
    }

    private fun getSizeIndex(tIndex: Int, size: Int): Int {
        val sIndex = sizes[tIndex].indexOf(size)

        return if (sIndex != -1) {
            sIndex
        } else {
            if (sizeIndices[tIndex] >= storageUnused[tIndex].size)
                grow(tIndex)

            val idx = sizeIndices[tIndex]++
            sizes[tIndex][idx] = size
            idx
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

    private fun create(type: ArrayTypes, size: Int): Any {
        return when (type) {
            ArrayTypes.ByteArrayType -> ByteArray(size)         // 8-bit signed
            ArrayTypes.UByteArrayType -> UByteArray(size)       // 8-bit unsigned
            ArrayTypes.ShortArrayType -> ShortArray(size)       // 16-bit signed
            ArrayTypes.UShortArrayType -> UShortArray(size)     // 16-bit unsigned
            ArrayTypes.IntArrayType -> IntArray(size)           // 32-bit signed
            ArrayTypes.UIntArrayType -> UIntArray(size)         // 32-bit unsigned
            ArrayTypes.LongArrayType -> LongArray(size)         // 64-bit signed
            ArrayTypes.ULongArrayType -> ULongArray(size)       // 64-bit unsigned
            ArrayTypes.FloatArrayType -> FloatArray(size)
            ArrayTypes.DoubleArrayType -> DoubleArray(size)
            ArrayTypes.BooleanArrayType -> BooleanArray(size)
            else -> throw IllegalArgumentException("Unsupported array type")
        }
    }

    private fun resetArray(array: Any): Unit =
        when (array) {
            is ByteArray -> array.fill(0)               // 8-bit signed
            is UByteArray -> array.fill(0u)             // 8-bit unsigned
            is ShortArray -> array.fill(0)              // 16-bit signed
            is UShortArray -> array.fill(0u)            // 16-bit unsigned
            is IntArray -> array.fill(0)                // 32-bit signed
            is UIntArray -> array.fill(0u)              // 32-bit unsigned
            is LongArray -> array.fill(0L)              // 64-bit signed
            is ULongArray -> array.fill(0U)             // 64-bit unsigned
            is FloatArray -> array.fill(0.0f)
            is DoubleArray -> array.fill(0.0)
            is BooleanArray -> array.fill(false)
            else -> throw IllegalArgumentException("Unsupported array type")
        }
}
