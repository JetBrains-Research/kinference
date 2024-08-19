package io.kinference.ndarray.arrays.memory

import io.kinference.primitives.types.DataType

internal abstract class BaseArrayStorage(typeLength: Int, sizeLength: Int, storageCount: Int) {
    /**
     * This is a storage for arrays.
     *
     * Structure is as follows:
     * 1. Array by predefined types (all types are known compiled time)
     * 2. Array by size.
     * Starting with 'INIT_SIZE_VALUE' element and grow it doubling (typically there are no more than 16 different sizes)
     * 3. Queue of array containers (used as FIFO)
     */
    protected var storage: Array<Array<Array<ArrayDeque<Any>>>> =
        Array(storageCount) { Array(typeLength) { Array(sizeLength) { ArrayDeque() } } }

    private var sizeIndices: IntArray = IntArray(typeLength)
    private var sizes: Array<IntArray> = Array(typeLength) { IntArray(sizeLength) }

    protected fun getSizeIndex(tIndex: Int, size: Int): Int {
        val sIndex = sizes[tIndex].indexOf(size)

        return if (sIndex != -1) {
            sIndex
        } else {
            if (sizeIndices[tIndex] >= storage[0][tIndex].size)
                grow(tIndex)

            val idx = sizeIndices[tIndex]++
            sizes[tIndex][idx] = size
            idx
        }
    }

    private fun grow(typeIndex: Int) {
        val newSize = sizes[typeIndex].size * 2
        for (i in storage.indices) {
            val newStorage: Array<ArrayDeque<Any>> = Array(newSize) { ArrayDeque() }

            for (j in storage[i][typeIndex].indices) {
                newStorage[j] = storage[i][typeIndex][j]
            }

            storage[i][typeIndex] = newStorage
        }

        sizes[typeIndex] = sizes[typeIndex].copyOf(newSize)
    }

    protected fun create(type: DataType, size: Int): Any {
        return when (type) {
            DataType.BYTE -> ByteArray(size)         // 8-bit signed
            DataType.SHORT -> ShortArray(size)       // 16-bit signed
            DataType.INT -> IntArray(size)           // 32-bit signed
            DataType.LONG -> LongArray(size)         // 64-bit signed

            DataType.UBYTE -> UByteArray(size)       // 8-bit unsigned
            DataType.USHORT -> UShortArray(size)     // 16-bit unsigned
            DataType.UINT -> UIntArray(size)         // 32-bit unsigned
            DataType.ULONG -> ULongArray(size)       // 64-bit unsigned

            DataType.FLOAT -> FloatArray(size)
            DataType.DOUBLE -> DoubleArray(size)

            DataType.BOOLEAN -> BooleanArray(size)
            else -> throw IllegalArgumentException("Unsupported array type")
        }
    }

    protected fun resetArray(array: Any): Unit =
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
            else -> error("Unsupported array type")
        }
}

internal class SingleArrayStorage(typeLength: Int, sizeLength: Int, private val limiter: MemoryLimiter) : BaseArrayStorage(typeLength, sizeLength, 1) {
    internal fun getArray(type: DataType, size: Int, fillZeros: Boolean = true): Any {
        return if (limiter.checkMemoryLimitAndAdd(type, size)) {
            val tIndex = type.ordinal
            val sIndex = getSizeIndex(tIndex, size)
            storage[0][tIndex][sIndex].removeFirstOrNull()?.takeIf { fillZeros }?.apply(::resetArray) ?: create(type, size)
        } else {
            create(type, size)
        }
    }

    internal fun returnArrays(type: DataType, size: Int, arrays: Array<Any>) {
        val tIndex = type.ordinal
        val sIndex = getSizeIndex(tIndex, size)
        val queue = storage[0][tIndex][sIndex]

        queue.addAll(arrays)
    }

    internal fun clear() {
        storage[0].forEach { arraysBySize ->
            arraysBySize.forEach { arrayDeque ->
                arrayDeque.clear()
            }
        }
        limiter.resetLimit()
    }
}

internal class ArrayStorage(typeLength: Int, sizeLength: Int, private val limiter: MemoryLimiter) : BaseArrayStorage(typeLength, sizeLength, 2) {
    internal fun getArray(type: DataType, size: Int, fillZeros: Boolean = true): Any {
        return if (limiter.checkMemoryLimitAndAdd(type, size)) {
            val tIndex = type.ordinal
            val sIndex = getSizeIndex(tIndex, size)
            val array = storage[0][tIndex][sIndex].removeFirstOrNull()?.takeIf { fillZeros }?.apply(::resetArray) ?: create(type, size)
            storage[1][tIndex][sIndex].add(array)
            array
        } else {
            create(type, size)
        }
    }

    internal fun moveArrays() {
        storage[1].forEachIndexed { typeIndex, arraysByType ->
            arraysByType.forEachIndexed { sizeIndex, arrayDeque ->
                storage[0][typeIndex][sizeIndex].addAll(arrayDeque)
                arrayDeque.clear()
            }
        }
        limiter.resetLimit()
    }
}
