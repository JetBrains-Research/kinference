package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*

sealed class ArrayContainer(
    val arrayTypeIndex: Int,
    val arraySizeIndex: Int,
    val sizeBytes: Int
) {
    var isOutput: Boolean = false
        private set

    var isNewlyCreated: Boolean = true
        private set

    val markAsOutput = {
        isOutput = true
    }

    companion object {
        private const val EMPTY_INDEX = -1

        operator fun invoke(type: ArrayTypes, size: Int, sizeIndex: Int = EMPTY_INDEX): ArrayContainer {
            val sizeBytes: Int = type.size * size
            return when (type) {
                ArrayTypes.ByteArray -> ByteArrayContainer(type.index, sizeIndex, sizeBytes, ByteArray(size))         // 8-bit signed
                ArrayTypes.UByteArray -> UByteArrayContainer(type.index, sizeIndex, sizeBytes, UByteArray(size))      // 8-bit unsigned
                ArrayTypes.ShortArray -> ShortArrayContainer(type.index, sizeIndex, sizeBytes, ShortArray(size))      // 16-bit signed
                ArrayTypes.UShortArray -> UShortArrayContainer(type.index, sizeIndex, sizeBytes, UShortArray(size))   // 16-bit unsigned
                ArrayTypes.IntArray -> IntArrayContainer(type.index, sizeIndex, sizeBytes, IntArray(size))            // 32-bit signed
                ArrayTypes.UIntArray -> UIntArrayContainer(type.index, sizeIndex, sizeBytes, UIntArray(size))         // 32-bit unsigned
                ArrayTypes.LongArray -> LongArrayContainer(type.index, sizeIndex, sizeBytes, LongArray(size))         // 64-bit signed
                ArrayTypes.ULongArray -> ULongArrayContainer(type.index, sizeIndex, sizeBytes, ULongArray(size))      // 64-bit unsigned
                ArrayTypes.FloatArray -> FloatArrayContainer(type.index, sizeIndex, sizeBytes, FloatArray(size))
                ArrayTypes.DoubleArray -> DoubleArrayContainer(type.index, sizeIndex, sizeBytes, DoubleArray(size))
                ArrayTypes.BooleanArray -> BooleanArrayContainer(type.index, sizeIndex, sizeBytes, BooleanArray(size))
                else -> throw IllegalArgumentException("Unsupported array type")
            }
        }

        fun resetArray(arrayContainer: ArrayContainer) {
            arrayContainer.isNewlyCreated = false
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
