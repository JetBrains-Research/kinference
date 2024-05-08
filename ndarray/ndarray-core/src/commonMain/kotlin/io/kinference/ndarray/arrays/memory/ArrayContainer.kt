package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*

internal sealed class ArrayContainer(
    val arrayTypeIndex: Int,
    val arraySizeIndex: Int,
    var marker: ArrayUsageMarker = ArrayUsageMarker.Used,
) {
    val markAsOutput: StateMarker = {
        marker = it
    }

    var next: ArrayContainer? = null

    private class EmptyArrayContainer : ArrayContainer(EMPTY_INDEX, EMPTY_INDEX)

    companion object {
        private const val EMPTY_INDEX = -1

        fun emptyContainer(): ArrayContainer = EmptyArrayContainer()

        operator fun invoke(type: ArrayTypes, size: Int, sizeIndex: Int = EMPTY_INDEX): ArrayContainer {
            return when (type) {
                ArrayTypes.ByteArray -> ByteArrayContainer(type.index, sizeIndex, ByteArray(size))         // 8-bit signed
                ArrayTypes.UByteArray -> UByteArrayContainer(type.index, sizeIndex, UByteArray(size))      // 8-bit unsigned
                ArrayTypes.ShortArray -> ShortArrayContainer(type.index, sizeIndex, ShortArray(size))      // 16-bit signed
                ArrayTypes.UShortArray -> UShortArrayContainer(type.index, sizeIndex, UShortArray(size))   // 16-bit unsigned
                ArrayTypes.IntArray -> IntArrayContainer(type.index, sizeIndex, IntArray(size))            // 32-bit signed
                ArrayTypes.UIntArray -> UIntArrayContainer(type.index, sizeIndex, UIntArray(size))         // 32-bit unsigned
                ArrayTypes.LongArray -> LongArrayContainer(type.index, sizeIndex, LongArray(size))         // 64-bit signed
                ArrayTypes.ULongArray -> ULongArrayContainer(type.index, sizeIndex, ULongArray(size))      // 64-bit unsigned
                ArrayTypes.FloatArray -> FloatArrayContainer(type.index, sizeIndex, FloatArray(size))
                ArrayTypes.DoubleArray -> DoubleArrayContainer(type.index, sizeIndex, DoubleArray(size))
                ArrayTypes.BooleanArray -> BooleanArrayContainer(type.index, sizeIndex, BooleanArray(size))
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
