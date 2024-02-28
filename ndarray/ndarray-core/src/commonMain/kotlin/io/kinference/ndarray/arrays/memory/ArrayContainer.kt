package io.kinference.ndarray.arrays.memory

import io.kinference.ndarray.arrays.*
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
