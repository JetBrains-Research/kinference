package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.extensions.primitives.plus
import org.jetbrains.research.kotlin.inference.extensions.primitives.times
import TensorProto

class LongNDArray(array: LongArray, strides: Strides = Strides.empty()) : NDArray(array, strides, TensorProto.DataType.INT64) {
    override fun clone(newStrides: Strides): LongNDArray {
        return LongNDArray(array as LongArray, newStrides)
    }

    override fun get(i: Int): Long {
        return (array as LongArray)[i]
    }

    override fun get(indices: IntArray): Long {
        return (array as LongArray)[strides.offset(indices)]
    }

    override fun plus(other: NDArray): NDArray {
        other as LongNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> plus(left as LongArray, right as LongArray) }
        }

        val sum = plus(array as LongArray, other.array as LongArray)
        return LongNDArray(sum, strides)
    }

    override fun times(other: NDArray): NDArray {
        other as LongNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> times(left as LongArray, right as LongArray) }
        }

        val sum = times(array as LongArray, other.array as LongArray)
        return LongNDArray(sum, strides)
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as LongArray; block as LongArray
        block.copyInto(array, startOffset)
    }
}
