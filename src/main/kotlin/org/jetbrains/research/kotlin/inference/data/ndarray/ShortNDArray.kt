package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.extensions.primitives.plus
import org.jetbrains.research.kotlin.inference.extensions.primitives.times
import TensorProto

class ShortNDArray(array: ShortArray, strides: Strides = Strides.empty()) : NDArray(array, strides, TensorProto.DataType.INT16) {
    override fun clone(newStrides: Strides): ShortNDArray {
        return ShortNDArray(array as ShortArray, newStrides)
    }

    override fun get(i: Int): Short {
        return (array as ShortArray)[i]
    }

    override fun get(indices: IntArray): Short {
        return (array as ShortArray)[strides.offset(indices)]
    }

    override fun plus(other: NDArray): NDArray {
        other as ShortNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> plus(left as ShortArray, right as ShortArray) }
        }

        val sum = plus(array as ShortArray, other.array as ShortArray)
        return ShortNDArray(sum, strides)
    }

    override fun times(other: NDArray): NDArray {
        other as ShortNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> times(left as ShortArray, right as ShortArray) }
        }

        val sum = times(array as ShortArray, other.array as ShortArray)
        return ShortNDArray(sum, strides)
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as ShortArray; block as ShortArray
        block.copyInto(array, startOffset)
    }
}
