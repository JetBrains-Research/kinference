package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.extensions.primitives.plus
import org.jetbrains.research.kotlin.inference.extensions.primitives.times
import TensorProto

class DoubleNDArray(array: DoubleArray, strides: Strides = Strides.empty()) : NDArray(array, strides, TensorProto.DataType.DOUBLE) {
    override fun clone(newStrides: Strides): DoubleNDArray {
        return DoubleNDArray(array as DoubleArray, newStrides)
    }

    override fun get(i: Int): Double {
        return (array as DoubleArray)[i]
    }

    override fun get(indices: IntArray): Double {
        return (array as DoubleArray)[strides.offset(indices)]
    }

    override fun plus(other: NDArray): NDArray {
        other as DoubleNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> plus(left as DoubleArray, right as DoubleArray) }
        }

        val sum = plus(array as DoubleArray, other.array as DoubleArray)
        return DoubleNDArray(sum, strides)
    }

    override fun times(other: NDArray): NDArray {
        other as DoubleNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> times(left as DoubleArray, right as DoubleArray) }
        }

        val sum = times(array as DoubleArray, other.array as DoubleArray)
        return DoubleNDArray(sum, strides)
    }


    override fun placeAll(startOffset: Int, block: Any) {
        array as DoubleArray; block as DoubleArray
        block.copyInto(array, startOffset)
    }
}
