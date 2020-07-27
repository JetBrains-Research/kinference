package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.extensions.primitives.plus
import org.jetbrains.research.kotlin.inference.extensions.primitives.times
import TensorProto

class IntNDArray(array: IntArray, strides: Strides = Strides.empty()) : NDArray(array, strides, TensorProto.DataType.INT32) {
    override fun clone(newStrides: Strides): IntNDArray {
        return IntNDArray(array as IntArray, newStrides)
    }

    override fun get(i: Int): Int {
        return (array as IntArray)[i]
    }

    override fun get(indices: IntArray): Int {
        return (array as IntArray)[strides.offset(indices)]
    }

    override fun plus(other: NDArray): NDArray {
        other as IntNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> plus(left as IntArray, right as IntArray) }
        }

        val sum = plus(array as IntArray, other.array as IntArray)
        return IntNDArray(sum, strides)
    }

    override fun times(other: NDArray): NDArray {
        other as IntNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> times(left as IntArray, right as IntArray) }
        }

        val sum = times(array as IntArray, other.array as IntArray)
        return IntNDArray(sum, strides)
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as IntArray; block as IntArray
        block.copyInto(array, startOffset)
    }
}
