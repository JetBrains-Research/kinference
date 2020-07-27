package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.extensions.primitives.plus
import org.jetbrains.research.kotlin.inference.extensions.primitives.times
import TensorProto

class FloatNDArray(array: FloatArray, strides: Strides = Strides.empty()) : NDArray(array, strides, TensorProto.DataType.FLOAT) {
    override fun clone(newStrides: Strides): FloatNDArray {
        return FloatNDArray(array as FloatArray, newStrides)
    }

    override fun get(i: Int): Float {
        return (array as FloatArray)[i]
    }

    override fun get(vararg indices: Int): Float {
        return (array as FloatArray)[strides.offset(indices)]
    }

    override fun plus(other: NDArray): NDArray {
        other as FloatNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> plus(left as FloatArray, right as FloatArray) }
        }

        val sum = plus(array as FloatArray, other.array as FloatArray)
        return FloatNDArray(sum, strides)
    }

    override fun times(other: NDArray): NDArray {
        other as FloatNDArray

        if (!shape.contentEquals(other.shape)) {
            return applyWithBroadcast(other) { left, right -> times(left as FloatArray, right as FloatArray) }
        }

        val sum = times(array as FloatArray, other.array as FloatArray)
        return FloatNDArray(sum, strides)
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as FloatArray; block as FloatArray
        block.copyInto(array, startOffset)
    }
}
