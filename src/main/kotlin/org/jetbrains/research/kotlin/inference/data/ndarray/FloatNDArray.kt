package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*

class FloatNDArray(array: FloatArray, strides: Strides = Strides.empty()) : NDArray(array, strides, TensorProto.DataType.FLOAT) {
    override fun clone(newStrides: Strides): FloatNDArray {
        return FloatNDArray((array as FloatArray).copyOf(), newStrides)
    }

    override fun get(i: Int): Float {
        return (array as FloatArray)[i]
    }

    override fun get(vararg indices: Int): Float {
        return (array as FloatArray)[strides.offset(indices)]
    }

    override fun plus(other: NDArray): NDArray {
        other as FloatNDArray

        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this[0] + other[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst as FloatArray, snd as FloatArray) }
        }
    }

    override fun times(other: NDArray): NDArray {
        other as FloatNDArray

        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this[0] * other[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst as FloatArray, snd as FloatArray) }
        }
    }

    override fun minus(other: NDArray): NDArray {
        other as FloatNDArray

        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this[0] - other[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst as FloatArray, snd as FloatArray) }
        }
    }

    override fun div(other: NDArray): NDArray {
        other as FloatNDArray

        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this[0] / other[0]))
        } else {
            this.combineWith(other) { fst, snd -> div(fst as FloatArray, snd as FloatArray) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as FloatArray; block as FloatArray
        block.copyInto(array, startOffset)
    }
}
