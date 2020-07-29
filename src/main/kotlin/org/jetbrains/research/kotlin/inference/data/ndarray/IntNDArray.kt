package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*

class IntNDArray(array: IntArray, strides: Strides = Strides.empty()) : NDArray(array, strides, TensorProto.DataType.INT32) {
    override fun clone(newStrides: Strides): IntNDArray {
        return IntNDArray((array as IntArray).copyOf(), newStrides)
    }

    override fun get(i: Int): Int {
        return (array as IntArray)[i]
    }

    override fun get(indices: IntArray): Int {
        return (array as IntArray)[strides.offset(indices)]
    }

    override fun plus(other: NDArray): NDArray {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this[0] + other[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst as IntArray, snd as IntArray) }
        }
    }

    override fun times(other: NDArray): NDArray {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this[0] * other[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst as IntArray, snd as IntArray) }
        }
    }

    override fun minus(other: NDArray): NDArray {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this[0] - other[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst as IntArray, snd as IntArray) }
        }
    }

    override fun div(other: NDArray): NDArray {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this[0] / other[0]))
        } else {
            this.combineWith(other) { fst, snd -> div(fst as IntArray, snd as IntArray) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as IntArray; block as IntArray
        block.copyInto(array, startOffset)
    }
}
