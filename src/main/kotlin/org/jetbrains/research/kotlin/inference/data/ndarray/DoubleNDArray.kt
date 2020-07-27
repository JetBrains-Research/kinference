package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.minus
import org.jetbrains.research.kotlin.inference.extensions.primitives.times

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

        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this[0] + other[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst as DoubleArray, snd as DoubleArray) }
        }
    }

    override fun minus(other: NDArray): NDArray {
        other as DoubleNDArray

        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this[0] - other[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst as DoubleArray, snd as DoubleArray) }
        }
    }

    override fun times(other: NDArray): NDArray {
        other as DoubleNDArray

        return if (this.isScalar() && other.isScalar()) {
            return DoubleNDArray(doubleArrayOf(this[0] * other[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst as DoubleArray, snd as DoubleArray) }
        }
    }

    override fun div(other: NDArray): NDArray {
        other as DoubleNDArray

        return if (this.isScalar() && other.isScalar()) {
            return DoubleNDArray(doubleArrayOf(this[0] / other[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst as DoubleArray, snd as DoubleArray) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as DoubleArray; block as DoubleArray
        block.copyInto(array, startOffset)
    }
}
