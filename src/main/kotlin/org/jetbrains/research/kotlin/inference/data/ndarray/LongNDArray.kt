package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*

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

        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this[0] + other[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst as LongArray, snd as LongArray) }
        }
    }

    override fun times(other: NDArray): NDArray {
        other as LongNDArray

        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this[0] * other[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst as LongArray, snd as LongArray) }
        }
    }

    override fun div(other: NDArray): NDArray {
        other as LongNDArray

        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this[0] / other[0]))
        } else {
            this.combineWith(other) { fst, snd -> div(fst as LongArray, snd as LongArray) }
        }
    }

    override fun minus(other: NDArray): NDArray {
        other as LongNDArray

        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this[0] - other[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst as LongArray, snd as LongArray) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as LongArray; block as LongArray
        block.copyInto(array, startOffset)
    }
}
