package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*

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

        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this[0] + other[0]).toShort()))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst as ShortArray, snd as ShortArray) }
        }
    }

    override fun times(other: NDArray): NDArray {
        other as ShortNDArray

        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this[0] * other[0]).toShort()))
        } else {
            this.combineWith(other) { fst, snd -> times(fst as ShortArray, snd as ShortArray) }
        }
    }

    override fun div(other: NDArray): NDArray {
        other as ShortNDArray

        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this[0] / other[0]).toShort()))
        } else {
            this.combineWith(other) { fst, snd -> div(fst as ShortArray, snd as ShortArray) }
        }
    }

    override fun minus(other: NDArray): NDArray {
        other as ShortNDArray

        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this[0] - other[0]).toShort()))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst as ShortArray, snd as ShortArray) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any) {
        array as ShortArray; block as ShortArray
        block.copyInto(array, startOffset)
    }
}
