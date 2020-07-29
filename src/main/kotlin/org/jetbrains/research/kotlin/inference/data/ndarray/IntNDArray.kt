package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*

class IntNDArray(array: IntArray, strides: Strides = Strides.empty()) : NDArray<IntArray>(array, strides, TensorProto.DataType.INT32) {
    override fun clone(newStrides: Strides): IntNDArray {
        return IntNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Int {
        return array[i]
    }

    override fun get(indices: IntArray): Int {
        return array[strides.offset(indices)]
    }

    override fun plus(other: NDArray<IntArray>): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst, snd) }
        }
    }

    override fun times(other: NDArray<IntArray>): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst, snd) }
        }
    }

    override fun minus(other: NDArray<IntArray>): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst, snd) }
        }
    }

    override fun div(other: NDArray<IntArray>): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> div(fst, snd) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as IntArray
        block.copyInto(array, startOffset)
    }
}
