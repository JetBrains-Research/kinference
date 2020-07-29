package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*

class DoubleNDArray(array: DoubleArray, strides: Strides = Strides.empty()) : NDArray<DoubleArray>(array, strides, TensorProto.DataType.DOUBLE) {
    override fun clone(newStrides: Strides): DoubleNDArray {
        return DoubleNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Double {
        return array[i]
    }

    override fun get(indices: IntArray): Double {
        return array[strides.offset(indices)]
    }

    override fun plus(other: NDArray<DoubleArray>): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst, snd) }
        }
    }

    override fun minus(other: NDArray<DoubleArray>): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst, snd) }
        }
    }

    override fun times(other: NDArray<DoubleArray>): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            return DoubleNDArray(doubleArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst, snd) }
        }
    }

    override fun div(other: NDArray<DoubleArray>): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            return DoubleNDArray(doubleArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> div(fst, snd) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as DoubleArray
        block.copyInto(array, startOffset)
    }
}
