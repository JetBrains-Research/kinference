package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*

class FloatNDArray(array: FloatArray, strides: Strides = Strides.empty()) : NDArray<FloatArray>(array, strides, TensorProto.DataType.FLOAT) {
    override fun clone(newStrides: Strides): FloatNDArray {
        return FloatNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Float {
        return array[i]
    }

    override fun get(vararg indices: Int): Float {
        return array[strides.offset(indices)]
    }

    override fun plus(other: NDArray<FloatArray>): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst, snd) }
        }
    }

    override fun times(other: NDArray<FloatArray>): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst, snd) }
        }
    }

    override fun minus(other: NDArray<FloatArray>): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst, snd) }
        }
    }

    override fun div(other: NDArray<FloatArray>): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> div(fst, snd) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as FloatArray
        block.copyInto(array, startOffset)
    }
}
