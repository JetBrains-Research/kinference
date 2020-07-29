package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

class ShortNDArray(array: ShortArray, strides: Strides = Strides.empty()) : NDArray<ShortArray>(array, strides, TensorProto.DataType.INT16) {
    override fun clone(newStrides: Strides): ShortNDArray {
        return ShortNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Short {
        return array[i]
    }

    override fun get(indices: IntArray): Short {
        return array[strides.offset(indices)]
    }

    override fun plus(other: NDArray<ShortArray>): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] + other.array[0]).toShort()))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst, snd) }
        }
    }

    override fun times(other: NDArray<ShortArray>): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] * other.array[0]).toShort()))
        } else {
            this.combineWith(other) { fst, snd -> times(fst, snd) }
        }
    }

    override fun div(other: NDArray<ShortArray>): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] / other.array[0]).toShort()))
        } else {
            this.combineWith(other) { fst, snd -> div(fst, snd) }
        }
    }

    override fun minus(other: NDArray<ShortArray>): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] - other.array[0]).toShort()))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst, snd) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as ShortArray
        block.copyInto(array, startOffset)
    }
}
