package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

class LongNDArray(array: LongArray, strides: Strides = Strides.empty()) : NDArray<LongArray>(array, strides, TensorProto.DataType.INT64) {
    override fun clone(newStrides: Strides): LongNDArray {
        return LongNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Long {
        return array[i]
    }

    override fun get(indices: IntArray): Long {
        return array[strides.offset(indices)]
    }

    override fun plus(other: NDArray<LongArray>, copy: Boolean): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst, snd, copy) }
        }
    }

    override fun times(other: NDArray<LongArray>, copy: Boolean): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst, snd, copy) }
        }
    }

    override fun div(other: NDArray<LongArray>): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> div(fst, snd) }
        }
    }

    override fun minus(other: NDArray<LongArray>): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst, snd) }
        }
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as LongArray
        block.copyInto(array, startOffset)
    }

    override fun mapElements(func: (Any) -> Any, copy: Boolean): NDArray<LongArray> {
        func as (Long) -> Long
        return if (copy) LongNDArray(map(array, func, copy), strides) else {
            map(array, func, copy); this
        }
    }

    override fun slice(sliceLength: Int, start: Int): LongArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun clean() {
        for (i in array.indices) array[i] = 0
    }
}
