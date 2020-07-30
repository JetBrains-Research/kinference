package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

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

    override fun plus(other: NDArray<FloatArray>, copy: Boolean): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst, snd, copy) }
        }
    }

    override fun times(other: NDArray<FloatArray>, copy: Boolean): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst, snd, copy) }
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

    override fun mapElements(func: (Any) -> Any, copy: Boolean): NDArray<FloatArray> {
        func as (Float) -> Float
        return if (copy) FloatNDArray(map(array, func, copy), strides) else {
            map(array, func, copy); this
        }
    }

    override fun slice(sliceLength: Int, start: Int): FloatArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun clean() {
        for (i in array.indices) array[i] = 0.0f
    }
}
