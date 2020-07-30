package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

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

    override fun plus(other: NDArray<DoubleArray>, copy: Boolean): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> plus(fst, snd, copy) }
        }
    }

    override fun minus(other: NDArray<DoubleArray>): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> minus(fst, snd) }
        }
    }

    override fun times(other: NDArray<DoubleArray>, copy: Boolean): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            return DoubleNDArray(doubleArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other) { fst, snd -> times(fst, snd, copy) }
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

    override fun mapElements(func: PrimitiveArrayFunction, copy: Boolean): NDArray<DoubleArray> {
        func as DoubleArrayToDoubleArray
        return if (copy) DoubleNDArray(map(array, func, copy), strides) else {
            map(array, func, copy); this
        }
    }

    override fun slice(sliceLength: Int, start: Int): DoubleArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun clean() {
        for (i in array.indices) array[i] = 0.0
    }
}
