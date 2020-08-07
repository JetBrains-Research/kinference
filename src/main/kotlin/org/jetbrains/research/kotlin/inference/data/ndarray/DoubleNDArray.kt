package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.DoubleArrayToDoubleArray
import org.jetbrains.research.kotlin.inference.extensions.functional.DoubleArrayWithDoubleArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
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

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitDoubleArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: NDArray<DoubleArray>, copy: Boolean): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other, DoubleArrayWithDoubleArray { array, otherArray -> plus(array, otherArray, copy) })
        }
    }

    override fun minus(other: NDArray<DoubleArray>, copy: Boolean): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other, DoubleArrayWithDoubleArray { array, otherArray -> minus(array, otherArray, copy) })
        }
    }

    override fun times(other: NDArray<DoubleArray>, copy: Boolean): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            return DoubleNDArray(doubleArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other, DoubleArrayWithDoubleArray { array, otherArray -> times(array, otherArray, copy) })
        }
    }

    override fun div(other: NDArray<DoubleArray>, copy: Boolean): NDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            return DoubleNDArray(doubleArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other, DoubleArrayWithDoubleArray { array, otherArray -> div(array, otherArray, copy) })
        }
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as DoubleArray
        block.copyInto(array, startOffset, startIndex, endIndex)
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

    override fun clean() = array.fill(0.0)
}
