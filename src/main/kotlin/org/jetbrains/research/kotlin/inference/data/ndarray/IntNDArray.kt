package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.IntArrayToIntArray
import org.jetbrains.research.kotlin.inference.extensions.functional.IntArrayWithIntArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

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

    override fun plus(other: NDArray<IntArray>, copy: Boolean): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other, IntArrayWithIntArray { array, otherArray -> plus(array, otherArray, copy) })
        }
    }

    override fun times(other: NDArray<IntArray>, copy: Boolean): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other, IntArrayWithIntArray { array, otherArray -> times(array, otherArray, copy) })
        }
    }

    override fun minus(other: NDArray<IntArray>, copy: Boolean): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other, IntArrayWithIntArray { array, otherArray -> minus(array, otherArray, copy) })
        }
    }

    override fun div(other: NDArray<IntArray>, copy: Boolean): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other, IntArrayWithIntArray { array, otherArray -> div(array, otherArray, copy) })
        }
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as IntArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as IntArray
        block.copyInto(array, startOffset)
    }

    override fun mapElements(func: PrimitiveArrayFunction, copy: Boolean): NDArray<IntArray> {
        func as IntArrayToIntArray
        return if (copy) IntNDArray(map(array, func, copy), strides) else {
            map(array, func, copy); this
        }
    }

    override fun slice(sliceLength: Int, start: Int): IntArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun clean() = array.fill(0)
}
