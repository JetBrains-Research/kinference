package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.LongArrayToLongArray
import org.jetbrains.research.kotlin.inference.extensions.functional.LongArrayWithLongArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
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
            this.combineWith(other, LongArrayWithLongArray { array, otherArray -> plus(array, otherArray, copy) })
        }
    }

    override fun times(other: NDArray<LongArray>, copy: Boolean): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other, LongArrayWithLongArray { array, otherArray -> times(array, otherArray, copy) })
        }
    }

    override fun div(other: NDArray<LongArray>, copy: Boolean): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other, LongArrayWithLongArray { array, otherArray -> div(array, otherArray, copy) })
        }
    }

    override fun minus(other: NDArray<LongArray>, copy: Boolean): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other, LongArrayWithLongArray { array, otherArray -> minus(array, otherArray, copy) })
        }
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as LongArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as LongArray
        block.copyInto(array, startOffset)
    }

    override fun mapElements(func: PrimitiveArrayFunction, copy: Boolean): NDArray<LongArray> {
        func as LongArrayToLongArray
        return if (copy) LongNDArray(map(array, func, copy), strides) else {
            map(array, func, copy); this
        }
    }

    override fun slice(sliceLength: Int, start: Int): LongArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun clean() = array.fill(0)
}
