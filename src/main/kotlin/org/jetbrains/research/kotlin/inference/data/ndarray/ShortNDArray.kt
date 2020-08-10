package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.functional.ShortArrayToShortArray
import org.jetbrains.research.kotlin.inference.extensions.functional.ShortArrayWithShortArray
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

    override fun set(i: Int, value: Any) {
        array[i] = value as Short
    }

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitShortArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: NDArray<ShortArray>, copy: Boolean): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] + other.array[0]).toShort()))
        } else {
            this.combineWith(other, ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, copy) })
        }
    }

    override fun times(other: NDArray<ShortArray>, copy: Boolean): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] * other.array[0]).toShort()))
        } else {
            this.combineWith(other, ShortArrayWithShortArray { array, otherArray -> times(array, otherArray, copy) })
        }
    }

    override fun div(other: NDArray<ShortArray>, copy: Boolean): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] / other.array[0]).toShort()))
        } else if (other.isScalar()) {
            ShortNDArray(div(this.array, other.array[0], copy), this.strides)
        } else {
            this.combineWith(other, ShortArrayWithShortArray { array, otherArray -> div(array, otherArray, copy) })
        }
    }

    override fun minus(other: NDArray<ShortArray>, copy: Boolean): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] - other.array[0]).toShort()))
        } else if (other.isScalar()) {
            ShortNDArray(minus(this.array, other.array[0], copy), this.strides)
        } else {
            this.combineWith(other, ShortArrayWithShortArray { array, otherArray -> minus(array, otherArray, copy) })
        }
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as ShortArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as ShortArray
        block.copyInto(array, startOffset)
    }

    override fun mapElements(func: PrimitiveArrayFunction, copy: Boolean): NDArray<ShortArray> {
        func as ShortArrayToShortArray
        return if (copy) ShortNDArray(map(array, func, copy), strides) else {
            map(array, func, copy); this
        }
    }

    override fun slice(sliceLength: Int, start: Int): ShortArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun clean() = array.fill(0)
}
