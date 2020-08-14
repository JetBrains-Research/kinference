package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.LongArrayToLongArray
import org.jetbrains.research.kotlin.inference.extensions.functional.LongArrayWithLongArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

class LongNDArray(array: LongArray, strides: Strides = Strides.empty(), offset: Int = 0) : NDArray<LongArray>(array, strides, TensorProto.DataType.INT64, offset) {
    override fun clone(newStrides: Strides): LongNDArray {
        return LongNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Long {
        return array[i]
    }

    override fun get(indices: IntArray): Long {
        return array[strides.offset(indices)]
    }

    override fun set(i: Int, value: Any) {
        array[i] = value as Long
    }

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitLongArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: NDArray<LongArray>, destination: NDArray<LongArray>?): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] + other.array[0]))
        } else {
            this.combineWith(other, destination,
                LongArrayWithLongArray { array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size ->
                    plus(array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size)
                })
        }
    }

    override fun times(other: NDArray<LongArray>, destination: NDArray<LongArray>?): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other, destination,
                LongArrayWithLongArray { array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size ->
                    times(array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size)
                })
        }
    }

    override fun div(other: NDArray<LongArray>, destination: NDArray<LongArray>?): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other, destination,
                LongArrayWithLongArray { array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size ->
                    div(array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size)
                })
        }
    }

    override fun minus(other: NDArray<LongArray>, destination: NDArray<LongArray>?): NDArray<LongArray> {
        return if (this.isScalar() && other.isScalar()) {
            LongNDArray(longArrayOf(this.array[0] - other.array[0]))
        } else {
            //this.combineWith(other, FloatArrayWithFloatArray { array, otherArray -> minus(array, otherArray, copy) })
            this.combineWith(other, destination,
                LongArrayWithLongArray { array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size ->
                    minus(array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size)
                })
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
