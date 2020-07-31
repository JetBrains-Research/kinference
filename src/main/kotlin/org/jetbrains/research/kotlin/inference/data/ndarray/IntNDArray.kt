package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
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
            this.combineWith(other, object : IntArrayWithIntArray {
                override fun apply(array: IntArray, otherArray: IntArray): IntArray {
                    return plus(array, otherArray, copy)
                }
            })
        }
    }

    override fun times(other: NDArray<IntArray>, copy: Boolean): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] * other.array[0]))
        } else {
            this.combineWith(other, object : IntArrayWithIntArray {
                override fun apply(array: IntArray, otherArray: IntArray): IntArray {
                    return times(array, otherArray, copy)
                }
            })
        }
    }

    override fun minus(other: NDArray<IntArray>, copy: Boolean): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] - other.array[0]))
        } else {
            this.combineWith(other, object : IntArrayWithIntArray {
                override fun apply(array: IntArray, otherArray: IntArray): IntArray {
                    return minus(array, otherArray, copy)
                }
            })
        }
    }

    override fun div(other: NDArray<IntArray>, copy: Boolean): NDArray<IntArray> {
        other as IntNDArray

        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other, object : IntArrayWithIntArray {
                override fun apply(array: IntArray, otherArray: IntArray): IntArray {
                    return div(array, otherArray, copy)
                }
            })
        }
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

    override fun clean() {
        for (i in array.indices) array[i] = 0
    }
}
