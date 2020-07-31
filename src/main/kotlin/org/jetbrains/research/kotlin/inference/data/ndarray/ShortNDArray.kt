package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
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

    override fun plus(other: NDArray<ShortArray>, copy: Boolean): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] + other.array[0]).toShort()))
        } else {
            this.combineWith(other, object : ShortArrayWithShortArray {
                override fun apply(array: ShortArray, otherArray: ShortArray): ShortArray {
                    return plus(array, otherArray, copy)
                }
            })
        }
    }

    override fun times(other: NDArray<ShortArray>, copy: Boolean): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] * other.array[0]).toShort()))
        } else {
            this.combineWith(other, object : ShortArrayWithShortArray {
                override fun apply(array: ShortArray, otherArray: ShortArray): ShortArray {
                    return times(array, otherArray, copy)
                }
            })
        }
    }

    override fun div(other: NDArray<ShortArray>): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] / other.array[0]).toShort()))
        } else {
            this.combineWith(other, object : ShortArrayWithShortArray {
                override fun apply(array: ShortArray, otherArray: ShortArray): ShortArray {
                    return div(array, otherArray)
                }
            })
        }
    }

    override fun minus(other: NDArray<ShortArray>): NDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] - other.array[0]).toShort()))
        } else {
            this.combineWith(other, object : ShortArrayWithShortArray {
                override fun apply(array: ShortArray, otherArray: ShortArray): ShortArray {
                    return minus(array, otherArray)
                }
            })
        }
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

    override fun clean() {
        for (i in array.indices) array[i] = 0
    }
}
