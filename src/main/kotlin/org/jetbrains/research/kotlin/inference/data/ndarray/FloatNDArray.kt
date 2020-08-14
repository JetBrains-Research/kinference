package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.FloatArrayToFloatArray
import org.jetbrains.research.kotlin.inference.extensions.functional.FloatArrayWithFloatArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

class FloatNDArray(array: FloatArray, strides: Strides = Strides.empty(), offset: Int = 0) : NDArray<FloatArray>(array, strides, TensorProto.DataType.FLOAT, offset) {
    /*init {
        require(array.size == strides.linearSize)
    }*/

    /*private companion object {
        val plusWithCopy = FloatArrayWithFloatArray { array, otherArray -> plus(array, otherArray, true) }
        val plusWithoutCopy = FloatArrayWithFloatArray { array, otherArray -> plus(array, otherArray, false) }
        val timesWithCopy = FloatArrayWithFloatArray { array, otherArray -> times(array, otherArray, true) }
        val timesWithoutCopy = FloatArrayWithFloatArray { array, otherArray -> times(array, otherArray, false) }
    }*/

    override fun clone(newStrides: Strides): FloatNDArray {
        return FloatNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Float {
        return array[i]
    }

    override fun get(vararg indices: Int): Float {
        return array[strides.offset(indices)]
    }

    override fun set(i: Int, value: Any) {
        array[i] = value as Float
    }

    // TODO check if step == 1 and use Arrays.copy
    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitFloatArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: NDArray<FloatArray>, destination: NDArray<FloatArray>?): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] + other.array[0]))
        } else {
            //this.combineWith(other, if (copy) plusWithCopy else plusWithoutCopy)
            this.combineWith(other, destination,
                FloatArrayWithFloatArray { array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size ->
                    plus(array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size)
                })
        }
    }

    override fun times(other: NDArray<FloatArray>, destination: NDArray<FloatArray>?): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] * other.array[0]))
        } else {
            //this.combineWith(other, if (copy) timesWithCopy else timesWithoutCopy)
            this.combineWith(other, destination,
                FloatArrayWithFloatArray { array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size ->
                    times(array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size)
                })
        }
    }

    override fun minus(other: NDArray<FloatArray>, destination: NDArray<FloatArray>?): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] - other.array[0]))
        } else {
            //this.combineWith(other, FloatArrayWithFloatArray { array, otherArray -> minus(array, otherArray, copy) })
            this.combineWith(other, destination,
                FloatArrayWithFloatArray { array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size ->
                    minus(array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size)
                })
        }
    }

    override fun div(other: NDArray<FloatArray>, destination: NDArray<FloatArray>?): NDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] / other.array[0]))
        } else {
            this.combineWith(other, destination,
                FloatArrayWithFloatArray { array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size ->
                    div(array, arrayOffset, otherArray, otherArrayOffset, destinationArray, destinationArrayOffset, size)
                })
        }
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as FloatArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as FloatArray
        block.copyInto(array, startOffset)
    }

    override fun mapElements(func: PrimitiveArrayFunction, copy: Boolean): NDArray<FloatArray> {
        func as FloatArrayToFloatArray
        return if (copy) FloatNDArray(map(array, func, copy), strides) else {
            map(array, func, copy); this
        }
    }

    override fun slice(sliceLength: Int, start: Int): FloatArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun clean() = array.fill(0.0f)
}
