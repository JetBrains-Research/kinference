package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class FloatNDArray(array: FloatArray, strides: Strides = Strides.empty()) : NDArray<FloatArray>(array, strides, TensorProto.DataType.FLOAT) {
    init {
        require(array.size == strides.linearSize)
    }

    private companion object {
        val plus = FloatArrayWithFloatArray { array, otherArray -> plus(array, otherArray, true) }
        val times = FloatArrayWithFloatArray { array, otherArray -> times(array, otherArray, true) }
        val minus = FloatArrayWithFloatArray { array, otherArray -> plus(array, otherArray, true) }
        val div = FloatArrayWithFloatArray { array, otherArray -> plus(array, otherArray, true) }
        val scalarPlus = FloatArrayWithFloat { array, value -> plus(array, value, true) }
        val scalarTimes = FloatArrayWithFloat { array, value -> times(array, value, true) }
        val scalarMinus = FloatArrayWithFloat { array, value -> minus(array, value, true) }
        val scalarDiv = FloatArrayWithFloat { array, value -> div(array, value, true) }
    }

    override fun clone(): TypedNDArray<FloatArray> {
        return FloatNDArray(array.copyOf(), strides)
    }

    override fun get(i: Int): Float {
        return array[i]
    }

    override fun get(vararg indices: Int): Float {
        return array[strides.offset(indices)]
    }

    // TODO check if step == 1 and use Arrays.copy
    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitFloatArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: TypedNDArray<FloatArray>): TypedNDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] + other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarPlus)
        } else {
            this.combineWith(other, plus)
        }
    }

    override fun times(other: TypedNDArray<FloatArray>): TypedNDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] * other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarTimes)
        } else {
            this.combineWith(other, times)
        }
    }

    override fun minus(other: TypedNDArray<FloatArray>): TypedNDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] - other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarMinus)
        } else {
            this.combineWith(other, minus)
        }
    }

    override fun div(other: TypedNDArray<FloatArray>): TypedNDArray<FloatArray> {
        return if (this.isScalar() && other.isScalar()) {
            FloatNDArray(floatArrayOf(this.array[0] / other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarDiv)
        } else {
            this.combineWith(other, div)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): NDArray<FloatArray> {
        func as FloatArrayToFloatArray
        return FloatNDArray(map(array, func, true), strides)
    }

    override fun slice(sliceLength: Int, start: Int): FloatArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun toMutable(): MutableTypedNDArray<FloatArray> {
        return MutableFloatNDArray(array, strides)
    }
}

class MutableFloatNDArray(array: FloatArray, strides: Strides = Strides.empty()) : FloatNDArray(array, strides), MutableTypedNDArray<FloatArray> {
    private companion object {
        val plus = FloatArrayWithFloatArray { array, otherArray -> plus(array, otherArray, false) }
        val times = FloatArrayWithFloatArray { array, otherArray -> times(array, otherArray, false) }
        val minus = FloatArrayWithFloatArray { array, otherArray -> plus(array, otherArray, false) }
        val div = FloatArrayWithFloatArray { array, otherArray -> plus(array, otherArray, false) }
        val scalarPlus = FloatArrayWithFloat { array, value -> plus(array, value, false) }
        val scalarTimes = FloatArrayWithFloat { array, value -> times(array, value, false) }
        val scalarMinus = FloatArrayWithFloat { array, value -> minus(array, value, false) }
        val scalarDiv = FloatArrayWithFloat { array, value -> div(array, value, false) }
    }

    override fun clean() = array.fill(0.0f)

    override fun clone(): MutableFloatNDArray {
        return MutableFloatNDArray(array.copyOf(), strides)
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as FloatArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as FloatArray
        block.copyInto(array, startOffset)
    }

    override fun toMutable(): MutableTypedNDArray<FloatArray> = this

    override fun set(i: Int, value: Any) {
        array[i] = value as Float
    }

    override fun plusAssign(other: TypedNDArray<FloatArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] += other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarPlus)
        } else {
            this.combineWith(other, plus)
        }
    }

    override fun minusAssign(other: TypedNDArray<FloatArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] -= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarMinus)
        } else {
            this.combineWith(other, minus)
        }
    }

    override fun timesAssign(other: TypedNDArray<FloatArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] *= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarTimes)
        } else {
            this.combineWith(other, times)
        }
    }

    override fun divAssign(other: TypedNDArray<FloatArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] /= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarDiv)
        } else {
            this.combineWith(other, div)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): NDArray<FloatArray> {
        func as FloatArrayToFloatArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<FloatArray> {
        this.strides = strides
        return this
    }
}
