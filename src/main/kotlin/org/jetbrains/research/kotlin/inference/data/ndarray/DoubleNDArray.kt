package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class DoubleNDArray(array: DoubleArray, strides: Strides = Strides.empty()) : NDArray<DoubleArray>(array, strides, TensorProto.DataType.DOUBLE) {
    init {
        require(array.size == strides.linearSize)
    }

    private companion object {
        val plus = DoubleArrayWithDoubleArray { array, otherArray -> plus(array, otherArray, true) }
        val times = DoubleArrayWithDoubleArray { array, otherArray -> times(array, otherArray, true) }
        val minus = DoubleArrayWithDoubleArray { array, otherArray -> plus(array, otherArray, true) }
        val div = DoubleArrayWithDoubleArray { array, otherArray -> plus(array, otherArray, true) }
        val scalarPlus = DoubleArrayWithDouble { array, value -> plus(array, value, true) }
        val scalarTimes = DoubleArrayWithDouble { array, value -> times(array, value, true) }
        val scalarMinus = DoubleArrayWithDouble { array, value -> minus(array, value, true) }
        val scalarDiv = DoubleArrayWithDouble { array, value -> div(array, value, true) }
    }
    
    override fun clone(): DoubleNDArray {
        return DoubleNDArray(array.copyOf(), strides)
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

    override fun plus(other: TypedNDArray<DoubleArray>): TypedNDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] + other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarPlus)
        } else {
            this.combineWith(other, plus)
        }
    }

    override fun minus(other: TypedNDArray<DoubleArray>): TypedNDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] - other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarMinus)
        } else {
            this.combineWith(other, minus)
        }
    }

    override fun times(other: TypedNDArray<DoubleArray>): TypedNDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] * other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarTimes)
        } else {
            this.combineWith(other, times)
        }
    }

    override fun div(other: TypedNDArray<DoubleArray>): TypedNDArray<DoubleArray> {
        return if (this.isScalar() && other.isScalar()) {
            DoubleNDArray(doubleArrayOf(this.array[0] / other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarDiv)
        } else {
            this.combineWith(other, div)
        }
    }
    
    override fun mapElements(func: PrimitiveArrayFunction): NDArray<DoubleArray> {
        func as DoubleArrayToDoubleArray
        return DoubleNDArray(map(array, func, false), strides)
    }

    override fun slice(sliceLength: Int, start: Int): DoubleArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun toMutable(): MutableTypedNDArray<DoubleArray> {
       return MutableDoubleNDArray(array.copyOf(), strides)
    }
}

class MutableDoubleNDArray(array: DoubleArray, strides: Strides = Strides.empty()) : DoubleNDArray(array, strides), MutableTypedNDArray<DoubleArray> {
    private companion object {
        val plus = DoubleArrayWithDoubleArray { array, otherArray -> plus(array, otherArray, false) }
        val times = DoubleArrayWithDoubleArray { array, otherArray -> times(array, otherArray, false) }
        val minus = DoubleArrayWithDoubleArray { array, otherArray -> plus(array, otherArray, false) }
        val div = DoubleArrayWithDoubleArray { array, otherArray -> plus(array, otherArray, false) }
        val scalarPlus = DoubleArrayWithDouble { array, value -> plus(array, value, false) }
        val scalarTimes = DoubleArrayWithDouble { array, value -> times(array, value, false) }
        val scalarMinus = DoubleArrayWithDouble { array, value -> minus(array, value, false) }
        val scalarDiv = DoubleArrayWithDouble { array, value -> div(array, value, false) }
    }

    override fun clean() = array.fill(0.0)

    override fun clone(): MutableDoubleNDArray {
        return MutableDoubleNDArray(array.copyOf(), strides)
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as DoubleArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as DoubleArray
        block.copyInto(array, startOffset)
    }

    override fun toMutable(): MutableTypedNDArray<DoubleArray> = this

    override fun set(i: Int, value: Any) {
        array[i] = value as Double
    }

    override fun plusAssign(other: TypedNDArray<DoubleArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] += other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarPlus)
        } else {
            this.combineWith(other, plus)
        }
    }

    override fun minusAssign(other: TypedNDArray<DoubleArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] -= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarMinus)
        } else {
            this.combineWith(other, minus)
        }
    }

    override fun timesAssign(other: TypedNDArray<DoubleArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] *= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarTimes)
        } else {
            this.combineWith(other, times)
        }
    }

    override fun divAssign(other: TypedNDArray<DoubleArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] /= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarDiv)
        } else {
            this.combineWith(other, div)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): NDArray<DoubleArray> {
        func as DoubleArrayToDoubleArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<DoubleArray> {
        this.strides = strides
        return this
    }
}
