package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class IntNDArray(array: IntArray, strides: Strides = Strides.empty()) : NDArray<IntArray>(array, strides, TensorProto.DataType.INT32) {
    init {
        require(array.size == strides.linearSize)
    }

    private companion object {
        val plus = IntArrayWithIntArray { array, otherArray -> plus(array, otherArray, true) }
        val times = IntArrayWithIntArray { array, otherArray -> times(array, otherArray, true) }
        val minus = IntArrayWithIntArray { array, otherArray -> plus(array, otherArray, true) }
        val div = IntArrayWithIntArray { array, otherArray -> plus(array, otherArray, true) }
        val scalarPlus = IntArrayWithInt { array, value -> plus(array, value, true) }
        val scalarTimes = IntArrayWithInt { array, value -> times(array, value, true) }
        val scalarMinus = IntArrayWithInt { array, value -> minus(array, value, true) }
        val scalarDiv = IntArrayWithInt { array, value -> div(array, value, true) }
    }
    
    override fun clone(): IntNDArray {
        return IntNDArray(array.copyOf(), strides)
    }

    override fun get(i: Int): Int {
        return array[i]
    }

    override fun get(indices: IntArray): Int {
        return array[strides.offset(indices)]
    }

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitIntArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: TypedNDArray<IntArray>): TypedNDArray<IntArray> {
        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] + other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarPlus)
        } else {
            this.combineWith(other, plus)
        }
    }

    override fun times(other: TypedNDArray<IntArray>): TypedNDArray<IntArray> {
        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] * other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarTimes)
        } else {
            this.combineWith(other, times)
        }
    }

    override fun minus(other: TypedNDArray<IntArray>): TypedNDArray<IntArray> {
        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] - other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarMinus)
        } else {
            this.combineWith(other, minus)
        }
    }

    override fun div(other: TypedNDArray<IntArray>): TypedNDArray<IntArray> {
        return if (this.isScalar() && other.isScalar()) {
            IntNDArray(intArrayOf(this.array[0] / other.array[0]))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarDiv)
        } else {
            this.combineWith(other, div)
        }
    }
    
    override fun mapElements(func: PrimitiveArrayFunction): NDArray<IntArray> {
        func as IntArrayToIntArray
        return IntNDArray(map(array, func, true), strides)
    }

    override fun slice(sliceLength: Int, start: Int): IntArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun toMutable(): MutableTypedNDArray<IntArray> {
        return MutableIntNDArray(array.copyOf(), strides)
    }
}

class MutableIntNDArray(array: IntArray, strides: Strides = Strides.empty()) : IntNDArray(array, strides), MutableTypedNDArray<IntArray> {
    private companion object {
        val plus = IntArrayWithIntArray { array, otherArray -> plus(array, otherArray, false) }
        val times = IntArrayWithIntArray { array, otherArray -> times(array, otherArray, false) }
        val minus = IntArrayWithIntArray { array, otherArray -> plus(array, otherArray, false) }
        val div = IntArrayWithIntArray { array, otherArray -> plus(array, otherArray, false) }
        val scalarPlus = IntArrayWithInt { array, value -> plus(array, value, false) }
        val scalarTimes = IntArrayWithInt { array, value -> times(array, value, false) }
        val scalarMinus = IntArrayWithInt { array, value -> minus(array, value, false) }
        val scalarDiv = IntArrayWithInt { array, value -> div(array, value, false) }
    }

    override fun clean() = array.fill(0)

    override fun clone(): MutableIntNDArray {
        return MutableIntNDArray(array.copyOf(), strides)
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as IntArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as IntArray
        block.copyInto(array, startOffset)
    }

    override fun toMutable(): MutableTypedNDArray<IntArray> = this

    override fun set(i: Int, value: Any) {
        array[i] = value as Int
    }

    override fun plusAssign(other: TypedNDArray<IntArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] += other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarPlus)
        } else {
            this.combineWith(other, plus)
        }
    }

    override fun minusAssign(other: TypedNDArray<IntArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] -= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarMinus)
        } else {
            this.combineWith(other, minus)
        }
    }

    override fun timesAssign(other: TypedNDArray<IntArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] *= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarTimes)
        } else {
            this.combineWith(other, times)
        }
    }

    override fun divAssign(other: TypedNDArray<IntArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] /= other.array[0]
        } else if (other.isScalar()) {
            this.combineWith(other, scalarDiv)
        } else {
            this.combineWith(other, div)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): NDArray<IntArray> {
        func as IntArrayToIntArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<IntArray> {
        this.strides = strides
        return this
    }
}
