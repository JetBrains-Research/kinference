package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class LongNDArray(array: LongArray, strides: Strides = Strides.empty()) : NDArray<LongArray>(array, strides, TensorProto.DataType.INT64) {
    init {
        require(array.size == strides.linearSize)
    }

    private companion object {
        val plus = LongArrayWithLongArray { array, otherArray -> plus(array, otherArray, true) }
        val times = LongArrayWithLongArray { array, otherArray -> times(array, otherArray, true) }
        val minus = LongArrayWithLongArray { array, otherArray -> minus(array, otherArray, true) }
        val div = LongArrayWithLongArray { array, otherArray -> div(array, otherArray, true) }
        val scalarPlus = LongArrayWithLong { array, value -> plus(array, value, true) }
        val scalarTimes = LongArrayWithLong { array, value -> times(array, value, true) }
        val scalarMinus = LongArrayWithLong { array, value -> minus(array, value, true) }
        val scalarDiv = LongArrayWithLong { array, value -> div(array, value, true) }
    }
    
    override fun clone(): TypedNDArray<LongArray> {
        return LongNDArray(array.copyOf(), strides)
    }

    override fun get(i: Int): Long {
        return array[i]
    }

    override fun get(indices: IntArray): Long {
        return array[strides.offset(indices)]
    }

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitLongArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: TypedNDArray<LongArray>): TypedNDArray<LongArray> {
        return when {
            this.isScalar() && other.isScalar() -> LongNDArray(longArrayOf(this.array[0] + other.array[0]))
            this.isScalar() || other.isScalar() -> this.combine(other, scalarPlus, ordered = false)
            else -> this.combine(other, plus, ordered = false)
        }
    }

    override fun minus(other: TypedNDArray<LongArray>): TypedNDArray<LongArray> {
        return when {
            this.isScalar() && other.isScalar() -> LongNDArray(longArrayOf(this.array[0] - other.array[0]))
            other.isScalar() -> this.combine(other, scalarMinus)
            else -> this.combine(other, minus)
        }
    }

    override fun times(other: TypedNDArray<LongArray>): TypedNDArray<LongArray> {
        return when {
            this.isScalar() && other.isScalar() -> LongNDArray(longArrayOf(this.array[0] * other.array[0]))
            this.isScalar() || other.isScalar() -> this.combine(other, scalarTimes, ordered = false)
            else -> this.combine(other, times, ordered = false)
        }
    }

    override fun div(other: TypedNDArray<LongArray>): TypedNDArray<LongArray> {
        return when {
            this.isScalar() && other.isScalar() -> LongNDArray(longArrayOf(this.array[0] / other.array[0]))
            other.isScalar() -> this.combine(other, scalarDiv)
            else -> this.combine(other, div)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<LongArray> {
        func as LongArrayToLongArray
        return LongNDArray(map(array, func, true), strides)
    }

    override fun slice(sliceLength: Int, start: Int): LongArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun toMutable(): MutableTypedNDArray<LongArray> {
        return MutableLongNDArray(array, strides)
    }
}

class MutableLongNDArray(array: LongArray, strides: Strides = Strides.empty()) : LongNDArray(array, strides), MutableTypedNDArray<LongArray> {
    private companion object {
        val plusAssign = LongArrayWithLongArray { array, otherArray -> plus(array, otherArray, false) }
        val timesAssign = LongArrayWithLongArray { array, otherArray -> times(array, otherArray, false) }
        val minusAssign = LongArrayWithLongArray { array, otherArray -> minus(array, otherArray, false) }
        val divAssign = LongArrayWithLongArray { array, otherArray -> div(array, otherArray, false) }
        val scalarPlusAssign = LongArrayWithLong { array, value -> plus(array, value, false) }
        val scalarTimesAssign = LongArrayWithLong { array, value -> times(array, value, false) }
        val scalarMinusAssign = LongArrayWithLong { array, value -> minus(array, value, false) }
        val scalarDivAssign = LongArrayWithLong { array, value -> div(array, value, false) }
    }

    override fun clean() = array.fill(0L)

    override fun clone(): MutableTypedNDArray<LongArray> {
        return MutableLongNDArray(array.copyOf(), strides)
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as LongArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as LongArray
        block.copyInto(array, startOffset)
    }

    override fun toMutable(): MutableTypedNDArray<LongArray> = MutableLongNDArray(array, strides)

    override fun set(i: Int, value: Any) {
        array[i] = value as Long
    }

    override fun plusAssign(other: TypedNDArray<LongArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] += other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarPlusAssign)
            else -> this.combineAssign(other, plusAssign)
        }
    }

    override fun minusAssign(other: TypedNDArray<LongArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] -= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarMinusAssign)
            else -> this.combineAssign(other, minusAssign)
        }
    }

    override fun timesAssign(other: TypedNDArray<LongArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] *= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarTimesAssign)
            else -> this.combineAssign(other, timesAssign)
        }
    }

    override fun divAssign(other: TypedNDArray<LongArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] /= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarDivAssign)
            else -> this.combineAssign(other, divAssign)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): MutableTypedNDArray<LongArray> {
        func as LongArrayToLongArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<LongArray> {
        this.strides = strides
        return this
    }
}
