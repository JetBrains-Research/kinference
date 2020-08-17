package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class ShortNDArray(array: ShortArray, strides: Strides = Strides.empty()) : NDArray<ShortArray>(array, strides, TensorProto.DataType.INT16) {
    init {
        require(array.size == strides.linearSize)
    }

    private companion object {
        val plus = ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, true) }
        val times = ShortArrayWithShortArray { array, otherArray -> times(array, otherArray, true) }
        val minus = ShortArrayWithShortArray { array, otherArray -> minus(array, otherArray, true) }
        val div = ShortArrayWithShortArray { array, otherArray -> div(array, otherArray, true) }
        val scalarPlus = ShortArrayWithShort { array, value -> plus(array, value, true) }
        val scalarTimes = ShortArrayWithShort { array, value -> times(array, value, true) }
        val scalarMinus = ShortArrayWithShort { array, value -> minus(array, value, true) }
        val scalarDiv = ShortArrayWithShort { array, value -> div(array, value, true) }
    }
    
    override fun clone(): TypedNDArray<ShortArray> {
        return ShortNDArray(array.copyOf(), strides)
    }

    override fun get(i: Int): Short {
        return array[i]
    }

    override fun get(indices: IntArray): Short {
        return array[strides.offset(indices)]
    }

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitShortArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: TypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        return when {
            this.isScalar() && other.isScalar() -> ShortNDArray(shortArrayOf((this.array[0] + other.array[0]).toShort()))
            this.isScalar() || other.isScalar() -> this.combine(other, scalarPlus, ordered = false)
            else -> this.combine(other, plus, ordered = false)
        }
    }

    override fun minus(other: TypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        return when {
            this.isScalar() && other.isScalar() -> ShortNDArray(shortArrayOf((this.array[0] - other.array[0]).toShort()))
            other.isScalar() -> this.combine(other, scalarMinus)
            else -> this.combine(other, minus)
        }
    }

    override fun times(other: TypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        return when {
            this.isScalar() && other.isScalar() -> ShortNDArray(shortArrayOf((this.array[0] * other.array[0]).toShort()))
            this.isScalar() || other.isScalar() -> this.combine(other, scalarTimes, ordered = false)
            else -> this.combine(other, times, ordered = false)
        }
    }

    override fun div(other: TypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        return when {
            this.isScalar() && other.isScalar() -> ShortNDArray(shortArrayOf((this.array[0] / other.array[0]).toShort()))
            other.isScalar() -> this.combine(other, scalarDiv)
            else -> this.combine(other, div)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<ShortArray> {
        func as ShortArrayToShortArray
        return ShortNDArray(map(array, func, true), strides)
    }

    override fun slice(sliceLength: Int, start: Int): ShortArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun toMutable(): MutableTypedNDArray<ShortArray> {
        return MutableShortNDArray(array.copyOf(), strides)
    }
}

class MutableShortNDArray(array: ShortArray, strides: Strides = Strides.empty()) : ShortNDArray(array, strides), MutableTypedNDArray<ShortArray> {
    private companion object {
        val plusAssign = ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, false) }
        val timesAssign = ShortArrayWithShortArray { array, otherArray -> times(array, otherArray, false) }
        val minusAssign = ShortArrayWithShortArray { array, otherArray -> minus(array, otherArray, false) }
        val divAssign = ShortArrayWithShortArray { array, otherArray -> div(array, otherArray, false) }
        val scalarPlusAssign = ShortArrayWithShort { array, value -> plus(array, value, false) }
        val scalarTimesAssign = ShortArrayWithShort { array, value -> times(array, value, false) }
        val scalarMinusAssign = ShortArrayWithShort { array, value -> minus(array, value, false) }
        val scalarDivAssign = ShortArrayWithShort { array, value -> div(array, value, false) }
    }

    override fun clean() = array.fill(0)

    override fun clone(): MutableTypedNDArray<ShortArray> {
        return MutableShortNDArray(array.copyOf(), strides)
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as ShortArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as ShortArray
        block.copyInto(array, startOffset)
    }

    override fun toMutable(): MutableTypedNDArray<ShortArray> = MutableShortNDArray(array, strides)

    override fun set(i: Int, value: Any) {
        array[i] = value as Short
    }

    override fun plusAssign(other: TypedNDArray<ShortArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] + other.array[0]).toShort()
            other.isScalar() -> this.combineAssign(other, scalarPlusAssign)
            else -> this.combineAssign(other, plusAssign)
        }
    }

    override fun minusAssign(other: TypedNDArray<ShortArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] - other.array[0]).toShort()
            other.isScalar() -> this.combineAssign(other, scalarMinusAssign)
            else -> this.combineAssign(other, minusAssign)
        }
    }

    override fun timesAssign(other: TypedNDArray<ShortArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] * other.array[0]).toShort()
            other.isScalar() -> this.combineAssign(other, scalarTimesAssign)
            else -> this.combineAssign(other, timesAssign)
        }
    }

    override fun divAssign(other: TypedNDArray<ShortArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] = (this.array[0] / other.array[0]).toShort()
            other.isScalar() -> this.combineAssign(other, scalarDivAssign)
            else -> this.combineAssign(other, divAssign)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): MutableTypedNDArray<ShortArray> {
        func as ShortArrayToShortArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<ShortArray> {
        this.strides = strides
        return this
    }
}
