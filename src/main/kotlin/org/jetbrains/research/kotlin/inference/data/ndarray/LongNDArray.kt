package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.LongArrayToLongArray
import org.jetbrains.research.kotlin.inference.extensions.functional.LongArrayWithLong
import org.jetbrains.research.kotlin.inference.extensions.functional.LongArrayWithLongArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combine
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineAssign
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class LongNDArray(array: LongArray, strides: Strides = Strides.empty(), offset: Int = 0) : NDArray<LongArray>(array, strides, TensorProto.DataType.INT64, offset) {
    /*init {
        require(array.size == strides.linearSize)
    }*/

    private companion object {
        val plus = LongArrayWithLongArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val times = LongArrayWithLongArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> times(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val minus = LongArrayWithLongArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val div = LongArrayWithLongArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> div(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val scalarPlus = LongArrayWithLong { array, offset, value, destination, destinationOffset, size -> plus(array, offset, value, destination, destinationOffset, size) }
        val scalarTimes = LongArrayWithLong { array, offset, value, destination, destinationOffset, size -> times(array, offset, value, destination, destinationOffset, size) }
        val scalarMinus = LongArrayWithLong { array, offset, value, destination, destinationOffset, size -> minus(array, offset, value, destination, destinationOffset, size) }
        val scalarDiv = LongArrayWithLong { array, offset, value, destination, destinationOffset, size -> div(array, offset, value, destination, destinationOffset, size) }
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

    override fun plus(other: TypedNDArray<LongArray>, destination: MutableTypedNDArray<LongArray>): TypedNDArray<LongArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] + other.array[0]
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarPlus, ordered = false)
            else -> this.combine(other, destination, plus, ordered = false)
        }
        return destination
    }

    override fun minus(other: TypedNDArray<LongArray>, destination: MutableTypedNDArray<LongArray>): TypedNDArray<LongArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] - other.array[0]
            other.isScalar() -> this.combine(other, destination, scalarMinus)
            else -> this.combine(other, destination, minus)
        }
        return destination
    }

    override fun times(other: TypedNDArray<LongArray>, destination: MutableTypedNDArray<LongArray>): TypedNDArray<LongArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] * other.array[0]
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarTimes, ordered = false)
            else -> this.combine(other, destination, times, ordered = false)
        }
        return destination
    }

    override fun div(other: TypedNDArray<LongArray>, destination: MutableTypedNDArray<LongArray>): TypedNDArray<LongArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] / other.array[0]
            other.isScalar() -> this.combine(other, destination, scalarDiv)
            else -> this.combine(other, destination, div)
        }
        return destination
    }

    override fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<LongArray> {
        func as LongArrayToLongArray
        return LongNDArray(map(array, func, true), strides)
    }

    override fun slice(sliceLength: Int, start: Int): LongArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun toMutable(): MutableTypedNDArray<LongArray> {
        return MutableLongNDArray(array.copyOf(), strides)
    }
}

class MutableLongNDArray(array: LongArray, strides: Strides = Strides.empty(), offset: Int = 0) : LongNDArray(array, strides, offset), MutableTypedNDArray<LongArray> {
    private companion object {
        val plusAssign = LongArrayWithLongArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val timesAssign = LongArrayWithLongArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> times(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val minusAssign = LongArrayWithLongArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val divAssign = LongArrayWithLongArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> div(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val scalarPlusAssign = LongArrayWithLong { array, offset, value, destination, destinationOffset, size -> plus(array, offset, value, destination, destinationOffset, size) }
        val scalarTimesAssign = LongArrayWithLong { array, offset, value, destination, destinationOffset, size -> times(array, offset, value, destination, destinationOffset, size) }
        val scalarMinusAssign = LongArrayWithLong { array, offset, value, destination, destinationOffset, size -> minus(array, offset, value, destination, destinationOffset, size) }
        val scalarDivAssign = LongArrayWithLong { array, offset, value, destination, destinationOffset, size -> div(array, offset, value, destination, destinationOffset, size) }
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
