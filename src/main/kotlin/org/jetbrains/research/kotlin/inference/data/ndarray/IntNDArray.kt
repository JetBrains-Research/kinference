package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.IntArrayToIntArray
import org.jetbrains.research.kotlin.inference.extensions.functional.IntArrayWithInt
import org.jetbrains.research.kotlin.inference.extensions.functional.IntArrayWithIntArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combine
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineAssign
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class IntNDArray(array: IntArray, strides: Strides = Strides.empty(), offset: Int = 0) : NDArray<IntArray>(array, strides, TensorProto.DataType.INT32, offset) {
    /*init {
        require(array.size == strides.linearSize)
    }*/

    private companion object {
        val plus = IntArrayWithIntArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val times = IntArrayWithIntArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> times(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val minus = IntArrayWithIntArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val div = IntArrayWithIntArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> div(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val scalarPlus = IntArrayWithInt { array, offset, value, destination, destinationOffset, size -> plus(array, offset, value, destination, destinationOffset, size) }
        val scalarTimes = IntArrayWithInt { array, offset, value, destination, destinationOffset, size -> times(array, offset, value, destination, destinationOffset, size) }
        val scalarMinus = IntArrayWithInt { array, offset, value, destination, destinationOffset, size -> minus(array, offset, value, destination, destinationOffset, size) }
        val scalarDiv = IntArrayWithInt { array, offset, value, destination, destinationOffset, size -> div(array, offset, value, destination, destinationOffset, size) }
    }
    
    override fun clone(): TypedNDArray<IntArray> {
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

    override fun plus(other: TypedNDArray<IntArray>, destination: MutableTypedNDArray<IntArray>): TypedNDArray<IntArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] + other.array[0]
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarPlus, ordered = false)
            else -> this.combine(other, destination, plus, ordered = false)
        }
        return destination
    }

    override fun minus(other: TypedNDArray<IntArray>, destination: MutableTypedNDArray<IntArray>): TypedNDArray<IntArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] - other.array[0]
            other.isScalar() -> this.combine(other, destination, scalarMinus)
            else -> this.combine(other, destination, minus)
        }
        return destination
    }

    override fun times(other: TypedNDArray<IntArray>, destination: MutableTypedNDArray<IntArray>): TypedNDArray<IntArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] * other.array[0]
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarTimes, ordered = false)
            else -> this.combine(other, destination, times, ordered = false)
        }
        return destination
    }

    override fun div(other: TypedNDArray<IntArray>, destination: MutableTypedNDArray<IntArray>): TypedNDArray<IntArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] / other.array[0]
            other.isScalar() -> this.combine(other, destination, scalarDiv)
            else -> this.combine(other, destination, div)
        }
        return destination
    }

    override fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<IntArray> {
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

class MutableIntNDArray(array: IntArray, strides: Strides = Strides.empty(), offset: Int = 0) : IntNDArray(array, strides, offset), MutableTypedNDArray<IntArray> {
    private companion object {
        val plusAssign = IntArrayWithIntArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val timesAssign = IntArrayWithIntArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> times(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val minusAssign = IntArrayWithIntArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val divAssign = IntArrayWithIntArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> div(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val scalarPlusAssign = IntArrayWithInt { array, offset, value, destination, destinationOffset, size -> plus(array, offset, value, destination, destinationOffset, size) }
        val scalarTimesAssign = IntArrayWithInt { array, offset, value, destination, destinationOffset, size -> times(array, offset, value, destination, destinationOffset, size) }
        val scalarMinusAssign = IntArrayWithInt { array, offset, value, destination, destinationOffset, size -> minus(array, offset, value, destination, destinationOffset, size) }
        val scalarDivAssign = IntArrayWithInt { array, offset, value, destination, destinationOffset, size -> div(array, offset, value, destination, destinationOffset, size) }
    }

    override fun clean() = array.fill(0)

    override fun clone(): MutableTypedNDArray<IntArray> {
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

    override fun toMutable(): MutableTypedNDArray<IntArray> = MutableIntNDArray(array, strides)

    override fun set(i: Int, value: Any) {
        array[i] = value as Int
    }

    override fun plusAssign(other: TypedNDArray<IntArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] += other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarPlusAssign)
            else -> this.combineAssign(other, plusAssign)
        }
    }

    override fun minusAssign(other: TypedNDArray<IntArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] -= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarMinusAssign)
            else -> this.combineAssign(other, minusAssign)
        }
    }

    override fun timesAssign(other: TypedNDArray<IntArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] *= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarTimesAssign)
            else -> this.combineAssign(other, timesAssign)
        }
    }

    override fun divAssign(other: TypedNDArray<IntArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] /= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarDivAssign)
            else -> this.combineAssign(other, divAssign)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): MutableTypedNDArray<IntArray> {
        func as IntArrayToIntArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<IntArray> {
        this.strides = strides
        return this
    }
}
