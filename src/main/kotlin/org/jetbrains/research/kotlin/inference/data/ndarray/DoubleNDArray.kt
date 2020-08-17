package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.DoubleArrayToDoubleArray
import org.jetbrains.research.kotlin.inference.extensions.functional.DoubleArrayWithDouble
import org.jetbrains.research.kotlin.inference.extensions.functional.DoubleArrayWithDoubleArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combine
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineAssign
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class DoubleNDArray(array: DoubleArray, strides: Strides = Strides.empty(), offset: Int = 0) : NDArray<DoubleArray>(array, strides, TensorProto.DataType.DOUBLE, offset) {
    /*init {
        require(array.size == strides.linearSize)
    }*/

    protected companion object {
        val plus = DoubleArrayWithDoubleArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val times = DoubleArrayWithDoubleArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> times(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val minus = DoubleArrayWithDoubleArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val div = DoubleArrayWithDoubleArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> div(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val scalarPlus = DoubleArrayWithDouble { array, offset, value, destination, destinationOffset, size -> plus(array, offset, value, destination, destinationOffset, size) }
        val scalarTimes = DoubleArrayWithDouble { array, offset, value, destination, destinationOffset, size -> times(array, offset, value, destination, destinationOffset, size) }
        val scalarMinus = DoubleArrayWithDouble { array, offset, value, destination, destinationOffset, size -> minus(array, offset, value, destination, destinationOffset, size) }
        val scalarDiv = DoubleArrayWithDouble { array, offset, value, destination, destinationOffset, size -> div(array, offset, value, destination, destinationOffset, size) }
    }

    override fun clone(): TypedNDArray<DoubleArray> {
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

    override fun plus(other: TypedNDArray<DoubleArray>, destination: MutableTypedNDArray<DoubleArray>): TypedNDArray<DoubleArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] + other.array[0]
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarPlus, ordered = false)
            else -> this.combine(other, destination, plus, ordered = false)
        }
        return destination
    }

    override fun minus(other: TypedNDArray<DoubleArray>, destination: MutableTypedNDArray<DoubleArray>): TypedNDArray<DoubleArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] - other.array[0]
            other.isScalar() -> this.combine(other, destination, scalarMinus)
            else -> this.combine(other, destination, minus)
        }
        return destination
    }

    override fun times(other: TypedNDArray<DoubleArray>, destination: MutableTypedNDArray<DoubleArray>): TypedNDArray<DoubleArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] * other.array[0]
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarTimes, ordered = false)
            else -> this.combine(other, destination, times, ordered = false)
        }
        return destination
    }

    override fun div(other: TypedNDArray<DoubleArray>, destination: MutableTypedNDArray<DoubleArray>): TypedNDArray<DoubleArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] / other.array[0]
            other.isScalar() -> this.combine(other, destination, scalarDiv)
            else -> this.combine(other, destination, div)
        }
        return destination
    }

    override fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<DoubleArray> {
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

class MutableDoubleNDArray(array: DoubleArray, strides: Strides = Strides.empty(), offset: Int = 0) : DoubleNDArray(array, strides, offset), MutableTypedNDArray<DoubleArray> {
    private companion object {
        val plusAssign = DoubleArrayWithDoubleArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val timesAssign = DoubleArrayWithDoubleArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> times(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val minusAssign = DoubleArrayWithDoubleArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val divAssign = DoubleArrayWithDoubleArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> div(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val scalarPlusAssign = DoubleArrayWithDouble { array, offset, value, destination, destinationOffset, size -> plus(array, offset, value, destination, destinationOffset, size) }
        val scalarTimesAssign = DoubleArrayWithDouble { array, offset, value, destination, destinationOffset, size -> times(array, offset, value, destination, destinationOffset, size) }
        val scalarMinusAssign = DoubleArrayWithDouble { array, offset, value, destination, destinationOffset, size -> minus(array, offset, value, destination, destinationOffset, size) }
        val scalarDivAssign = DoubleArrayWithDouble { array, offset, value, destination, destinationOffset, size -> div(array, offset, value, destination, destinationOffset, size) }
    }

    override fun clean() = array.fill(0.0)

    override fun clone(): MutableTypedNDArray<DoubleArray> {
        return MutableDoubleNDArray(array, strides)
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as DoubleArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as DoubleArray
        block.copyInto(array, startOffset)
    }

    override fun toMutable(): MutableTypedNDArray<DoubleArray> = MutableDoubleNDArray(array, strides)

    override fun set(i: Int, value: Any) {
        array[i] = value as Double
    }

    override fun plusAssign(other: TypedNDArray<DoubleArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] += other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarPlusAssign)
            else -> this.combineAssign(other, plusAssign)
        }
    }

    override fun minusAssign(other: TypedNDArray<DoubleArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] -= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarMinusAssign)
            else -> this.combineAssign(other, minusAssign)
        }
    }

    override fun timesAssign(other: TypedNDArray<DoubleArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] *= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarTimesAssign)
            else -> this.combineAssign(other, timesAssign)
        }
    }

    override fun divAssign(other: TypedNDArray<DoubleArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] /= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarDivAssign)
            else -> this.combineAssign(other, divAssign)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): MutableTypedNDArray<DoubleArray> {
        func as DoubleArrayToDoubleArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<DoubleArray> {
        this.strides = strides
        return this
    }
}
