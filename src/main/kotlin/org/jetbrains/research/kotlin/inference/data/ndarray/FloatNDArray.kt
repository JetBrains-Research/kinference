package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.FloatArrayToFloatArray
import org.jetbrains.research.kotlin.inference.extensions.functional.FloatArrayWithFloat
import org.jetbrains.research.kotlin.inference.extensions.functional.FloatArrayWithFloatArray
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combine
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineAssign
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class FloatNDArray(array: FloatArray, strides: Strides = Strides.empty(), offset: Int = 0) : NDArray<FloatArray>(array, strides, TensorProto.DataType.FLOAT, offset) {
    /*init {
        require(array.size == strides.linearSize)
    }*/

    private companion object {
        val plus = FloatArrayWithFloatArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val times = FloatArrayWithFloatArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> times(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val minus = FloatArrayWithFloatArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val div = FloatArrayWithFloatArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> div(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val scalarPlus = FloatArrayWithFloat { array, offset, value, destination, destinationOffset, size -> plus(array, offset, value, destination, destinationOffset, size) }
        val scalarTimes = FloatArrayWithFloat { array, offset, value, destination, destinationOffset, size -> times(array, offset, value, destination, destinationOffset, size) }
        val scalarMinus = FloatArrayWithFloat { array, offset, value, destination, destinationOffset, size -> minus(array, offset, value, destination, destinationOffset, size) }
        val scalarDiv = FloatArrayWithFloat { array, offset, value, destination, destinationOffset, size -> div(array, offset, value, destination, destinationOffset, size) }
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

    override fun plus(other: TypedNDArray<FloatArray>, destination: MutableTypedNDArray<FloatArray>): TypedNDArray<FloatArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] + other.array[0]
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarPlus)
            else -> this.combine(other, destination, plus)
        }
        return destination
    }

    override fun minus(other: TypedNDArray<FloatArray>, destination: MutableTypedNDArray<FloatArray>): TypedNDArray<FloatArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] - other.array[0]
            other.isScalar() -> this.combine(other, destination, scalarMinus, ordered = true)
            else -> this.combine(other, destination, minus, ordered = true)
        }
        return destination
    }

    override fun times(other: TypedNDArray<FloatArray>, destination: MutableTypedNDArray<FloatArray>): TypedNDArray<FloatArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] * other.array[0]
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarTimes)
            else -> this.combine(other, destination, times)
        }
        return destination
    }

    override fun div(other: TypedNDArray<FloatArray>, destination: MutableTypedNDArray<FloatArray>): TypedNDArray<FloatArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = this.array[0] / other.array[0]
            other.isScalar() -> this.combine(other, destination, scalarDiv, ordered = true)
            else -> this.combine(other, destination, div, ordered = true)
        }
        return destination
    }

    override fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<FloatArray> {
        func as FloatArrayToFloatArray
        return FloatNDArray(map(array, func, true), strides, offset)
    }

    override fun slice(sliceLength: Int, start: Int): FloatArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun toMutable(): MutableTypedNDArray<FloatArray> {
        return MutableFloatNDArray(array.copyOf(), strides, offset)
    }
}

class MutableFloatNDArray(array: FloatArray, strides: Strides = Strides.empty(), offset: Int = 0) : FloatNDArray(array, strides, offset), MutableTypedNDArray<FloatArray> {
    private companion object {
        val plusAssign = FloatArrayWithFloatArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val timesAssign = FloatArrayWithFloatArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> times(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val minusAssign = FloatArrayWithFloatArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val divAssign = FloatArrayWithFloatArray { left, leftOffset, right, rightOffset, destination, destinationOffset, size -> div(left, leftOffset, right, rightOffset, destination, destinationOffset, size) }
        val scalarPlusAssign = FloatArrayWithFloat { array, offset, value, destination, destinationOffset, size -> plus(array, offset, value, destination, destinationOffset, size) }
        val scalarTimesAssign = FloatArrayWithFloat { array, offset, value, destination, destinationOffset, size -> times(array, offset, value, destination, destinationOffset, size) }
        val scalarMinusAssign = FloatArrayWithFloat { array, offset, value, destination, destinationOffset, size -> minus(array, offset, value, destination, destinationOffset, size) }
        val scalarDivAssign = FloatArrayWithFloat { array, offset, value, destination, destinationOffset, size -> div(array, offset, value, destination, destinationOffset, size) }
    }

    override fun clean() = array.fill(0.0f)

    override fun clone(): MutableTypedNDArray<FloatArray> {
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

    override fun toMutable(): MutableTypedNDArray<FloatArray> = MutableFloatNDArray(array, strides)

    override fun set(i: Int, value: Any) {
        array[i] = value as Float
    }

    override fun plusAssign(other: TypedNDArray<FloatArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] += other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarPlusAssign)
            else -> this.combineAssign(other, plusAssign)
        }
    }

    override fun minusAssign(other: TypedNDArray<FloatArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] -= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarMinusAssign)
            else -> this.combineAssign(other, minusAssign)
        }
    }

    override fun timesAssign(other: TypedNDArray<FloatArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] *= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarTimesAssign)
            else -> this.combineAssign(other, timesAssign)
        }
    }

    override fun divAssign(other: TypedNDArray<FloatArray>) {
        when {
            this.isScalar() && other.isScalar() -> this.array[0] /= other.array[0]
            other.isScalar() -> this.combineAssign(other, scalarDivAssign)
            else -> this.combineAssign(other, divAssign)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): MutableTypedNDArray<FloatArray> {
        func as FloatArrayToFloatArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<FloatArray> {
        this.strides = strides
        return this
    }
}
