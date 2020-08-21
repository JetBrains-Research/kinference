package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.functional.ShortArrayToShortArray
import org.jetbrains.research.kotlin.inference.extensions.functional.ShortArrayWithShort
import org.jetbrains.research.kotlin.inference.extensions.functional.ShortArrayWithShortArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combine
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineAssign
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.math.LateInitArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class ShortNDArray(array: ShortArray, strides: Strides = Strides.empty(), offset: Int = 0) : NDArray<ShortArray>(array, strides, TensorProto.DataType.INT16, offset) {
    /*init {
        require(array.size == strides.linearSize)
    }*/

    private companion object {
        val plus = object : ShortArrayWithShortArray {
            override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
                return plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size)
            }
        }
        val times = object : ShortArrayWithShortArray {
            override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
                return times(left, leftOffset, right, rightOffset, destination, destinationOffset, size)
            }
        }
        val minus = object : ShortArrayWithShortArray {
            override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
                return minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size)
            }
        }
        val div = object : ShortArrayWithShortArray {
            override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
                return div(left, leftOffset, right, rightOffset, destination, destinationOffset, size)
            }
        }
        val scalarPlus = object : ShortArrayWithShort {
            override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int) {
                plus(array, offset, value, destination, destinationOffset, size)
            }
        }
        val scalarTimes = object : ShortArrayWithShort {
            override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int) {
                times(array, offset, value, destination, destinationOffset, size)
            }
        }
        val scalarMinus = object : ShortArrayWithShort {
            override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int) {
                minus(array, offset, value, destination, destinationOffset, size)
            }
        }
        val scalarDiv = object : ShortArrayWithShort {
            override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int) {
                div(array, offset, value, destination, destinationOffset, size)
            }
        }
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

    override fun plus(other: TypedNDArray<ShortArray>, destination: MutableTypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = (this.array[0] + other.array[0]).toShort()
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarPlus)
            else -> this.combine(other, destination, plus)
        }
        return destination
    }

    override fun minus(other: TypedNDArray<ShortArray>, destination: MutableTypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = (this.array[0] - other.array[0]).toShort()
            other.isScalar() -> this.combine(other, destination, scalarMinus, ordered = true)
            else -> this.combine(other, destination, minus, ordered = true)
        }
        return destination
    }

    override fun times(other: TypedNDArray<ShortArray>, destination: MutableTypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = (this.array[0] * other.array[0]).toShort()
            this.isScalar() || other.isScalar() -> this.combine(other, destination, scalarTimes)
            else -> this.combine(other, destination, times)
        }
        return destination
    }

    override fun div(other: TypedNDArray<ShortArray>, destination: MutableTypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        when {
            this.isScalar() && other.isScalar() -> destination.array[0] = (this.array[0] / other.array[0]).toShort()
            other.isScalar() -> this.combine(other, destination, scalarDiv, ordered = true)
            else -> this.combine(other, destination, div, ordered = true)
        }
        return destination
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

class MutableShortNDArray(array: ShortArray, strides: Strides = Strides.empty(), offset: Int = 0) : ShortNDArray(array, strides, offset), MutableTypedNDArray<ShortArray> {
    private companion object {
        val plusAssign = object : ShortArrayWithShortArray {
            override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
                return plus(left, leftOffset, right, rightOffset, destination, destinationOffset, size)
            }
        }
        val timesAssign = object : ShortArrayWithShortArray {
            override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
                return times(left, leftOffset, right, rightOffset, destination, destinationOffset, size)
            }
        }
        val minusAssign = object : ShortArrayWithShortArray {
            override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
                return minus(left, leftOffset, right, rightOffset, destination, destinationOffset, size)
            }
        }
        val divAssign = object : ShortArrayWithShortArray {
            override fun apply(left: ShortArray, leftOffset: Int, right: ShortArray, rightOffset: Int, destination: ShortArray, destinationOffset: Int, size: Int): ShortArray {
                return div(left, leftOffset, right, rightOffset, destination, destinationOffset, size)
            }
        }
        val scalarPlusAssign = object : ShortArrayWithShort {
            override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int) {
                plus(array, offset, value, destination, destinationOffset, size)
            }
        }
        val scalarTimesAssign = object : ShortArrayWithShort {
            override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int) {
                times(array, offset, value, destination, destinationOffset, size)
            }
        }
        val scalarMinusAssign = object : ShortArrayWithShort {
            override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int) {
                minus(array, offset, value, destination, destinationOffset, size)
            }
        }
        val scalarDivAssign = object : ShortArrayWithShort {
            override fun apply(array: ShortArray, offset: Int, value: Short, destination: ShortArray, destinationOffset: Int, size: Int) {
                div(array, offset, value, destination, destinationOffset, size)
            }
        }
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
