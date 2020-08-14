package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.combineWith
import org.jetbrains.research.kotlin.inference.extensions.ndarray.isScalar
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class ShortNDArray(array: ShortArray, strides: Strides = Strides.empty()) : NDArray<ShortArray>(array, strides, TensorProto.DataType.INT16) {
    init {
        require(array.size == strides.linearSize)
    }

    private companion object {
        val plus = ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, true) }
        val times = ShortArrayWithShortArray { array, otherArray -> times(array, otherArray, true) }
        val minus = ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, true) }
        val div = ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, true) }
        val scalarPlus = ShortArrayWithShort { array, value -> plus(array, value, true) }
        val scalarTimes = ShortArrayWithShort { array, value -> times(array, value, true) }
        val scalarMinus = ShortArrayWithShort { array, value -> minus(array, value, true) }
        val scalarDiv = ShortArrayWithShort { array, value -> div(array, value, true) }
    }
    
    override fun clone(): ShortNDArray {
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
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] + other.array[0]).toShort()))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarPlus)
        } else {
            this.combineWith(other, plus)
        }
    }

    override fun times(other: TypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] * other.array[0]).toShort()))
        } else if (other.isScalar()) {
            this.combineWith(other, ShortNDArray.scalarTimes)
        } else {
            this.combineWith(other, ShortNDArray.times)
        }
    }

    override fun minus(other: TypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] - other.array[0]).toShort()))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarMinus)
        } else {
            this.combineWith(other, minus)
        }
    }

    override fun div(other: TypedNDArray<ShortArray>): TypedNDArray<ShortArray> {
        return if (this.isScalar() && other.isScalar()) {
            ShortNDArray(shortArrayOf((this.array[0] / other.array[0]).toShort()))
        } else if (other.isScalar()) {
            this.combineWith(other, scalarDiv)
        } else {
            this.combineWith(other, div)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): NDArray<ShortArray> {
        func as ShortArrayToShortArray
        return ShortNDArray(map(array, func, true), strides)
    }

    override fun slice(sliceLength: Int, start: Int): ShortArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun toMutable(): MutableTypedNDArray<ShortArray> {
        return MutableShortNDArray(array, strides)
    }
}

class MutableShortNDArray(array: ShortArray, strides: Strides = Strides.empty()) : ShortNDArray(array, strides), MutableTypedNDArray<ShortArray> {
    private companion object {
        val plus = ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, false) }
        val times = ShortArrayWithShortArray { array, otherArray -> times(array, otherArray, false) }
        val minus = ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, false) }
        val div = ShortArrayWithShortArray { array, otherArray -> plus(array, otherArray, false) }
        val scalarPlus = ShortArrayWithShort { array, value -> plus(array, value, false) }
        val scalarTimes = ShortArrayWithShort { array, value -> times(array, value, false) }
        val scalarMinus = ShortArrayWithShort { array, value -> minus(array, value, false) }
        val scalarDiv = ShortArrayWithShort { array, value -> div(array, value, false) }
    }

    override fun clean() = array.fill(0)

    override fun clone(): MutableShortNDArray {
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

    override fun toMutable(): MutableTypedNDArray<ShortArray> = this

    override fun set(i: Int, value: Any) {
        array[i] = value as Short
    }

    override fun plusAssign(other: TypedNDArray<ShortArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] = (this.array[0] + other.array[0]).toShort()
        } else if (other.isScalar()) {
            this.combineWith(other, scalarPlus)
        } else {
            this.combineWith(other, plus)
        }
    }

    override fun minusAssign(other: TypedNDArray<ShortArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] = (this.array[0] - other.array[0]).toShort()
        } else if (other.isScalar()) {
            this.combineWith(other, scalarMinus)
        } else {
            this.combineWith(other, minus)
        }
    }

    override fun timesAssign(other: TypedNDArray<ShortArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0]  = (this.array[0] * other.array[0]).toShort()
        } else if (other.isScalar()) {
            this.combineWith(other, scalarTimes)
        } else {
            this.combineWith(other, times)
        }
    }

    override fun divAssign(other: TypedNDArray<ShortArray>) {
        if (this.isScalar() && other.isScalar()) {
            this.array[0] = (this.array[0] / other.array[0]).toShort()
        } else if (other.isScalar()) {
            this.combineWith(other, scalarDiv)
        } else {
            this.combineWith(other, div)
        }
    }

    override fun mapElements(func: PrimitiveArrayFunction): NDArray<ShortArray> {
        func as ShortArrayToShortArray
        map(array, func, false)
        return this
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<ShortArray> {
        this.strides = strides
        return this
    }
}
