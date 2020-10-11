package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.extensions.slice
import io.kinference.primitives.types.DataType
import kotlin.math.abs

class LateInitBooleanArray(size: Int) : LateInitArray {
    private val array = BooleanArray(size)
    private var index = 0

    fun putNext(value: Boolean) {
        array[index] = value
        index++
    }

    fun getArray(): BooleanArray {
        require(index == array.size) { "LateInitArray not initialized yet" }
        return array
    }
}


interface BooleanMap : PrimitiveToPrimitiveFunction {
    fun apply(value: Boolean): Boolean
}

open class BooleanNDArray(val array: BooleanArray, strides: Strides = Strides.empty()) : NDArray {
    override val type: DataType = DataType.BOOLEAN

    final override var strides: Strides = strides
        protected set

    override fun get(index: Int): Boolean {
        return array[index]
    }

    override fun get(indices: IntArray): Boolean {
        return array[strides.offset(indices)]
    }

    override fun allocateNDArray(strides: Strides): MutableNDArray {
        return MutableBooleanNDArray(BooleanArray(strides.linearSize), strides)
    }

    override fun reshapeView(newShape: IntArray): NDArray {
        return BooleanNDArray(array, Strides(newShape))
    }

    override fun toMutable(newStrides: Strides): MutableNDArray {
        return MutableBooleanNDArray(array.copyOf(), strides)
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutableBooleanNDArray(array, strides)
    }

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, additionalOffset: Int) {
        array as LateInitBooleanArray
        for (index in range) {
            array.putNext(this.array[additionalOffset + index])
        }
    }

    override fun map(function: PrimitiveToPrimitiveFunction): MutableNDArray {
        function as BooleanMap
        val destination = allocateNDArray(strides) as MutableBooleanNDArray
        for (index in 0 until destination.linearSize) {
            destination.array[index] = function.apply(this.array[index])
        }

        return destination
    }

    override fun row(row: Int): MutableNDArray {
        val rowLength: Int = linearSize / shape[0]
        val start = row * rowLength
        val dims = shape.copyOfRange(1, rank)

        return MutableBooleanNDArray(array.copyOfRange(start, start + rowLength), Strides(dims))
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNDArray {
        val newShape = IntArray(shape.size) {
            val length = abs(ends[it] - starts[it])
            val rest = length % abs(steps[it])
            (length / abs(steps[it])) + if (rest != 0) 1 else 0
        }

        val newStrides = Strides(newShape)
        val newArray = LateInitBooleanArray(newStrides.linearSize)

        slice(newArray, 0, 0, shape, starts, ends, steps)

        return MutableBooleanNDArray(newArray.getArray(), newStrides)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BooleanNDArray) return false

        if (type != other.type) return false
        if (strides != other.strides) return false
        if (array != other.array) return false

        return true
    }
}

class MutableBooleanNDArray(array: BooleanArray, strides: Strides = Strides.empty()): BooleanNDArray(array, strides), MutableNDArray {
    override fun set(index: Int, value: Any) {
        array[index] = value as Boolean
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutableBooleanNDArray(array, strides)
    }

    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNDArray {
        function as BooleanMap
        for (index in 0 until linearSize) {
            array[index] = function.apply(array[index])
        }

        return this
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as BooleanNDArray
        other.array.copyInto(this.array, offset, startInOther, endInOther)
    }

    override fun fill(value: Any, from: Int, to: Int) {
        array.fill(value as Boolean)
    }
    
    override fun reshape(strides: Strides): MutableNDArray {
        this.strides = strides
        return  this
    }

    private fun transposeRec(prevArray: BooleanArray, newArray: BooleanArray, prevStrides: Strides, newStrides: Strides, index: Int, prevOffset: Int, newOffset: Int, permutation: IntArray) {
        if (index != newStrides.shape.lastIndex) {
            val temp = prevStrides.strides[permutation[index]]
            val temp2 = newStrides.strides[index]
            for (i in 0 until newStrides.shape[index])
                transposeRec(prevArray, newArray, prevStrides, newStrides, index + 1, prevOffset + temp * i,
                    newOffset + temp2 * i, permutation)
        } else {
            val temp = prevStrides.strides[permutation[index]]
            if (temp == 1) {
                prevArray.copyInto(newArray, newOffset, prevOffset, prevOffset + newStrides.shape[index])
            } else {
                for (i in 0 until newStrides.shape[index]) {
                    newArray[newOffset + i] = prevArray[prevOffset + i * temp]
                }
            }
        }
    }

    override fun transpose(permutations: IntArray): MutableNDArray {
        val newStrides = strides.transpose(permutations)
        transposeRec(array.copyOf(), array, strides, newStrides, 0, 0, 0, permutations)
        return this.reshape(newStrides)
    }

    override fun transpose2D(): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun clean() {
        array.fill(false)
    }

    fun not(): MutableNDArray {
        return mapMutable(object : BooleanMap {
            override fun apply(value: Boolean): Boolean = value.not()
        })
    }
}
