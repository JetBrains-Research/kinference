package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.isScalar
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

open class BooleanNDArray(var array: BooleanTiledArray, strides: Strides = Strides.empty()) : NDArray {
    constructor(array: BooleanArray, strides: Strides = Strides.empty()) : this(BooleanTiledArray(array, strides), strides)

    override val type: DataType = DataType.BOOLEAN

    final override var strides: Strides = strides
        protected set

    protected val blocksInRow: Int
        get() = when {
            strides.linearSize == 0 -> 0
            strides.shape.isEmpty() -> 1
            else -> strides.shape.last() / array.blockSize
        }

    override fun view(vararg axes: Int): NDArray {
        TODO("Not yet implemented")
    }

    override fun get(index: Int): Boolean {
        return array[index]
    }

    override fun get(indices: IntArray): Boolean {
        return array[strides.offset(indices)]
    }

    override fun singleValue(): Boolean {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array.blocks[0][0]
    }

    override fun copyOfRange(start: Int, end: Int): Any {
        return array.copyOfRange(start, end)
    }

    override fun allocateNDArray(strides: Strides): MutableNDArray {
        return MutableBooleanNDArray(BooleanTiledArray(strides), strides)
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

        return MutableBooleanNDArray(BooleanTiledArray(Strides(dims)) { array[start + it] }, Strides(dims))
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

        return MutableBooleanNDArray(BooleanTiledArray(newArray.getArray(), newStrides), newStrides)
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

class MutableBooleanNDArray(array: BooleanTiledArray, strides: Strides = Strides.empty()): BooleanNDArray(array, strides), MutableNDArray {
    override fun set(index: Int, value: Any) {
        array[index] = value as Boolean
    }

    override fun viewMutable(vararg axes: Int): MutableNDArray {
        TODO()
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

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        array as BooleanNDArray
        this.array.fill(array[index], from, to)
    }

    override fun reshape(strides: Strides): MutableNDArray {
        this.strides = strides
        return  this
    }

    // TODO separate from PrimitiveArray (maybe LateInitArray will help)
    private fun transposeRec(prevArray: BooleanTiledArray, newArray: BooleanTiledArray, prevStrides: Strides, newStrides: Strides, index: Int, prevOffset: Int, newOffset: Int, permutation: IntArray) {
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
                var (newArrayBlock, newArrayOffset) = newArray.indexFor(newOffset)
                var (prevArrayBlock, prevArrayOffset) = prevArray.indexFor(prevOffset)

                val (deltaBlock, deltaOffset) = prevArray.indexFor(temp)

                var tempNewBlock = newArray.blocks[newArrayBlock]
                var tempPrevBlock = prevArray.blocks[prevArrayBlock]

                for (i in 0 until newStrides.shape[index]) {
                    tempNewBlock[newArrayOffset++] = tempPrevBlock[prevArrayOffset]

                    prevArrayBlock += deltaBlock
                    prevArrayOffset += deltaOffset

                    if (prevArrayOffset >= prevArray.blockSize) {
                        prevArrayBlock += prevArrayOffset / prevArray.blockSize
                        prevArrayOffset %= prevArray.blockSize
                    }

                    if (prevArrayBlock < prevArray.blocksNum)
                        tempPrevBlock = prevArray.blocks[prevArrayBlock]

                    if (newArrayOffset >= newArray.blockSize) {
                        newArrayOffset = 0

                        if (++newArrayBlock < newArray.blocksNum)
                            tempNewBlock = newArray.blocks[newArrayBlock]
                    }
                }
            }
        }
    }

    override fun transpose(permutations: IntArray): MutableNDArray {
        val newStrides = strides.transpose(permutations)
        val newArray = BooleanTiledArray(newStrides)
        array.copyInto(newArray)

        transposeRec(array, newArray, strides, newStrides, 0, 0, 0, permutations)

        this.strides = newStrides
        this.array = newArray
        return this
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
