package io.kinference.ndarray.arrays

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.tiled.LongTiledArray
import io.kinference.ndarray.extensions.isScalar
import io.kinference.primitives.types.DataType

open class StringNDArray(var array: Array<String>, strides: Strides) : NDArray {
    constructor(shape: IntArray) : this(Array(shape.reduce(Int::times)) { "" }, Strides(shape))
    constructor(shape: IntArray, init: (Int) -> String) : this(Array<String>(shape.reduce(Int::times), init), Strides(shape))

    override val type: DataType = DataType.ALL

    final override var strides: Strides = strides
        protected set

    override fun singleValue(): String {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array[0]
    }

    override fun allocateNDArray(strides: Strides): MutableNDArray {
        return MutableStringNDArray(Array(strides.linearSize) { "" }, strides)
    }

    override fun view(vararg axes: Int): NDArray {
        TODO("Not yet implemented")
    }

    override fun reshapeView(newShape: IntArray): NDArray {
        return StringNDArray(array, Strides(newShape))
    }

    override fun toMutable(newStrides: Strides): MutableNDArray {
        return MutableStringNDArray(array.copyOf(), newStrides)
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutableStringNDArray(array, strides)
    }

    override fun map(function: PrimitiveToPrimitiveFunction): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun row(row: Int): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun splitHorizontalByBlocks(parts: Int): Array<NDArray> {
        TODO("Not yet implemented")
    }

    companion object {
        fun scalar(value: String): StringNDArray {
            return StringNDArray(arrayOf(value), Strides.EMPTY)
        }
    }
}

class MutableStringNDArray(array: Array<String>, strides: Strides = Strides.EMPTY): StringNDArray(array, strides), MutableNDArray {
    constructor(shape: IntArray) : this(Array(shape.reduce(Int::times)) { "" }, Strides(shape))

    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        TODO("Not yet implemented")
    }

    override fun fill(value: Any, from: Int, to: Int) {
        array.fill(value as String, from ,to)
    }

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        array as StringNDArray
        this.array.fill(array.array[index], from, to)
    }

    override fun reshape(strides: Strides): MutableNDArray {
        this.strides = strides
        return this
    }

    override fun transpose(permutations: IntArray): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun transpose2D(): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun clean() {
        TODO("Not yet implemented")
    }

    override fun viewMutable(vararg axes: Int): MutableNDArray {
        TODO("Not yet implemented")
    }

    companion object {
        fun scalar(value: String): MutableStringNDArray {
            return MutableStringNDArray(arrayOf(value), Strides.EMPTY)
        }
    }
}
