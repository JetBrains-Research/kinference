package io.kinference.ndarray.arrays

import io.kinference.ndarray.Strides
import io.kinference.ndarray.extensions.isScalar
import io.kinference.ndarray.extensions.ndIndexed
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import kotlin.jvm.JvmName

private fun emptyStringArrFromShape(shape: IntArray) = Array(shape.fold(1, Int::times)) { "" }
private fun initStringArr(shape: IntArray, init: (Int) -> String) = Array(shape.fold(1, Int::times), init)

open class StringNDArray(var array: Array<String>, strides: Strides) : NDArray {
    constructor(shape: IntArray) : this(emptyStringArrFromShape(shape), Strides(shape))
    constructor(shape: IntArray, init: (Int) -> String) : this(initStringArr(shape, init), Strides(shape))

    constructor(strides: Strides) : this(emptyStringArrFromShape(strides.shape), strides)
    constructor(strides: Strides, init: (Int) -> String) : this(initStringArr(strides.shape, init), strides)

    override val type: DataType = DataType.ALL

    final override var strides: Strides = strides
        protected set

    override fun singleValue(): String {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array[0]
    }

    override fun get(index: IntArray): String {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.strides.reduceIndexed { idx, acc, i -> acc + i * index[idx] }
        return array[linearIndex]
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

    override fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun row(row: Int): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun concatenate(others: List<NDArray>, axis: Int): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun expand(shape: IntArray): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun pad(pads: Array<Pair<Int, Int>>, mode: String, constantValue: NDArray?): NDArray {
        TODO("Not yet implemented")
    }

    override fun nonZero(): LongNDArray {
        TODO("Not yet implemented")
    }

    override fun tile(repeats: IntArray): NDArray {
        TODO("Not yet implemented")
    }

    override fun reshape(strides: Strides): StringNDArray {
        require(strides.linearSize == this.strides.linearSize)

        return StringNDArray(array, strides)
    }

    override fun transpose(permutations: IntArray): NDArray {
        TODO("Not yet implemented")
    }

    override fun transpose2D(): NDArray {
        TODO("Not yet implemented")
    }

    companion object {
        fun scalar(value: String): StringNDArray {
            return StringNDArray(arrayOf(value), Strides.EMPTY)
        }

        operator fun invoke(strides: Strides, init: (IntArray) -> String): StringNDArray {
            return MutableStringNDArray(strides.shape).apply { this.ndIndexed { this[it] = init(it) } }
        }

        operator fun invoke(shape: IntArray, init: (IntArray) -> String): StringNDArray {
            return invoke(Strides(shape), init)
        }

        operator fun invoke(vararg shape: Int): StringNDArray {
            return StringNDArray(emptyStringArrFromShape(shape), Strides(shape))
        }

        @JvmName("invokeVarArg")
        operator fun invoke(vararg shape: Int, init: (Int) -> String): StringNDArray {
            return StringNDArray(Array(shape.fold(1, Int::times), init), Strides(shape))
        }

        @JvmName("invokeNDVarArg")
        operator fun invoke(vararg shape: Int, init: (IntArray) -> String): StringNDArray {
            return invoke(shape, init)
        }
    }
}

class MutableStringNDArray(array: Array<String>, strides: Strides = Strides.EMPTY): StringNDArray(array, strides), MutableNDArray {
    constructor(shape: IntArray) : this(emptyStringArrFromShape(shape), Strides(shape))
    constructor(shape: IntArray, init: (Int) -> String) : this(initStringArr(shape, init), Strides(shape))

    constructor(strides: Strides) : this(emptyStringArrFromShape(strides.shape), strides)
    constructor(strides: Strides, init: (Int) -> String) : this(initStringArr(strides.shape, init), strides)

    override fun set(index: IntArray, value: Any) {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.strides.reduceIndexed { idx, acc, i -> acc + i * index[idx] }
        array[linearIndex] = value as String
    }

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

    /*override fun transpose(permutations: IntArray): MutableNDArray {
        TODO("Not yet implemented")
    }*/

    /*override fun transpose2D(): MutableNDArray {
        TODO("Not yet implemented")
    }*/

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

        operator fun invoke(strides: Strides, init: (IntArray) -> String): MutableStringNDArray {
            return MutableStringNDArray(strides.shape).apply { this.ndIndexed { this[it] = init(it) } }
        }

        operator fun invoke(shape: IntArray, init: (IntArray) -> String): MutableStringNDArray {
            return MutableStringNDArray(shape).apply { this.ndIndexed { this[it] = init(it) }  }
        }

        operator fun invoke(vararg shape: Int): MutableStringNDArray {
            return MutableStringNDArray(emptyStringArrFromShape(shape), Strides(shape))
        }

        @JvmName("invokeVarArg")
        operator fun invoke(vararg shape: Int, init: (Int) -> String): MutableStringNDArray {
            return MutableStringNDArray(Array(shape.fold(1, Int::times), init), Strides(shape))
        }

        @JvmName("invokeNDVarArg")
        operator fun invoke(vararg shape: Int, init: (IntArray) -> String): MutableStringNDArray {
            return invoke(shape, init)
        }
    }
}
