package io.kinference.ndarray.arrays

import io.kinference.primitives.types.DataType
import kotlin.jvm.JvmName

open class StringNDArray(var array: Array<String>, strides: Strides) : NDArrayCore {
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
        val linearIndex = strides.offset(index)
        return array[linearIndex]
    }

    override fun getLinear(index: Int): String {
        return array[index]
    }

    override fun view(vararg axes: Int): StringNDArray {
        TODO("Not yet implemented")
    }

    override fun row(i: Int): MutableStringNDArray {
        TODO("Not yet implemented")
    }

    override fun clone(): StringNDArray {
        return StringNDArray(array.copyOf(), Strides(shape))
    }

    override fun close() = Unit

    override fun toMutable(): MutableStringNDArray {
        return MutableStringNDArray(array.copyOf(), strides)
    }

    override fun copyIfNotMutable(): MutableStringNDArray {
        return MutableStringNDArray(array, strides)
    }

    override fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableStringNDArray {
        TODO("Not yet implemented")
    }

    override fun map(function: PrimitiveToPrimitiveFunction) = map(function, MutableStringNDArray(strides))

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableStringNDArray {
        TODO("Not yet implemented")
    }

    override fun concat(others: List<NDArray>, axis: Int): MutableStringNDArray {
        TODO("Not yet implemented")
    }

    override fun expand(shape: IntArray): MutableStringNDArray {
        TODO("Not yet implemented")
    }

    override fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray?): StringNDArray {
        TODO("Not yet implemented")
    }

    override fun nonZero(): LongNDArray {
        TODO("Not yet implemented")
    }

    override fun tile(repeats: IntArray): StringNDArray {
        TODO("Not yet implemented")
    }

    override fun reshape(strides: Strides): StringNDArray {
        require(strides.linearSize == this.strides.linearSize)

        return StringNDArray(array, strides)
    }

    override fun transpose(permutations: IntArray): StringNDArray {
        TODO("Not yet implemented")
    }

    override fun transpose2D(): StringNDArray {
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

        internal fun emptyStringArrFromShape(shape: IntArray) = Array(shape.fold(1, Int::times)) { "" }
        internal fun initStringArr(shape: IntArray, init: (Int) -> String) = Array(shape.fold(1, Int::times), init)
    }
}

class MutableStringNDArray(array: Array<String>, strides: Strides = Strides.EMPTY): StringNDArray(array, strides), MutableNDArrayCore {
    constructor(shape: IntArray) : this(emptyStringArrFromShape(shape), Strides(shape))
    constructor(shape: IntArray, init: (Int) -> String) : this(initStringArr(shape, init), Strides(shape))

    constructor(strides: Strides) : this(emptyStringArrFromShape(strides.shape), strides)
    constructor(strides: Strides, init: (Int) -> String) : this(initStringArr(strides.shape, init), strides)

    override fun set(index: IntArray, value: Any) {
        require(index.size == rank) { "Index size should contain $rank elements, but ${index.size} given" }
        val linearIndex = strides.offset(index)
        array[linearIndex] = value as String
    }

    override fun setLinear(index: Int, value: Any) {
        array[index] = value as String
    }

    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNDArrayCore {
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

    override fun viewMutable(vararg axes: Int): MutableNDArrayCore {
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
