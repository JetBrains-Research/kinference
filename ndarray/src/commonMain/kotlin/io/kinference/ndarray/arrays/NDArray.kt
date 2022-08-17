package io.kinference.ndarray.arrays

import io.kinference.ndarray.Strides
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.primitives.types.DataType
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.math.min

interface PrimitiveToPrimitiveFunction

interface NDArray {
    val type: DataType

    val strides: Strides

    val linearSize: Int
        get() = strides.linearSize
    val shape: IntArray
        get() = strides.shape
    val rank: Int
        get() = shape.size

    operator fun get(index: IntArray): Any
    operator fun set(index: IntArray, value: Any)

    fun singleValue(): Any

    fun view(vararg axes: Int): NDArray
    @Deprecated(message = "Use reshape() instead", replaceWith = ReplaceWith("reshape()"))
    fun reshapeView(newShape: IntArray): NDArray
    fun reshape(strides: Strides): NDArray
    fun reshape(shape: IntArray): NDArray = reshape(Strides(shape))
    fun toMutable(newStrides: Strides = strides): MutableNDArray

    fun copyIfNotMutable(): MutableNDArray

    fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNDArray
    fun map(function: PrimitiveToPrimitiveFunction) = map(function, allocateNDArray(type, strides))

    fun row(row: Int): MutableNDArray
    fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNDArray
    fun expand(shape: IntArray): MutableNDArray
    fun pad(pads: Array<Pair<Int, Int>>, mode: String, constantValue: NDArray?): NDArray
    fun nonZero(): LongNDArray

    fun concatenate(others: List<NDArray>, axis: Int): MutableNDArray
    fun tile(repeats: IntArray): NDArray
    fun transpose(permutations: IntArray): NDArray
    fun transpose2D(): NDArray
}

interface MutableNDArray : NDArray {
    fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNDArray

    fun copyFrom(offset: Int, other: NDArray, startInOther: Int = 0, endInOther: Int = min(other.linearSize, linearSize))
    fun fill(value: Any, from: Int = 0, to: Int = linearSize)

    fun fillByArrayValue(array: NDArray, index: Int, from: Int = 0, to: Int = linearSize)

    fun clean()

    fun viewMutable(vararg axes: Int): MutableNDArray
}

interface NumberNDArray : NDArray {
    override fun toMutable(newStrides: Strides): MutableNumberNDArray

    override fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNumberNDArray
    override fun map(function: PrimitiveToPrimitiveFunction) = map(function, allocateNDArray(type, strides))

    override fun row(row: Int): MutableNumberNDArray
    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArray

    fun min(): Any
    fun max(): Any
    fun max(axis: Int, keepDims: Boolean): NumberNDArray
    fun sum(): Any
    fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutableNumberNDArray
    fun erf(): NumberNDArray


    operator fun plus(other: NumberNDArray): MutableNumberNDArray
    fun plus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    operator fun minus(other: NumberNDArray): MutableNumberNDArray
    fun minus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    operator fun times(other: NumberNDArray): MutableNumberNDArray
    fun times(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    operator fun div(other: NumberNDArray): MutableNumberNDArray
    fun div(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    fun dot(other: NumberNDArray, destination: MutableNumberNDArray, coroutineContext: CoroutineContext = EmptyCoroutineContext): MutableNumberNDArray
    fun dotTransposedWithAlpha(alpha: Double, other: NumberNDArray, destination: MutableNumberNDArray, coroutineContext: CoroutineContext = EmptyCoroutineContext): MutableNumberNDArray

    fun gemm(m: Int, n: Int, k: Int, alpha: Double, lda: Int, b: NDArray, ldb: Int, beta: Double, c: MutableNDArray,
             ldc: Int, aOffset: Int, bOffset: Int, cOffset: Int, transposeA: Boolean = false, transposeB: Boolean = false) : MutableNDArray

    fun argmax(axis: Int = 0, keepDims: Boolean = true, selectLastIndex: Boolean = false): IntNDArray
    fun reduceSum(axes: IntArray, keepDims: Boolean = true): NDArray
    fun topK(axis: Int, k: Int, largest: Boolean, sorted: Boolean): Pair<NumberNDArray, LongNDArray>

    override fun reshape(strides: Strides): NumberNDArray
    override fun reshape(shape: IntArray): NumberNDArray = reshape(Strides(shape))

    override fun view(vararg axes: Int): NumberNDArray

    override fun transpose(permutations: IntArray): NumberNDArray
}

interface MutableNumberNDArray : MutableNDArray, NumberNDArray {
    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNumberNDArray

    override fun viewMutable(vararg axes: Int): MutableNumberNDArray

    operator fun plusAssign(other: NDArray)
    operator fun minusAssign(other: NDArray)
    operator fun timesAssign(other: NDArray)
    operator fun divAssign(other: NDArray)
}
