package io.kinference.ndarray.arrays

import io.kinference.ndarray.Strides
import io.kinference.primitives.types.DataType
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

    fun singleValue(): Any

    fun allocateNDArray(strides: Strides): MutableNDArray

    fun view(vararg axes: Int): NDArray
    fun reshapeView(newShape: IntArray): NDArray
    fun toMutable(newStrides: Strides = strides): MutableNDArray

    fun copyIfNotMutable(): MutableNDArray

    fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNDArray
    fun map(function: PrimitiveToPrimitiveFunction) = map(function, allocateNDArray(strides))

    fun row(row: Int): MutableNDArray
    fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNDArray
    fun expand(shape: IntArray): MutableNDArray
    fun pad(pads: Array<Pair<Int, Int>>, mode: String, constantValue: NDArray?): NDArray
    fun nonZero(): LongNDArray
}

interface MutableNDArray : NDArray {
    fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNDArray

    fun copyFrom(offset: Int, other: NDArray, startInOther: Int = 0, endInOther: Int = min(other.linearSize, linearSize))
    fun fill(value: Any, from: Int = 0, to: Int = linearSize)
    fun fillByArrayValue(array: NDArray, index: Int, from: Int = 0, to: Int = linearSize)

    fun reshape(strides: Strides): MutableNDArray
    fun reshape(shape: IntArray): MutableNDArray = reshape(Strides(shape))
    fun transpose(permutations: IntArray): MutableNDArray
    fun transpose2D(): MutableNDArray

    fun clean()

    fun viewMutable(vararg axes: Int): MutableNDArray
}

interface NumberNDArray : NDArray {
    override fun allocateNDArray(strides: Strides): MutableNumberNDArray

    fun dequantize(zeroPoint: NDArray?, scale: NDArray, axis: Int? = null): NDArray

    override fun toMutable(newStrides: Strides): MutableNumberNDArray

    override fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNumberNDArray
    override fun map(function: PrimitiveToPrimitiveFunction) = map(function, allocateNDArray(strides))

    override fun row(row: Int): MutableNumberNDArray
    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArray

    fun min(): Any
    fun max(): Any
    fun sum(): Any
    fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutableNumberNDArray
    fun withZeroPoint(zeroPoint: NumberNDArray): IntNDArray

    fun erfFor(value: Any): Any

    operator fun plus(other: NumberNDArray): MutableNumberNDArray
    fun plus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    operator fun minus(other: NumberNDArray): MutableNumberNDArray
    fun minus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    operator fun times(other: NumberNDArray): MutableNumberNDArray
    fun times(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    operator fun div(other: NumberNDArray): MutableNumberNDArray
    fun div(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    fun dot(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray
    fun dotTransposedWithAlpha(alpha: Double, other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray

    fun gemm(m: Int, n: Int, k: Int, alpha: Double, lda: Int, b: NDArray, ldb: Int, beta: Double, c: MutableNDArray,
             ldc: Int, aOffset: Int, bOffset: Int, cOffset: Int, transposeA: Boolean = false, transposeB: Boolean = false) : MutableNDArray

    fun argmax(axis: Int = 0, keepDims: Boolean = true, selectLastIndex: Boolean = false): IntNDArray
    fun reduceSum(axes: IntArray, keepDims: Boolean = true): NDArray
    fun topK(axis: Int, k: Int, largest: Boolean, sorted: Boolean): Pair<NumberNDArray, LongNDArray>

    override fun view(vararg axes: Int): NumberNDArray
}

interface MutableNumberNDArray : MutableNDArray, NumberNDArray {
    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNumberNDArray

    override fun reshape(strides: Strides): MutableNumberNDArray
    override fun reshape(shape: IntArray): MutableNumberNDArray = reshape(Strides(shape))
    override fun transpose(permutations: IntArray): MutableNumberNDArray
    override fun viewMutable(vararg axes: Int): MutableNumberNDArray

    fun erf(): MutableNumberNDArray

    operator fun plusAssign(other: NDArray)
    operator fun minusAssign(other: NDArray)
    operator fun timesAssign(other: NDArray)
    operator fun divAssign(other: NDArray)
}
