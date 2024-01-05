package io.kinference.ndarray.arrays

import io.kinference.primitives.types.DataType
import io.kinference.utils.ArrayUsageMarker
import io.kinference.utils.Closeable
import kotlin.math.min

enum class PadMode {
    CONSTANT,
    EDGE,
    REFLECT
}

/**
 * This interface defines base operations available for each NDArray implementation.
 */
interface NDArray : Closeable {
    /**
     * Elements' data type
     */
    val type: DataType

    val strides: Strides

    val linearSize: Int
        get() = strides.linearSize

    val shape: IntArray
        get() = strides.shape

    val rank: Int
        get() = shape.size

    /**
     * Gets an element by ND index. Index length must be equal to [rank].
     */
    operator fun get(index: IntArray): Any

    /**
     * Gets an element by linear index. Index must be less than [linearSize].
     */
    fun getLinear(index: Int): Any

    /**
     * Converts an array to a single value of a type specified in [type]. [linearSize] of the array must be 1.
     */
    fun singleValue(): Any

    fun markOutput(marker: ArrayUsageMarker)

    /**
     * Creates an array containing the same data with the new strides.
     */
    suspend fun reshape(strides: Strides): NDArray

    /**
     * Creates an array containing the same data with the new shape.
     */
    suspend fun reshape(shape: IntArray): NDArray = reshape(Strides(shape))

    /**
     * Copies current array data to a new instance of [MutableNDArray].
     */
    fun toMutable(): MutableNDArray

    /**
     * Removes single-dimensional entries from the array shape on specified in [axes] positions.
     *
     * @param axes positions of axes to remove. If it is empty, removes all single-dimensional entries from shape.
     */
    suspend fun squeeze(vararg axes: Int): NDArray

    /**
     * Inserts single-dimensional entries to the array shape on the positions specified in [axes].
     */
    suspend fun unsqueeze(vararg axes: Int): NDArray

    /**
     * Joins an array with a sequence of [others] along a new [axis].
     * Each array must have the same shape.
     */
    suspend fun stack(others: List<NDArray>, axis: Int): NDArray

    /**
     * Joins an array with a sequence of [others] along existing [axis].
     * Each array must have the same shape (except in the dimension corresponding to [axis])
     */
    suspend fun concat(others: List<NDArray>, axis: Int): NDArray

    /**
     * Splits an array into multiple sub-arrays of length [parts] along [axis].
     */
    suspend fun split(parts: Int, axis: Int = 0): List<NDArray>

    /**
     * Split an array into multiple sub-arrays on given indices [split] along [axis].
     */
    suspend fun split(split: IntArray, axis: Int = 0): List<NDArray>

    /**
     * Gathers values along given [axis] specified by [indices].
     */
    suspend fun gather(indices: NDArray, axis: Int = 0, batchDims: Int = 0): NDArray

    /**
     * If a current array is not an instance of [MutableNDArray], creates [MutableNDArray] with deep copy of array data.
     * In any other case it just creates a new instance of [MutableNDArray] with the same data.
     */
    fun copyIfNotMutable(): MutableNDArray

    fun clone(): NDArray

    /**
     * Makes a slice of an array along multiple axes with elements from [starts] to [ends] with [steps].
     *
     * @param starts start slice indices along each axis.
     * @param ends end slice indices along each axis.
     * @param steps step between elements along each axis.
     */
    suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): NDArray

    /**
     * Broadcasts an array to the given shape.
     */
    suspend fun expand(shape: IntArray): MutableNDArray

    /**
     * Pads an array with given [pads] values (pairs of start and end pad values for each axis).
     * [constantValue] is optional and used only for [PadMode.CONSTANT]
     */
    suspend fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray? = null): NDArray

    /**
     * Computes indices of the elements that are non-zero.
     * These indices are represented as a 2D array of shape [[rank], numNonZeros],
     * where each row contains the indices of non-zero elements in corresponding dimension.
     */
    suspend fun nonZero(): NumberNDArray

    /**
     * Constructs a new array by repeating a current array the number of times specified in [repeats] along each axis.
     * [repeats] size should be equal to [rank].
     */
    suspend fun tile(repeats: IntArray): NDArray

    /**
     * Permutes the axes of an array in the given order specified in [permutations]
     * [permutations] size should be equal to [rank].
     */
    suspend fun transpose(permutations: IntArray): NDArray

    /**
     * Transposes an array if it is a matrix (Could be more time-efficient for matrices than common [transpose])
     */
    suspend fun transpose2D(): NDArray

    /**
     * Returns a view on an array using specified indices.
     */
    fun view(vararg axes: Int): NDArray
}

/**
 * This interface defines mutable operations available for NDArrays.
 */
interface MutableNDArray : NDArray {
    /**
     * Sets an element by ND index. Index length must be equal to [rank].
     */
    operator fun set(index: IntArray, value: Any)

    /**
     * Sets an element by linear index. Index must be less than [linearSize].
     */
    fun setLinear(index: Int, value: Any)

    /**
     * Returns a mutable view on an array using specified indices.
     */
    fun viewMutable(vararg axes: Int): MutableNDArray

    /**
     * Copies a block of data from [other] array to the current array.
     *
     * @param offset start index in the current array to copy new data.
     * @param other source data array.
     * @param startInOther inclusive start index of the copy range in [other].
     * @param endInOther exclusive end index of the copy range in [other].
     */
    fun copyFrom(offset: Int, other: NDArray, startInOther: Int = 0, endInOther: Int = min(other.linearSize, linearSize))

    /**
     * Fills the array with scalar [value] within specified range of linear indices.
     * @param [from] inclusive start index of the fill range.
     * @param [to] exclusive end index of the fill range.
     */
    fun fill(value: Any, from: Int = 0, to: Int = linearSize)

    /**
     * Fills the array with [array] value within specified range of linear indices.
     * @param [from] inclusive start index of the fill range.
     * @param [to] exclusive end index of the fill range.
     */
    fun fillByArrayValue(array: NDArray, index: Int, from: Int = 0, to: Int = linearSize)

    /**
     * Fills the array with zero-like values.
     */
    fun clean()
}

/**
 * This interface defines base operations available for NDArrays containing numeric data.
 */
interface NumberNDArray : NDArray {
    override fun toMutable(): MutableNumberNDArray

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArray

    suspend fun abs(): NumberNDArray

    /**
     * Computes the minimum value of an array.
     */
    suspend fun min(): Any

    /**
     * Computes the minimum value along an axis.
     * If [keepDims] flag is true, target axis will be set to 1. In any other case, it will be reduced.
     */
    suspend fun min(axis: Int, keepDims: Boolean): NumberNDArray

    /**
     * Computes maximum value of an array.
     */
    suspend fun max(): Any

    /**
     * Computes the minimum value along an axis.
     * If [keepDims] flag is true, target axis will be set to 1. In any other case, it will be reduced.
     */
    suspend fun max(axis: Int, keepDims: Boolean): NumberNDArray

    /**
     * Computes a sum of all array elements.
     */
    suspend fun sum(): Any

    /**
     * Computes the cumulative sum of array elements along the given [axis].
     *
     * @param axis axis along which the cumulative sum is computed.
     * @param exclusive if this flag is true, an exclusive sum (top element is not included) is computed.
     * @param reverse if this flag is true, sums are made in a reverse direction.
     */
    suspend fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutableNumberNDArray

    /**
     * Computes error function element-wise.
     */
    suspend fun erf(): NumberNDArray

    /**
     * Computes softmax along the given axis.
     */
    suspend fun softmax(axis: Int = 0): NumberNDArray

    /**
     * Computes logSoftmax along the given axis.
     */
    suspend fun logSoftmax(axis: Int = 0): NumberNDArray

    /**
     * Element-wise addition (supports broadcasting).
     */
    suspend operator fun plus(other: NumberNDArray): MutableNumberNDArray

    /**
     * Element-wise subtraction (supports broadcasting).
     */
    suspend operator fun minus(other: NumberNDArray): MutableNumberNDArray

    /**
     * Element-wise multiplication (supports broadcasting).
     */
    suspend operator fun times(other: NumberNDArray): MutableNumberNDArray

    /**
     * Element-wise division (supports broadcasting).
     */
    suspend operator fun div(other: NumberNDArray): MutableNumberNDArray

    /**
     * Computes dot product. Supports coroutines to speed up the computations.
     */
    suspend fun dot(other: NumberNDArray): MutableNumberNDArray

    /**
     * Computes matmul. Supports coroutines to speed up the computations.
     */
    suspend fun matmul(other: NumberNDArray): MutableNumberNDArray

    /**
     * Computes the indices of the maximum values along the given axis.
     *
     * @param axis the axis along which the indices are computed.
     * @param keepDims if this flag is true, the target axis will be set to 1. In any other case, it will be reduced.
     * @param selectLastIndex this flag determines whether the first occurrence of the maximum index should be selected,
     *                        or the last one (True -- last, False -- first).
     */
    suspend fun argmax(axis: Int = 0, keepDims: Boolean = true, selectLastIndex: Boolean = false): NumberNDArray

    /**
     * Computes the indices of the minimum values along the given axis.
     *
     * @param axis the axis along which the indices are computed.
     * @param keepDims if this flag is true, the target axis will be set to 1. In any other case, it will be reduced.
     * @param selectLastIndex this flag determines whether the first occurrence of the maximum index should be selected,
     *                        or the last one (True -- last, False -- first).
     */
    suspend fun argmin(axis: Int = 0, keepDims: Boolean = true, selectLastIndex: Boolean = false): NumberNDArray

    /**
     * Computes the sum of array elements along provided axes.
     * If [keepDims] flag is true, target axes will be set to 1. In any other case, they will be reduced.
     */
    suspend fun reduceSum(axes: IntArray, keepDims: Boolean = true): NDArray

    /**
     * Retrieves the top-k largest (or smallest) elements along the provided axis.
     * Function returns two arrays. The first array contains top-k elements and the second one -- their indices in the original array.
     *
     * @param axis the axis along which to search for the elements.
     * @param k value representing the number of elements to retrieve.
     * @param largest this flag determines whether to return the top-k largest or smallest elements.
     * @param sorted this flag determines whether to sort elements.
     */
    suspend fun topK(axis: Int, k: Int, largest: Boolean, sorted: Boolean): Pair<NumberNDArray, NumberNDArray>

    override suspend fun reshape(strides: Strides): NumberNDArray
    override suspend fun reshape(shape: IntArray): NumberNDArray = reshape(Strides(shape))

    override suspend fun transpose(permutations: IntArray): NumberNDArray
    override suspend fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray?): NumberNDArray

    override fun view(vararg axes: Int): NumberNDArray
}

/**
 * This interface defines mutable operations available for NDArrays containing numeric data.
 */
interface MutableNumberNDArray : MutableNDArray, NumberNDArray {
    /**
     * Assigns the element-wise addition result to the current array.
     * Broadcasting is only available for [other] array.
     */
    suspend operator fun plusAssign(other: NumberNDArray)

    /**
     * Assigns the element-wise subtraction result to the current array.
     * Broadcasting is only available for [other] array.
     */
    suspend operator fun minusAssign(other: NumberNDArray)

    /**
     * Assigns the element-wise multiplication result to the current array.
     * Broadcasting is only available for [other] array.
     */
    suspend operator fun timesAssign(other: NumberNDArray)

    /**
     * Assigns the element-wise division result to the current array.
     * Broadcasting is only available for [other] array.
     */
    suspend operator fun divAssign(other: NumberNDArray)

    override fun viewMutable(vararg axes: Int): MutableNumberNDArray
}

operator fun NDArray.get(vararg index: Int) = this[index]
operator fun MutableNDArray.set(vararg index: Int, value: Any) { this[index] = value }
