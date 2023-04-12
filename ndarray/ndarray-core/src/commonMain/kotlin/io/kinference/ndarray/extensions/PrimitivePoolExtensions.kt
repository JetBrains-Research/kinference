@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE, DataType.UBYTE, DataType.BYTE)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.LongPointer
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.extensions.utils.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*

fun PrimitiveNDArray.maxPool(
    kernel: IntArray,
    pads: IntArray,
    strides: IntArray,
    dilations: IntArray,
    ceilMode: Int,
    storageOrder: Int,
    minValue: PrimitiveType = PrimitiveType.MIN_VALUE
): List<NumberNDArrayCore> {
    val xShapeWithPads = getShapeWithPads(shape, pads)
    val kShapeWithDilations = IntArray(kernel.size + 2) { i -> if (i < 2) 1 else (kernel[i - 2] - 1) * dilations[i - 2] + 1 }

    val resultShape = IntArray(shape.size) {
        when (it) {
            0, 1 -> shape[it]
            else -> {
                if (ceilMode == 1)
                    ((xShapeWithPads[it] - kShapeWithDilations[it]) divCeil strides[it - 2]) + 1
                else
                    ((xShapeWithPads[it] - kShapeWithDilations[it]) / strides[it - 2]) + 1
            }
        }
    }

    if (ceilMode == 1) {
        for (i in kernel.size until 2 * kernel.size)
            pads[i]++
    }

    val result = MutablePrimitiveNDArray(resultShape)

    val xShrankShape = shape - kShapeWithDilations + IntArray(shape.size) { 1 }
    val xIterator = PrimitiveTensorIterator(this, xShrankShape, pads, strides)

    val w = PrimitiveNDArray(IntArray(kernel.size + 2) { i -> if (i < 2) 1 else kernel[i - 2] })
    val wIterator = PrimitiveTensorIterator(w, w.shape)
    val wSize = calculateInnerShapeSize(w.shape)
    val rawShifts = IntArray(wSize)
    val indexShifts = Array(wSize) {
        val cur = wIterator.next() * dilations
        rawShifts[it] = calculateInnerShift(shape, cur)
        cur
    }

    val indices = if (storageOrder == -1) MutableLongNDArray.scalar(0) else MutableLongNDArray(resultShape)

    val resultPointer = PrimitivePointer(result.array)
    val indicesPointer = LongPointer(indices.array)
    while (xIterator.hasNext()) {
        var res: PrimitiveType = minValue
        xIterator.next()
        indexShifts.forEachIndexed { i: Int, index: IntArray ->
            val cur = xIterator.getShifted(index, rawShifts[i], minValue)
            if (res < cur) {
                res = cur
                when (storageOrder) {
                    0 -> indicesPointer.set((xIterator.rawCurrent + rawShifts[i]).toLong())
                    1 -> indicesPointer.set(calcRawIndexWithColMajor(xIterator.current, shape) + calcRawIndexWithColMajor(index, shape))
                }
            }
        }
        indicesPointer.increment()
        resultPointer.setAndIncrement(res)
    }

    return listOf(result, indices)
}

@SpecifyPrimitives(include = [])
fun calcRawIndexWithColMajor(index: IntArray, shape: IntArray): Long {
    var shift = 1L
    var result = 0L
    for (i in 2 .. index.lastIndex) {
        result += index[i] * shift
        shift *= shape[i]
    }
    result += index[1] * shift
    shift *= shape[1]
    result += index[0] * shift
    return result
}

// Suppress error in line `res = maxOf(res, xIterator.getShifted(index, rawShifts[i]))`
@SpecifyPrimitives(include = [])
fun maxOf(a: PrimitiveType, b: PrimitiveType): PrimitiveType {
    return 0.toPrimitive()
}

/***
 * Provides the ability to iterate over a tensor, maintaining the n-dimensional and raw indexes.
 */
@GenerateNameFromPrimitives
internal class PrimitiveTensorIterator(
    private val x: PrimitiveNDArray,
    shape: IntArray,
    pads: IntArray = IntArray(shape.size * 2) { 0 },
    strides: IntArray = IntArray(shape.size) { 1 }
) : Iterator<IntArray> {
    private val indexes: List<IntProgression> = MutableList(shape.size) { i ->
        when (i) {
            0, 1 -> 0 until shape[i] step 1
            else -> 0 - pads[i - 2] until shape[i] + pads[i + shape.size - 4] step strides[i - 2]
        }
    }

    private val iterator = indexes.map { ProgressionIterator(it) }
    var current = IntArray(iterator.size) { iterator[it].next() }
    private var firstIteration = true

    private val jumps = IntArray(shape.size)
    var rawCurrent = 0

    init {
        jumps[jumps.lastIndex] = 1
        rawCurrent += current[current.lastIndex]
        for (i in (jumps.lastIndex - 1) downTo 0) {
            jumps[i] = jumps[i + 1] * x.shape[i + 1]
            rawCurrent += jumps[i] * current[i]
        }
    }

    fun isInPadding(index: IntArray): Boolean {
        if (index.any { i -> i < 0 })
            return true
        index.forEachIndexed { ind, i -> if (i >= x.shape[ind]) return true }
        return false
    }

    override fun hasNext(): Boolean {
        return iterator.any { it.hasNext() }
    }

    override fun next(): IntArray {
        if (firstIteration) {
            firstIteration = false
            return current
        }

        var currentIndex = iterator.lastIndex
        while (!iterator[currentIndex].hasNext())
            currentIndex--

        current[currentIndex] = iterator[currentIndex].next()
        rawCurrent += jumps[currentIndex] * indexes[currentIndex].step
        for (i in currentIndex + 1..iterator.lastIndex) {
            iterator[i].reset()
            val next = iterator[i].next()
            rawCurrent -= jumps[i] * (current[i] - next)
            current[i] = next
        }

        return current
    }

    fun getShifted(indexShift: IntArray, rawShift: Int, default: PrimitiveType = 0.toPrimitive()): PrimitiveType {
        current.add(indexShift)
        val inPadding = isInPadding(current)
        current.subtract(indexShift)
        if (inPadding)
            return default
        return x.array[rawCurrent + rawShift]
    }
}

