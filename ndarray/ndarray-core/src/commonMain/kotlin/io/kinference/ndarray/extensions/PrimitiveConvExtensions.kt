@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)
package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import io.kinference.ndarray.extensions.utils.*


/***
 * Performs convolution as described by ONNX docs.
 */
fun PrimitiveNDArray.conv(
    w: PrimitiveNDArray,
    b: PrimitiveNDArray?,
    pads: IntArray,
    strides: IntArray,
    dilations: IntArray,
    groups: Int
): PrimitiveNDArray {
    val xShapeWithPads = getShapeWithPads(this.shape, pads)
    val wShapeWithDilations = getShapeWithDilations(w.shape, dilations)

    val resultShape = IntArray(this.shape.size) {
        when (it) {
            0 -> this.shape[0]
            1 -> w.shape[0]
            else -> (xShapeWithPads[it] - wShapeWithDilations[it] + 1) divCeil strides[it - 2]
        }
    }

    val result = MutablePrimitiveNDArray(resultShape) { pos: IntArray -> b?.get(intArrayOf(pos[1])) ?: 0.toPrimitive() }

    val xShrankShape = this.shape - wShapeWithDilations + IntArray(this.shape.size) { 1 }
    xShrankShape[1] = this.shape[1]
    xShrankShape[0] = this.shape[0]
    val xIterator = PrimitiveTensorIterator(this, xShrankShape, pads, strides)
    var totalIterations = 1
    for (i in this.shape.size - 1 downTo 2)
        totalIterations *= (xShrankShape[i] + pads[i - 2] + pads[i + this.shape.size - 4]) divCeil strides[i - 2]


    val resultSize = calculateInnerShapeSize(result.shape)
    val wSize = calculateInnerShapeSize(w.shape)

    val wIterator = PrimitiveTensorIterator(w, w.shape)
    val rawShifts = IntArray(wSize)
    val indexShifts = Array(wSize) {
        val cur = wIterator.next() * dilations
        rawShifts[it] = calculateInnerShift(this.shape, cur)
        cur
    }

    val lastFeature = w.shape[0] / groups
    for (item in 0 until this.shape[0]) {
        for (feature in 0 until lastFeature - 1) {
            this.internalConvolve(w, result, xIterator, lastFeature, item, feature, resultSize, wSize, indexShifts, rawShifts, totalIterations)
            xIterator.resetPos1()
        }
        this.internalConvolve(w, result, xIterator, lastFeature, item, lastFeature - 1, resultSize, wSize, indexShifts, rawShifts, totalIterations)
    }

    return result
}

/***
 * Performs convolution with a fixed item in the batch and a fixed outChannel.
 */
private fun PrimitiveNDArray.internalConvolve(
    w: PrimitiveNDArray,
    result: MutablePrimitiveNDArray,
    xIterator: PrimitiveTensorIterator,
    outChannels: Int,
    item: Int,
    feature: Int,
    resultSize: Int,
    wSize: Int,
    indexShift: Array<IntArray>,
    rawShifts: IntArray,
    innerIterations: Int
) {
    var outShift = 0
    var wChannel = 0
    for (i in 0 until this.shape[1]) {
        val outFeature = feature + outShift * outChannels
        val resultPointer = PrimitivePointer(result.array, calculateSignificantShift(item, outFeature, result.shape[1], resultSize))

        val wStartIndex = calculateSignificantShift(outFeature, wChannel, w.shape[1], wSize)
        repeat(innerIterations) {
            val wPointer = PrimitivePointer(w.array, wStartIndex)

            xIterator.next()

            repeat(wSize) {
                resultPointer.add(xIterator.getShifted(indexShift[it], rawShifts[it]) * wPointer.getAndIncrement())
            }

            resultPointer.increment()
        }

        wChannel++
        if (wChannel == w.shape[1]) {
            wChannel = 0
            outShift++
        }
    }
}

private fun PrimitivePointer.add(value: PrimitiveType) {
    currentBlock[indexInBlock] += value
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
    private var current = IntArray(iterator.size) { iterator[it].next() }
    private var firstIteration = true

    private val jumps = IntArray(shape.size)
    private var rawCurrent = 0

    init {
        jumps[jumps.lastIndex] = 1
        rawCurrent += current[current.lastIndex]
        for (i in (jumps.lastIndex - 1) downTo 0) {
            jumps[i] = jumps[i + 1] * x.shape[i + 1]
            rawCurrent += jumps[i] * current[i]
        }
    }

    private fun isInPadding(index: IntArray): Boolean {
        if (index.any { i -> i < 0 })
            return true
        index.forEachIndexed { ind, i -> if (i >= x.shape[ind]) return true }
        return false
    }

    override fun hasNext(): Boolean {
        return true
    }

    fun resetPos1() {
        iterator[0].decrement()
        rawCurrent -= jumps[0]
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

    fun getShifted(indexShift: IntArray, rawShift: Int): PrimitiveType {
        if (isInPadding(current + indexShift))
            return 0.toPrimitive()
        return x.array[rawCurrent + rawShift]
    }
}
