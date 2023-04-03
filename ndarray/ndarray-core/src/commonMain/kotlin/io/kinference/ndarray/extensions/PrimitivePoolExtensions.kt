@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)
package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.extensions.utils.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.SpecifyPrimitives
import io.kinference.primitives.types.*

fun PrimitiveNDArray.maxPool(kernel: IntArray, pads: IntArray, strides: IntArray, dilations: IntArray, ceilMode: Int) : PrimitiveNDArray {
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

    val resultPointer = PrimitivePointer(result.array)
    val min = (-1000).toPrimitive()
    while (xIterator.hasNext()) {
        var res: PrimitiveType = min
        xIterator.next()
        indexShifts.forEachIndexed { i: Int, index: IntArray ->
            res = maxOf(res, xIterator.getShifted(index, rawShifts[i], min))
            //println(xIterator.getShifted(index, rawShifts[i], PrimitiveType.MIN_VALUE))
        }
        resultPointer.setAndIncrement(res)
    }

    return result
}

// Suppress error in line `res = maxOf(res, xIterator.getShifted(index, rawShifts[i]))`
@SpecifyPrimitives(include = [])
fun maxOf(a: PrimitiveType, b: PrimitiveType): PrimitiveType {
    return 0.toPrimitive()
}
