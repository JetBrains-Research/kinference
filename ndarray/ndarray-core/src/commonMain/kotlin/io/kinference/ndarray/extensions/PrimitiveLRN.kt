@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.extensions.utils.calculateInnerShapeSize
import io.kinference.ndarray.extensions.utils.divCeil
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.SpecifyPrimitives
import io.kinference.primitives.types.*
import kotlin.math.pow

fun PrimitiveNDArray.lrn(alpha: Float, beta: Float, bias: Float, size: Int) : PrimitiveNDArray {
    val result = PrimitiveNDArray(shape)
    val resultPointer = PrimitivePointer(result.array)

    val shift = calculateInnerShapeSize(shape)
    val cur = PrimitivePointer(array)
    var curChannel = 0
    var curInnerIndex = 0

    while (cur.isValid()) {
        val lowerBound = maxOf(0, curChannel - (size - 1) / 2)
        val upperBound = minOf(shape[1] - 1, curChannel + ((size - 1) divCeil 2))

        var squareSum = 0.toPrimitive()
        for (i in lowerBound..upperBound)
            squareSum += array[cur.linearIndex + (i - curChannel) * shift] * array[cur.linearIndex + (i - curChannel) * shift]

        resultPointer.setAndIncrement(cur.getAndIncrement() / (bias.toPrimitive() + (alpha.toPrimitive() / size.toPrimitive() * squareSum)).pow(beta.toPrimitive()))

        curInnerIndex++
        if (curInnerIndex == shift) {
            curInnerIndex = 0
            curChannel++
        }
    }

    return result
}

@SpecifyPrimitives(include = [])
private fun PrimitiveType.pow(toPrimitive: PrimitiveType): PrimitiveType {
    return 0.toPrimitive()
}
