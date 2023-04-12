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
    var resultIndex = 0

    val shift = calculateInnerShapeSize(shape)
    var curIndex = 0

    val halfSize = (size - 1) / 2
    val halfSizeCeil = (size - 1) divCeil 2

    repeat (shape[0]) {
        repeat(shift) {
            var lowerBound = 0
            var upperBound = minOf(shape[1] - 1, halfSizeCeil)
            var squareSum = 0.toPrimitive()

            val startCurIndex = curIndex
            for (channel in lowerBound..upperBound) {
                val value = array[startCurIndex + channel * shift]
                squareSum += value * value
            }

            result.array[resultIndex] = array[curIndex] / (bias.toPrimitive() + (alpha.toPrimitive() / size.toPrimitive() * squareSum)).pow(beta.toPrimitive())

            for (channel in 1 until shape[1]) {
                resultIndex += shift
                curIndex += shift

                val newLowerBound = channel - halfSize
                val newUpperBound = minOf(shape[1] - 1, channel + halfSizeCeil)

                if (newLowerBound > lowerBound) {
                    val value = array[startCurIndex + lowerBound * shift]
                    squareSum -= value * value
                    lowerBound = newLowerBound
                }

                if (newUpperBound > upperBound) {
                    val value = array[startCurIndex + newUpperBound * shift]
                    squareSum += value * value
                    upperBound = newUpperBound
                }

                result.array[resultIndex] = array[curIndex] / (bias.toPrimitive() + (alpha.toPrimitive() / size.toPrimitive() * squareSum)).pow(beta.toPrimitive())
            }

            resultIndex -= (shape[1] - 1) * shift
            curIndex -= (shape[1] - 1) * shift

            curIndex++
            resultIndex++
        }

        curIndex -= shift
        resultIndex -= shift

        curIndex += shape[1] * shift
        resultIndex += shape[1] * shift
    }

    return result
}

@SpecifyPrimitives(include = [])
private fun PrimitiveType.pow(toPrimitive: PrimitiveType): PrimitiveType {
    return 0.toPrimitive()
}
