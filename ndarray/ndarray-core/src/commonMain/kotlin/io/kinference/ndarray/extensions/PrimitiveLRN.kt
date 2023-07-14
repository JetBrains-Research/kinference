@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.stubs.pow
import io.kinference.ndarray.extensions.utils.calculateInnerShapeSize
import io.kinference.ndarray.extensions.utils.divCeil
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*
import kotlin.math.pow

suspend fun PrimitiveNDArray.lrn(alpha: Float, beta: Float, bias: Float, size: Int) : PrimitiveNDArray {
    val result = PrimitiveNDArray(shape)

    val inputSize = calculateInnerShapeSize(shape)

    val halfSize = (size - 1) / 2
    val halfSizeCeil = (size - 1) divCeil 2

    val input = this.reshape(intArrayOf(shape[0], shape[1], inputSize))

    repeat (shape[0]) { batch ->
        val currentImage = input.view(batch)
        repeat(inputSize) { linearIndex ->
            val valuesAcrossChannel = currentImage.gather(IntNDArray.scalar(linearIndex), axis = 1)
            var lowerBound = 0
            var upperBound = minOf(shape[1] - 1, halfSizeCeil)
            var squareSum = 0.toPrimitive()

            for (channel in lowerBound..upperBound) {
                val value = valuesAcrossChannel[channel] as PrimitiveType
                squareSum += value * value
            }

            for (channel in 0 until shape[1]) {
                val newLowerBound = channel - halfSize
                val newUpperBound = minOf(shape[1] - 1, channel + halfSizeCeil)

                if (newLowerBound > lowerBound) {
                    val value = valuesAcrossChannel[lowerBound] as PrimitiveType
                    squareSum -= value * value
                    lowerBound = newLowerBound
                }

                if (newUpperBound > upperBound) {
                    val value = valuesAcrossChannel[newUpperBound] as PrimitiveType
                    squareSum += value * value
                    upperBound = newUpperBound
                }

                val currentIndex = batch * shape[1] * inputSize + channel * inputSize + linearIndex
                result.array[currentIndex] = array[currentIndex] / (bias.toPrimitive() + (alpha.toPrimitive() / size.toPrimitive() * squareSum)).pow(beta.toPrimitive())
            }
        }
    }

    return result
}
