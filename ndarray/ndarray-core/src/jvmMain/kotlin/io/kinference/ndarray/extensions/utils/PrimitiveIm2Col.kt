@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE, DataType.UBYTE, DataType.BYTE, DataType.LONG)

package io.kinference.ndarray.extensions.utils

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.*

@MakePublic
internal fun primitiveIm2Col(
    im: PrimitiveNDArray,
    inputInfo: InputInfo,
    channelsTotal: Int,
    col: PrimitiveNDArray,
    padValue: PrimitiveType = 0.toPrimitive()
) {
    val dOffset = IntArray(inputInfo.rank)
    val dIter = IntArray(inputInfo.rank)

    repeat(channelsTotal) { colChannel ->
        var offset = colChannel
        for (dI in inputInfo.rank - 1 downTo 0) {
            if (dI < inputInfo.rank - 1) {
                offset /= inputInfo.kernelShape[dI + 1]
            }
            dOffset[dI] = offset % inputInfo.kernelShape[dI]
        }

        var hasNextOutput: Boolean
        do {
            var indexCol = colChannel
            var indexIm = colChannel / inputInfo.kernelSize
            var isPadding = false

            for (dI in 0 until inputInfo.rank) {
                val d = dIter[dI]
                val dIm = d * inputInfo.strides[dI] - inputInfo.padBegin(dI) + dOffset[dI] * inputInfo.dilations[dI]
                if (dIm < 0 || dIm >= inputInfo.inputShape[dI])
                    isPadding = true

                indexCol *= inputInfo.outputShape[dI]
                indexCol += d

                indexIm *= inputInfo.inputShape[dI]
                indexIm += dIm
            }

            if (isPadding)
                col.array[indexCol] = padValue
            else
                col.array[indexCol] = im.array[indexIm]

            hasNextOutput = false
            for (j in inputInfo.rank - 1 downTo 0) {
                val dMax = inputInfo.outputShape[j] - 1
                if (dIter[j] >= dMax) {
                    dIter[j] = 0
                } else {
                    ++dIter[j]
                    hasNextOutput = true
                    break
                }
            }
        } while (hasNextOutput)
    }
}

@MakePublic
internal fun primitiveIm2ColRank2(
    im: PrimitiveNDArray,
    inputInfo: InputInfo,
    channelsCol: Int,
    col: PrimitiveNDArray,
    padValue: PrimitiveType = 0.toPrimitive()
) {
    val colPointer = col.array.pointer()
    var linearIndex = 0

    val channelSize = inputInfo.inputSize
    repeat(channelsCol) {
        for (kernelRow in 0 until inputInfo.kernelShape[0]) {
            for (kernelCol in 0 until inputInfo.kernelShape[1]) {
                var inputRow = -inputInfo.padBegin(0) + kernelRow * inputInfo.dilations[0]
                repeat(inputInfo.outputShape[0]) {
                    if (isInPadding(inputRow, inputInfo.inputShape[0])) {
                        colPointer.map(inputInfo.outputShape[1]) { padValue }
                    } else {
                        var inputCol = -inputInfo.padBegin(1) + kernelCol * inputInfo.dilations[1]
                        val imOffset = inputRow * inputInfo.inputShape[1] + inputCol
                        var i = 0
                        while (i < inputInfo.outputShape[1]) {
                            if (isInPadding(inputCol, inputInfo.inputShape[1])) {
                                colPointer.setAndIncrement(padValue)
                                i++
                                inputCol += inputInfo.strides[1]
                            } else {
                                if (inputInfo.strides[1] == 1) {
                                    val cnt = minOf(inputInfo.inputShape[1] - inputCol, inputInfo.outputShape[1] - i)
                                    val imPointer = im.array.pointer(linearIndex + imOffset + i)
                                    imPointer.mapTo(colPointer, cnt) { it }
                                    i += cnt
                                    inputCol += cnt
                                } else {
                                    colPointer.setAndIncrement(im.array[linearIndex + imOffset + i * inputInfo.strides[1]])
                                    i++
                                    inputCol += inputInfo.strides[1]
                                }
                            }
                        }
                    }
                    inputRow += inputInfo.strides[0]
                }
            }
        }

        linearIndex += channelSize
    }
}
