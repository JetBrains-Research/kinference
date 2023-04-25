@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE, DataType.UBYTE, DataType.BYTE)

package io.kinference.ndarray.extensions.utils

import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.SpecifyPrimitives
import io.kinference.primitives.types.*

fun primitiveIm2Col(
    im: PrimitiveNDArray,
    inputInfo: InputInfo,
    channelsCol: Int,
    col: PrimitiveNDArray,
    padValue: PrimitiveType = 0.toPrimitive(),
    storageOrder: Int = -1,
    indCol: IntNDArray = IntNDArray.scalar(-1)
) {
    val dOffset = IntArray(inputInfo.rank)
    val dIter = IntArray(inputInfo.rank)

    for (cCol in 0 until channelsCol) {
        var offset = cCol
        for (dI in inputInfo.rank - 1 downTo 0) {
            if (dI < inputInfo.rank - 1) {
                offset /= inputInfo.kernelShape[dI + 1]
            }
            dOffset[dI] = offset % inputInfo.kernelShape[dI]
        }

        var hasNextOutput: Boolean
        do {
            var indexCol = cCol
            var indexIm = cCol / inputInfo.kernelSize
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

            var indexImR = 0
            if (storageOrder == 1)
                indexImR = computeColumnMajorIndex(inputInfo, dIter, dOffset, cCol / inputInfo.kernelSize)

            if (isPadding) {
                col.array[indexCol] = padValue
                if (storageOrder != -1)
                    indCol.array[indexCol] = -1
            }
            else {
                col.array[indexCol] = im.array[indexIm]
                when (storageOrder) {
                    0 -> indCol.array[indexCol] = indexIm
                    1 -> indCol.array[indexCol] = indexImR
                }
            }

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

fun primitiveIm2ColRank2(
    im: PrimitiveNDArray,
    inputInfo: InputInfo,
    channels: Int,
    col: PrimitiveNDArray,
    padValue: PrimitiveType = 0.toPrimitive()
) {
    var colInd = 0
    var linearIndex = 0

    val channelSize = inputInfo.inputSize
    repeat(channels) {
        for (kernelRow in 0 until inputInfo.kernelShape[0]) {
            for (kernelCol in 0 until inputInfo.kernelShape[1]) {
                var inputRow = -inputInfo.padBegin(0) + kernelRow * inputInfo.dilations[0]
                repeat(inputInfo.outputShape[0]) {
                    if (isInPadding(inputRow, inputInfo.inputShape[0])) {
                        for (i in 0 until inputInfo.outputShape[1]) {
                            col.array[colInd] = padValue
                            colInd++
                        }
                    } else {
                        var inputCol = -inputInfo.padBegin(1) + kernelCol * inputInfo.dilations[1]
                        val imOffset = inputRow * inputInfo.inputShape[1] + inputCol
                        for (i in 0 until inputInfo.outputShape[1]) {  // TODO(CASES)
                            if (!isInPadding(inputCol, inputInfo.inputShape[1])) {
                                val localIm = im.array.pointer(linearIndex)
                                localIm.linearIndex += imOffset + i * inputInfo.strides[1]
                                col.array[colInd] = localIm.get()
                                colInd++
                            } else {
                                col.array[colInd] = padValue
                                colInd++
                            }

                            inputCol += inputInfo.strides[1]
                        }
                    }
                    inputRow += inputInfo.strides[0]
                }
            }
        }
        if (it != channels - 1)
            linearIndex += channelSize
    }
}
