@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.SpecifyPrimitives
import io.kinference.primitives.types.*

fun primitiveIm2Col(
    im: PrimitiveNDArray,
    imShift: Int,
    imageShape: IntArray,
    outputShape: IntArray,
    channelsCol: Int,
    kernelShape: IntArray,
    stride: IntArray,
    dilation: IntArray,
    pad: IntArray,
    rank: Int,
    col: PrimitiveNDArray
) {
    val kernelSize = kernelShape.reduce { size, el -> size * el }
    val dOffset = IntArray(rank)
    val dIter = IntArray(rank)

    for (i in 0 until channelsCol) {
        var offset = i
        for (j in rank - 1 downTo 0) {
            if (j < rank - 1) {
                offset /= kernelShape[j + 1]
            }
            dOffset[j] = offset % kernelShape[j]
        }

        var hasNextOutput: Boolean
        do {
            var indexCol = i
            var indexIm = i / kernelSize
            var isPadding = false

            for (j in 0 until rank) {
                val d = dIter[j]
                val dIm = d * stride[j] - pad[j] + dOffset[j] * dilation[j]
                if (dIm < 0 || dIm >= imageShape[j])
                    isPadding = true

                indexCol *= outputShape[j]
                indexCol += d

                indexIm *= imageShape[j]
                indexIm += dIm
            }

            if (isPadding)
                col.array[indexCol] = 0.toPrimitive()
            else
                col.array[indexCol] = im.array[indexIm + imShift]

            hasNextOutput = false
            for (j in rank - 1 downTo 0) {
                val dMax = outputShape[j]
                if (dIter[j] < dMax) {
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

fun primitiveIm2Col(
    im: PrimitivePointer,
    channels: Int,
    height: Int,
    width: Int,
    kernelHeight: Int,
    kernelWidth: Int,
    dilationHeight: Int,
    dilationWidth: Int,
    padTop: Int,
    padLeft: Int,
    padBottom: Int,
    padRight: Int,
    strideHeight: Int,
    strideWidth: Int,
    col: PrimitivePointer,
    padValue: PrimitiveType = 0.toPrimitive()
) {
    val outputHeight = (height + padBottom + padTop - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1
    val outputWidth = (width + padLeft + padRight - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1

    val channelSize = height * width
    repeat(channels) {
        for (kernelRow in 0 until kernelHeight) {
            for (kernelCol in 0 until kernelWidth) {
                var inputRow = -padTop + kernelRow * dilationHeight
                repeat(outputHeight) {
                    if (isInPadding(inputRow, height)) {
                        for (i in 0 until outputWidth) {
                            col.setAndIncrement(padValue)
                        }
                    } else {
                        var inputCol = -padLeft + kernelCol * dilationWidth
                        val imOffset = inputRow * width + inputCol
                        for (i in 0 until outputWidth) {  // TODO(CASES)
                            if (!isInPadding(inputCol, width)) {
                                val localIm = PrimitivePointer(im)
                                localIm.linearIndex += imOffset + i * strideWidth
                                col.setAndIncrement(localIm.get())
                            } else {
                                col.setAndIncrement(padValue)
                            }

                            inputCol += strideWidth
                        }
                    }
                    inputRow += strideHeight
                }
            }
        }
        if (it != channels - 1)
            im.linearIndex += channelSize
    }
}

@SpecifyPrimitives(include = [])
fun isInPadding(actual: Int, bound: Int) : Boolean {
    return actual < 0 || actual >= bound
}


fun primitiveIm2ColArray(
    im: PrimitivePointer,
    channels: Int,
    height: Int,
    width: Int,
    kernelHeight: Int,
    kernelWidth: Int,
    dilationHeight: Int,
    dilationWidth: Int,
    padTop: Int,
    padLeft: Int,
    padBottom: Int,
    padRight: Int,
    strideHeight: Int,
    strideWidth: Int,
    col: FloatArray,
    padValue: PrimitiveType = 0.toPrimitive()
) {
    val outputHeight = (height + padBottom + padTop - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1
    val outputWidth = (width + padLeft + padRight - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1

    var colInd = 0

    val channelSize = height * width
    repeat(channels) {
        for (kernelRow in 0 until kernelHeight) {
            for (kernelCol in 0 until kernelWidth) {
                var inputRow = -padTop + kernelRow * dilationHeight
                repeat(outputHeight) {
                    if (isInPadding(inputRow, height)) {
                        for (i in 0 until outputWidth) {
                            col[colInd] = padValue.toFloat()
                            colInd++
                        }
                    } else {
                        var inputCol = -padLeft + kernelCol * dilationWidth
                        val imOffset = inputRow * width + inputCol
                        for (i in 0 until outputWidth) {  // TODO(CASES)
                            if (!isInPadding(inputCol, width)) {
                                val localIm = PrimitivePointer(im)
                                localIm.linearIndex += imOffset + i * strideWidth
                                col[colInd] = localIm.get().toFloat()
                                colInd++
                            } else {
                                col[colInd] = padValue.toFloat()
                                colInd++
                            }

                            inputCol += strideWidth
                        }
                    }
                    inputRow += strideHeight
                }
            }
        }
        if (it != channels - 1)
            im.linearIndex += channelSize
    }
}
