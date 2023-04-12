@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.SpecifyPrimitives
import io.kinference.primitives.types.*

fun primitiveIm2Col(
    im: PrimitivePointer,
    imageShape: IntArray,
    outputShape: IntArray,
    channelsCol: Int,
    kernelShape: IntArray,
    stride: IntArray,
    dilation: IntArray,
    pad: IntArray,
    rank: Int,
    col: PrimitiveTiledArray,
    padValue: PrimitiveType = 0.toPrimitive()
) {
    val kernelSize = kernelShape.shapeSize()
    val dOffset = IntArray(rank)
    val dIter = IntArray(rank)

    for (cCol in 0 until channelsCol) {
        var offset = cCol
        for (dI in rank - 1 downTo 0) {
            if (dI < rank - 1) {
                offset /= kernelShape[dI + 1]
            }
            dOffset[dI] = offset % kernelShape[dI]
        }

        var hasNextOutput: Boolean
        do {
            var indexCol = cCol
            var indexIm = cCol / kernelSize
            var isPadding = false

            for (dI in 0 until rank) {
                val d = dIter[dI]
                val dIm = d * stride[dI] - pad[dI] + dOffset[dI] * dilation[dI]
                if (dIm < 0 || dIm >= imageShape[dI])
                    isPadding = true

                indexCol *= outputShape[dI]
                indexCol += d

                indexIm *= imageShape[dI]
                indexIm += dIm
            }

            if (isPadding)
                col[indexCol] = padValue
            else
                col[indexCol] = im.array[indexIm + im.linearIndex]

            hasNextOutput = false
            for (j in rank - 1 downTo 0) {
                val dMax = outputShape[j] - 1
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
