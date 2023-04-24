@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions.conv

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.extensions.utils.*
import io.kinference.ndarray.extensions.utils.inferShapeSize
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

suspend fun PrimitiveNDArray.conv(
    w: PrimitiveNDArray,
    b: PrimitiveNDArray?,
    inputInfo: InputInfo
): PrimitiveNDArray {
    val resultBatchSize = shape[0]
    val resultChannels = w.shape[0]
    val resultShape = IntArray(rank) {
        when (it) {
            0 -> resultBatchSize
            1 -> resultChannels
            else -> inputInfo.outputShape[it - 2]
        }
    }
    val resultSize = resultShape.inferShapeSize()
    val result = PrimitiveArray(resultSize)

    val xOffset = shape[1] / inputInfo.groups * inputInfo.inputSize
    val yOffset = resultSize / resultBatchSize / inputInfo.groups
    val wOffset = (inputInfo.kernelSize * w.shape[0] * w.shape[1]) / inputInfo.groups

    val kernelDim = shape[1] / inputInfo.groups * inputInfo.kernelSize
    val col = PrimitiveArray(inputInfo.outputSize * kernelDim)

    val xPointer = array.pointer()
    var yPointer = 0

    for (imageId in 0 until shape[0]) {
        for (groupId in 0 until inputInfo.groups) {
            val prevLiX = xPointer.linearIndex
            xPointer.linearIndex += xOffset * groupId
            if (inputInfo.rank == 2)
                primitiveIm2ColRank2(xPointer, inputInfo, shape[1] / inputInfo.groups, col)
            else
                primitiveIm2Col(xPointer, inputInfo, kernelDim, col)
            xPointer.linearIndex = prevLiX

            val prevLiY = yPointer
            val wPointer = w.array.pointer(wOffset * groupId)
            yPointer += yOffset * groupId
            gemmConvCoroutines(
                wPointer,
                col,
                yPointer,
                result,
                w.shape[0] / inputInfo.groups,
                inputInfo.outputSize,
                kernelDim
            )
            yPointer = prevLiY
        }

        if (imageId != shape[0] - 1) {
            xPointer.linearIndex += xOffset * inputInfo.groups
            yPointer += yOffset * inputInfo.groups
        }
    }

    if (b != null) {
        var resultPointer = 0

        repeat(resultBatchSize) {
            val bPointer = PrimitivePointer(b.array)

            repeat(resultChannels) {
                repeat(inputInfo.outputSize) {
                    result[resultPointer] += bPointer.get()
                    resultPointer++
                }
                bPointer.increment()
            }
        }
    }

    return PrimitiveNDArray(resultShape) { i -> result[i] }
}

suspend fun gemmConvCoroutines(
    aPointer: PrimitivePointer,
    b: PrimitiveArray,
    cPointer: Int,
    c: PrimitiveArray,
    m: Int,
    n: Int,
    k: Int
) {
    val corNum = 8
    val step = m / corNum

    if (step < 8) {
        gemmConv(aPointer, b, cPointer, c, m, n, k)
        return
    }

    coroutineScope {
        repeat(corNum) {
            launch {
                val start = it * step
                var end = it * step + step
                if (it == corNum - 1)
                    end = m

                val aLocalPointer = PrimitivePointer(aPointer)
                aLocalPointer.linearIndex += start * k
                val cLocalPointer = cPointer + start * n

                gemmConv(aLocalPointer, b, cLocalPointer, c, m, n, k, start, end)
            }
        }
    }
}

fun gemmConv(
    aPointer: PrimitivePointer,
    b: PrimitiveArray,
    cPointer: Int,
    c: PrimitiveArray,
    m: Int,
    n: Int,
    k: Int,
    start: Int = 0,
    end: Int = m
) {
    var bLocalPointer = 0
    var cLocalPointer = cPointer

    for (t in start until end) {
        bLocalPointer = 0

        repeat(k) {
            val temp = aPointer.getAndIncrement()

            repeat(n) {
                c[cLocalPointer + it] += b[bLocalPointer + it] * temp
            }

            bLocalPointer += n
        }

        cLocalPointer += n
    }
}

