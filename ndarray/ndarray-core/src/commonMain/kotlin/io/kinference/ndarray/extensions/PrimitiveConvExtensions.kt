@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.extensions.utils.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.SpecifyPrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

suspend fun PrimitiveNDArray.conv(
    w: PrimitiveNDArray,
    b: PrimitiveNDArray?,
    pads: IntArray,
    strides: IntArray,
    dilations: IntArray,
    groups: Int
): PrimitiveNDArray {
    val inputShape = IntArray(shape.size - 2) { shape[it + 2] }
    val xShapeWithPads = getShapeWithPads(this.shape, pads)
    val wShapeWithDilations = getShapeWithDilations(w.shape, dilations)

    val resultShape = IntArray(this.shape.size) {
        when (it) {
            0 -> this.shape[0]
            1 -> w.shape[0]
            else -> ((xShapeWithPads[it] - wShapeWithDilations[it]) divCeil strides[it - 2]) + 1
        }
    }
    val result = PrimitiveArray(resultShape.shapeSize())
    val outputShape = IntArray(resultShape.size - 2) { resultShape[it + 2] }

    val kernel = IntArray(w.shape.size - 2) { w.shape[it + 2] }
    val xOffset = shape[1] / groups * inputShape.shapeSize()
    val yOffset = resultShape.shapeSize() / resultShape[0] / groups
    val wOffset = w.shape.shapeSize() / groups
    val kernelDim = shape[1] / groups * kernel.shapeSize()
    val col = PrimitiveArray(outputShape.shapeSize() * kernelDim)

    val xPointer = array.pointer()
    var yPointer = 0

    for (imageId in 0 until shape[0]) {
        for (groupId in 0 until groups) {
            val prevLiX = xPointer.linearIndex
            xPointer.linearIndex += xOffset * groupId
            if (kernel.size == 2) {
                primitiveIm2Col(
                    xPointer,
                    shape[1] / groups,
                    inputShape[0],
                    inputShape[1],
                    kernel[0],
                    kernel[1],
                    dilations[0],
                    dilations[1],
                    pads[0],
                    pads[1],
                    pads[2],
                    pads[3],
                    strides[0],
                    strides[1],
                    col
                )
            } else {
                primitiveIm2Col(
                    xPointer,
                    inputShape,
                    outputShape,
                    kernelDim,
                    kernel,
                    strides,
                    dilations,
                    pads,
                    kernel.size,
                    col
                )
            }
            xPointer.linearIndex = prevLiX

            val prevLiY = yPointer
            val wPointer = w.array.pointer(wOffset * groupId)
            yPointer += yOffset * groupId
            gemmConvCoroutines(
                wPointer,
                col,
                yPointer,
                result,
                w.shape[0] / groups,
                outputShape.shapeSize(),
                kernelDim
            )
            yPointer = prevLiY
        }

        if (imageId != shape[0] - 1) {
            xPointer.linearIndex += xOffset * groups
            yPointer += yOffset * groups
        }
    }

    if (b != null) {
        var resultPointer = 0
        repeat(shape[0]) {
            val bPointer = PrimitivePointer(b.array)
            repeat(w.shape[0]) {
                repeat(outputShape.shapeSize()) {
                    result[resultPointer] += bPointer.get()
                    resultPointer++
                }
                bPointer.increment()
            }
        }
    }

    return PrimitiveNDArray(resultShape) { i -> result[i].toPrimitive() }
}

@SpecifyPrimitives(include = [])
fun IntArray.shapeSize(): Int {
    return this.reduce { size, i -> size * i }
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

