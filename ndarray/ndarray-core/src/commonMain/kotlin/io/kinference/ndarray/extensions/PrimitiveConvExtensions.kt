@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import io.kinference.ndarray.extensions.utils.*
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.random.Random

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
    val result = MutablePrimitiveNDArray(resultShape)
    val outputShape = IntArray(resultShape.size - 2) { resultShape[it + 2] }

    val kernel = IntArray(w.shape.size - 2) { w.shape[it + 2] }
    val xOffset = shape[1] / groups * inputShape.shapeSize()
    val yOffset = result.shape.shapeSize() / result.shape[0] / groups
    val wOffset = w.shape.shapeSize() / groups
    val kernelDim = shape[1] / groups * kernel.shapeSize()
    val col = PrimitiveNDArray(outputShape.shapeSize() * kernelDim)

    val xPointer = array.pointer()
    val wPointer = w.array.pointer()
    val yPointer = result.array.pointer()

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
                    PrimitivePointer(col.array)
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
                    col.array
                )
            }
            xPointer.linearIndex = prevLiX

            val prevLiW = wPointer.linearIndex
            val prevLiY = yPointer.linearIndex
            wPointer.linearIndex += wOffset * groupId
            yPointer.linearIndex += yOffset * groupId
            myGemm(
                wPointer,
                PrimitivePointer(col.array),
                yPointer,
                w.shape[0] / groups,
                outputShape.shapeSize(),
                kernelDim
            )
            wPointer.linearIndex = prevLiW
            yPointer.linearIndex = prevLiY
        }

        if (imageId != shape[0] - 1) {
            xPointer.linearIndex += xOffset * groups
            yPointer.linearIndex += yOffset * groups
        }
    }

    if (b != null) {
        val resultPointer = PrimitivePointer(result.array)
        repeat(shape[0]) {
            val bPointer = PrimitivePointer(b.array)
            repeat(w.shape[0]) {
                resultPointer.map(outputShape.shapeSize()) { it + bPointer.get() }
                bPointer.increment()
            }
        }
    }

    return result
}

suspend fun myGemm(
    aPointer: PrimitivePointer,
    bPointer: PrimitivePointer,
    cPointer: PrimitivePointer,
    m: Int,
    n: Int,
    k: Int,
    ldb: Int = n,
    lda: Int = k,
    ldc: Int = n
) {
    if (m > 128) {
        myGemmCoroutines(aPointer, bPointer, cPointer, m, n, k, ldb, lda, ldc)
        return
    }

    val aLocalPointer = PrimitivePointer(aPointer)
    val bLocalPointer = PrimitivePointer(bPointer)
    val cLocalPointer = PrimitivePointer(cPointer)

    //coroutineScope {
    //val bPointerIndex = bPointer.linearIndex
    for (t in 0 until m) {
        val cIdx = t * ldc
        aLocalPointer.linearIndex = aPointer.linearIndex + t * lda

        //val cPointerIndex = cPointer.linearIndex
        for (i in 0 until k) {
            val temp = aLocalPointer.getAndIncrement()

            bLocalPointer.linearIndex = bPointer.linearIndex + i * ldb
            cLocalPointer.linearIndex = cPointer.linearIndex + cIdx

//                launch {
//                    val bLocalPointer = PrimitivePointer(bPointer)
//                    val cLocalPointer = PrimitivePointer(cPointer)
            //cLocalPointer.myAccept(bLocalPointer, n, temp)
            //}

            cLocalPointer.myAccept(bLocalPointer, n, temp)

            //cPointer.linearIndex = cPointerIndex
        }

//            bPointer.linearIndex = bPointerIndex
//            if (t != m - 1) {
//                cPointer.linearIndex = cPointerIndex + ldc
        //}
    }
    //}
}

@SpecifyPrimitives(include = [])
fun IntArray.shapeSize(): Int {
    return this.reduce { size, i -> size * i }
}

fun PrimitivePointer.myAccept(
    other: PrimitivePointer,
    count: Int,
    temp: PrimitiveType
) {
    var end = count
    var dstBlock = this.currentBlock
    var dstOffset = this.indexInBlock

    var srcBlock = other.currentBlock
    var srcOffset = other.indexInBlock

    var dstShift: Int = 0
    var srcShift: Int = 0
    var iter: Int = -1

    while (end > 0) {
        if (iter == dstShift) {
            this.unsafeAcceptBlockIncrement()
            dstBlock = this.currentBlock
            dstOffset = 0
        }

        if (iter == srcShift) {
            other.unsafeAcceptBlockIncrement()
            srcBlock = other.currentBlock
            srcOffset = 0
        }

        dstShift = dstBlock.size - dstOffset
        srcShift = srcBlock.size - srcOffset
        iter = minOf(dstShift, srcShift, end)

        extracted(iter, dstBlock, dstOffset, srcBlock, srcOffset, temp)
        dstOffset += iter
        srcOffset += iter

        end -= iter
    }

    this.indexInBlock = dstOffset
    other.indexInBlock = srcOffset
}

fun PrimitivePointer.unsafeAcceptBlockIncrement() {
    blockNum++
    currentBlock = array.blocks[blockNum]
}

private fun extracted(
    iter: Int,
    dstBlock: PrimitiveArray,
    dstOffset: Int,
    srcBlock: PrimitiveArray,
    srcOffset: Int,
    temp: PrimitiveType
) {
    for (i in 0 until iter) {
        dstBlock[dstOffset + i] += srcBlock[srcOffset + i] * temp
    }
}

suspend fun myGemmCoroutines(
    aPointer: PrimitivePointer,
    bPointer: PrimitivePointer,
    cPointer: PrimitivePointer,
    m: Int,
    n: Int,
    k: Int,
    ldb: Int = n,
    lda: Int = k,
    ldc: Int = n
) {
    coroutineScope {
        for (startT in 0 until m step 32) {
            launch {
                val aLocalPointer = PrimitivePointer(aPointer)
                val bLocalPointer = PrimitivePointer(bPointer)
                val cLocalPointer = PrimitivePointer(cPointer)
                val endT = minOf(startT + 32, m)
                for (t in startT until endT) {
                    val cIdx = t * ldc
                    aLocalPointer.linearIndex = aPointer.linearIndex + t * lda

                    for (i in 0 until k) {
                        val temp = aLocalPointer.getAndIncrement()

                        bLocalPointer.linearIndex = bPointer.linearIndex + i * ldb
                        cLocalPointer.linearIndex = cPointer.linearIndex + cIdx
                        cLocalPointer.myAccept(bLocalPointer, n, temp)
                    }
                }
            }
        }
    }
}



/*
Everything below this commit was used for test purposes.
 */
suspend fun PrimitiveNDArray.conv2Array(
    w: PrimitiveNDArray,
    b: PrimitiveNDArray?,
    pads: IntArray,
    strides: IntArray,
    dilations: IntArray,
    groups: Int
): PrimitiveNDArray {
    val inputShape = shape.slice(2..shape.lastIndex).toIntArray()
    val xShapeWithPads = getShapeWithPads(this.shape, pads)
    val wShapeWithDilations = getShapeWithDilations(w.shape, dilations)

    val resultShape = IntArray(this.shape.size) {
        when (it) {
            0 -> this.shape[0]
            1 -> w.shape[0]
            else -> (xShapeWithPads[it] - wShapeWithDilations[it] + 1) divCeil strides[it - 2]
        }
    }
    val result = FloatArray(resultShape.shapeSize())
    val outputShape = resultShape.slice(2..resultShape.lastIndex).toIntArray()

    val kernel = w.shape.slice(2..shape.lastIndex).toIntArray()
    val xOffset = shape[1] / groups * calculateInnerShapeSize(shape)
    val yOffset = resultShape.shapeSize() / resultShape[0] / groups
    val wOffset = w.shape.shapeSize() / groups
    val kernelDim = shape[1] / groups * calculateInnerShapeSize(w.shape)
    val colSize = kernelDim * calculateInnerShapeSize(w.shape)
    val col = FloatArray(outputShape.shapeSize() * kernelDim)

    val xPointer = array.pointer()
    val wPointer = w.array.pointer()
    //val yPointer = result.array.pointer()
    var yPointer = 0

    for (imageId in 0 until shape[0]) {
        for (groupId in 0 until groups) {
            val prevLiX = xPointer.linearIndex
            xPointer.linearIndex += xOffset * groupId
            primitiveIm2ColArray(
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
            xPointer.linearIndex = prevLiX

            val prevLiW = wPointer.linearIndex
            val prevLiY = yPointer
            wPointer.linearIndex += wOffset * groupId
            yPointer += yOffset * groupId
            myGemmArray(
                wPointer,
                col,
                yPointer,
                result,
                w.shape[0] / groups,
                outputShape.shapeSize(),
                kernelDim
            )
            wPointer.linearIndex = prevLiW
            yPointer = prevLiY
        }

        if (imageId != shape[0] - 1) {
            xPointer.linearIndex += xOffset * groups
            yPointer += yOffset * groups
        }
    }

    return PrimitiveNDArray(resultShape) { i -> result[i].toPrimitive() }
}

suspend fun myGemmArray(
    aPointer: PrimitivePointer,
    bPointer: FloatArray,
    cPointer: Int,
    c: FloatArray,
    m: Int,
    n: Int,
    k: Int,
    ldb: Int = n,
    lda: Int = k,
    ldc: Int = n
) {
    if (m > 128) {
        myGemmArrayCor(aPointer, bPointer, cPointer, c, m, n, k, ldb, lda, ldc)
        return
    }

    val aLocalPointer = PrimitivePointer(aPointer)
    var bLocalPointer = 0
    var cLocalPointer = cPointer

    for (t in 0 until m) {
        val cIdx = t * ldc
        aLocalPointer.linearIndex = aPointer.linearIndex + t * lda

        for (i in 0 until k) {
            val temp = aLocalPointer.getAndIncrement()

            bLocalPointer = i * ldb
            cLocalPointer = cPointer + cIdx

            repeat(n) {
                c[cLocalPointer + it] += bPointer[bLocalPointer + it] * temp.toFloat()
            }
        }
    }
}

suspend fun myGemmArrayCor(
    aPointer: PrimitivePointer,
    bPointer: FloatArray,
    cPointer: Int,
    c: FloatArray,
    m: Int,
    n: Int,
    k: Int,
    ldb: Int = n,
    lda: Int = k,
    ldc: Int = n
) {
    val stepM = m / 7
    coroutineScope {
        for (startT in 0 until m step stepM) {
            launch {
                val endT = minOf(startT + stepM, m)
                val aLocalPointer = PrimitivePointer(aPointer)
                var bLocalPointer = 0
                var cLocalPointer = cPointer

                for (t in startT until endT) {
                    val cIdx = t * ldc
                    aLocalPointer.linearIndex = aPointer.linearIndex + t * lda

                    for (i in 0 until k) {
                        val temp = aLocalPointer.getAndIncrement()

                        bLocalPointer = i * ldb
                        cLocalPointer = cPointer + cIdx

                        repeat(n) {
                            c[cLocalPointer + it] += bPointer[bLocalPointer + it] * temp.toFloat()
                        }
                    }
                }
            }
        }
    }
}
