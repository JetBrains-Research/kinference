@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE, DataType.UBYTE, DataType.BYTE)

package io.kinference.ndarray.extensions.pool

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.utils.InputInfo
import io.kinference.ndarray.extensions.utils.primitiveIm2Col
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*

fun PrimitiveNDArray.maxPool(
    inputInfo: InputInfo,
    storageOrder: Int,
    minValue: PrimitiveType = PrimitiveType.MIN_VALUE
): List<NumberNDArrayCore> {
    val resultShape = IntArray(rank) {
        when (it) {
            0, 1 -> shape[it]
            else -> inputInfo.outputShape[it - 2]
        }
    }
    val result = MutablePrimitiveNDArray(resultShape)
    val indices = if (storageOrder == -1) MutableLongNDArray.scalar(0) else MutableLongNDArray(resultShape)

    val kernelDim = shape[1] / inputInfo.groups * inputInfo.kernelSize
    val col = PrimitiveArray(inputInfo.outputSize * kernelDim)
    val indCol = IntArray(inputInfo.outputSize * kernelDim)

    val resultPointer = result.array.pointer()
    val indicesPointer = indices.array.pointer()

    repeat(shape[0]) {
        val xPointer = array.pointer(it * inputInfo.inputSize * shape[1])
        primitiveIm2Col(xPointer, inputInfo, kernelDim, col, minValue, storageOrder, indCol)

        repeat(shape[1]) { channel ->
            val start = channel * inputInfo.outputSize * inputInfo.kernelSize
            repeat(inputInfo.outputSize) { j ->
                resultPointer.set(minValue)
                repeat(inputInfo.kernelSize) { i ->
                    val cur = col[start + i * inputInfo.outputSize + j]
                    if (resultPointer.get() < cur) {
                        resultPointer.set(cur)
                        if (storageOrder != -1)
                            indicesPointer.set((indCol[start + i * inputInfo.outputSize + j]).toLong())
                    }
                }
                resultPointer.increment()
                indicesPointer.increment()
            }
        }
    }

    return listOf(result, indices)
}
