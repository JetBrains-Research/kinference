@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE, DataType.UBYTE, DataType.BYTE)

package io.kinference.ndarray.extensions.pool

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.ndarray.arrays.pointers.forEachWith
import io.kinference.ndarray.stubs.*
import io.kinference.ndarray.extensions.utils.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import io.kinference.ndarray.extensions.*
import io.kinference.utils.InlineInt


@MakePublic
internal suspend fun PrimitiveNDArray.maxPool(
    inputInfo: InputInfo,
    storageOrder: Int,
    minValue: PrimitiveType = PrimitiveType.MIN_VALUE_FOR_MAX
): List<NumberNDArrayCore> {
    val resultShape = IntArray(rank) {
        when (it) {
            0, 1 -> shape[it]
            else -> inputInfo.outputShape[it - 2]
        }
    }
    val result = PrimitiveNDArray(resultShape)
    val kernelDim = shape[1] * inputInfo.kernelSize
    val col = PrimitiveNDArray(kernelDim, inputInfo.outputSize)

    var resultIndices: LongNDArray? = null
    var indCol: LongNDArray? = null

    if (storageOrder != -1) {
        resultIndices = LongNDArray(resultShape)
        indCol = LongNDArray(kernelDim, inputInfo.outputSize)
    }

    val defaultIndices = when (storageOrder) {
        0 -> {
            val typedLambda: (InlineInt) -> Long = { it.value.toLong() }
            LongNDArray(shape, typedLambda)
        }
        1 -> {
            val typedLambda: (InlineInt) -> Long = { computeColumnMajorIndex(it.value, shape).toLong() }
            LongNDArray(shape, typedLambda)
        }
        else -> null
    }

    repeat(shape[0]) { batch ->
        val batchOffset = batch  * shape[1] * inputInfo.inputSize

        if (inputInfo.rank == 2) {
            primitiveIm2ColRank2(this.view(batch), inputInfo, shape[1], col, minValue)
            if (defaultIndices != null && indCol != null)
                primitiveIm2ColRank2(defaultIndices.view(batch), inputInfo, shape[1], indCol, -1L)
        } else {
            primitiveIm2Col(this.view(batch), inputInfo, kernelDim, col, minValue)
            if (defaultIndices != null && indCol != null)
                primitiveIm2Col(defaultIndices.view(batch), inputInfo, kernelDim, indCol, -1L)
        }
        val transposedCol = col.transpose2D()
        val transposedIndCol = indCol?.transpose2D()

        repeat(inputInfo.outputSize) { i ->
            val currentCol = transposedCol.view(i)
            val currentInd = transposedIndCol?.view(i)
            repeat(shape[1]) { channel ->
                val resultIndex = i + inputInfo.outputSize * channel + inputInfo.outputSize * shape[1] * batch
                val curColPointer =  currentCol.array.pointer(channel * inputInfo.kernelSize)
                val curIndColPointer = currentInd?.array?.pointer(channel * inputInfo.kernelSize)
                var curResult = minValue
                var curResultIndex = -1L

                if (curIndColPointer != null) {
                    curColPointer.forEachWith(curIndColPointer, inputInfo.kernelSize) { it, ind ->
                        if (curResult < it) {
                            curResult = it
                            curResultIndex = ind
                        }
                    }
                } else {
                    curColPointer.forEach(inputInfo.kernelSize) {
                        if (curResult < it)
                            curResult = it
                    }
                }

                //curResultIndex += batchOffset
                result.array[resultIndex] = curResult
                if (resultIndices != null)
                    resultIndices.array[resultIndex] = curResultIndex
            }
        }
    }

    if (resultIndices != null)
        return listOf(result, resultIndices)
    return listOf(result)
}
