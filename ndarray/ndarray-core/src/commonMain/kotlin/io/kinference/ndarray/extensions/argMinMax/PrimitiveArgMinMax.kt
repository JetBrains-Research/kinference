@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions.argMinMax

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

private val lessOrEqual = { a: PrimitiveType, b: PrimitiveType -> a <= b  }
private val less = { a: PrimitiveType, b: PrimitiveType -> a < b  }
private val greaterOrEqual = { a: PrimitiveType, b: PrimitiveType -> a >= b  }
private val greater = { a: PrimitiveType, b: PrimitiveType -> a > b  }

@GenerateNameFromPrimitives
internal fun argMinMaxPrimitive(input: PrimitiveNDArray, axis: Int, keepDims: Boolean, selectLastIndex: Boolean, mode: ArgMinMaxMode = ArgMinMaxMode.MAX): IntNDArray {
    val actualAxis = input.indexAxis(axis)

    val iterations = input.computeBlockSize(toDim = actualAxis)
    val elementsPerDim = input.computeBlockSize(fromDim = actualAxis + 1)
    val dimsToReduce = input.shape[actualAxis]

    val outputShape = if (keepDims) input.shape.copyOf().apply { set(actualAxis, 1) } else input.shape.sliceArray(input.shape.indices.minus(actualAxis))
    val outputArray = MutableIntNDArray(outputShape)

    val comparator = when(mode) {
        ArgMinMaxMode.MIN -> if (selectLastIndex) lessOrEqual else less
        ArgMinMaxMode.MAX -> if (selectLastIndex) greaterOrEqual else greater
    }

    return when {
        // If we have nothing to reduce, then we should return an array of zeros
        input.shape[actualAxis] == 1 -> outputArray
        actualAxis == input.shape.lastIndex -> argMinMaxAlongLastAxis(input, outputArray, iterations, comparator)
        else -> argMinMaxDefault(input, outputArray, iterations, dimsToReduce, elementsPerDim, comparator)
    }
}

private fun argMinMaxAlongLastAxis(
    input: PrimitiveNDArray,
    output: MutableIntNDArray,
    iterations: Int,
    comparator: (PrimitiveType, PrimitiveType) -> Boolean
): MutableIntNDArray {

    val outputPointer = output.array.pointer()
    val blocksPerIteration = input.blocksInRow
    val inputBlocks = input.array.blocks
    val blockSize = input.array.blockSize

    repeat(iterations) { iter ->
        val startBlock = iter * blocksPerIteration

        var minMaxValue = inputBlocks[startBlock][0]
        var minMaxIndex = 0

        for (blockIdx in 0 until blocksPerIteration) {
            val block = inputBlocks[startBlock + blockIdx]
            val indexOffset = blockIdx * blockSize

            for (idx in 0 until blockSize) {
                val value = block[idx]

                if (comparator(value, minMaxValue)) {
                    minMaxValue = value
                    minMaxIndex = indexOffset + idx
                }
            }
        }

        outputPointer.setAndIncrement(minMaxIndex)
    }

    return output
}

private fun argMinMaxDefault(
    input: PrimitiveNDArray,
    output: MutableIntNDArray,
    iterations: Int,
    dimsToReduce: Int,
    elementsPerDim: Int,
    comparator: (PrimitiveType, PrimitiveType) -> Boolean
): MutableIntNDArray {
    val inputBlocks = input.array.blocks
    val outputBlocks = output.array.blocks

    val minMaxValuesArray = PrimitiveTiledArray(elementsPerDim, output.array.blockSize)
    val minMaxValuesBlocks = minMaxValuesArray.blocks

    val blocksPerDim = elementsPerDim / input.array.blockSize
    val blockSize = output.array.blockSize

    repeat(iterations) { iter ->
        val outputBlockIterOffset = iter * blocksPerDim

        val inputBlockIterOffset = blocksPerDim * dimsToReduce * iter

        // Copy values from first dim
        for (blockIdx in 0 until blocksPerDim) {
            val inputBlock = inputBlocks[inputBlockIterOffset + blockIdx]
            val minMaxValuesBlock = minMaxValuesBlocks[blockIdx]

            for (idx in 0 until blockSize) {
                minMaxValuesBlock[idx] = inputBlock[idx]
            }
        }

        for (dimIdx in 1 until dimsToReduce) {
            val inputBlockFullOffset = inputBlockIterOffset + dimIdx * blocksPerDim

            for (blockIdx in 0 until blocksPerDim) {
                val inputBlock = inputBlocks[inputBlockFullOffset + blockIdx]
                val minMaxValuesBlock = minMaxValuesBlocks[blockIdx]
                val outputBlock = outputBlocks[outputBlockIterOffset + blockIdx]

                for (idx in 0 until blockSize) {
                    val value = inputBlock[idx]
                    if (comparator(value, minMaxValuesBlock[idx])) {
                        minMaxValuesBlock[idx] = value
                        outputBlock[idx] = dimIdx
                    }
                }
            }
        }
    }

    return output
}
