@file:GeneratePrimitives(DataType.ALL)
package io.kinference.ndarray.extensions.reduce.primitive

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.utils.divCeil
import io.kinference.ndarray.parallelizeByBlocks
import io.kinference.ndarray.parallelizeByRows
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.utils.inlines.InlinePrimitive
import kotlinx.coroutines.coroutineScope

@GenerateNameFromPrimitives
internal suspend fun reduceOneAxisPrimitive(
    array: PrimitiveNDArray,
    axis: Int,
    keepDims: Boolean,
    initOutputValue: PrimitiveType? = null,
    operation: PrimitiveBinaryOperation
): PrimitiveNDArray {
    val actualAxis = array.indexAxis(axis)

    if (actualAxis == array.shape.lastIndex) return reduceAlongLastAxis(array, keepDims, initOutputValue, operation)

    val outputShape = if (keepDims)
        array.shape.copyOf().apply { set(axis, 1) }
    else
        IntArray(array.rank - 1).apply {
            array.shape.copyInto(this, endIndex = actualAxis)
            array.shape.copyInto(this, destinationOffset = actualAxis, startIndex = actualAxis + 1)
        }

    val output = MutablePrimitiveNDArray(outputShape)

    if (initOutputValue != null) {
        output.fill(initOutputValue)
    }

    val batchSize = array.computeBlockSize(toDim = actualAxis)
    val reduceSize = array.shape[actualAxis]
    val rowSize = array.computeBlockSize(fromDim = actualAxis + 1)

    val blockInRow = rowSize / array.array.blockSize

    val inputBatchBlocksRow = blockInRow * reduceSize

    val inputBlocks = array.array.blocks
    val outputBlocks = output.array.blocks

    for (batchNum in 0 until batchSize) {
        val inputBatchBlocksOffset = inputBatchBlocksRow * batchNum
        val outputBatchBlockOffset = blockInRow * batchNum

        for (reduceIdx in 0 until reduceSize) {
            val inputFullOffset = inputBatchBlocksOffset + reduceIdx * blockInRow

            for (blockIdx in 0 until blockInRow) {
                val inputBlock = inputBlocks[inputFullOffset + blockIdx]
                val outputBlock = outputBlocks[outputBatchBlockOffset + blockIdx]

                for (idx in outputBlock.indices) {
                    outputBlock[idx] = operation(outputBlock[idx], inputBlock[idx])
                }
            }
        }
    }

    return output
}

private suspend fun reduceAlongLastAxis(
    array: PrimitiveNDArray,
    keepDims: Boolean,
    initOutputValue: PrimitiveType? = null,
    operation: PrimitiveBinaryOperation
): PrimitiveNDArray {
    val outputShape = if (keepDims) {
        array.shape.copyOf().apply { this[lastIndex] = 1 }
    } else {
        array.shape.copyOfRange(fromIndex = 0, toIndex = array.rank - 1)
    }

    val output = MutablePrimitiveNDArray(outputShape)

    if (initOutputValue != null) {
        output.fill(initOutputValue)
    }

    val batchSize = array.computeBlockSize(fromDim = 0, toDim = array.rank - 1)

    val blocksInRow = array.blocksInRow
    val blockSize = array.array.blockSize
    val blocks = array.array.blocks

    val outputPointer = output.array.pointer()

    for (batchNum in 0 until batchSize) {
        val blockOffset = batchNum * blocksInRow
        var destValue = outputPointer.get()

        for (blockIdx in blockOffset until blockOffset + blocksInRow) {
            val block = blocks[blockIdx]
            for (idx in 0 until blockSize) {
                destValue = operation(destValue, block[idx])
            }
        }

        outputPointer.setAndIncrement(destValue)
    }

    return output
}
