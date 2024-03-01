@file:GeneratePrimitives(
    DataType.DOUBLE,
    DataType.FLOAT
)

package io.kinference.ndarray.extensions.hardmax

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.stubs.MIN_VALUE_FOR_MAX
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.*
import io.kinference.ndarray.extensions.*

@MakePublic
internal suspend fun PrimitiveNDArray.hardmax(axis: Int = 1): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(strides)

    val actualAxis = indexAxis(axis)
    val rows = computeBlockSize(toDim = actualAxis)
    val columns = computeBlockSize(fromDim = actualAxis)

    val blockSize = this.array.blockSize
    val blocksPerColumn = columns / blockSize

    val inputBlocks = this.array.blocks
    val outputBlocks = output.array.blocks

    repeat(rows) { row ->
        var maxValue = PrimitiveType.MIN_VALUE_FOR_MAX
        var maxBlockIdx = 0
        var maxIndex = 0

        val startBlockOffset = blocksPerColumn * row

        for (blockIdx in startBlockOffset until startBlockOffset + blocksPerColumn) {
            val block = inputBlocks[blockIdx]

            for (idx in block.indices) {
                if (block[idx] > maxValue) {
                    maxValue = block[idx]
                    maxIndex = idx
                    maxBlockIdx = blockIdx
                }
            }
        }

        outputBlocks[maxBlockIdx][maxIndex] = PrimitiveConstants.ONE
    }

    return output
}
