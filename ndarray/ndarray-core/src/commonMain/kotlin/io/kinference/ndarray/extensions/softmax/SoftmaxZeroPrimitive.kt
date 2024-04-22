@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions.softmax

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.stubs.max
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.ndarray.stubs.MIN_VALUE_FOR_MAX
import kotlin.math.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants


@GenerateNameFromPrimitives
internal suspend fun softmaxZeroPrimitive(input: PrimitiveNDArray, dest: MutablePrimitiveNDArray, rows: Int, columns: Int): MutablePrimitiveNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks

    val outputArray = dest.array
    val maxesArray = PrimitiveArray(inputBlocks.size)

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 65536) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            maxesArray[blockNum] = inputBlocks[blockNum].max()
        }
    }

    val blocksInRow = columns / inputBlockSize

    //Minus maximum from input and store in output
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 2048) { rowStart, rowEnd, _ ->
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localMax = PrimitiveType.MIN_VALUE_FOR_MAX
            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                localMax = max(localMax, maxesArray[rowBlockIdx])
            }

            val expNegLocalMax = FastMath.exp(-localMax)

            var localSum = PrimitiveConstants.ZERO
            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                val inputBlock = inputBlocks[rowBlockIdx]
                val outputBlock = outputArray.blocks[rowBlockIdx]

                for (j in outputBlock.indices) {
                    val inputValue = inputBlock[j]
                    if (inputValue > (0.0000001f).toPrimitive() || inputValue < (-0.0000001f).toPrimitive()) {
                        val outputValue = FastMath.exp(inputValue - localMax)
                        outputBlock[j] = outputValue
                        localSum += outputValue
                    } else {
                        outputBlock[j] = inputValue * expNegLocalMax
                    }
                }
            }

            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                val outputBlock = outputArray.blocks[rowBlockIdx]

                for (j in outputBlock.indices) {
                    outputBlock[j] /= localSum
                }
            }
        }
    }

    return dest
}

@GenerateNameFromPrimitives
internal suspend fun softmaxZeroPrimitive(input: PrimitiveNDArray, rows: Int, columns: Int): MutablePrimitiveNDArray =
    softmaxZeroPrimitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides), rows, columns)
