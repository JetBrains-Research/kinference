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
import kotlin.math.*

@GenerateNameFromPrimitives
internal suspend fun softmaxPrimitive(input: PrimitiveNDArray, rows: Int, columns: Int): MutablePrimitiveNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks

    val outputArray = PrimitiveTiledArray(input.linearSize, input.array.blockSize)
    val maxesArray = PrimitiveArray(inputBlocks.size)

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 65536) { blockStart, blockEnd ->
        for (blockNum in blockStart until blockEnd) {
            maxesArray[blockNum] = inputBlocks[blockNum].max()
        }
    }

    val blocksInRow = columns / inputBlockSize

    //Minus maximum from input and store in output
    // Constant 1048576 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd ->
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localMax = PrimitiveType.MIN_VALUE
            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                localMax = max(localMax, maxesArray[rowBlockIdx])
            }

            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                val inputBlock = inputBlocks[rowBlockIdx]
                val outputBlock = outputArray.blocks[rowBlockIdx]

                for (j in outputBlock.indices) {
                    outputBlock[j] = inputBlock[j] - localMax
                }
            }
        }
    }

    // Apply exp for output array
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 2048) { blockStart, blockEnd ->
        for (blockNum in blockStart until blockEnd) {
            val outputBlock = outputArray.blocks[blockNum]

            for (j in outputBlock.indices) {
                outputBlock[j] = FastMath.exp(outputBlock[j])
            }
        }
    }

    val sumsArray = PrimitiveArray(outputArray.blocks.size)

    // Calculate sum for each block
    // Constant 131072 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 131072) { blockStart, blockEnd ->
        for (blockNum in blockStart until blockEnd) {
            sumsArray[blockNum] = outputArray.blocks[blockNum].sum()
        }
    }

    // Div by sum in output array
    // Constant 1048576 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd ->
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localSum = (0).toPrimitive()
            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                localSum += sumsArray[rowBlockIdx]
            }

            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                val outputBlock = outputArray.blocks[rowBlockIdx]

                for (j in outputBlock.indices) {
                    outputBlock[j] /= localSum
                }
            }
        }
    }

    return MutablePrimitiveNDArray(outputArray, input.strides)
}
