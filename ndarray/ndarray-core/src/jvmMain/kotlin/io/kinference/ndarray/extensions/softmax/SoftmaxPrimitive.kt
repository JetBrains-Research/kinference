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


@GenerateNameFromPrimitives
internal suspend fun softmaxPrimitive(input: PrimitiveNDArray, dest: MutablePrimitiveNDArray, rows: Int, columns: Int, stride: Int): MutablePrimitiveNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks

    val outputArray = dest.array
    val maxesArray = PrimitiveTiledArray(rows * stride, inputBlockSize)
    maxesArray.fill(PrimitiveType.MIN_VALUE_FOR_MAX)

    val blocksInStride = stride/inputBlockSize
    val stridesInRow = columns / stride

    //Finding Max along the axis
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
        for (rowNum in rowStart until rowEnd) {
            val rowStrideStart = rowNum * stridesInRow
            val maxBlockStart = rowNum * blocksInStride

            for (strideNum in rowStrideStart until rowStrideStart + stridesInRow) {
                val strideStartBlock = strideNum * blocksInStride

                for (blockNum in 0 until blocksInStride) {
                    val inputBlock = inputBlocks[strideStartBlock + blockNum]
                    val maxBlock = maxesArray.blocks[maxBlockStart + blockNum]

                    for (j in  inputBlock.indices) {
                        maxBlock[j] = max(inputBlock[j], maxBlock[j])
                    }
                }
            }

            for (strideNum in rowStrideStart until rowStrideStart + stridesInRow) {
                val strideStartBlock = strideNum * blocksInStride

                for (blockNum in 0 until blocksInStride) {
                    val maxBlock = maxesArray.blocks[maxBlockStart + blockNum]
                    val inputBlock = inputBlocks[strideStartBlock + blockNum]
                    val outputBlock = outputArray.blocks[strideStartBlock + blockNum]

                    for (j in  inputBlock.indices) {
                        outputBlock[j] = inputBlock[j] - maxBlock[j]
                    }
                }
            }
        }
    }

    // Apply exp for output array
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 2048) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            val outputBlock = outputArray.blocks[blockNum]

            for (j in outputBlock.indices) {
                outputBlock[j] = FastMath.exp(outputBlock[j])
            }
        }
    }

    val sumsArray = PrimitiveTiledArray(rows*stride, inputBlockSize)
    sumsArray.fill((0).toPrimitive())

    // Compute sums along axis and divide output by them
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
        for (rowNum in rowStart until rowEnd) {
            val rowStrideStart = rowNum * stridesInRow
            val maxBlockStart = rowNum * blocksInStride

            for (strideNum in rowStrideStart until rowStrideStart + stridesInRow) {
                val strideStartBlock = strideNum * blocksInStride

                for (blockNum in 0 until blocksInStride) {
                    val outputBlock = outputArray.blocks[strideStartBlock + blockNum]
                    val sumBlock = maxesArray.blocks[maxBlockStart + blockNum]

                    for (j in outputBlock.indices) {
                        sumBlock[j] += outputBlock[j]
                    }
                }
            }

            for (strideNum in rowStrideStart until rowStrideStart + stridesInRow) {
                val strideStartBlock = strideNum * blocksInStride

                for (blockNum in 0 until blocksInStride) {
                    val outputBlock = outputArray.blocks[strideStartBlock + blockNum]
                    val sumBlock = maxesArray.blocks[maxBlockStart + blockNum]

                    for (j in outputBlock.indices) {
                        outputBlock[j] /= sumBlock[j]
                    }
                }
            }
        }
    }

    return dest
}

@GenerateNameFromPrimitives
internal suspend fun softmaxPrimitive(input: PrimitiveNDArray, rows: Int, columns: Int, stride: Int): MutablePrimitiveNDArray =
    softmaxPrimitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides), rows, columns, stride)
