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
import io.kinference.ndarray.stubs.MIN_VALUE_FOR_MAX
import kotlin.math.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.math.exp


@GenerateNameFromPrimitives
internal suspend fun softmaxVer13Primitive(
    input: PrimitiveNDArray,
    dest: MutablePrimitiveNDArray,
    rows: Int,
    columns: Int,
    stride: Int
): MutablePrimitiveNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks
    val outputArray = dest.array
    val outputBlocks = outputArray.blocks

    val blocksInStride = stride / inputBlockSize
    val blocksInRow = columns / inputBlockSize
    val stridesInRow = columns / stride

    //Finding Max along the axis
    if (stride > 1) {
        val maxesArray = PrimitiveTiledArray(rows * stride, inputBlockSize)
        maxesArray.fill(PrimitiveType.MIN_VALUE_FOR_MAX)
        parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
            for (rowNum in rowStart until rowEnd) {
                val rowStrideStart = rowNum * stridesInRow
                val maxBlockStart = rowNum * blocksInStride

                for (strideNum in rowStrideStart until rowStrideStart + stridesInRow) {
                    val strideStartBlock = strideNum * blocksInStride

                    for (blockNum in 0 until blocksInStride) {
                        val inputBlock = inputBlocks[strideStartBlock + blockNum]
                        val maxBlock = maxesArray.blocks[maxBlockStart + blockNum]

                        for (j in inputBlock.indices) {
                            maxBlock[j] = max(inputBlock[j], maxBlock[j])
                        }
                    }
                }

                for (strideNum in rowStrideStart until rowStrideStart + stridesInRow) {
                    val strideStartBlock = strideNum * blocksInStride

                    for (blockNum in 0 until blocksInStride) {
                        val maxBlock = maxesArray.blocks[maxBlockStart + blockNum]
                        val inputBlock = inputBlocks[strideStartBlock + blockNum]
                        val outputBlock = outputBlocks[strideStartBlock + blockNum]

                        for (j in inputBlock.indices) {
                            outputBlock[j] = inputBlock[j] - maxBlock[j]
                        }
                    }
                }

            }
        }
    } else {
        parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
            for (row in rowStart until rowEnd) {
                val maxBlock = PrimitiveArray(inputBlockSize) { PrimitiveType.MIN_VALUE_FOR_MAX }
                for (blockIdx in row * blocksInRow until (row + 1) * blocksInRow) {
                    val inputBlock = inputBlocks[blockIdx]
                    for (j in inputBlock.indices) {
                        maxBlock[j] = max(inputBlock[j], maxBlock[j])
                    }
                }
                maxBlock.fill(maxBlock.max())
                for (blockIdx in row * blocksInRow until (row + 1) * blocksInRow) {
                    val inputBlock = inputBlocks[blockIdx]
                    val outputBlock = outputBlocks[blockIdx]
                    for (j in inputBlock.indices) {
                        outputBlock[j] = inputBlock[j] - maxBlock[j]
                    }
                }
            }
        }
    }

    // Apply exp for output array
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 2048) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            val outputBlock = outputBlocks[blockNum]

            for (j in outputBlock.indices) {
                outputBlock[j] = FastMath.exp(outputBlock[j])
            }
        }
    }

    if (stride > 1) {
        val sumsArray = PrimitiveTiledArray(rows * stride, inputBlockSize)
        val sumBlocks = sumsArray.blocks
        sumsArray.fill((0).toPrimitive())

        parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
            for (rowNum in rowStart until rowEnd) {
                val rowStrideStart = rowNum * stridesInRow
                val sumBlockStart = rowNum * blocksInStride

                for (strideNum in rowStrideStart until rowStrideStart + stridesInRow) {
                    val strideStartBlock = strideNum * blocksInStride

                    for (blockNum in 0 until blocksInStride) {
                        val outputBlock = outputBlocks[strideStartBlock + blockNum]
                        val sumBlock = sumBlocks[sumBlockStart + blockNum]

                        for (j in outputBlock.indices) {
                            sumBlock[j] += outputBlock[j]
                        }
                    }
                }

                for (strideNum in rowStrideStart until rowStrideStart + stridesInRow) {
                    val strideStartBlock = strideNum * blocksInStride

                    for (blockNum in 0 until blocksInStride) {
                        val outputBlock = outputBlocks[strideStartBlock + blockNum]
                        val sumBlock = sumBlocks[sumBlockStart + blockNum]

                        for (j in outputBlock.indices) {
                            outputBlock[j] /= sumBlock[j]
                        }
                    }
                }
            }
        }
    } else {
        parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
            val sumsArray = PrimitiveArray(inputBlockSize)
            for (row in rowStart until rowEnd) {
                sumsArray.fill((0).toPrimitive())
                for (blockIdx in row * blocksInRow until (row + 1) * blocksInRow) {
                    val outputBlock = outputBlocks[blockIdx]
                    for (j in outputBlock.indices) {
                        sumsArray[j] += outputBlock[j]
                    }
                }

                sumsArray.fill(sumsArray.sum())
                for (blockIdx in row * blocksInRow until (row + 1) * blocksInRow) {
                    val outputBlock = outputBlocks[blockIdx]
                    for (j in outputBlock.indices) {
                        outputBlock[j] /= sumsArray[j]
                    }
                }
            }
        }

    }

    return dest
}

@GenerateNameFromPrimitives
internal suspend fun softmaxVer13Primitive(input: PrimitiveNDArray, rows: Int, columns: Int, stride: Int): MutablePrimitiveNDArray =
    softmaxVer13Primitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides), rows, columns, stride)

@GenerateNameFromPrimitives
internal suspend fun softmaxVer1Primitive(input: PrimitiveNDArray, dest: MutablePrimitiveNDArray, rows: Int, columns: Int): MutablePrimitiveNDArray {
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
    // Constant 1048576 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localMax = PrimitiveType.MIN_VALUE_FOR_MAX
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
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 2048) { blockStart, blockEnd, _ ->
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
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 131072) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            sumsArray[blockNum] = outputArray.blocks[blockNum].sum()
        }
    }

    // Div by sum in output array
    // Constant 1048576 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
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

    return dest
}

@GenerateNameFromPrimitives
internal suspend fun softmaxVer1Primitive(input: PrimitiveNDArray, rows: Int, columns: Int): MutablePrimitiveNDArray =
    softmaxVer1Primitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides), rows, columns)
