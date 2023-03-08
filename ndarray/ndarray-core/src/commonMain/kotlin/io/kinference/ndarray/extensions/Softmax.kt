package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.FloatTiledArray
import io.kinference.primitives.types.DataType
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.*

suspend fun softmaxFloat(input: FloatNDArray, rows: Int, columns: Int): MutableFloatNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks

    val outputArray = FloatTiledArray(input.linearSize, input.array.blockSize)
    val maxesArray = FloatArray(inputBlocks.size)

    val maxBatchSize = run {
        var batchSize = 1
        while (batchSize < inputBlocks.size && batchSize * inputBlockSize < 65536) {
            batchSize++
        }
        batchSize
    }

    fun maxWrapper(startBlock: Int, endBlock: Int) {
        for (blockNum in startBlock until endBlock) {
            maxesArray[blockNum] = inputBlocks[blockNum].max()
        }
    }

    if (maxBatchSize == inputBlocks.size) {
        maxWrapper(0, inputBlocks.size)
    } else {
        coroutineScope {
            for (blockStart in inputBlocks.indices step maxBatchSize) {
                launch {
                    maxWrapper(blockStart, min(blockStart + maxBatchSize, inputBlocks.size))
                }
            }
        }
    }

    val blocksInRow = columns / inputBlockSize


    val minusBatchSize = run {
        var batchSize = 1
        while (batchSize < rows && batchSize * columns < 1048576) {
            batchSize++
        }
        batchSize
    }

    fun minusWrapper(rowStart: Int, rowEnd: Int) {
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localMax = Float.MIN_VALUE
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

    if (minusBatchSize == rows) {
        minusWrapper(0, rows)
    } else {
        coroutineScope {
            for (rowStart in 0 until rows step minusBatchSize) {
                launch {
                    minusWrapper(rowStart, min(rowStart + minusBatchSize, rows))
                }
            }
        }
    }

    val expBatchSize = run {
        var batchSize = 1
        while (batchSize < outputArray.blocks.size && batchSize * outputArray.blockSize < 512) {
            batchSize++
        }
        batchSize
    }

    fun expWrapper(blockStart: Int, blockEnd: Int) {
        for (blockNum in blockStart until blockEnd) {
            val outputBlock = outputArray.blocks[blockNum]

            for (j in outputBlock.indices) {
                outputBlock[j] = exp(outputBlock[j])
            }
        }
    }

    if (expBatchSize == outputArray.blocks.size) {
        expWrapper(0, outputArray.blocks.size)
    } else {
        coroutineScope {
            for (blockStart in 0 until outputArray.blocks.size step expBatchSize) {
                launch {
                    expWrapper(blockStart, min(blockStart + expBatchSize, outputArray.blocks.size))
                }
            }
        }
    }

    val sumsArray = FloatArray(outputArray.blocks.size)

    val sumBatchSize = run {
        var batchSize = 1
        while (batchSize < outputArray.blocks.size && batchSize * outputArray.blockSize < 131072) {
            batchSize++
        }
        batchSize
    }

    fun sumWrapper(blockStart: Int, blockEnd: Int) {
        for (blockNum in blockStart until blockEnd) {
            sumsArray[blockNum] = outputArray.blocks[blockNum].sum()
        }
    }

    if (sumBatchSize == outputArray.blocks.size) {
        sumWrapper(0, outputArray.blocks.size)
    } else {
        coroutineScope {
            for (blockStart in 0 until outputArray.blocks.size step sumBatchSize) {
                launch {
                    sumWrapper(blockStart, min(blockStart + sumBatchSize, outputArray.blocks.size))
                }
            }
        }
    }

    fun divWrapper(rowStart: Int, rowEnd: Int) {
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localSum = 0f
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

    if (minusBatchSize == rows) {
        divWrapper(0, rows)
    } else {
        coroutineScope {
            for (rowStart in 0 until rows step minusBatchSize) {
                launch {
                    divWrapper(rowStart, min(rowStart + minusBatchSize, rows))
                }
            }
        }
    }

    return MutableFloatNDArray(outputArray, input.strides)
}

suspend fun softmax(
    input: NDArrayCore,
    axis: Int = 0,
    strides: Strides = input.strides
): MutableNDArrayCore {
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)

    val actualAxis = input.indexAxis(axis)
    val shape = input.shape
    val rows = run {
        var rows = 1
        for (idx in 0 until actualAxis) {
            rows *= shape[idx]
        }
        rows
    }

    val columns = run {
        var columns = 1
        for (idx in actualAxis until shape.size) {
            columns *= shape[idx]
        }
        columns
    }

    return if (input.type == DataType.FLOAT) {
        softmaxFloat(input as FloatNDArray, rows, columns)
    } else {
        error("Unsupported Data Type")
    }
}
