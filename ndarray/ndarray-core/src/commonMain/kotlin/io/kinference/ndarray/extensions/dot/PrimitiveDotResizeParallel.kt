@file:GeneratePrimitives(DataType.NUMBER)
package io.kinference.ndarray.extensions.dot

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*
import kotlinx.coroutines.*
import kotlin.math.min

private val PAGE_PRIMITIVE = DotUtils.PAGE_BYTES / PrimitiveType.SIZE_BYTES

internal suspend fun dotResizeParallel(left: PrimitiveNDArray, right: PrimitiveNDArray, dest: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val n = left.shape[0]
    val t = right.shape[0]
    val m = right.shape[1]

    val nBatchSize: Int
    val lBlockSize: Int
    val rdBlockSize = PAGE_PRIMITIVE

    if (n / t >= 10) {
        nBatchSize = 256
        lBlockSize = 24
    } else {
        nBatchSize = 24
        lBlockSize = 30
    }

    val (destCopy, needToCopyBack) = coroutineScope {
        val leftBlocksInProgress = async {
            if (left.array.blockSize > lBlockSize) {
                copyArray(left, blockSize = lBlockSize)
            } else {
                left.array.copyOfBlocks() to left.blocksInRow
            }
        }

        val rightBlocksInProgress = async {
            if (right.array.blockSize > rdBlockSize) {
                copyArray(right, blockSize = rdBlockSize)
            } else {
                right.array.copyOfBlocks() to right.blocksInRow
            }
        }

        val destBlocksInProgress = async {
            if (dest.array.blockSize > rdBlockSize) {
                emptyBlocks(dest.shape, blockSize = rdBlockSize).first to true
            } else {
                dest.array.copyOfBlocks() to false
            }
        }

        val (leftBlocks, lBlocksInRow) = leftBlocksInProgress.await()
        val aBlockSize = leftBlocks[0].size

        val (rightBlocks, rdBlocksInRow) = rightBlocksInProgress.await()

        val (destBlocks, needToCopyBack) = destBlocksInProgress.await()

        for (iStart in 0 until n step nBatchSize) {
            val iEnd = min(iStart + nBatchSize, n)
            for (rdCol in 0 until rdBlocksInRow) launch {
                for (lCol in 0 until lBlocksInRow) {
                    val rightOffset = lCol * aBlockSize
                    for (i in iStart until iEnd) {
                        val destBlock = destBlocks[i * rdBlocksInRow + rdCol]
                        val leftBlock = leftBlocks[i * lBlocksInRow + lCol]

                        for (k in leftBlock.indices) {
                            val rightBlock = rightBlocks[(rightOffset + k) * rdBlocksInRow + rdCol]
                            val leftValue = leftBlock[k]
                            for (j in destBlock.indices) {
                                destBlock[j] = (destBlock[j] + leftValue * rightBlock[j]).toPrimitive()
                            }
                        }
                    }
                }
            }
        }

        return@coroutineScope destBlocks to needToCopyBack
    }

    if (needToCopyBack) {
        copyBlocks(destCopy, dest.array)
    }

    return dest
}


private fun emptyBlocks(shape: IntArray, blockSize: Int): Pair<Array<PrimitiveArray>, Int> {
    val blocksInRow: Int
    val lastBlockSize: Int

    if (shape[1] % blockSize == 0) {
        blocksInRow = shape[1] / blockSize
        lastBlockSize = blockSize
    } else {
        blocksInRow = shape[1] / blockSize + 1
        lastBlockSize = shape[1] % blockSize
    }

    val array = Array(shape[0] * blocksInRow) { blockIdx ->
        val actualBlockSize = if ((blockIdx + 1) % blocksInRow == 0) lastBlockSize else blockSize
        PrimitiveArray(actualBlockSize)
    }

    return array to blocksInRow
}

// copyArray is a merge of emptyBlocks and copyBlocks
private fun copyArray(array: PrimitiveNDArray, blockSize: Int): Pair<Array<PrimitiveArray>, Int> {
    val shape = array.shape

    val blocksInRow: Int
    val lastBlockSize: Int

    if (shape[1] % blockSize == 0) {
        blocksInRow = shape[1] / blockSize
        lastBlockSize = blockSize
    } else {
        blocksInRow = shape[1] / blockSize + 1
        lastBlockSize = shape[1] % blockSize
    }

    val inputArray = array.array
    val inputBlockSize = array.array.blockSize

    var inputBlockIdx = 0
    var inputBlockOffset = 0

    val outputArray = Array(shape[0] * blocksInRow) { blockIdx ->
        val actualBlockSize = if ((blockIdx + 1) % blocksInRow == 0) lastBlockSize else blockSize

        val primitiveArray = PrimitiveArray(actualBlockSize)

        var copied = 0
        while (copied < actualBlockSize) {
            val inputBlock = inputArray.getBlock(inputBlockIdx)
            val copySize = min(inputBlockSize - inputBlockOffset, actualBlockSize - copied)

            inputBlock.copyInto(
                primitiveArray,
                destinationOffset = copied,
                startIndex = inputBlockOffset,
                endIndex = inputBlockOffset + copySize
            )

            copied += copySize
            inputBlockOffset += copySize

            if (inputBlockOffset == inputBlockSize) {
                inputBlockIdx++
                inputBlockOffset = 0
            }
        }

        primitiveArray
    }

    return outputArray to blocksInRow
}

private fun copyBlocks(srcBlocks: Array<PrimitiveArray>, dstArray: PrimitiveTiledArray) {
    var srcBlockIdx = 0
    var srcBlockOffset = 0

    var dstBlockIdx = 0
    var dstBlockOffset = 0

    while (srcBlockIdx < srcBlocks.size && dstBlockIdx < dstArray.blocksNum) {
        val srcBlock = srcBlocks[srcBlockIdx]
        val dstBlock = dstArray.getBlock(dstBlockIdx)

        val copySize = minOf(srcBlock.size - srcBlockOffset, dstBlock.size - dstBlockOffset)

        srcBlock.copyInto(
            dstBlock,
            destinationOffset = dstBlockOffset,
            startIndex = srcBlockOffset,
            endIndex = srcBlockOffset + copySize
        )

        srcBlockOffset += copySize
        if (srcBlockOffset == srcBlock.size) {
            srcBlockOffset = 0
            srcBlockIdx++
        }

        dstBlockOffset += copySize
        if (dstBlockOffset == dstBlock.size) {
            dstBlockOffset = 0
            dstBlockIdx++
        }
    }
}
