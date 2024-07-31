@file:GeneratePrimitives(DataType.NUMBER)
package io.kinference.ndarray.extensions.dot

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*
import io.kinference.utils.launchWithLimitOrDefault
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

    val destCopy = coroutineScope {
        val leftBlocksInProgress = async {
            if (left.array.blockSize > lBlockSize) {
                copyArray(left, blockSize = lBlockSize)
            } else {
                left.array.blocks to left.blocksInRow
            }
        }

        val rightBlocksInProgress = async {
            if (right.array.blockSize > rdBlockSize) {
                copyArray(right, blockSize = rdBlockSize)
            } else {
                right.array.blocks to right.blocksInRow
            }
        }

        val destBlocksInProgress = async {
            if (dest.array.blockSize > rdBlockSize) {
                emptyBlocks(dest.shape, blockSize = rdBlockSize).first
            } else {
                dest.array.blocks
            }
        }

        val (leftBlocks, lBlocksInRow) = leftBlocksInProgress.await()
        val aBlockSize = leftBlocks[0].size

        val (rightBlocks, rdBlocksInRow) = rightBlocksInProgress.await()

        val destBlocks = destBlocksInProgress.await()

        for (iStart in 0 until n step nBatchSize) {
            val iEnd = min(iStart + nBatchSize, n)
            for (rdCol in 0 until rdBlocksInRow) launchWithLimitOrDefault {
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

        return@coroutineScope destBlocks
    }

    copyBlocks(destCopy, dest.array.blocks)

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

    val inputBlocks = array.array.blocks
    val inputBlockSize = array.array.blockSize

    var inputBlockIdx = 0
    var inputBlockOffset = 0

    val outputArray = Array(shape[0] * blocksInRow) { blockIdx ->
        val actualBlockSize = if ((blockIdx + 1) % blocksInRow == 0) lastBlockSize else blockSize

        val primitiveArray = PrimitiveArray(actualBlockSize)

        var copied = 0
        while (copied < actualBlockSize) {
            val inputBlock = inputBlocks[inputBlockIdx]
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

private fun copyBlocks(srcBlocks: Array<PrimitiveArray>, dstBlocks: Array<PrimitiveArray>) {
    if (srcBlocks === dstBlocks) return

    var srcBlockIdx = 0
    var srcBlockOffset = 0

    var dstBlockIdx = 0
    var dstBlockOffset = 0

    while (srcBlockIdx < srcBlocks.size && dstBlockIdx < dstBlocks.size) {
        val srcBlock = srcBlocks[srcBlockIdx]
        val dstBlock = dstBlocks[dstBlockIdx]

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
