@file:GeneratePrimitives(DataType.NUMBER)
package io.kinference.ndarray.extensions.dot

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.parallelizeByRows
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType

internal suspend fun dotParallelN(left: PrimitiveNDArray, right: PrimitiveNDArray, dest: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val n = left.shape[0]
    val t = left.shape[1]
    val m = right.shape[1]


    val lBlocksInRow = left.blocksInRow
    val rdBlocksInRow = right.blocksInRow

    val leftArray = left.array
    val rightArray = right.array
    val destArray = dest.array

    val lBlockSize = left.array.blockSize

    val nRowFlop = t * m

    // Constant 261120 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(nRowFlop, n, DotUtils.MIN_DATA_PER_LAUNCH) { nStart, nEnd ->
        for (i in nStart until nEnd) {
            val leftBlockOffset = i * lBlocksInRow
            val destBlockOffset = i * rdBlocksInRow
//            val rightBlockIterator = rightBlocks.iterator()

            for (lCol in 0 until lBlocksInRow) {
                val leftBlock = leftArray.getBlock(leftBlockOffset + lCol)

                for (k in 0 until lBlockSize) {
                    val temp = leftBlock[k]

                    for (rdCol in 0 until rdBlocksInRow) {
                        val destBlock = destArray.getBlock(destBlockOffset + rdCol)
                        val rightBlock = rightArray.getBlock((lCol * lBlockSize + k) * rdBlocksInRow + rdCol)

                        for (j in destBlock.indices) {
                            destBlock[j] = (destBlock[j] + temp * rightBlock[j]).toPrimitive()
                        }
                    }
                }
            }
        }
    }

    return dest
}
