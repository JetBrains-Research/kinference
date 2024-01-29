@file:GeneratePrimitives(DataType.NUMBER)
package io.kinference.ndarray.extensions.dot

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.parallelizeByRows
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType

internal suspend fun dotParallelM(left: PrimitiveNDArray, right: PrimitiveNDArray, dest: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val n = left.shape[0]
    val t = left.shape[1]

    val lBlocksInRow = left.blocksInRow
    val rdBlocksInRow = right.blocksInRow

    val leftArray = left.array
    val rightArray = right.array
    val destArray = dest.array

    val lBlockSize = left.array.blockSize
    val rdBlockSize = right.array.blockSize

    val rdBlockFlop = rdBlockSize * n * t

    // Constant 261120 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(rdBlockFlop, rdBlocksInRow, DotUtils.MIN_DATA_PER_LAUNCH) { rdColStart, rdColEnd ->
        for (rdCol in rdColStart until rdColEnd) {
            for (i in 0 until n) {
                val destBlock = destArray.getBlock(i * rdBlocksInRow + rdCol)
                val leftBlockOffset = i * lBlocksInRow

                for (lCol in 0 until lBlocksInRow) {
                    val leftBlock = leftArray.getBlock(leftBlockOffset + lCol)
                    val rightOffset = lCol * lBlockSize

                    for (k in 0 until lBlockSize) {
                        val temp = leftBlock[k]
                        val rightBlock = rightArray.getBlock((rightOffset + k) * rdBlocksInRow + rdCol)

                        for (j in 0 until rdBlockSize) {
                            destBlock[j] = (destBlock[j] + temp * rightBlock[j]).toPrimitive()
                        }
                    }
                }
            }
        }
    }

    return dest
}
