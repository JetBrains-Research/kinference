@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)
package io.kinference.ndarray.extensions.det

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.pow
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*
import kotlin.math.pow

private val ZERO = (0).toPrimitive()
private val ONE = (1).toPrimitive()
private val MINUS_ONE = (-1).toPrimitive()


//TODO: Research coroutines there
suspend fun PrimitiveNDArray.det(): PrimitiveNDArray {
    val batchSize = this.computeBlockSize(0, shape.size - 2)

    val n = shape.last()
    val reshapedInput = this.reshape(intArrayOf(batchSize, n, n)) as PrimitiveNDArray

    val outputShape = this.shape.sliceArray(0 until shape.size - 2)
    val outputArray = if (outputShape.isEmpty()) MutablePrimitiveNDArray.scalar(ZERO) else MutablePrimitiveNDArray(outputShape)

    val blocksInRow = reshapedInput.blocksInRow

    fun findNonZeroRow(i: Int, matrix: PrimitiveNDArray): Int {
        for (idx in i until n) {
            if (matrix[idx, i] != ZERO) return idx
        }

        return n
    }

    for (batchIdx in 0 until batchSize) {
        val inputMatrix = reshapedInput.view(batchIdx).clone()
        val inputBlocks = inputMatrix.array.blocks

        var result = ONE
        var swapsCount = 0

        for (i in 0 until n) {
            val rowForI = findNonZeroRow(i, inputMatrix)

            if (rowForI == n) {
                result = ZERO
                break
            }

            val iBlocksOffset = i * blocksInRow

            if (rowForI != i) {
                swapsCount++

                val iRowBlockOffset = rowForI * blocksInRow

                for (blockIdx in 0 until blocksInRow) {
                    val iBlock = inputBlocks[iBlocksOffset + blockIdx]
                    inputBlocks[iBlocksOffset + blockIdx] = inputBlocks[iRowBlockOffset + blockIdx]
                    inputBlocks[iRowBlockOffset + blockIdx] = iBlock
                }
            }

            val iScale = inputMatrix[i, i] as PrimitiveType

            result *= iScale

            for (j in i + 1 until n) {
                val jScale = inputMatrix[j, i] as PrimitiveType
                val jBlockOffset = j * blocksInRow

                val scale = jScale / iScale

                for (blockIdx in 0 until blocksInRow) {
                    val iBlock = inputBlocks[iBlocksOffset + blockIdx]
                    val jBlock = inputBlocks[jBlockOffset + blockIdx]

                    for (idx in jBlock.indices) {
                        jBlock[idx] = jBlock[idx] - iBlock[idx] * scale
                    }
                }
            }
        }

        outputArray.setLinear(batchIdx, result * MINUS_ONE.pow(swapsCount % 2))
    }

    return outputArray
}
