@file:GeneratePrimitives(
    DataType.ALL
)
package io.kinference.ndarray.extensions.broadcasting

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@GenerateNameFromPrimitives
internal fun broadcastTwoTensorsPrimitive(
    left: PrimitiveNDArray,
    right: PrimitiveNDArray,
    dest: MutablePrimitiveNDArray,
    op: PrimitiveBinaryOperation
): MutablePrimitiveNDArray {
    val broadcastingInfo = BroadcastingInfo.create(listOf(left, right))
    require(dest.shape.contentEquals(broadcastingInfo.destShape)) { "Destination has incorrect shape, expected: ${broadcastingInfo.destShape.joinToString()}, actual ${dest.shape.joinToString()}" }

    if (broadcastingInfo.broadcastingAxes.isEmpty()) {
        return executeWithoutBroadcasting(left, right, dest, op)
    }

    val totalAxesToBroadcast = if (broadcastingInfo.broadcastAlongLastAxis)
        broadcastingInfo.broadcastingAxes.size - 1
    else
        broadcastingInfo.broadcastingAxes.size

    val (leftBroadcastingShape, rightBroadcastingShape) = broadcastingInfo.broadcastingShapes
    val destBroadcastingShape = broadcastingInfo.broadcastingDestShape

    val destBlocksInRow = destBroadcastingShape.last() / dest.array.blockSize

    val leftOffsets = makeOffsets(leftBroadcastingShape, leftBroadcastingShape.last() / left.array.blockSize)
    val rightOffsets = makeOffsets(rightBroadcastingShape, rightBroadcastingShape.last() / right.array.blockSize)
    val destOffsets = makeOffsets(destBroadcastingShape, destBlocksInRow)

    val leftIsScalar = broadcastingInfo.broadcastAlongLastAxis && leftBroadcastingShape.last() == 1
    val rightIsScalar = broadcastingInfo.broadcastAlongLastAxis && rightBroadcastingShape.last() == 1

    val leftBlocks = left.array.blocks
    val rightBlocks = right.array.blocks
    val destBlocks = dest.array.blocks

    val leftIsScalarFun = { leftOffset: Int, rightOffset: Int, destOffset: Int, axisToBroadcastIdx: Int ->
        val shapeIdx = axisToBroadcastIdx * 2
        val batchSize = destBroadcastingShape[shapeIdx]

        for (batchIdx in 0 until batchSize) {
            val leftScalar = leftBlocks[leftOffset][0]

            for (blockIdx in 0 until destBlocksInRow) {
                val destBlock = destBlocks[destOffset + blockIdx]
                val rightBlock = rightBlocks[rightOffset + blockIdx]

                for (idx in destBlock.indices) {
                    destBlock[idx] = op(leftScalar, rightBlock[idx])
                }
            }
        }
    }

    val rightIsScalarFun = { leftOffset: Int, rightOffset: Int, destOffset: Int, axisToBroadcastIdx: Int ->
        val shapeIdx = axisToBroadcastIdx * 2
        val batchSize = destBroadcastingShape[shapeIdx]

        for (batchIdx in 0 until batchSize) {
            val rightScalar = rightBlocks[rightOffset][0]

            for (blockIdx in 0 until destBlocksInRow) {
                val destBlock = destBlocks[destOffset + blockIdx]
                val leftBlock = leftBlocks[leftOffset + blockIdx]

                for (idx in destBlock.indices) {
                    destBlock[idx] = op(leftBlock[idx], rightScalar)
                }
            }
        }
    }

    val defaultFun = { leftOffset: Int, rightOffset: Int, destOffset: Int, axisToBroadcastIdx: Int ->
        for (blockIdx in 0 until destBlocksInRow) {
            val leftBlock = leftBlocks[leftOffset + blockIdx]
            val rightBlock = rightBlocks[rightOffset + blockIdx]
            val destBlock = destBlocks[destOffset + blockIdx]

            for (idx in destBlock.indices) {
                destBlock[idx] = op(leftBlock[idx], rightBlock[idx])
            }
        }
    }

    val broadcastingFun = when {
        leftIsScalar -> leftIsScalarFun
        rightIsScalar -> rightIsScalarFun
        else -> defaultFun
    }

    fun broadcast(leftOffset: Int, rightOffset: Int, destOffset: Int, axisToBroadcastIdx: Int) {
        if (axisToBroadcastIdx == totalAxesToBroadcast) {
            broadcastingFun(leftOffset, rightOffset, destOffset, axisToBroadcastIdx)
        } else {
            val shapeIdx = axisToBroadcastIdx * 2

            val batchSize = destBroadcastingShape[shapeIdx]
            val dimSize = destBroadcastingShape[shapeIdx + 1]

            for (batchIdx in 0 until batchSize) {
                val leftBatchOffset = leftOffset + leftOffsets[shapeIdx] * batchIdx
                val rightBatchOffset = rightOffset + rightOffsets[shapeIdx] * batchIdx
                val destBatchOffset = destOffset + destOffsets[shapeIdx] * batchIdx

                for (dimIdx in 0 until dimSize) {
                    val leftFullOffset = leftBatchOffset + (dimIdx % leftBroadcastingShape[shapeIdx + 1]) * leftOffsets[shapeIdx + 1]
                    val rightFullOffset = rightBatchOffset + (dimIdx % rightBroadcastingShape[shapeIdx + 1]) * rightOffsets[shapeIdx + 1]
                    val destFullOffset = destBatchOffset + dimIdx * destOffsets[shapeIdx + 1]

                    broadcast(leftFullOffset, rightFullOffset, destFullOffset, axisToBroadcastIdx + 1)
                }
            }
        }
    }

    broadcast(0, 0, 0, 0)

    return dest
}

private fun executeWithoutBroadcasting(
    left: PrimitiveNDArray,
    right: PrimitiveNDArray,
    dest: MutablePrimitiveNDArray,
    op: PrimitiveBinaryOperation
): MutablePrimitiveNDArray {
    val leftBlocks = left.array.blocks
    val rightBlocks = right.array.blocks
    val destBlocks = dest.array.blocks

    for (blockIdx in destBlocks.indices) {
        val destBlock = destBlocks[blockIdx]
        val leftBlock = leftBlocks[blockIdx]
        val rightBlock = rightBlocks[blockIdx]

        for (idx in destBlock.indices) {
            destBlock[idx] = op(leftBlock[idx], rightBlock[idx])
        }
    }

    return dest
}
