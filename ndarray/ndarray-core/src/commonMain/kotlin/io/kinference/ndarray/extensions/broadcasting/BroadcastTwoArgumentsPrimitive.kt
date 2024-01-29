@file:GeneratePrimitives(
    DataType.ALL
)
package io.kinference.ndarray.extensions.broadcasting

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.applyWithBroadcast
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@GenerateNameFromPrimitives
internal suspend fun broadcastTwoTensorsPrimitive(
    left: PrimitiveNDArray,
    right: PrimitiveNDArray,
    dest: MutablePrimitiveNDArray,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType
): MutablePrimitiveNDArray {
    when {
        left.isScalar() && right.isScalar() -> dest.array[0] = op(left.singleValue(), right.singleValue())
        right.isScalar() -> broadcastRightScalar(left.array, right.singleValue(), dest.array, op)
        left.isScalar() -> broadcastLeftScalar(left.singleValue(), right.array, dest.array, op)
        else -> left.applyWithBroadcast(right, dest) { left, right, dest ->
            left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

            for (blockNum in 0 until dest.array.blocksNum) {
                val leftBlock = left.array.getBlock(blockNum)
                val rightBlock = right.array.getBlock(blockNum)
                val destBlock = dest.array.getBlock(blockNum)

                for (idx in destBlock.indices) {
                    destBlock[idx] = op(leftBlock[idx], rightBlock[idx])
                }
            }
        }
    }
    return dest
}

private fun broadcastRightScalar(
    left: PrimitiveTiledArray,
    rightScalar: PrimitiveType,
    dest: PrimitiveTiledArray,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType
) {
    require(left.blocksNum == dest.blocksNum && left.blockSize == dest.blockSize)

    for (blockNum in 0 until dest.blocksNum) {
        val leftBlock = left.getBlock(blockNum)
        val destBlock = dest.getBlock(blockNum)

        for (idx in destBlock.indices) {
            destBlock[idx] = op(leftBlock[idx], rightScalar)
        }
    }
}

private fun broadcastLeftScalar(
    leftScalar: PrimitiveType,
    right: PrimitiveTiledArray,
    dest: PrimitiveTiledArray,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType
) {
    require(right.blocksNum == dest.blocksNum && right.blockSize == dest.blockSize)

    for (blockNum in 0 until dest.blocksNum) {
        val rightBlock = right.getBlock(blockNum)
        val destBlock = dest.getBlock(blockNum)

        for (idx in destBlock.indices) {
            destBlock[idx] = op(leftScalar, rightBlock[idx])
        }
    }
}
