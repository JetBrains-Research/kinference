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
import io.kinference.utils.inlines.*

@GenerateNameFromPrimitives
internal suspend fun broadcastTwoTensorsPrimitive(
    left: PrimitiveNDArray,
    right: PrimitiveNDArray,
    dest: MutablePrimitiveNDArray,
    op: (InlinePrimitive, InlinePrimitive) -> InlinePrimitive
): MutablePrimitiveNDArray {
    when {
        left.isScalar() && right.isScalar() -> dest.array.blocks[0][0] = op(InlinePrimitive(left.singleValue()), InlinePrimitive(right.singleValue())).value
        right.isScalar() -> broadcastRightScalar(left.array, right.singleValue(), dest.array, op)
        left.isScalar() -> broadcastLeftScalar(left.singleValue(), right.array, dest.array, op)
        else -> left.applyWithBroadcast(right, dest) { left, right, dest ->
            left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

            for (blockNum in 0 until dest.array.blocksNum) {
                val leftBlock = left.array.blocks[blockNum]
                val rightBlock = right.array.blocks[blockNum]
                val destBlock = dest.array.blocks[blockNum]

                for (idx in destBlock.indices) {
                    destBlock[idx] = op(InlinePrimitive(leftBlock[idx]), InlinePrimitive(rightBlock[idx])).value
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
    op: (InlinePrimitive, InlinePrimitive) -> InlinePrimitive
) {
    require(left.blocksNum == dest.blocksNum && left.blockSize == dest.blockSize)

    for (blockNum in 0 until dest.blocksNum) {
        val leftBlock = left.blocks[blockNum]
        val destBlock = dest.blocks[blockNum]

        for (idx in destBlock.indices) {
            destBlock[idx] = op(InlinePrimitive(leftBlock[idx]), InlinePrimitive(rightScalar)).value
        }
    }
}

private fun broadcastLeftScalar(
    leftScalar: PrimitiveType,
    right: PrimitiveTiledArray,
    dest: PrimitiveTiledArray,
    op: (InlinePrimitive, InlinePrimitive) -> InlinePrimitive
) {
    require(right.blocksNum == dest.blocksNum && right.blockSize == dest.blockSize)

    for (blockNum in 0 until dest.blocksNum) {
        val rightBlock = right.blocks[blockNum]
        val destBlock = dest.blocks[blockNum]

        for (idx in destBlock.indices) {
            destBlock[idx] = op(InlinePrimitive(leftScalar), InlinePrimitive(rightBlock[idx])).value
        }
    }
}
