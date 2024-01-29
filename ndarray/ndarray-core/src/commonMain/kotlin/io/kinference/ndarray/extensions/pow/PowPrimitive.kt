@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE, DataType.INT, DataType.LONG)

package io.kinference.ndarray.extensions.pow

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.applyWithBroadcast
import io.kinference.ndarray.stubs.pow
import io.kinference.primitives.annotations.*
import kotlin.math.pow
import io.kinference.primitives.types.*

@MakePublic
internal suspend fun PrimitiveNDArray.powArray(powArray: NumberNDArrayCore): MutablePrimitiveNDArray {
    val outputShape = broadcastShape(listOf(this.shape, powArray.shape))
    val doublePowArray = powArray.toDoubleNDArray()
    val output = MutablePrimitiveNDArray(outputShape)
    return broadcastTwoTensorsPrimitivePow(this, doublePowArray, output)
}

@GenerateNameFromPrimitives
internal suspend fun broadcastTwoTensorsPrimitivePow(
    left: PrimitiveNDArray,
    right: DoubleNDArray,
    dest: MutablePrimitiveNDArray
): MutablePrimitiveNDArray {
    when {
        left.isScalar() && right.isScalar() -> dest.array[0] = left.singleValue().pow(right.singleValue())
        right.isScalar() -> broadcastRightScalarPow(left.array, right.singleValue(), dest.array)
        left.isScalar() -> broadcastLeftScalarPow(left.singleValue(), right.array, dest.array)
        else -> left.applyWithBroadcast(right, dest) { left, right, dest ->
            left as PrimitiveNDArray; right as DoubleNDArray; dest as MutablePrimitiveNDArray

            for (blockNum in 0 until dest.array.blocksNum) {
                val leftBlock = left.array.getBlock(blockNum)
                val rightBlock = right.array.getBlock(blockNum)
                val destBlock = dest.array.getBlock(blockNum)

                for (idx in destBlock.indices) {
                    destBlock[idx] = leftBlock[idx].pow(rightBlock[idx])
                }
            }
        }
    }
    return dest
}

private fun broadcastRightScalarPow(
    left: PrimitiveTiledArray,
    rightScalar: Double,
    dest: PrimitiveTiledArray,
) {
    require(left.blocksNum == dest.blocksNum && left.blockSize == dest.blockSize)

    for (blockNum in 0 until dest.blocksNum) {
        val leftBlock = left.getBlock(blockNum)
        val destBlock = dest.getBlock(blockNum)

        for (idx in destBlock.indices) {
            destBlock[idx] = leftBlock[idx].pow(rightScalar)
        }
    }
}

private fun broadcastLeftScalarPow(
    leftScalar: PrimitiveType,
    right: DoubleTiledArray,
    dest: PrimitiveTiledArray,
) {
    require(right.blocksNum == dest.blocksNum && right.blockSize == dest.blockSize)

    for (blockNum in 0 until dest.blocksNum) {
        val rightBlock = right.getBlock(blockNum)
        val destBlock = dest.getBlock(blockNum)

        for (idx in destBlock.indices) {
            destBlock[idx] = leftScalar.pow(rightBlock[idx])
        }
    }
}
