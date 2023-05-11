@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.UBYTE,
    DataType.USHORT,
    DataType.UINT,
    DataType.ULONG,
)

package io.kinference.ndarray.extensions.bitwise.and

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.applyWithBroadcast
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import kotlin.experimental.and

suspend fun PrimitiveNDArray.bitAnd(other: PrimitiveNDArray): MutablePrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, other.shape))
    return bitAnd(other, MutablePrimitiveNDArray(destShape))
}

suspend fun PrimitiveNDArray.bitAnd(other: PrimitiveNDArray, dest: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    when {
        this.isScalar() && other.isScalar() -> dest.array.blocks[0][0] = this.singleValue() and other.singleValue()
        other.isScalar() -> bitAndScalar(this.array, other.singleValue(), dest.array)
        this.isScalar() -> bitAndFromScalar(this.singleValue(), other.array, dest.array)
        else -> this.applyWithBroadcast(other, dest) { left, right, dest ->
            left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

            for (blockNum in 0 until dest.array.blocksNum) {
                val leftBlock = left.array.blocks[blockNum]
                val rightBlock = right.array.blocks[blockNum]
                val destBlock = dest.array.blocks[blockNum]

                for (idx in destBlock.indices) {
                    destBlock[idx] = leftBlock[idx] and rightBlock[idx]
                }
            }
        }
    }

    return dest
}

private fun bitAndScalar(
    inputArray: PrimitiveTiledArray,
    scalar: PrimitiveType,
    dest: PrimitiveTiledArray
) {
    require(inputArray.blocksNum == dest.blocksNum && inputArray.blockSize == dest.blockSize)

    for (blockNum in 0 until dest.blocksNum) {
        val inputBlock = inputArray.blocks[blockNum]
        val destBlock = dest.blocks[blockNum]

        for (idx in destBlock.indices) {
            destBlock[idx] = inputBlock[idx] and scalar
        }
    }
}

private fun bitAndFromScalar(
    inputScalar: PrimitiveType,
    otherArray: PrimitiveTiledArray,
    dest: PrimitiveTiledArray
) {
    require(otherArray.blocksNum == dest.blocksNum && otherArray.blockSize == dest.blockSize)

    for (blockNum in 0 until dest.blocksNum) {
        val otherBlock = otherArray.blocks[blockNum]
        val destBlock = dest.blocks[blockNum]

        for (idx in destBlock.indices) {
            destBlock[idx] = inputScalar and otherBlock[idx]
        }
    }
}
