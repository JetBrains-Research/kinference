@file:GeneratePrimitives(
    DataType.UBYTE,
    DataType.USHORT,
    DataType.UINT,
    DataType.ULONG,
)
package io.kinference.ndarray.extensions.bitwise.shift

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.applyWithBroadcast
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

suspend fun PrimitiveNDArray.bitShift(amountsOfShift: PrimitiveNDArray, direction: BitShiftDirection): MutablePrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, amountsOfShift.shape))
    return bitShift(amountsOfShift, direction, MutablePrimitiveNDArray(destShape))
}

suspend fun PrimitiveNDArray.bitShift(amountsOfShift: PrimitiveNDArray, direction: BitShiftDirection, destination: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val shiftFunction = when(direction) {
        BitShiftDirection.LEFT -> PrimitiveType::shl
        BitShiftDirection.RIGHT -> PrimitiveType::shr
    } as PrimitiveType.(Int) -> PrimitiveType

    when {
        this.isScalar() && amountsOfShift.isScalar() -> destination.array.blocks[0][0] = this.singleValue().shiftFunction(amountsOfShift.singleValue().toInt())
        amountsOfShift.isScalar() -> bitShiftScalar(this.array, amountsOfShift.singleValue().toInt(), destination.array, shiftFunction)
        this.isScalar() -> bitShiftFromScalar(this.singleValue(), amountsOfShift.array, destination.array, shiftFunction)
        else -> this.applyWithBroadcast(amountsOfShift, destination) { left, right, dest ->
            left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

            for (blockNum in 0 until left.array.blocksNum) {
                val leftBlock = left.array.blocks[blockNum]
                val rightBlock = right.array.blocks[blockNum]
                val destBlock = dest.array.blocks[blockNum]

                for (idx in destBlock.indices) {
                    destBlock[idx] = leftBlock[idx].shiftFunction(rightBlock[idx].toInt())
                }
            }
        }
    }

    return destination
}


private fun bitShiftScalar(
    inputArray: PrimitiveTiledArray,
    shiftScalar: Int,
    destination: PrimitiveTiledArray,
    shiftFunction: PrimitiveType.(Int) -> PrimitiveType
) {
    require(inputArray.blocksNum == destination.blocksNum && inputArray.blockSize == destination.blockSize)

    for (blockNum in 0 until inputArray.blocksNum) {
        val arrayBlock = inputArray.blocks[blockNum]
        val destBlock = destination.blocks[blockNum]

        for (idx in destBlock.indices) {
            destBlock[idx] = arrayBlock[idx].shiftFunction(shiftScalar)
        }
    }
}

private fun bitShiftFromScalar(
    inputScalar: PrimitiveType,
    shiftArray: PrimitiveTiledArray,
    destination: PrimitiveTiledArray,
    shiftFunction: PrimitiveType.(Int) -> PrimitiveType
) {
    require(shiftArray.blocksNum == destination.blocksNum && shiftArray.blockSize == destination.blockSize)

    for (blockNum in 0 until shiftArray.blocksNum) {
        val shiftBlock = shiftArray.blocks[blockNum]
        val destBlock = destination.blocks[blockNum]

        for (idx in destBlock.indices) {
            destBlock[idx] = inputScalar.shiftFunction(shiftBlock[idx].toInt())
        }
    }
}
