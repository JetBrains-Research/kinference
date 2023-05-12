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
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsPrimitive
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

    return broadcastTwoTensorsPrimitive(this, amountsOfShift, destination) { left: PrimitiveType, right: PrimitiveType ->
        left.shiftFunction(right.toInt())
    }
}

