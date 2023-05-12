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

package io.kinference.ndarray.extensions.bitwise.xor

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsPrimitive
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import kotlin.experimental.xor


suspend fun PrimitiveNDArray.bitXor(other: PrimitiveNDArray): MutablePrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, other.shape))
    return bitXor(other, MutablePrimitiveNDArray(destShape))
}

suspend fun PrimitiveNDArray.bitXor(other: PrimitiveNDArray, dest: MutablePrimitiveNDArray) =
    broadcastTwoTensorsPrimitive(this, other, dest) { left: PrimitiveType, right: PrimitiveType -> left xor right }
