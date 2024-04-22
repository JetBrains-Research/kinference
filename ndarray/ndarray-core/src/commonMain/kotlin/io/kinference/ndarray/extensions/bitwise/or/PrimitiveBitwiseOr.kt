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

package io.kinference.ndarray.extensions.bitwise.or

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsPrimitive
import io.kinference.ndarray.inlines.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.utils.InlineInt

@MakePublic
internal suspend fun PrimitiveNDArray.bitOr(other: PrimitiveNDArray): MutablePrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, other.shape))
    return bitOr(other, MutablePrimitiveNDArray(destShape))
}

@MakePublic
internal suspend fun PrimitiveNDArray.bitOr(other: PrimitiveNDArray, dest: MutablePrimitiveNDArray) =
    broadcastTwoTensorsPrimitive(this, other, dest) { left: InlinePrimitive, right: InlinePrimitive -> left or right }
