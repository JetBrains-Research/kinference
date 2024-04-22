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
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsPrimitive
import io.kinference.ndarray.inlines.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.utils.InlineInt

@MakePublic
internal suspend fun PrimitiveNDArray.bitAnd(other: PrimitiveNDArray): MutablePrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, other.shape))
    return bitAnd(other, MutablePrimitiveNDArray(destShape))
}

@MakePublic
internal suspend fun PrimitiveNDArray.bitAnd(other: PrimitiveNDArray, dest: MutablePrimitiveNDArray) =
    broadcastTwoTensorsPrimitive(this, other, dest) { left: InlinePrimitive, right: InlinePrimitive -> left and right }
