@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.mod

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsPrimitive
import io.kinference.ndarray.inlines.*
import io.kinference.ndarray.math.Math
import io.kinference.ndarray.math.floorMod
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.utils.InlineInt

@MakePublic
internal suspend fun PrimitiveNDArray.mod(other: PrimitiveNDArray, dest: MutablePrimitiveNDArray) =
    broadcastTwoTensorsPrimitive(this, other, dest) { left: InlinePrimitive, right: InlinePrimitive -> InlinePrimitive(Math.floorMod(left.value, right.value)) }

@MakePublic
internal suspend fun PrimitiveNDArray.mod(other: PrimitiveNDArray): PrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, other.shape))
    return mod(other, MutablePrimitiveNDArray(destShape))
}
