@file:GeneratePrimitives(
    DataType.NUMBER
)

package io.kinference.ndarray.extensions.mod

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsPrimitive
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
internal suspend fun PrimitiveNDArray.fmod(other: PrimitiveNDArray, dest: MutablePrimitiveNDArray) =
    broadcastTwoTensorsPrimitive(this, other, dest) { left: PrimitiveType, right: PrimitiveType -> (left % right).toPrimitive() }

@MakePublic
internal suspend fun PrimitiveNDArray.fmod(other: PrimitiveNDArray): PrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, other.shape))
    return fmod(other, MutablePrimitiveNDArray(destShape))
}
