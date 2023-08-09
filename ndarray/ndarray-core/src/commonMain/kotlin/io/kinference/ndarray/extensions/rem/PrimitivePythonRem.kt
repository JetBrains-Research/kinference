@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
)

package io.kinference.ndarray.extensions.rem

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.broadcasting.broadcastTwoTensorsPrimitive
import io.kinference.ndarray.math.Math
import io.kinference.ndarray.math.floorMod
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
internal suspend fun PrimitiveNDArray.pythonRem(other: PrimitiveNDArray, dest: MutablePrimitiveNDArray) =
    broadcastTwoTensorsPrimitive(this, other, dest) { left: PrimitiveType, right: PrimitiveType -> Math.floorMod(left, right) }

@MakePublic
internal suspend fun PrimitiveNDArray.pythonRem(other: PrimitiveNDArray): PrimitiveNDArray {
    val destShape = broadcastShape(listOf(this.shape, other.shape))
    return pythonRem(other, MutablePrimitiveNDArray(destShape))
}
