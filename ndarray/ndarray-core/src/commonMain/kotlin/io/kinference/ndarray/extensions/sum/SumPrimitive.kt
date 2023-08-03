@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions.sum

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.fold.fold
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType

@MakePublic
internal suspend fun List<PrimitiveNDArray>.sum(): PrimitiveNDArray {
    val newShape = broadcastShape(this.map { it.shape })
    val destination = MutablePrimitiveNDArray(newShape)
    return fold(destination) { first, second -> first + second }
}

@MakePublic
internal suspend fun Array<out PrimitiveNDArray>.sum() = toList().sum()

@MakePublic
internal suspend fun sumOf(vararg inputs: PrimitiveNDArray) = inputs.sum()
