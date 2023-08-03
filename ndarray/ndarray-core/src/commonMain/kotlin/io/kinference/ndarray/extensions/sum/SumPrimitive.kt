@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions.sum

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.extensions.fold.fold
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType

@MakePublic
internal suspend fun List<PrimitiveNDArray>.sum(): PrimitiveNDArray {
    return fold(initialValue = PrimitiveConstants.ZERO) { first, second -> first + second }
}

@MakePublic
internal suspend fun Array<out PrimitiveNDArray>.sum() = toList().sum()

@MakePublic
internal suspend fun sumOf(vararg inputs: PrimitiveNDArray) = inputs.sum()
