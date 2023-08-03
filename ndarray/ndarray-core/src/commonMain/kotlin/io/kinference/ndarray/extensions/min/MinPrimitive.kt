@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions.min

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.extensions.fold.fold
import io.kinference.ndarray.stubs.minOf
import io.kinference.ndarray.stubs.MAX_VALUE_FOR_MIN
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
internal suspend fun List<PrimitiveNDArray>.min(): PrimitiveNDArray {
    return fold(initialValue = PrimitiveType.MAX_VALUE_FOR_MIN) { first, second -> minOf(first, second) }
}

@MakePublic
internal suspend fun Array<out PrimitiveNDArray>.min() = this.toList().min()

@MakePublic
internal suspend fun minOf(vararg inputs: PrimitiveNDArray) = inputs.min()
