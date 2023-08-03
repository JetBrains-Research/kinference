@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions.max

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.extensions.compare.compare
import io.kinference.ndarray.stubs.maxOf
import io.kinference.ndarray.stubs.MIN_VALUE_FOR_MAX
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
internal suspend fun List<PrimitiveNDArray>.max(): PrimitiveNDArray {
    val newShape = broadcastShape(this.map { it.shape })
    val destination = MutablePrimitiveNDArray(newShape) { PrimitiveType.MIN_VALUE_FOR_MAX }
    return compare(destination) { first, second -> maxOf(first, second) }
}

@MakePublic
internal suspend fun Array<out PrimitiveNDArray>.max() = this.toList().max()

@MakePublic
internal suspend fun maxOf(vararg inputs: PrimitiveNDArray) = inputs.max()
