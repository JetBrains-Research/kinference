@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions.mean

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.sum.sum
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.toPrimitive

suspend fun List<PrimitiveNDArray>.mean(): PrimitiveNDArray {
    if (isEmpty()) error("Array for mean operation must have at least one element")
    if (size == 1) return single()

    val countTensors = size.toPrimitive()
    val sumOfTensors = this.sum() as MutablePrimitiveNDArray

    sumOfTensors.divAssign(PrimitiveNDArray.scalar(countTensors))

    return sumOfTensors
}

suspend fun Array<out PrimitiveNDArray>.mean() = toList().mean()

suspend fun meanOf(vararg inputs: PrimitiveNDArray) = inputs.mean()
