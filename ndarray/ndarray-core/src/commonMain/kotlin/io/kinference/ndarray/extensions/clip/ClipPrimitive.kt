@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions.clip

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.stubs.MAX_VALUE_FOR_MIN
import io.kinference.ndarray.stubs.MIN_VALUE_FOR_MAX
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
internal suspend fun PrimitiveNDArray.clip(min: PrimitiveType? = null, max: PrimitiveType? = null): PrimitiveNDArray {
    if (min == null && max == null || min == PrimitiveType.MIN_VALUE_FOR_MAX && max == PrimitiveType.MAX_VALUE_FOR_MIN)
        return this.clone()

    if (min == null || min == PrimitiveType.MIN_VALUE_FOR_MAX) return clipMax(max!!)
    if (max == null || max == PrimitiveType.MAX_VALUE_FOR_MIN) return clipMin(min)

    return applyElementWise {
        when {
            it < min -> min
            it > max -> max
            else -> it
        }
    }
}

private suspend fun PrimitiveNDArray.clipMin(min: PrimitiveType): MutablePrimitiveNDArray {
    return applyElementWise { if (it < min) min else it }
}

private suspend fun PrimitiveNDArray.clipMax(max: PrimitiveType): MutablePrimitiveNDArray {
    return applyElementWise { if (it > max) max else it }
}
