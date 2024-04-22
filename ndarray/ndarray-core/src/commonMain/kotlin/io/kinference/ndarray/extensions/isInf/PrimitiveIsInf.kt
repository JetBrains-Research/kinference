@file:GeneratePrimitives(
    DataType.DOUBLE,
    DataType.FLOAT
)

package io.kinference.ndarray.extensions.isInf

import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.predicateElementWise
import io.kinference.ndarray.stubs.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import kotlin.isInfinite


private val ONLY_NEGATIVE = { x: PrimitiveType -> x == PrimitiveType.NEGATIVE_INFINITY }
private val ONLY_POSITIVE = { x: PrimitiveType -> x == PrimitiveType.POSITIVE_INFINITY }

/**
 * Checks whether each element of the current NDArray is infinite or not.
 *
 * @param detectNegative - If true, detects negative infinity. Defaults to true.
 * @param detectPositive - If true, detects positive infinity. Defaults to true.
 * @return A BooleanNDArray with the same shape as the current NDArray where each element
 *         indicates whether the corresponding element in the current NDArray is infinite or not.
 */
@MakePublic
internal suspend fun PrimitiveNDArray.isInf(detectNegative: Boolean = true, detectPositive: Boolean = true): BooleanNDArray {
    val detector = when {
        detectNegative && detectPositive -> PrimitiveType::isInfinite
        detectNegative -> ONLY_NEGATIVE
        detectPositive -> ONLY_POSITIVE
        else -> error("At least one of detectNegative or detectPositive must be true")
    }

    return predicateElementWise(detector)
}
