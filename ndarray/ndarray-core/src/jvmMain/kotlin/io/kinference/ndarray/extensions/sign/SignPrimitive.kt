@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions.sign

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.stubs.isNaN
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*

@GenerateNameFromPrimitives
private fun signNotNaNPrimitive(x: PrimitiveType) = when {
    x > PrimitiveConstants.ZERO -> PrimitiveConstants.ONE
    x == PrimitiveConstants.ZERO -> PrimitiveConstants.ZERO
    x < PrimitiveConstants.ZERO -> PrimitiveConstants.MINUS_ONE
    else -> error("This branch should be unreachable")
}

@GenerateNameFromPrimitives
@FilterPrimitives(exclude = [DataType.FLOAT, DataType.DOUBLE])
internal suspend fun signIntegerPrimitive(array: PrimitiveNDArray): PrimitiveNDArray {
    return array.applyElementWise { signNotNaNPrimitive(it) }
}

@GenerateNameFromPrimitives
@SpecifyPrimitives(include = [DataType.FLOAT, DataType.DOUBLE])
internal suspend fun signFPPrimitive(array: PrimitiveNDArray): PrimitiveNDArray = array.applyElementWise {
    if (it.isNaN()) PrimitiveConstants.ZERO else signNotNaNPrimitive(it)
}
