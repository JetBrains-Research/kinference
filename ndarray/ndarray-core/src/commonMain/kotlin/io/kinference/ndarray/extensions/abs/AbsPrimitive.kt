@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.abs

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.ndarray.stubs.abs
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.abs

@GenerateNameFromPrimitives
internal suspend fun absPrimitive(array: PrimitiveNDArray): PrimitiveNDArray = array.applyElementWise { abs(it) }
