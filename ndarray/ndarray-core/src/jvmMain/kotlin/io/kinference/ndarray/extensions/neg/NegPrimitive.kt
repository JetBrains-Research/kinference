@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.neg

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType

@GenerateNameFromPrimitives
@SpecifyPrimitives(include = [DataType.INT, DataType.LONG,DataType.FLOAT, DataType.DOUBLE])
internal suspend fun negPrimitive(array: PrimitiveNDArray): PrimitiveNDArray = array.applyElementWise { -it }

@GenerateNameFromPrimitives
@SpecifyPrimitives(include = [DataType.SHORT, DataType.BYTE])
internal suspend fun negIntegerPrimitive(array: PrimitiveNDArray): PrimitiveNDArray = array.applyElementWise { (-it).toPrimitive() }
