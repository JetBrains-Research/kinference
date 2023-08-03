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
internal fun negPrimitive(array: PrimitiveNDArray): PrimitiveNDArray = array.applyElementWise { (-it).toPrimitive() }

@MakePublic
internal operator fun PrimitiveNDArray.unaryMinus() = negPrimitive(this)
