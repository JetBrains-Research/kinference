@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT,
    DataType.LONG,
    DataType.UINT,
    DataType.ULONG,
)
package io.kinference.ndarray.extensions.reduce.sumSquare

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.reduce.primitive.reduceOperationPrimitive
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.primitives.annotations.MakePublic

@MakePublic
internal suspend fun PrimitiveNDArray.reduceSumSquare(axes: IntArray, keepDims: Boolean) =
    reduceOperationPrimitive(axes, keepDims) { out: PrimitiveType, inp: PrimitiveType -> out + inp * inp }
