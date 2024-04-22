@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT,
    DataType.LONG,
    DataType.UINT,
    DataType.ULONG,
)
package io.kinference.ndarray.extensions.reduce.l1

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.reduce.primitive.reduceOperationPrimitive
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.primitives.annotations.MakePublic
import io.kinference.ndarray.stubs.abs
import io.kinference.ndarray.extensions.abs
import kotlin.math.abs

@MakePublic
internal suspend fun PrimitiveNDArray.reduceL1(axes: IntArray, keepDims: Boolean) =
    reduceOperationPrimitive(axes, keepDims) { out: PrimitiveType, inp: PrimitiveType -> out + abs(inp) }
