@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT,
    DataType.LONG,
    DataType.UINT,
    DataType.ULONG,
)
package io.kinference.ndarray.extensions.reduce.prod

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.extensions.reduce.primitive.reduceOperationPrimitive
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.primitives.annotations.MakePublic
import io.kinference.utils.inlines.InlinePrimitive

@MakePublic
internal suspend fun PrimitiveNDArray.reduceProd(axes: IntArray, keepDims: Boolean) =
    reduceOperationPrimitive(axes, keepDims, initOutputValue = PrimitiveConstants.ONE) { out: InlinePrimitive, inp: InlinePrimitive -> out * inp }
