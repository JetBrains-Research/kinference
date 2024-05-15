@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT,
    DataType.LONG,
    DataType.UINT,
    DataType.ULONG,
)
package io.kinference.ndarray.extensions.reduce.logSumExp

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.reduce.primitive.reduceOperationPrimitive
import io.kinference.ndarray.stubs.ln
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.primitives.annotations.MakePublic
import io.kinference.ndarray.extensions.ln
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.utils.inlines.InlinePrimitive
import kotlin.math.ln

@MakePublic
internal suspend fun PrimitiveNDArray.reduceLogSumExp(axes: IntArray, keepDims: Boolean): PrimitiveNDArray {
    val sumTensor = reduceOperationPrimitive(axes, keepDims)
        { out: InlinePrimitive, inp: InlinePrimitive -> out + InlinePrimitive(FastMath.exp(inp.value)) } as MutablePrimitiveNDArray

    sumTensor.mapMutable(object : PrimitiveMap {
        override fun apply(value: PrimitiveType) = ln(value)
    })


    return sumTensor
}
