@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT,
    DataType.LONG,
    DataType.UINT,
    DataType.ULONG,
)
package io.kinference.ndarray.extensions.reduce.logSum

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.stubs.ln
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.PrimitiveType
import io.kinference.ndarray.extensions.ln
import kotlin.math.ln


@MakePublic
internal suspend fun PrimitiveNDArray.reduceLogSum(axes: IntArray, keepDims: Boolean): PrimitiveNDArray {
    val sumTensor = this.reduceSum(axes, keepDims) as MutablePrimitiveNDArray
    sumTensor.mapMutable(object : PrimitiveMap {
        override fun apply(value: PrimitiveType) = ln(value)
    })

    return sumTensor
}
