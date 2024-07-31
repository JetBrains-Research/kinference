@file:GeneratePrimitives(
    DataType.UINT,
    DataType.ULONG,
    DataType.INT,
    DataType.LONG,
    DataType.FLOAT,
    DataType.DOUBLE
)
package io.kinference.ndarray.extensions.reduce.mean

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.toPrimitive

@MakePublic
internal suspend fun PrimitiveNDArray.reduceMean(axes: IntArray, keepDims: Boolean): PrimitiveNDArray {
    val reducedSum = this.reduceSum(axes, keepDims) as MutablePrimitiveNDArray

    val actualAxes = axes.map { indexAxis(it) }.toSet()
    val reducedDims = actualAxes.map { this.shape[it] }.reduce(Int::times)
    reducedSum.divAssign(PrimitiveNDArray.scalar(reducedDims.toPrimitive()))

    return reducedSum
}
