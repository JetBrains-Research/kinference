@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.INT,
    DataType.LONG,
    DataType.UINT,
    DataType.ULONG,
)

package io.kinference.ndarray.extensions.reduce.l2

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.reduce.sumSquare.reduceSumSquare
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import kotlin.math.sqrt
import io.kinference.ndarray.extensions.sqrt
import io.kinference.ndarray.stubs.sqrt
import io.kinference.primitives.types.PrimitiveType

@MakePublic
internal suspend fun PrimitiveNDArray.reduceL2(axes: IntArray, keepDims: Boolean): PrimitiveNDArray {
    if (axes.isEmpty()) return this

    val squaredSum = reduceSumSquare(axes, keepDims) as MutablePrimitiveNDArray

    squaredSum.mapMutable(object : PrimitiveMap {
        override fun apply(value: PrimitiveType) = sqrt(value)
    })

    return squaredSum
}
