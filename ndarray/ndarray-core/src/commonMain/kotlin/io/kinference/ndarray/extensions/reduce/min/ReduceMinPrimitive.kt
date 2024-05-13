@file:GeneratePrimitives(DataType.NUMBER)
package io.kinference.ndarray.extensions.reduce.min

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.reduce.primitive.reduceOperationPrimitive
import io.kinference.ndarray.stubs.*
import io.kinference.ndarray.stubs.minOf
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.primitives.annotations.MakePublic
import io.kinference.ndarray.extensions.MAX_VALUE_FOR_MIN
import io.kinference.utils.inlines.InlinePrimitive
import kotlin.comparisons.minOf

@MakePublic
internal suspend fun PrimitiveNDArray.reduceMin(axes: IntArray, keepDims: Boolean) =
    reduceOperationPrimitive(axes, keepDims, initOutputValue = PrimitiveType.MAX_VALUE_FOR_MIN) {
        out: InlinePrimitive, inp: InlinePrimitive -> InlinePrimitive(minOf(out.value, inp.value))
    }
