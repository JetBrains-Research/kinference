@file:GeneratePrimitives(DataType.NUMBER)
package io.kinference.ndarray.extensions.reduce.max

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.reduce.primitive.reduceOperationPrimitive
import io.kinference.ndarray.stubs.*
import io.kinference.ndarray.stubs.maxOf
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.primitives.annotations.MakePublic
import kotlin.comparisons.maxOf
import io.kinference.ndarray.extensions.*
import io.kinference.utils.inlines.InlinePrimitive

@MakePublic
internal suspend fun PrimitiveNDArray.reduceMax(axes: IntArray, keepDims: Boolean) =
    reduceOperationPrimitive(axes, keepDims, initOutputValue = PrimitiveType.MIN_VALUE_FOR_MAX) { out: InlinePrimitive, inp: InlinePrimitive -> InlinePrimitive(maxOf(out.value, inp.value)) }
