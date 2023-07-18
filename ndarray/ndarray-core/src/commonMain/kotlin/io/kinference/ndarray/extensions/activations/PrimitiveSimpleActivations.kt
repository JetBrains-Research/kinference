@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.activations

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.ndarray.stubs.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.*

fun PrimitiveNDArray.acos(): PrimitiveNDArray = applyElementWise { acos(it) }
fun PrimitiveNDArray.acosh(): PrimitiveNDArray = applyElementWise { acosh(it) }
fun PrimitiveNDArray.cos(): PrimitiveNDArray = applyElementWise { cos(it) }
fun PrimitiveNDArray.cosh(): PrimitiveNDArray = applyElementWise { cosh(it) }

fun PrimitiveNDArray.asin(): PrimitiveNDArray = applyElementWise { asin(it) }
fun PrimitiveNDArray.asinh(): PrimitiveNDArray = applyElementWise { asinh(it) }
fun PrimitiveNDArray.sinh(): PrimitiveNDArray = applyElementWise { sinh(it) }
fun PrimitiveNDArray.sin(): PrimitiveNDArray = applyElementWise { sin(it) }

fun PrimitiveNDArray.atan(): PrimitiveNDArray = applyElementWise { atan(it) }
fun PrimitiveNDArray.atanh(): PrimitiveNDArray = applyElementWise { atanh(it) }
fun PrimitiveNDArray.tan(): PrimitiveNDArray = applyElementWise { tan(it) }
