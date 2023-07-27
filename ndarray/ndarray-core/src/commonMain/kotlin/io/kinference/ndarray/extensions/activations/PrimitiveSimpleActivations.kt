@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.activations

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.ndarray.stubs.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import kotlin.math.*

@MakePublic
internal fun PrimitiveNDArray.acos(): PrimitiveNDArray = applyElementWise { acos(it) }
@MakePublic
internal fun PrimitiveNDArray.acosh(): PrimitiveNDArray = applyElementWise { acosh(it) }
@MakePublic
internal fun PrimitiveNDArray.cos(): PrimitiveNDArray = applyElementWise { cos(it) }
@MakePublic
internal fun PrimitiveNDArray.cosh(): PrimitiveNDArray = applyElementWise { cosh(it) }

@MakePublic
internal fun PrimitiveNDArray.asin(): PrimitiveNDArray = applyElementWise { asin(it) }
@MakePublic
internal fun PrimitiveNDArray.asinh(): PrimitiveNDArray = applyElementWise { asinh(it) }
@MakePublic
internal fun PrimitiveNDArray.sinh(): PrimitiveNDArray = applyElementWise { sinh(it) }
@MakePublic
internal fun PrimitiveNDArray.sin(): PrimitiveNDArray = applyElementWise { sin(it) }

@MakePublic
internal fun PrimitiveNDArray.atan(): PrimitiveNDArray = applyElementWise { atan(it) }
@MakePublic
internal fun PrimitiveNDArray.atanh(): PrimitiveNDArray = applyElementWise { atanh(it) }
@MakePublic
internal fun PrimitiveNDArray.tan(): PrimitiveNDArray = applyElementWise { tan(it) }
