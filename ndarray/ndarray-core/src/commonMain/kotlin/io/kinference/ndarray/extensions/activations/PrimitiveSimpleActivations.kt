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
internal suspend fun PrimitiveNDArray.acos(): PrimitiveNDArray = applyElementWise { acos(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.acosh(): PrimitiveNDArray = applyElementWise { acosh(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.cos(): PrimitiveNDArray = applyElementWise { cos(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.cosh(): PrimitiveNDArray = applyElementWise { cosh(it) }

@MakePublic
internal suspend fun PrimitiveNDArray.asin(): PrimitiveNDArray = applyElementWise { asin(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.asinh(): PrimitiveNDArray = applyElementWise { asinh(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.sinh(): PrimitiveNDArray = applyElementWise { sinh(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.sin(): PrimitiveNDArray = applyElementWise { sin(it) }

@MakePublic
internal suspend fun PrimitiveNDArray.atan(): PrimitiveNDArray = applyElementWise { atan(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.atanh(): PrimitiveNDArray = applyElementWise { atanh(it) }
@MakePublic
internal suspend fun PrimitiveNDArray.tan(): PrimitiveNDArray = applyElementWise { tan(it) }
