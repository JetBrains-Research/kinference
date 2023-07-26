@file:GeneratePrimitives(
    DataType.DOUBLE,
    DataType.FLOAT
)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.stubs.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.*

fun PrimitiveNDArray.isNaN(): BooleanNDArray = predicateElementWise { it.isNaN() }

fun PrimitiveNDArray.ceil(): PrimitiveNDArray = applyElementWise { ceil(it) }
fun PrimitiveNDArray.floor(): PrimitiveNDArray = applyElementWise { floor(it) }
fun PrimitiveNDArray.round(): PrimitiveNDArray = applyElementWise { round(it) }

fun PrimitiveNDArray.sqrt(): PrimitiveNDArray = applyElementWise { sqrt(it) }
