@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.UBYTE,
    DataType.USHORT,
    DataType.UINT,
    DataType.ULONG,
)

package io.kinference.ndarray.extensions.bitwise.not

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.ndarray.stubs.inv
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import kotlin.experimental.inv

@MakePublic
internal fun PrimitiveNDArray.bitNot(): PrimitiveNDArray = applyElementWise { it.inv() }
