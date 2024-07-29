@file:GeneratePrimitives(DataType.ALL)
@file:Suppress("DuplicatedCode")

package io.kinference.ndarray.arrays.memory

import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray

@GenerateNameFromPrimitives
internal class PrimitiveArrayContainer(
    arrayTypeIndex: Int,
    arraySizeIndex: Int,
    sizeBytes: Int,
    val array: PrimitiveArray
) : ArrayContainer(arrayTypeIndex, arraySizeIndex, sizeBytes)
