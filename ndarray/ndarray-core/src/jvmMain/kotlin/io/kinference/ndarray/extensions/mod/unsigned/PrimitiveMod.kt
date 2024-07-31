@file:GeneratePrimitives(
    DataType.UBYTE,
    DataType.USHORT,
    DataType.UINT,
    DataType.ULONG,
)

package io.kinference.ndarray.extensions.mod.unsigned

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.mod.fmod
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType

@MakePublic
internal suspend fun PrimitiveNDArray.mod(other: PrimitiveNDArray, dest: MutablePrimitiveNDArray) = fmod(other, dest)

@MakePublic
internal suspend fun PrimitiveNDArray.mod(other: PrimitiveNDArray) = fmod(other)
