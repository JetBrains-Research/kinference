@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions.onehot

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.utils.InlineInt

@GenerateNameFromPrimitives
internal suspend fun getOneHotIndices(indices: PrimitiveNDArray, depth: Int): IntNDArray {
    val pointer = indices.array.pointer()
    return IntNDArray(indices.shape) { _: InlineInt ->
        val intIndex = pointer.getAndIncrement().toInt()
        if (intIndex < 0) intIndex + depth else intIndex
    }
}
