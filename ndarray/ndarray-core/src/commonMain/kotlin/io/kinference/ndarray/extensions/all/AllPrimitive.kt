@file:GeneratePrimitives(DataType.ALL)

package io.kinference.ndarray.extensions.all

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
@GenerateNameFromPrimitives
internal fun PrimitiveNDArray.all(predicate: (input: PrimitiveType) -> Boolean): Boolean {
    for (blockIdx in array.indices) {
        val block = array.getBlock(blockIdx)
        for (idx in block.indices) {
            if (!predicate(block[idx])) {
                return false
            }
        }
    }

    return true
}
