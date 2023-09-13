@file:GeneratePrimitives(DataType.ALL)

package io.kinference.ndarray.extensions.all

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
@GenerateNameFromPrimitives
internal fun PrimitiveNDArray.all(predicate: (input: PrimitiveType) -> Boolean): Boolean {
    val blocks = array.blocks
    for (block in blocks) {
        for (idx in block.indices) {
            if (!predicate(block[idx])) {
                return false
            }
        }
    }

    return true
}
