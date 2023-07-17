@file:GeneratePrimitives(
    DataType.DOUBLE,
    DataType.FLOAT
)

package io.kinference.ndarray.extensions.isNaN

import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.stubs.isNaN
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType

fun PrimitiveNDArray.isNaN(): BooleanNDArray {
    val output = BooleanNDArray(strides)

    val inputBlockIter = this.array.blocks.iterator()
    val outputBlockIter = output.array.blocks.iterator()

    for (blockIdx in 0 until this.array.blocksNum) {
        val inputBlock = inputBlockIter.next()
        val outputBlock = outputBlockIter.next()

        for (idx in outputBlock.indices) {
            outputBlock[idx] = inputBlock[idx].isNaN()
        }
    }

    return output
}
