@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions.sum

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType

@MakePublic
internal suspend fun List<PrimitiveNDArray>.sum(): PrimitiveNDArray {
    if (isEmpty()) error("Array for sum operation must have at least one element")
    if (size == 1) return single()

    return Broadcasting.applyWithBroadcast(this, this.first().type) { inputs, output ->
        output as MutablePrimitiveNDArray
        for (input in inputs) {
            input as PrimitiveNDArray

            val inputBlocksIter = input.array.blocks.iterator()
            val outputBlocksIter = output.array.blocks.iterator()

            for (blockIdx in 0 until input.array.blocksNum) {
                val inputBlock = inputBlocksIter.next()
                val outputBlock = outputBlocksIter.next()

                for (idx in outputBlock.indices) {
                    outputBlock[idx] = inputBlock[idx] + outputBlock[idx]
                }
            }
        }
    } as PrimitiveNDArray
}

@MakePublic
internal suspend fun Array<out PrimitiveNDArray>.sum() = toList().sum()

@MakePublic
internal suspend fun sumOf(vararg inputs: PrimitiveNDArray) = inputs.sum()
