@file:GeneratePrimitives(DataType.ALL)

package io.kinference.ndarray.extensions.fold

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

internal suspend fun List<PrimitiveNDArray>.fold(
    initial: MutablePrimitiveNDArray,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType
): PrimitiveNDArray {
    require(isNotEmpty()) { "Input array must have at least one element" }
    if (size == 1) return single()

    return Broadcasting.applyWithBroadcast(this, initial) { inputs, output ->
        output as MutablePrimitiveNDArray
        for (input in inputs) {
            input as PrimitiveNDArray

            val inputBlocksIter = input.array.blocks.iterator()
            val outputBlocksIter = output.array.blocks.iterator()

            for (blockIdx in 0 until input.array.blocksNum) {
                val inputBlock = inputBlocksIter.next()
                val outputBlock = outputBlocksIter.next()

                for (idx in outputBlock.indices) {
                    outputBlock[idx] = op(inputBlock[idx], outputBlock[idx])
                }
            }
        }
    } as PrimitiveNDArray
}

internal suspend fun Array<out PrimitiveNDArray>.fold(
    initial: MutablePrimitiveNDArray,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType
) = this.toList().fold(initial, op)

internal suspend fun fold(
    initial: MutablePrimitiveNDArray,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType,
    vararg inputs: PrimitiveNDArray
) = inputs.fold(initial, op)
