@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions.compare

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

internal suspend fun List<PrimitiveNDArray>.compare(
    dest: MutablePrimitiveNDArray,
    comparator: (PrimitiveType, PrimitiveType) -> PrimitiveType
): PrimitiveNDArray {
    require(isNotEmpty()) { "Input array must have at least one element" }
    if (size == 1) return single()

    return Broadcasting.applyWithBroadcast(this, dest) { inputs, output ->
        output as MutablePrimitiveNDArray
        for (input in inputs) {
            input as PrimitiveNDArray

            val inputBlocksIter = input.array.blocks.iterator()
            val outputBlocksIter = output.array.blocks.iterator()

            for (blockIdx in 0 until input.array.blocksNum) {
                val inputBlock = inputBlocksIter.next()
                val outputBlock = outputBlocksIter.next()

                for (idx in outputBlock.indices) {
                    outputBlock[idx] = comparator(inputBlock[idx], outputBlock[idx])
                }
            }
        }
    } as PrimitiveNDArray
}

internal suspend fun Array<out PrimitiveNDArray>.compare(
    dest: MutablePrimitiveNDArray,
    comparator: (PrimitiveType, PrimitiveType) -> PrimitiveType
) = this.toList().compare(dest, comparator)

internal suspend fun compare(
    dest: MutablePrimitiveNDArray,
    comparator: (PrimitiveType, PrimitiveType) -> PrimitiveType,
    vararg inputs: PrimitiveNDArray
) = inputs.compare(dest, comparator)
