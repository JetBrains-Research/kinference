@file:GeneratePrimitives(DataType.ALL)

package io.kinference.ndarray.extensions.fold

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

internal suspend fun List<PrimitiveNDArray>.fold(
    initialValue: PrimitiveType? = null,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType
): PrimitiveNDArray {
    require(isNotEmpty()) { "Input array must have at least one element" }
    if (size == 1) return single()

    val newShape = broadcastShape(this.map { it.shape })
    val destination = MutablePrimitiveNDArray(newShape)

    if (initialValue != null) {
        destination.fill(initialValue)
    }

    return Broadcasting.applyWithBroadcast(this, destination) { inputs, output ->
        output as MutablePrimitiveNDArray
        for (input in inputs) {
            input as PrimitiveNDArray

            val inputArray = input.array
            val outputArray = output.array

            for (blockIdx in 0 until input.array.blocksNum) {
                val inputBlock = inputArray.getBlock(blockIdx)
                val outputBlock = outputArray.getBlock(blockIdx)

                for (idx in outputBlock.indices) {
                    outputBlock[idx] = op(inputBlock[idx], outputBlock[idx])
                }
            }
        }
    } as PrimitiveNDArray
}

internal suspend fun Array<out PrimitiveNDArray>.fold(
    initialValue: PrimitiveType,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType
) = this.toList().fold(initialValue, op)

internal suspend fun fold(
    initialValue: PrimitiveType,
    op: (PrimitiveType, PrimitiveType) -> PrimitiveType,
    vararg inputs: PrimitiveNDArray
) = inputs.fold(initialValue, op)
