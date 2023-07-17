@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions.max

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.stubs.maxOf
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.max

suspend fun List<PrimitiveNDArray>.max(): PrimitiveNDArray {
    if (isEmpty()) error("Array for max operation must have at least one element")
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
                    outputBlock[idx] = maxOf(inputBlock[idx], outputBlock[idx])
                }
            }
        }
    } as PrimitiveNDArray
}

suspend fun Array<out PrimitiveNDArray>.max() = this.toList().max()

suspend fun maxOf(vararg inputs: PrimitiveNDArray) = inputs.max()
