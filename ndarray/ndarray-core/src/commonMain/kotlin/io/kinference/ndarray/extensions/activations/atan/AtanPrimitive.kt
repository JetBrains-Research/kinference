@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.activations.atan

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.atan

fun PrimitiveNDArray.atan(): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(this.strides)

    val outputIter = output.array.blocks.iterator()
    val inputIter = this.array.blocks.iterator()

    repeat(this.array.blocksNum) {
        val inputBlock = inputIter.next()
        val outputBlock = outputIter.next()

        for (j in outputBlock.indices) {
            outputBlock[j] = atan(inputBlock[j])
        }
    }

    return output
}
