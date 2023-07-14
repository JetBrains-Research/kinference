@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.activations.atanh

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.stubs.atanh
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.atanh

fun PrimitiveNDArray.atanh(): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(this.strides)

    val outputIter = output.array.blocks.iterator()
    val inputIter = this.array.blocks.iterator()

    repeat(this.array.blocksNum) {
        val inputBlock = inputIter.next()
        val outputBlock = outputIter.next()

        for (j in outputBlock.indices) {
            outputBlock[j] = atanh(inputBlock[j])
        }
    }

    return output
}
