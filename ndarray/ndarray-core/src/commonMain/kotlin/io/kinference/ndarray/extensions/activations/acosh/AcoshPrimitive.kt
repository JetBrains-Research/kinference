@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.activations.acosh

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.acosh

fun PrimitiveNDArray.acosh(): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(this.strides)

    val outputIter = output.array.blocks.iterator()
    val inputIter = this.array.blocks.iterator()
    val blocksNum = this.array.blocksNum

    repeat(blocksNum) {
        val inputBlock = inputIter.next()
        val outputBlock = outputIter.next()

        for (idx in outputBlock.indices) {
            outputBlock[idx] = acosh(inputBlock[idx])
        }
    }

    return output
}
