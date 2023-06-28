@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.activations.cosh

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.cosh

fun PrimitiveNDArray.cosh(): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(this.strides)

    val outputIter = output.array.blocks.iterator()
    val inputIter = this.array.blocks.iterator()
    val blocksNum = this.array.blocksNum

    repeat(blocksNum) {
        val inputBlock = inputIter.next()
        val outputBlock = outputIter.next()

        for (idx in outputBlock.indices) {
            outputBlock[idx] = cosh(inputBlock[idx])
        }
    }

    return output
}
