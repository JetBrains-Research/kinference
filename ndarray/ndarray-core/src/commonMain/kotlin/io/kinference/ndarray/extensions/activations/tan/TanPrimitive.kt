@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.activations.tan

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.stubs.tan
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.tan

fun PrimitiveNDArray.tan(): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(this.strides)

    val outputIter = output.array.blocks.iterator()
    val inputIter = this.array.blocks.iterator()
    val blocksNum = this.array.blocksNum

    repeat(blocksNum) {
        val inputBlock = inputIter.next()
        val outputBlock = outputIter.next()

        for (idx in outputBlock.indices) {
            outputBlock[idx] = tan(inputBlock[idx])
        }
    }

    return output
}
