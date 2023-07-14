@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.ceil

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.stubs.ceil
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.ceil

fun PrimitiveNDArray.ceil(): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(strides)

    val inputBlockIter = array.blocks.iterator()
    val outputBlockIter = output.array.blocks.iterator()
    val blockSize = output.array.blockSize

    repeat(output.array.blocksNum) {
        val inputBlock = inputBlockIter.next()
        val outputBlock = outputBlockIter.next()

        for (idx in 0 until blockSize) {
            outputBlock[idx] = ceil(inputBlock[idx])
        }
    }

    return output
}
