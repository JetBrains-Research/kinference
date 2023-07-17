@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.floor

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.stubs.floor
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.floor

fun PrimitiveNDArray.floor(): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(strides)

    val inputBlockIter = array.blocks.iterator()
    val outputBlockIter = output.array.blocks.iterator()
    val blockSize = output.array.blockSize

    repeat(output.array.blocksNum) {
        val inputBlock = inputBlockIter.next()
        val outputBlock = outputBlockIter.next()

        for (idx in 0 until blockSize) {
            outputBlock[idx] = floor(inputBlock[idx])
        }
    }

    return output
}
