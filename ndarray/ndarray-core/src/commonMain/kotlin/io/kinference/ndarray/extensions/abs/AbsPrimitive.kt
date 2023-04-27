@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.abs

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.math.abs

@GenerateNameFromPrimitives
internal fun absPrimitive(array: PrimitiveNDArray): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(array.strides)

    val inputBlockIter = array.array.blocks.iterator()
    val outputBlockIter = output.array.blocks.iterator()
    val blockSize = output.array.blockSize

    repeat(output.array.blocksNum) {
        val inputBlock = inputBlockIter.next()
        val outputBlock = outputBlockIter.next()

        for (idx in 0 until blockSize) {
            outputBlock[idx] = abs(inputBlock[idx])
        }
    }

    return output
}
