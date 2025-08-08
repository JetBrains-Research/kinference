@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.FLOAT,
    DataType.DOUBLE
)
@file:GenerateVector

package io.kinference.ndarray.extensions.abs

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray.Companion.invoke
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.ndarray.stubs.abs
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.GenerateVector
import io.kinference.primitives.types.DataType
import io.kinference.primitives.vector.*
import kotlin.math.abs

@GenerateNameFromPrimitives
internal suspend fun absPrimitive(array: PrimitiveNDArray): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(array.strides)

    val inputBlockIter = array.array.blocks.iterator()
    val outputBlockIter = output.array.blocks.iterator()
    val blockSize = output.array.blockSize

    repeat(output.array.blocksNum) {
        val inputBlock = inputBlockIter.next()
        val outputBlock = outputBlockIter.next()

        Abs(PrimitiveSlice(inputBlock)).into(outputBlock, 0, blockSize)
    }

    return output
}
