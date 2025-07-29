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

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.ndarray.stubs.abs
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.GenerateVector
import io.kinference.primitives.types.DataType
import io.kinference.primitives.vector.Abs
import io.kinference.primitives.vector.PrimitiveSlice
import kotlin.math.abs

@GenerateNameFromPrimitives
internal suspend fun absPrimitive(array: PrimitiveNDArray): PrimitiveNDArray = array.applyElementWise { abs(it) }

@GenerateNameFromPrimitives
internal suspend fun vecAbsPrimitive(array: PrimitiveNDArray): PrimitiveNDArray {
    val output = PrimitiveNDArray(array.strides)
    val blockSize = array.array.blockSize
    for (blockIdx in 0 until array.array.blocksNum) {
        val inputBlock = array.array.blocks[blockIdx]
        val outputBlock = output.array.blocks[blockIdx]
        Abs(PrimitiveSlice(inputBlock)).into(outputBlock, 0, blockSize)
    }
    return output
}
