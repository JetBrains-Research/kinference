@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)
@file:GenerateVector

package io.kinference.ndarray.extensions.activations.exp

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.ndarray.parallelizeByBlocks
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.GenerateVector
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.vector.*

@MakePublic
internal suspend fun PrimitiveNDArray.exp(): PrimitiveNDArray {
    val output = PrimitiveNDArray(strides)

    val inputBlocks = this.array.blocks
    val outputBlocks = output.array.blocks

    val blockSize = this.array.blockSize
    val blocksNum = this.array.blocksNum

    parallelizeByBlocks(blockSize, blocksNum, 2048) { blockStart, blockEnd, _ ->
        for (blockIdx in blockStart until blockEnd) {
            val inputBlock = inputBlocks[blockIdx]
            val outputBlock = outputBlocks[blockIdx]
            Exp(PrimitiveSlice(inputBlock)).into(outputBlock, 0, blockSize)
        }
    }

    return output
}
