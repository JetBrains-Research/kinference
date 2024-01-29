@file:GeneratePrimitives(
    DataType.FLOAT,
    DataType.DOUBLE
)

package io.kinference.ndarray.extensions.activations.exp

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.ndarray.parallelizeByBlocks
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType

@MakePublic
internal suspend fun PrimitiveNDArray.exp(): PrimitiveNDArray {
    val output = PrimitiveNDArray(strides)

    val inputArray = this.array
    val outputArray = output.array

    val blockSize = this.array.blockSize
    val blocksNum = this.array.blocksNum

    parallelizeByBlocks(blockSize, blocksNum, 2048) { blockStart, blockEnd ->
        for (blockIdx in blockStart until blockEnd) {
            val inputBlock = inputArray.getBlock(blockIdx)
            val outputBlock = outputArray.getBlock(blockIdx)

            for (idx in outputBlock.indices) {
                outputBlock[idx] = FastMath.exp(inputBlock[idx])
            }
        }
    }

    return output
}
