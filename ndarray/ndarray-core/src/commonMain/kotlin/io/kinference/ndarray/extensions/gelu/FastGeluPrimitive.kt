@file:GeneratePrimitives(DataType.DOUBLE, DataType.FLOAT)
@file:Suppress("UnusedImport")

package io.kinference.ndarray.extensions.gelu

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.parallelizeByBlocks
import io.kinference.ndarray.stubs.min
import io.kinference.primitives.types.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import kotlin.math.*

@GenerateNameFromPrimitives
internal suspend fun fastGeluPrimitive(input: PrimitiveNDArray, bias: PrimitiveNDArray?): MutablePrimitiveNDArray {
    val output = MutablePrimitiveNDArray(input.strides)

    val inputArray = input.array
    val outputArray = output.array
    val biasArray = bias?.array

    val blockSize = input.array.blockSize

    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(blockSize, inputArray.blocksNum, 2048) { blockStart, blockEnd ->
        val temporaryBlockExp = PrimitiveArray(blockSize)
        for (blockIdx in blockStart until blockEnd) {
            val outputBlock = outputArray.getBlock(blockIdx)
            val block = inputArray.getBlock(blockIdx)

            if (biasArray != null) {
                val biasBlock = biasArray.getBlock(blockIdx % biasArray.blocksNum)
                for (j in outputBlock.indices) {
                    outputBlock[j] = block[j] + biasBlock[j]
                }
            } else {
                for (j in outputBlock.indices) {
                    outputBlock[j] = block[j]
                }
            }

            for (j in temporaryBlockExp.indices) {
                val temp = outputBlock[j]
                temporaryBlockExp[j] = FastMath.exp(PrimitiveConstants.TWO * temp * (PrimitiveConstants.FGELU_COEF_1 * temp * temp + PrimitiveConstants.FGELU_COEF_2))
            }

            for (j in temporaryBlockExp.indices) {
                temporaryBlockExp[j] =
                    min(temporaryBlockExp[j], PrimitiveType.MAX_VALUE)
            }

            for (j in outputBlock.indices) {
                outputBlock[j] = outputBlock[j] * (PrimitiveConstants.HALF + PrimitiveConstants.HALF * (temporaryBlockExp[j] - PrimitiveConstants.ONE) / (temporaryBlockExp[j] + PrimitiveConstants.ONE))
            }
        }
    }

    return output
}
