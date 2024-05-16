@file:GeneratePrimitives(DataType.DOUBLE, DataType.FLOAT)
package io.kinference.ndarray.extensions.gelu

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.ArrayDispatcher
import io.kinference.ndarray.arrays.memory.PrimitiveArrayContainer
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.stubs.absoluteValue
import io.kinference.ndarray.stubs.pow
import io.kinference.ndarray.math.*
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*
import kotlin.math.*

@GenerateNameFromPrimitives
internal suspend fun computeGeluPrimitive(input: PrimitiveNDArray, bias: PrimitiveNDArray): MutablePrimitiveNDArray {
    val output = MutablePrimitiveNDArray(input.strides)

    val inputBlocks = input.array.blocks
    val biasBlocks = bias.array.blocks
    val outputBlocks = output.array.blocks

    val blockSize = input.array.blockSize

    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(blockSize, inputBlocks.size, 2048) { blockStart, blockEnd, coroutineIndex ->
        val temporaryBlock = PrimitiveArray(blockSize)
        val temporaryBlockAbs = PrimitiveArray(blockSize)

        for (blockIdx in blockStart until blockEnd) {
            val outputBlock = outputBlocks[blockIdx]
            val block = inputBlocks[blockIdx]
            val biasBlock = biasBlocks[blockIdx % biasBlocks.size]

            for (j in temporaryBlock.indices) {
                temporaryBlock[j] = block[j] + biasBlock[j]
            }

            for (j in temporaryBlockAbs.indices) {
                temporaryBlockAbs[j] = temporaryBlock[j] * PrimitiveConstants.SQRT_1_2
            }

            for (j in temporaryBlock.indices) {
                temporaryBlock[j] = temporaryBlock[j] * PrimitiveConstants.HALF
            }

            for (j in temporaryBlockAbs.indices) {
                temporaryBlockAbs[j] = temporaryBlockAbs[j].absoluteValue
            }

            for (j in outputBlock.indices) {
                outputBlock[j] = PrimitiveConstants.ONE / (temporaryBlockAbs[j] * PrimitiveConstants.ERF_P_VALUE + PrimitiveConstants.ONE)
            }

            for (j in temporaryBlockAbs.indices) {
                temporaryBlockAbs[j] = FastMath.exp(-(temporaryBlockAbs[j].pow(2)))
            }

            for (j in outputBlock.indices) {
                outputBlock[j] = outputBlock[j] * (PrimitiveConstants.ERF_COEF_1 + outputBlock[j] * (PrimitiveConstants.ERF_COEF_2 + outputBlock[j] * (PrimitiveConstants.ERF_COEF_3 + outputBlock[j] * (PrimitiveConstants.ERF_COEF_4 + outputBlock[j] * PrimitiveConstants.ERF_COEF_5))))
            }

            for (j in outputBlock.indices) {
                outputBlock[j] = PrimitiveConstants.ONE - temporaryBlockAbs[j] * outputBlock[j]
            }

            for (j in outputBlock.indices) {
                outputBlock[j] = FastMath.copySign(outputBlock[j], temporaryBlock[j])
            }

            for (j in outputBlock.indices) {
                outputBlock[j] = (PrimitiveConstants.ONE + outputBlock[j]) * temporaryBlock[j]
            }
        }
    }

    return output
}
