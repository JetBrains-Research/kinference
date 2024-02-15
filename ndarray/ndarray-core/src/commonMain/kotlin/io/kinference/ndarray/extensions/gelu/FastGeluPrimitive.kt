@file:GeneratePrimitives(DataType.DOUBLE, DataType.FLOAT)
@file:Suppress("UnusedImport")

package io.kinference.ndarray.extensions.gelu

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.countCoroutinesByData
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

    val inputBlocks = input.array.blocks
    val outputBlocks = output.array.blocks

    val blockSize = input.array.blockSize

    val coroutineCount = countCoroutinesByData(blockSize, inputBlocks.size, 2048)
    val containerArray = ArrayDispatcher.getArraysAndMarkers(PrimitiveTiledArray.type, blockSize, coroutineCount)
    val temporaryBlockExpArrays = Array(containerArray.size) { i -> (containerArray[i] as PrimitiveArrayContainer).array }

    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(blockSize, inputBlocks.size, 2048) { blockStart, blockEnd, coroutineIndex ->
//        val containerTemporaryBlockExpArrays = ArraysDispatcher.getArrays<PrimitiveArray>(PrimitiveTiledArray.type, blockSize, coroutineCount)
//        val temporaryBlockExpArrays = Array(coroutineCount) { i -> containerTemporaryBlockExpArrays[i] }
        val temporaryBlockExp = temporaryBlockExpArrays[coroutineIndex]
        for (blockIdx in blockStart until blockEnd) {
            val outputBlock = outputBlocks[blockIdx]
            val block = inputBlocks[blockIdx]

            if (bias != null) {
                val biasBlocks = bias.array.blocks
                val biasBlock = biasBlocks[blockIdx % biasBlocks.size]
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
