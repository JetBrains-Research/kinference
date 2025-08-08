@file:GeneratePrimitives(DataType.DOUBLE, DataType.FLOAT)
@file:GenerateVector
@file:Suppress("UnusedImport")

package io.kinference.ndarray.extensions.gelu

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.memory.contexts.AutoAllocatorContext
import io.kinference.ndarray.arrays.memory.storage.*
import io.kinference.ndarray.countCoroutinesByData
import io.kinference.ndarray.parallelizeByBlocks
import io.kinference.primitives.types.*
import io.kinference.primitives.vector.*
import io.kinference.ndarray.math.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.GenerateVector
import kotlin.coroutines.coroutineContext

@GenerateNameFromPrimitives
internal suspend fun fastGeluPrimitive(input: PrimitiveNDArray, bias: PrimitiveNDArray?): MutablePrimitiveNDArray {
    val output = MutablePrimitiveNDArray(input.strides)

    val inputBlocks = input.array.blocks
    val outputBlocks = output.array.blocks

    val blockSize = input.array.blockSize

    val coroutineCount = countCoroutinesByData(blockSize, inputBlocks.size, 2048)
    val temporaryBlocksExp = coroutineContext[AutoAllocatorContext]?.getPrimitiveBlock(coroutineCount, blockSize)
        ?: Array(coroutineCount) { PrimitiveArray(blockSize) }

    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(blockSize, inputBlocks.size, 2048) { blockStart, blockEnd, coroutineIndex ->
        val temporaryBlockExp = temporaryBlocksExp[coroutineIndex]
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

            val blk = PrimitiveSlice(outputBlock)
            val tmp = Add(Mul(Mul(blk, blk), Value(PrimitiveConstants.FGELU_COEF_1)), Value(PrimitiveConstants.FGELU_COEF_2))
            Exp(Mul(Mul(blk, tmp), Value(PrimitiveConstants.TWO))).into(temporaryBlockExp, 0, blockSize)
            Min(PrimitiveSlice(temporaryBlockExp), Value(PrimitiveType.MAX_VALUE)).into(temporaryBlockExp, 0, blockSize)

            val txp = PrimitiveSlice(temporaryBlockExp)
            Mul(
                PrimitiveSlice(outputBlock),
                Add(
                    Mul(
                        Div(
                            Sub(PrimitiveSlice(temporaryBlockExp), Value(PrimitiveConstants.ONE)), Add(
                                PrimitiveSlice(temporaryBlockExp), Value(PrimitiveConstants.ONE)
                            )
                        ),
                        Value(PrimitiveConstants.HALF)
                    ), Value(PrimitiveConstants.HALF)
                )
            ).into(outputBlock, 0, blockSize)
        }
    }

    return output
}
