@file:GeneratePrimitives(DataType.DOUBLE, DataType.FLOAT)
@file:GenerateVector

package io.kinference.ndarray.extensions.gelu

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.contexts.AutoAllocatorContext
import io.kinference.ndarray.arrays.memory.storage.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.stubs.absoluteValue
import io.kinference.ndarray.stubs.pow
import io.kinference.ndarray.math.*
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.GenerateVector
import io.kinference.primitives.types.*
import io.kinference.primitives.vector.*
import java.util.PrimitiveIterator
import kotlin.coroutines.coroutineContext
import kotlin.math.*

@GenerateNameFromPrimitives
internal suspend fun computeGeluPrimitive(input: PrimitiveNDArray, bias: PrimitiveNDArray, output: MutablePrimitiveNDArray): MutablePrimitiveNDArray {

    val inputBlocks = input.array.blocks
    val biasBlocks = bias.array.blocks
    val outputBlocks = output.array.blocks

    val blockSize = input.array.blockSize

    val coroutineCount = countCoroutinesByData(blockSize, inputBlocks.size, 2048)
    val temporaryBlocks = coroutineContext[AutoAllocatorContext]?.getPrimitiveBlock(coroutineCount, blockSize)
        ?: Array(coroutineCount) { PrimitiveArray(blockSize) }
    val temporaryBlocksAbs = coroutineContext[AutoAllocatorContext]?.getPrimitiveBlock(coroutineCount, blockSize)
        ?: Array(coroutineCount) { PrimitiveArray(blockSize) }


    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(blockSize, inputBlocks.size, 2048) { blockStart, blockEnd, coroutineIndex ->
        val temporaryBlock = temporaryBlocks[coroutineIndex]
        val temporaryBlockAbs = temporaryBlocksAbs[coroutineIndex]

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
                outputBlock[j] =
                    outputBlock[j] * (PrimitiveConstants.ERF_COEF_1 + outputBlock[j] * (PrimitiveConstants.ERF_COEF_2 + outputBlock[j] * (PrimitiveConstants.ERF_COEF_3 + outputBlock[j] * (PrimitiveConstants.ERF_COEF_4 + outputBlock[j] * PrimitiveConstants.ERF_COEF_5))))
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

@GenerateNameFromPrimitives
internal suspend fun vecGeluPrimitive(input: PrimitiveNDArray, bias: PrimitiveNDArray, output: MutablePrimitiveNDArray): MutablePrimitiveNDArray {

    val inputBlocks = input.array.blocks
    val biasBlocks = bias.array.blocks
    val outputBlocks = output.array.blocks

    val blockSize = input.array.blockSize

    val coroutineCount = countCoroutinesByData(blockSize, inputBlocks.size, 2048)
    val temporaryBlocks = coroutineContext[AutoAllocatorContext]?.getPrimitiveBlock(coroutineCount, blockSize)
        ?: Array(coroutineCount) { PrimitiveArray(blockSize) }
    val temporaryBlocksAbs = coroutineContext[AutoAllocatorContext]?.getPrimitiveBlock(coroutineCount, blockSize)
        ?: Array(coroutineCount) { PrimitiveArray(blockSize) }
    parallelizeByBlocks(blockSize, inputBlocks.size, 2048) { blockStart, blockEnd, coroutineIndex ->
        val temporaryBlock = temporaryBlocks[coroutineIndex]
        val temporaryBlockAbs = temporaryBlocksAbs[coroutineIndex]

        for (blockIdx in blockStart until blockEnd) {
            val outputBlock = outputBlocks[blockIdx]
            val block = inputBlocks[blockIdx]
            val biasBlock = biasBlocks[blockIdx % biasBlocks.size]


            Add(PrimitiveSlice(block), PrimitiveSlice(biasBlock)).into(temporaryBlock, 0, blockSize)
            Abs(Mul(PrimitiveSlice(temporaryBlock), Value(PrimitiveConstants.SQRT_1_2))).into(temporaryBlockAbs, 0, blockSize)
            Mul(PrimitiveSlice(temporaryBlock), Value(PrimitiveConstants.HALF)).into(temporaryBlock, 0, blockSize)
            Div(
                Value(PrimitiveConstants.ONE),
                Add(Mul(PrimitiveSlice(temporaryBlockAbs), Value(PrimitiveConstants.ERF_P_VALUE)), Value(PrimitiveConstants.ONE))
            ).into(outputBlock, 0, blockSize)
            Exp(Neg(Mul(PrimitiveSlice(temporaryBlockAbs), PrimitiveSlice(temporaryBlockAbs)))).into(temporaryBlockAbs, 0, blockSize)

            val ob = PrimitiveSlice(outputBlock)
            Mul(
                ob,
                Add(
                    Value(PrimitiveConstants.ERF_COEF_1),
                    Mul(
                        ob, Add(
                            Value(PrimitiveConstants.ERF_COEF_2), Mul(
                                ob, Add(
                                    Value(PrimitiveConstants.ERF_COEF_3), Mul(
                                        ob, Add(
                                            Value(PrimitiveConstants.ERF_COEF_4), Mul(
                                                ob,
                                                Value(PrimitiveConstants.ERF_COEF_5)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ).into(outputBlock, 0, blockSize)
            Sub(
                Value(PrimitiveConstants.ONE),
                Mul(PrimitiveSlice(temporaryBlockAbs), PrimitiveSlice(outputBlock))
            ).into(outputBlock, 0, blockSize)

            for (j in outputBlock.indices) {
                outputBlock[j] = FastMath.copySign(outputBlock[j], temporaryBlock[j])
            }

            Mul(Add(Value(PrimitiveConstants.ONE), PrimitiveSlice(outputBlock)), PrimitiveSlice(temporaryBlock)).into(outputBlock, 0, blockSize)
        }
    }
    return output
}

@GenerateNameFromPrimitives
internal suspend fun computeGeluPrimitive(input: PrimitiveNDArray, bias: PrimitiveNDArray): MutablePrimitiveNDArray {
    return computeGeluPrimitive(input, bias, MutablePrimitiveNDArray(input.strides))
}

@GenerateNameFromPrimitives
internal suspend fun vecGeluPrimitive(input: PrimitiveNDArray, bias: PrimitiveNDArray): MutablePrimitiveNDArray {
    return vecGeluPrimitive(input, bias, MutablePrimitiveNDArray(input.strides))
}
