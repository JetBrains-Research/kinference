@file:GeneratePrimitives(DataType.DOUBLE, DataType.FLOAT)
@file: GenerateVector

package io.kinference.ndarray.extensions.probit

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.parallelizeByBlocks
import io.kinference.ndarray.stubs.ln
import io.kinference.ndarray.stubs.sqrt
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.GenerateVector
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import kotlin.math.ln
import kotlin.math.sqrt
import io.kinference.primitives.vector.*

@GenerateNameFromPrimitives
internal suspend fun vecProbitPrimitive(input: PrimitiveNDArray, dest: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val inputBlocks = input.array.blocks
    val outputBlocks = dest.array.blocks
    val blockSize = input.array.blockSize

    parallelizeByBlocks(blockSize, inputBlocks.size, 2048) { blockStart, blockEnd, _ ->
        val temporaryBlockOne = PrimitiveArray(blockSize)
        val temporaryBlockTwo = PrimitiveArray(blockSize)

        for (blockIdx in blockStart until blockEnd) {
            val inputBlock = inputBlocks[blockIdx]
            val outputBlock = outputBlocks[blockIdx]
            val two = PrimitiveConstants.TWO
            val one = PrimitiveConstants.ONE

            Sub(Mul(PrimitiveSlice(inputBlock), Value(two)), Value(one)).into(outputBlock, 0, blockSize)

            Log(
                Mul(
                    Sub(Value(one), PrimitiveSlice(outputBlock)),
                    Add(PrimitiveSlice(outputBlock), Value(one))
                )
            ).into(temporaryBlockOne, 0, blockSize)

            val inv_erf_1 = PrimitiveConstants.INV_ERF_COEF_1
            val half = PrimitiveConstants.HALF

            Add(Mul(PrimitiveSlice(temporaryBlockOne), Value(half)), Value(inv_erf_1)).into(temporaryBlockTwo, 0, blockSize)

            for (j in outputBlock.indices) {
                outputBlock[j] = PrimitiveConstants.SQRT_2 * FastMath.copySign(
                    sqrt(sqrt(temporaryBlockTwo[j] * temporaryBlockTwo[j] - PrimitiveConstants.INV_ERF_COEF_2 * temporaryBlockOne[j]) - temporaryBlockTwo[j]),
                    outputBlock[j]
                )
            }
        }
    }


    return dest
}

@GenerateNameFromPrimitives
internal suspend fun vecProbitPrimitive(input: PrimitiveNDArray) =
    vecProbitPrimitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides))

@GenerateNameFromPrimitives
internal suspend fun probitPrimitive(input: PrimitiveNDArray, dest: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val inputBlocks = input.array.blocks
    val outputBlocks = dest.array.blocks
    val blockSize = input.array.blockSize

    parallelizeByBlocks(blockSize, inputBlocks.size, 2048) { blockStart, blockEnd, _ ->
        val temporaryBlockOne = PrimitiveArray(blockSize)
        val temporaryBlockTwo = PrimitiveArray(blockSize)

        for (blockIdx in blockStart until blockEnd) {
            val inputBlock = inputBlocks[blockIdx]
            val outputBlock = outputBlocks[blockIdx]

            for (j in outputBlock.indices) {
                outputBlock[j] = PrimitiveConstants.TWO * inputBlock[j] - PrimitiveConstants.ONE
            }

            for (j in temporaryBlockOne.indices) {
                val inputValue = outputBlock[j]
                temporaryBlockOne[j] = ln((PrimitiveConstants.ONE - inputValue) * (PrimitiveConstants.ONE + inputValue))
            }

            for (j in temporaryBlockTwo.indices) {
                temporaryBlockTwo[j] = PrimitiveConstants.INV_ERF_COEF_1 + PrimitiveConstants.HALF * temporaryBlockOne[j]
            }

            for (j in outputBlock.indices) {
                outputBlock[j] = PrimitiveConstants.SQRT_2 * FastMath.copySign(sqrt(sqrt(temporaryBlockTwo[j] * temporaryBlockTwo[j] - PrimitiveConstants.INV_ERF_COEF_2 * temporaryBlockOne[j]) - temporaryBlockTwo[j]), outputBlock[j])
            }
        }
    }


    return dest
}

@GenerateNameFromPrimitives
internal suspend fun probitPrimitive(input: PrimitiveNDArray) =
    probitPrimitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides))
