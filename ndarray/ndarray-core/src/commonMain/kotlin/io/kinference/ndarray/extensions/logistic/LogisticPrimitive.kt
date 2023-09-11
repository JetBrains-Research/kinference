@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions.logistic

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.ndarray.stubs.abs
import kotlin.math.abs


@GenerateNameFromPrimitives
internal suspend fun logisticPrimitive(input: PrimitiveNDArray, dest: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks

    val outputBlocks = dest.array.blocks

    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 2048) { blockStart, blockEnd ->
        for (blockNum in blockStart until blockEnd) {
            val inputBlock = inputBlocks[blockNum]
            val outputBlock = outputBlocks[blockNum]

            for (j in outputBlock.indices) {
                val inputValue = inputBlock[j]
                val midValue = PrimitiveConstants.ONE / (PrimitiveConstants.ONE + FastMath.exp(-abs(inputValue)))

                if (inputValue < PrimitiveConstants.ZERO)
                    outputBlock[j] = PrimitiveConstants.ONE - midValue
                else
                    outputBlock[j] = midValue
            }
        }
    }

    return dest
}

@GenerateNameFromPrimitives
internal suspend fun logisticPrimitive(input: PrimitiveNDArray): MutablePrimitiveNDArray =
    logisticPrimitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides))
