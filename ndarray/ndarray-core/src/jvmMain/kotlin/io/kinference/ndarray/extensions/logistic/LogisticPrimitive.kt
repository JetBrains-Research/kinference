@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)
@file:Suppress("unused")
@file:GenerateVector

package io.kinference.ndarray.extensions.logistic

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.primitives.annotations.GenerateVector
import io.kinference.primitives.vector.*
import io.kinference.ndarray.math.*
import kotlin.math.exp
import kotlin.math.abs


@GenerateNameFromPrimitives
internal suspend fun logisticPrimitive(input: PrimitiveNDArray, dest: MutablePrimitiveNDArray): MutablePrimitiveNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks

    val outputBlocks = dest.array.blocks

    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 2048) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            val inputBlock = inputBlocks[blockNum]
            val outputBlock = outputBlocks[blockNum]

            val mid =
                Div(Value(PrimitiveConstants.ONE), Add(Value(PrimitiveConstants.ONE + PrimitiveConstants.ZERO), Exp(Neg(Abs(PrimitiveSlice(inputBlock))))))
            IfElse(
                GE(PrimitiveSlice(inputBlock), Value(PrimitiveConstants.ZERO)),
                mid,
                Sub(Value(PrimitiveConstants.ONE), mid)
            ).into(outputBlock, 0, inputBlockSize)


        }
    }

    return dest
}

@GenerateNameFromPrimitives
internal suspend fun logisticPrimitive(input: PrimitiveNDArray): MutablePrimitiveNDArray =
    logisticPrimitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides))
