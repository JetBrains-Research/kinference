@file:GeneratePrimitives(DataType.DOUBLE, DataType.FLOAT)
@file:GenerateVector
@file:Suppress("unused")

package io.kinference.ndarray.extensions.probit

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.tiled.PrimitiveTiledArray
import io.kinference.ndarray.extensions.constants.PrimitiveConstants
import io.kinference.ndarray.parallelizeByBlocks
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.GenerateVector
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import io.kinference.primitives.vector.*
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.math.abs

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

            Sub(Mul(PrimitiveSlice(inputBlock), Value(PrimitiveConstants.TWO)), Value(PrimitiveConstants.ONE)).into(outputBlock, 0, blockSize)

            Log(
                Mul(
                    Sub(Value(PrimitiveConstants.ONE), PrimitiveSlice(outputBlock)),
                    Add(PrimitiveSlice(outputBlock), Value(PrimitiveConstants.ONE))
                )
            ).into(temporaryBlockOne, 0, blockSize)

            Add(Mul(PrimitiveSlice(temporaryBlockOne), Value(PrimitiveConstants.HALF)), Value(PrimitiveConstants.INV_ERF_COEF_1)).into(
                temporaryBlockTwo,
                0,
                blockSize
            )

            val tbt = PrimitiveSlice(temporaryBlockTwo)

            val a = Sqrt(Sub(Sqrt(Sub(Mul(tbt, tbt), Mul(PrimitiveSlice(temporaryBlockOne), Value(PrimitiveConstants.INV_ERF_COEF_2)))), tbt))
            Mul(
                Value(PrimitiveConstants.SQRT_2), IfElse(
                    GE(PrimitiveSlice(outputBlock), Value(PrimitiveConstants.ZERO)), Abs(a), Neg(Abs(a))
                )
            ).into(outputBlock, 0, blockSize)

        }
    }


    return dest
}

@GenerateNameFromPrimitives
internal suspend fun probitPrimitive(input: PrimitiveNDArray) =
    probitPrimitive(input, MutablePrimitiveNDArray(PrimitiveTiledArray(input.linearSize, input.array.blockSize), input.strides))
