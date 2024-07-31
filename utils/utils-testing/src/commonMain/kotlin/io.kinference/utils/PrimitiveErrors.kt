@file:GeneratePrimitives(DataType.ALL)
package io.kinference.utils

import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveArray
import kotlin.math.abs

@MakePublic
internal fun Errors.computeErrors(expected: Array<PrimitiveArray>, actual: Array<PrimitiveArray>): Errors.ErrorsData {
    val blockSize = if (expected.isNotEmpty()) expected.first().size else 0
    val blocksNum = expected.size

    val errors = DoubleArray(blockSize * blocksNum)

    for (blockIdx in expected.indices) {
        val leftBlock = expected[blockIdx]
        val rightBlock = actual[blockIdx]

        val offset = blockIdx * leftBlock.size

        for (idx in leftBlock.indices) {
            errors[offset + idx] = abs(leftBlock[idx].toDouble() - rightBlock[idx].toDouble())
        }
    }

    return this.computeErrors(errors)
}

@MakePublic
internal fun Errors.computeErrors(expected: PrimitiveArray, actual: PrimitiveArray): Errors.ErrorsData {
    val errors = DoubleArray(expected.size)

    for (idx in errors.indices) {
        errors[idx] = abs(expected[idx].toDouble() - actual[idx].toDouble())
    }

    return this.computeErrors(errors)
}
