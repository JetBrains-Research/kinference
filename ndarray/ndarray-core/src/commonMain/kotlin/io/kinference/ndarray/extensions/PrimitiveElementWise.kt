@file:GeneratePrimitives(DataType.ALL)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
internal suspend fun PrimitiveNDArray.applyElementWise(func: (PrimitiveType) -> PrimitiveType): MutablePrimitiveNDArray {
    val output = MutablePrimitiveNDArray(strides)

    val inputBlockIter = array.blocks.iterator()
    val outputBlockIter = output.array.blocks.iterator()
    val blockSize = output.array.blockSize

    repeat(output.array.blocksNum) {
        val inputBlock = inputBlockIter.next()
        val outputBlock = outputBlockIter.next()

        for (idx in 0 until blockSize) {
            outputBlock[idx] = func(inputBlock[idx])
        }
    }

    return output
}

@MakePublic
internal suspend fun PrimitiveNDArray.predicateElementWise(predicate: (PrimitiveType) -> Boolean): BooleanNDArray {
    val output = MutableBooleanNDArray(strides)

    val inputBlockIter = array.blocks.iterator()
    val outputBlockIter = output.array.blocks.iterator()
    val blockSize = output.array.blockSize

    repeat(output.array.blocksNum) {
        val inputBlock = inputBlockIter.next()
        val outputBlock = outputBlockIter.next()

        for (idx in 0 until blockSize) {
            outputBlock[idx] = predicate(inputBlock[idx])
        }
    }

    return output
}
