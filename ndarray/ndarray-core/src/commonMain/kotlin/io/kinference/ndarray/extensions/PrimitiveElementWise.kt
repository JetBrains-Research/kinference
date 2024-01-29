@file:GeneratePrimitives(DataType.ALL)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType

@MakePublic
internal fun PrimitiveNDArray.applyElementWise(func: (PrimitiveType) -> PrimitiveType): MutablePrimitiveNDArray {
    val output = MutablePrimitiveNDArray(strides)

//    val inputBlockIter = array.blocks.iterator()
//    val outputBlockIter = output.array.blocks.iterator()
    val blockSize = output.array.blockSize

    repeat(output.array.blocksNum) { blockIdx ->
        val inputBlock = this.array.getBlock(blockIdx)
        val outputBlock = output.array.getBlock(blockIdx)

        for (idx in 0 until blockSize) {
            outputBlock[idx] = func(inputBlock[idx])
        }
    }

    return output
}

@MakePublic
internal fun PrimitiveNDArray.predicateElementWise(predicate: (PrimitiveType) -> Boolean): BooleanNDArray {
    val output = MutableBooleanNDArray(strides)

//    val inputBlockIter = array.blocks.iterator()
//    val outputBlockIter = output.array.blocks.iterator()
    val blockSize = output.array.blockSize

    repeat(output.array.blocksNum) { blockIdx ->
        val inputBlock = this.array.getBlock(blockIdx)
        val outputBlock = output.array.getBlock(blockIdx)

        for (idx in 0 until blockSize) {
            outputBlock[idx] = predicate(inputBlock[idx])
        }
    }

    return output
}
