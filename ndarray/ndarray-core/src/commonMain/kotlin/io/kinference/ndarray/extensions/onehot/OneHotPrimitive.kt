@file:GeneratePrimitives(DataType.ALL)

package io.kinference.ndarray.extensions.onehot

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyWithBroadcast
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import io.kinference.utils.InlineInt

private inline fun getValue(left: Int, right: Int, offValue: PrimitiveType, onValue: PrimitiveType): PrimitiveType {
    return if (left == right) onValue else offValue
}

@MakePublic
@GenerateNameFromPrimitives
internal suspend fun PrimitiveNDArray.Companion.oneHot(axis: Int, indices: NumberNDArrayCore, depth: Int, offValue: PrimitiveType, onValue: PrimitiveType): PrimitiveNDArray {
    val actualAxis = if (axis < 0) (indices.rank + 1) + axis else axis

    val arrayIndicesShape = IntArray(indices.rank + 1) { if (it != actualAxis) 1 else depth }
    val typedLambda: (InlineInt) -> Int = { it.value }
    val arrayIndices = IntNDArray(arrayIndicesShape, typedLambda)

    val oneHotIndices = (indices.unsqueeze(actualAxis) as NumberNDArrayCore).getOneHotIndices(depth)

    val outputShape = indices.shape.toMutableList().apply { add(actualAxis, depth) }.toIntArray()
    val output = MutablePrimitiveNDArray(outputShape)
    return arrayIndices.applyWithBroadcast(oneHotIndices, output) { arrayIdx, oneHotIdx, dest ->
        arrayIdx as IntNDArray; oneHotIdx as IntNDArray; dest as MutablePrimitiveNDArray

        for (blockNum in 0 until dest.array.blocksNum) {
            val arrayIdxBlock = arrayIdx.array.blocks[blockNum]
            val oneHotIdxBlock = oneHotIdx.array.blocks[blockNum]
            val destBlock = dest.array.blocks[blockNum]

            for (i in destBlock.indices) {
                destBlock[i] = getValue(arrayIdxBlock[i], oneHotIdxBlock[i], offValue, onValue)
            }
        }
    } as PrimitiveNDArray
}

@MakePublic
@GenerateNameFromPrimitives
internal suspend fun PrimitiveNDArray.Companion.oneHot(axis: Int, indices: NumberNDArrayCore, depth: Int, values: PrimitiveNDArray): PrimitiveNDArray {
    require(values.rank == 1 && values.linearSize == 2)  {
        "\"values\" must be two-element array of format [off_value, on_value], current array rank=${indices.rank}, linearSize=${indices.linearSize}"
    }
    val valuesArray = values.array.toArray()
    val offValue = valuesArray[0]
    val onValue = valuesArray[1]
    return oneHot(axis, indices, depth, offValue, onValue)
}
