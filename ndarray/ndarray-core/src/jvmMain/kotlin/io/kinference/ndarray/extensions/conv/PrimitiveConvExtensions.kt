@file:GeneratePrimitives(DataType.FLOAT, DataType.DOUBLE)

package io.kinference.ndarray.extensions.conv

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.ndarray.extensions.utils.*
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.DataType

@MakePublic
internal suspend fun PrimitiveNDArray.conv(
    w: PrimitiveNDArray,
    b: PrimitiveNDArray?,
    inputInfo: InputInfo
): PrimitiveNDArray {
    val resultBatchSize = shape[0]
    val resultChannels = w.shape[0]
    val resultShape = IntArray(rank) {
        when (it) {
            0 -> resultBatchSize
            1 -> resultChannels
            else -> inputInfo.outputShape[it - 2]
        }
    }
    val result = MutablePrimitiveNDArray(resultBatchSize, inputInfo.groups, resultChannels / inputInfo.groups, inputInfo.outputSize)

    val x = this.reshape(intArrayOf(shape[0], inputInfo.groups, shape[1] / inputInfo.groups, inputInfo.inputSize))

    val kernelDim = w.shape[1] * inputInfo.kernelSize
    val col = PrimitiveNDArray(kernelDim, inputInfo.outputSize)

    val reshapedWeights = w.reshape(intArrayOf(inputInfo.groups, w.shape[0] / inputInfo.groups, kernelDim))

    for (imageId in 0 until shape[0]) {
        for (groupId in 0 until inputInfo.groups) {
            val xCur = x.view(imageId, groupId) as PrimitiveNDArray
            if (inputInfo.rank == 2)
                primitiveIm2ColRank2(xCur, inputInfo, shape[1] / inputInfo.groups, col)
            else
                primitiveIm2Col(xCur, inputInfo, kernelDim, col)

            reshapedWeights.view(groupId).dot(col, result.viewMutable(imageId, groupId))
        }
    }

    if (b != null) {
        val resultPointer = result.array.pointer()
        repeat(resultBatchSize) {
            val bPointer = PrimitivePointer(b.array)
            bPointer.forEach(resultChannels) { bias ->
                resultPointer.map(inputInfo.outputSize) { it + bias }
            }
        }
    }

    return result.reshape(resultShape) as PrimitiveNDArray
}
