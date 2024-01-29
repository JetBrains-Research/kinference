@file:GeneratePrimitives(DataType.DOUBLE, DataType.FLOAT)

package io.kinference.ndarray.extensions.batchNorm

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.stubs.sqrt
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.annotations.MakePublic
import io.kinference.primitives.types.*
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.*

@MakePublic
internal suspend fun PrimitiveNDArray.batchNorm(
    scale: PrimitiveNDArray,
    bias: PrimitiveNDArray,
    mean: PrimitiveNDArray,
    variance: PrimitiveNDArray,
    epsilon: PrimitiveType,
): PrimitiveNDArray {
    val numBatches = this.shape[0]
    val numChannels = this.shape.getOrNull(1) ?: 1

    require(scale.rank == 1 && scale.shape[0] == numChannels) { "\"scale\" must be a tensor of shape [$numChannels]" }
    require(bias.rank == 1 && bias.shape[0] == numChannels) { "\"bias\" must be a tensor of shape [$numChannels]" }
    require(mean.rank == 1 && mean.shape[0] == numChannels) { "\"mean\" must be a tensor of shape [$numChannels]" }
    require(variance.rank == 1 && variance.shape[0] == numChannels) { "\"variance\" must be a tensor of shape [$numChannels]" }

    val output = MutablePrimitiveNDArray(this.strides)

    val inputArray = this.array
    val outputArray = output.array

    val scaleArray = scale.array
    val biasArray = bias.array
    val meanArray = mean.array
    val varArray = variance.array

    val paramsBlockSize = scale.array.blockSize
    val numInputBlocks = inputArray.blocksNum

    val blocksPerBatch = numInputBlocks / numBatches
    val blocksPerChannel = blocksPerBatch / numChannels

    coroutineScope {
        for (batchStartBlockIdx in 0 until numInputBlocks step blocksPerBatch) {
            launch {
                val blocksPerBatchLimit = min(numInputBlocks, batchStartBlockIdx + blocksPerBatch)

                for (channelStartBlockIdx in batchStartBlockIdx until blocksPerBatchLimit step blocksPerChannel) {
                    val channel = channelStartBlockIdx % numChannels
                    val channelArrBlockIdx = channel / paramsBlockSize
                    val inBlockIdx = channel % paramsBlockSize

                    val scaleScalar = scaleArray.getBlock(channelArrBlockIdx)[inBlockIdx]
                    val biasScalar = biasArray.getBlock(channelArrBlockIdx)[inBlockIdx]
                    val meanScalar = meanArray.getBlock(channelArrBlockIdx)[inBlockIdx]
                    val varScalar = varArray.getBlock(channelArrBlockIdx)[inBlockIdx]

                    val blocksPerChannelLimit = min(blocksPerBatchLimit, channelStartBlockIdx + blocksPerChannel)

                    for (i in channelStartBlockIdx until blocksPerChannelLimit) {
                        val inputBlock = inputArray.getBlock(i)
                        val outputBlock = outputArray.getBlock(i)
                        val tempBlockSqrt = PrimitiveArray(outputBlock.size)

                        for (j in outputBlock.indices) {
                            outputBlock[j] = inputBlock[j] - meanScalar
                        }

                        for (j in tempBlockSqrt.indices) {
                            tempBlockSqrt[j] = sqrt(varScalar + epsilon)
                        }

                        for (j in outputBlock.indices) {
                            outputBlock[j] = outputBlock[j] / tempBlockSqrt[j]
                        }

                        for (j in outputBlock.indices) {
                            outputBlock[j] = outputBlock[j] * scaleScalar + biasScalar
                        }
                    }
                }
            }
        }
    }
    return output
}
