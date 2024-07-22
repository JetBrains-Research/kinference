package io.kinference.core.operators.ml.svm

import io.kinference.core.operators.ml.utils.LabelsInfo
import io.kinference.ndarray.arrays.*

internal class SvmLinear(info: SvmInfo, val labelsInfo: LabelsInfo<*>): SvmCommon(info) {
    init {
        require(info.svmMode == SvmMode.LINEAR) { "Incorrect svmMode in info" }
    }

    private suspend fun calculateScores(input: FloatNDArray): FloatNDArray {
        val kernel = batchedKernelDot(input)
        kernel.plusAssign(FloatNDArray.scalar(svmInfo.rho[0]))

        return kernel
    }

    // Scores shape should be [batchSize, 2]
    private suspend fun writeLabelsLongTwoClasses(scores: FloatNDArray): LongNDArray {
        val batchSize = scores.shape[0]

        val labels = (labelsInfo as LabelsInfo.LongLabelsInfo).labels
        val futureOutput = MutableLongNDArray(batchSize)
        val futureOutputPointer = futureOutput.array.pointer()

        val scoresBlocks = scores.array.blocks

        if (svmInfo.weightsAreAllPositive) {
            for (batchNum in 0 until batchSize) {
                val scoresBlock = scoresBlocks[batchNum]
                val maxWeight = maxOf(scoresBlock[0], scoresBlock[1])
                if (maxWeight >= 0.5f) {
                    futureOutputPointer.setAndIncrement(labels[1])
                } else {
                    val maxClass = scoresBlock.indexOfFirst { it == maxWeight }
                    futureOutputPointer.setAndIncrement(labels[maxClass])
                }
            }
        } else {
            for (batchNum in 0 until batchSize) {
                val scoresBlock = scoresBlocks[batchNum]
                val maxWeight = maxOf(scoresBlock[0], scoresBlock[1])
                if (maxWeight > 0f) {
                    futureOutputPointer.setAndIncrement(labels[1])
                } else {
                    val maxClass = scoresBlock.indexOfFirst { it == maxWeight }
                    futureOutputPointer.setAndIncrement(labels[maxClass])
                }
            }
        }

        return futureOutput
    }

    // Scores shape should be [batchSize, 2]
    private fun writeLabelsStringTwoClasses(scores: FloatNDArray): StringNDArray {
        val batchSize = scores.shape[0]

        val labels = (labelsInfo as LabelsInfo.StringLabelsInfo).labels
        val futureOutput = MutableStringNDArray(batchSize)

        val scoresBlocks = scores.array.blocks

        if (svmInfo.weightsAreAllPositive) {
            for (batchNum in 0 until batchSize) {
                val scoresBlock = scoresBlocks[batchNum]
                val maxWeight = maxOf(scoresBlock[0], scoresBlock[1])
                if (maxWeight >= 0.5f) {
                    futureOutput.setLinear(batchNum, labels[1])
                } else {
                    val maxClass = scoresBlock.indexOfFirst { it == maxWeight }
                    futureOutput.setLinear(batchNum, labels[maxClass])
                }
            }
        } else {
            for (batchNum in 0 until batchSize) {
                val scoresBlock = scoresBlocks[batchNum]
                val maxWeight = maxOf(scoresBlock[0], scoresBlock[1])
                if (maxWeight > 0f) {
                    futureOutput.setLinear(batchNum, labels[1])
                } else {
                    val maxClass = scoresBlock.indexOfFirst { it == maxWeight }
                    futureOutput.setLinear(batchNum, labels[maxClass])
                }
            }
        }

        return futureOutput
    }

    //scores shape [batchSize, classCount]
    private suspend fun writeLabelsLong(scores: FloatNDArray): LongNDArray {
        val (batchSize, classCount) = scores.shape
        val labels = (labelsInfo as LabelsInfo.LongLabelsInfo).labels

        // It has another realisation in ONNX, but not in ONNXRuntime
        if (classCount == 1) return MutableLongNDArray(batchSize).also { it.fill(labels.first()) }
        if (classCount == 2) return writeLabelsLongTwoClasses(scores)


        val futureOutput = MutableLongNDArray(batchSize)
        val futureOutputPointer = futureOutput.array.pointer()

        val scoresBlockSize = scores.array.blockSize

        for (batchNum in 0 until batchSize) {
            val scoresBatch = scores.view(batchNum)
            val scoresBlocks = scoresBatch.array.blocks

            var maxClass = -1
            var maxWeight = Float.NEGATIVE_INFINITY

            for (blockNum in scoresBlocks.indices) {
                val indexOffset = blockNum * scoresBlockSize
                val scoresBlock = scoresBlocks[blockNum]

                for (j in scoresBlock.indices) {
                    if (scoresBlock[j] > maxWeight) {
                        maxWeight = scoresBlock[j]
                        maxClass = indexOffset + j
                    }
                }
            }

            futureOutputPointer.setAndIncrement(labels[maxClass])
        }

        return futureOutput
    }

    //scores shape [batchSize, classCount]
    private fun writeLabelsString(scores: FloatNDArray): StringNDArray {
        val labels = (labelsInfo as LabelsInfo.StringLabelsInfo).labels

        val (batchSize, classCount) = scores.shape
        if (classCount == 1) return MutableStringNDArray(batchSize).also { it.fill(labels.first()) }
        if (classCount == 2) return writeLabelsStringTwoClasses(scores)


        val futureOutput = MutableStringNDArray(batchSize)

        val scoresBlockSize = scores.array.blockSize

        for (batchNum in 0 until batchSize) {
            val scoresBatch = scores.view(batchNum)
            val scoresBlocks = scoresBatch.array.blocks

            var maxClass = -1
            var maxWeight = Float.NEGATIVE_INFINITY

            for (blockNum in scoresBlocks.indices) {
                val indexOffset = blockNum * scoresBlockSize
                val scoresBlock = scoresBlocks[blockNum]

                for (j in scoresBlock.indices) {
                    if (scoresBlock[j] > maxWeight) {
                        maxWeight = scoresBlock[j]
                        maxClass = indexOffset + j
                    }
                }
            }

            futureOutput.setLinear(batchNum, labels[maxClass])
        }

        return futureOutput
    }

    private suspend fun writeLabels(scores: FloatNDArray): NDArrayCore {
        return when (labelsInfo) {
            is LabelsInfo.LongLabelsInfo -> writeLabelsLong(scores)
            is LabelsInfo.StringLabelsInfo -> writeLabelsString(scores)
        }
    }

    // input shape: [batchSize, featuresCount]
    override suspend fun run(input: FloatNDArray): LabelsAndScores {
        //scores shape: [batchSize, clasCount]
        val scores = calculateScores(input)
        val labels = writeLabels(scores)

        val finalScores = updateScoresInplace(scores)

        return LabelsAndScores(labels, finalScores)
    }

}
