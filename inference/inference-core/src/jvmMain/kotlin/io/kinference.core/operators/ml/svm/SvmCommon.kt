package io.kinference.core.operators.ml.svm

import io.kinference.core.operators.ml.utils.LabelsInfo
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import kotlin.jvm.JvmStatic
import kotlin.math.*

internal abstract class SvmCommon(protected val svmInfo: SvmInfo) {
    data class LabelsAndScores(
        val labels: NDArrayCore,
        val scores: FloatNDArray
    )

    abstract suspend fun run(input: FloatNDArray): LabelsAndScores

    protected suspend fun batchedKernelDot(input: FloatNDArray): MutableFloatNDArray {
        if (svmInfo.kernelType == KernelType.RBF) return batchedKernelDotRbf(input)

        // If it SVC support supportVectors shape: [featuresCount, vectorCount]
        // If it LINEAR coefficients shape: [featureCount, classCount]
        val rightTensor = when (svmInfo.svmMode) {
            SvmMode.SVC -> svmInfo.supportVectors
            SvmMode.LINEAR -> svmInfo.coefficients
        }

        val futureOutput = input.dot(rightTensor)

        if (svmInfo.kernelType != KernelType.LINEAR) {
            if (svmInfo.gamma != 1f) {
                futureOutput.timesAssign(FloatNDArray.scalar(svmInfo.gamma))
            }

            if (svmInfo.coefZero != 0f) {
                futureOutput.plusAssign(FloatNDArray.scalar(svmInfo.coefZero))
            }
        }

        if (svmInfo.kernelType == KernelType.POLY) {
            if (svmInfo.degree != 1f) {
                for (block in futureOutput.array.blocks) {
                    for (idx in block.indices) {

                        block[idx] = block[idx].pow(svmInfo.degree)
                    }
                }
            }
        } else if (svmInfo.kernelType == KernelType.SIGMOID) {
            for (block in futureOutput.array.blocks) {
                for (idx in block.indices) {
                    // TODO: Add inplace tanh for MutableArrays
                    block[idx] = tanh(block[idx])
                }
            }
        }

        return futureOutput
    }

    // In this case input has shape [batchSize, featuresCount],
    // supportVectors has shape [vectorsCount, featuresCount]
    private suspend fun batchedKernelDotRbf(input: FloatNDArray): MutableFloatNDArray {
        require(svmInfo.svmMode == SvmMode.SVC) { "KernelType RBF Supported only for SVC" }

        val (batchSize, featuresCount) = input.shape
        val vectorsCount = svmInfo.supportVectors.shape[0]

        val output = MutableFloatNDArray(batchSize, vectorsCount)
        val outputPointer = output.array.pointer()

        for (batchNum in 0 until batchSize) {
            val inputBatch = input.view(batchNum)

            for (vectorNum in 0 until vectorsCount) {
                val supportVector = svmInfo.supportVectors.view(vectorNum)

                // TODO: Optimize without new array
                val diff = inputBatch - supportVector

                var sum = 0f

                for (diffBlock in diff.array.blocks) {
                    for (idx in diffBlock.indices) {
                        val value = diffBlock[idx]
                        sum += value * value
                    }
                }

                outputPointer.setAndIncrement(FastMath.exp(-svmInfo.gamma * sum))
            }
        }

        return output
    }

    private fun addAdditionalScore(scores: FloatNDArray, updateScoreFunction: (Float) -> Pair<Float, Float>): FloatNDArray {
        val batchSize = scores.shape[0]
        val blocks = scores.array.blocks

        for (blockIdx in 0 until batchSize) {
            val block = blocks[blockIdx]

            val (leftScore, rightScore) = updateScoreFunction(block[0])
            block[0] = leftScore
            block[1] = rightScore
        }

        return scores
    }

    protected suspend fun updateScoresInplace(scores: FloatNDArray): FloatNDArray {
        if (svmInfo.writeAdditionalScores == null) return svmInfo.postTransform.apply(scores as MutableFloatNDArray)

        return when (svmInfo.writeAdditionalScores) {
            WriteAdditionalScores.WITH_POST_TRANSFORM -> addAdditionalScore(scores) { 1f - it to it }
            WriteAdditionalScores.WITHOUT_POST_TRANSFORM -> addAdditionalScore(scores) { -it to it }
        }
    }


    companion object {
        fun fromInfo(info: SvmInfo, labelsInfo: LabelsInfo<*>): SvmCommon {
            return when (info.svmMode) {
                SvmMode.SVC -> SvmSVC(info, labelsInfo)
                SvmMode.LINEAR -> SvmLinear(info, labelsInfo)
            }
        }
    }
}
