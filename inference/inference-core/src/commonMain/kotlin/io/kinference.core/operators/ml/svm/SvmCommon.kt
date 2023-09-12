package io.kinference.core.operators.ml.svm

import io.kinference.core.operators.ml.utils.LabelsInfo
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import kotlin.jvm.JvmStatic
import kotlin.math.*

abstract class SvmCommon(protected val svmInfo: SvmInfo) {
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

    protected suspend fun updateScoresInplace(scores: FloatNDArray): FloatNDArray {
        if (svmInfo.writeAdditionalScores == null) return svmInfo.postTransform.apply(scores as MutableFloatNDArray)

        // Add additional score
        val batchSize = scores.shape[0]

        val scoresPointer = scores.array.pointer()
        if (svmInfo.writeAdditionalScores == WriteAdditionalScores.WITH_POST_TRANSFORM) {
            repeat(batchSize) {
                val score = scoresPointer.get()
                scoresPointer.setAndIncrement(1f - score)
                scoresPointer.setAndIncrement(score)
            }

            return scores
        }

        if (svmInfo.writeAdditionalScores == WriteAdditionalScores.WITHOUT_POST_TRANSFORM) {
            repeat(batchSize) {
                val score = scoresPointer.get()
                scoresPointer.setAndIncrement(-score)
                scoresPointer.setAndIncrement(score)
            }

            return scores
        }

        return scores
    }


    companion object {
        fun fromInfo(info: SvmInfo, labelsInfo: LabelsInfo<*>): SvmCommon {
            return when (info.svmMode) {
                SvmMode.SVC -> SvmSVC(info, labelsInfo)
                SvmMode.LINEAR -> SvmLinear(info, labelsInfo)
            }
        }

        @JvmStatic
        protected fun computeLogistic(value: Float): Float {
            val result = 1f / (1f + FastMath.exp(-abs(value)))
            return if (value < 0) 1f - result else result
        }
    }
}
