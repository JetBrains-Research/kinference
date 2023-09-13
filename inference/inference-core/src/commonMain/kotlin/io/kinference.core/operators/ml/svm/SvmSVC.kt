package io.kinference.core.operators.ml.svm

import io.kinference.core.operators.ml.utils.LabelsInfo
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.forEachWith
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import kotlin.math.abs
import kotlin.math.pow

internal class SvmSVC(info: SvmInfo, val labelsInfo: LabelsInfo<*>): SvmCommon(info) {
    init {
        require(info.svmMode == SvmMode.SVC) { "Incorrect svmMode in info" }
    }

    private data class ScoresAndVotes(
        val scores: FloatNDArray,
        val votes: Array<IntArray>
    )

    // kernel shape is [batchSize, vectorCount]
    private fun calculateScoresAndVotes(kernel: FloatNDArray): ScoresAndVotes {
        val (batchSize, vectorsCount) = kernel.shape

        val scoresPerBatch = if (!svmInfo.haveProba && svmInfo.classCount <= 2) 2 else svmInfo.numClassifier

        val votes = Array(batchSize) { IntArray(svmInfo.classCount) }
        val scores = MutableFloatNDArray(batchSize, scoresPerBatch)

        for (batchNum in 0 until batchSize) {
            var classifierIdx = 0


            val kernelBatch = kernel.view(batchNum)
            val scoresBatch = scores.viewMutable(batchNum)
            val votesBatch = votes[batchNum]

            val scoresBatchPointer = scoresBatch.array.pointer()

            for (i in 0 until svmInfo.classCount - 1) {
                val startIndexI = svmInfo.startingVector[i]
                val countValuesInI = svmInfo.vectorsPerClass[i]
                val coeffsViewByI = svmInfo.coefficients.view(i)

                for (j in i + 1 until svmInfo.classCount) {
                    val startIndexJ = svmInfo.startingVector[j]
                    val countValuesInJ = svmInfo.vectorsPerClass[j]
                    val coeffsViewByJ = svmInfo.coefficients.view(j - 1)

                    var sum = svmInfo.rho[classifierIdx++]

                    val iKernelPointer = kernelBatch.array.pointer(startIndexI)
                    val iCoeffsPointer = coeffsViewByJ.array.pointer(startIndexI)

                    iKernelPointer.forEachWith(iCoeffsPointer, countValuesInI) { kernelVal, coeffVal -> sum += kernelVal * coeffVal }

                    val jKernelPointer = kernelBatch.array.pointer(startIndexJ)
                    val jCoeffsPointer = coeffsViewByI.array.pointer(startIndexJ)

                    jKernelPointer.forEachWith(jCoeffsPointer, countValuesInJ) { kernelVal, coeffVal -> sum += kernelVal * coeffVal }

                    scoresBatchPointer.setAndIncrement(sum)
                    if (sum > 0) {
                        votesBatch[i]++
                    } else {
                        votesBatch[j]++
                    }
                }
            }
        }

        return ScoresAndVotes(scores, votes)
    }

    private fun sigmoidProbability(score: Float, proba: Float, probb: Float): Float {
        val value = score * proba + probb
        return 1f - computeLogistic(value)
    }

    private fun computeLogistic(value: Float): Float {
        val result = 1f / (1f + FastMath.exp(-abs(value)))
        return if (value < 0) 1f - result else result
    }

    private suspend fun calculateProbabilities(scores: FloatNDArray): FloatNDArray {
        if (!svmInfo.haveProba) return scores

        val (batchSize, numClassifiers) = scores.shape

        val newScores = MutableFloatNDArray(batchSize, svmInfo.classCount)
        for (batchNum in 0 until batchSize) {
            val scoresBatch = scores.view(batchNum)
            val scoresBatchPointer = scoresBatch.array.pointer()

            val newScoresBatch = newScores.viewMutable(batchNum)

            val probsp2 = FloatArray(svmInfo.classCount * svmInfo.classCount)
            var index = 0

            for (i in 0 until svmInfo.classCount - 1) {
                var pointer1 = i * svmInfo.classCount + i + 1
                var pointer2 = (i + 1) * svmInfo.classCount + i
                for (j in i + 1 until svmInfo.classCount) {
                    val val1 = sigmoidProbability(scoresBatchPointer.getAndIncrement(), svmInfo.probA[index], svmInfo.probB[index])
                    val val2 = minOf(maxOf(val1, 1.0e-7f), 1f - 1.0e-7f)
                    probsp2[pointer1] = val2
                    probsp2[pointer2] = 1f - val2

                    index++
                    pointer1++
                    pointer2 += svmInfo.classCount
                }
            }

            multiclassProbability(probsp2, newScoresBatch)
        }

        return newScores
    }

    // votes shape: [batchSize, classCount]
    private fun writeLabels(votes: Array<IntArray>): NDArrayCore {
        return when (labelsInfo) {
            is LabelsInfo.LongLabelsInfo -> writeLabelsLong(votes)
            is LabelsInfo.StringLabelsInfo -> writeLabelsString(votes)
        }
    }

    private fun writeLabelsLong(votes: Array<IntArray>): LongNDArray {
        val labels = (labelsInfo as LabelsInfo.LongLabelsInfo).labels
        val batchSize = votes.size
        val futureOutput = MutableLongNDArray(batchSize)
        val futureOutputPointer = futureOutput.array.pointer()

        for (batchArray in votes) {
            var maxVote = batchArray.first()
            var maxIndex = 0

            for (idx in 1 until batchArray.size) {
                if (batchArray[idx] > maxVote) {
                    maxVote = batchArray[idx]
                    maxIndex = idx
                }
            }

            futureOutputPointer.setAndIncrement(labels[maxIndex])
        }

        return futureOutput
    }

    private fun writeLabelsString(votes: Array<IntArray>): StringNDArray {
        val labels = (labelsInfo as LabelsInfo.StringLabelsInfo).labels
        val batchSize = votes.size
        val futureOutput = MutableStringNDArray(batchSize)
        val futureOutputArray = futureOutput.array

        for ((batchNum, batchArray) in votes.withIndex()) {
            var maxVote = batchArray.first()
            var maxIndex = 0

            for (idx in 1 until batchArray.size) {
                if (batchArray[idx] > maxVote) {
                    maxVote = batchArray[idx]
                    maxIndex = idx
                }
            }

            futureOutputArray[batchNum] = labels[maxIndex]
        }

        return futureOutput
    }

    // ONNX implementation: https://github.com/onnx/onnx/blob/02a41bf1031defac78bd482328381f137ca99137/onnx/reference/ops/aionnxml/op_svm_classifier.py#L20
    // ONNXRuntime implementation: https://github.com/microsoft/onnxruntime/blob/c0a4fe777fcc1311bf1379651ca68dfde176d94d/onnxruntime/core/providers/cpu/ml/ml_common.h#L189
    // Reference article: https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf
    //r shape: [classCount, classCount]
    //p shape: [classCount]
    private suspend fun multiclassProbability(r: FloatArray, p: MutableFloatNDArray) {
        val maxIter = maxOf(100, svmInfo.classCount)

        val Q = FloatArray(svmInfo.classCount * svmInfo.classCount)
        val Qp = FloatArray(svmInfo.classCount)

        val eps = 0.005f / svmInfo.classCount

        p.fill(1f / svmInfo.classCount)

        for (i in 0 until svmInfo.classCount) {
            val iOffset = i * svmInfo.classCount
            val iDiagOffset = i * svmInfo.classCount + i

            for (j in 0 until i) {
                val jOffset = j * svmInfo.classCount

                Q[iDiagOffset] += r[jOffset + i].pow(2)// * r[jOffset + i]
                Q[iOffset + j] = Q[jOffset + i]
            }

            for (j in i + 1 until svmInfo.classCount) {
                val jOffset = j * svmInfo.classCount

                Q[iDiagOffset] += r[jOffset + i].pow(2)
                Q[iOffset + j] = -r[jOffset + i] * r[iOffset + j]
            }
        }

        for (loop in 0 until maxIter) {
            Qp.fill(0f)

            var pQp = 0f
            for (i in 0 until svmInfo.classCount) {
                val iOffset = i * svmInfo.classCount
                val pPointer = p.array.pointer()
                for (j in 0 until svmInfo.classCount) {
                    Qp[i] += Q[iOffset + j] * pPointer.getAndIncrement()
                }
                pPointer.linearIndex = i
                pQp += pPointer.get() * Qp[i]
            }

            var maxError = 0f
            for (i in 0 until svmInfo.classCount) {
                val error = abs(Qp[i] - pQp)
                if (error > maxError) {
                    maxError = error
                }
            }

            if (maxError < eps) break

            val pPointer = p.array.pointer()
            for (i in 0 until svmInfo.classCount) {
                val iDiagOffset = i * svmInfo.classCount + i
                val iOffset  = i * svmInfo.classCount

                val diff = (-Qp[i] + pQp) / Q[iDiagOffset]
                pPointer.setAndIncrement(pPointer.get() + diff)

                val onePlusDiff = 1f + diff
                pQp = (pQp + diff * (diff * Q[iDiagOffset] + 2f * Qp[i])) / onePlusDiff.pow(2)

                for (j in 0 until svmInfo.classCount) {
                    Qp[j] = (Qp[j] + diff * Q[iOffset + j]) / onePlusDiff
                }
                p.divAssign(FloatNDArray.scalar(onePlusDiff))
            }
        }
    }

    // input shape: [batch, features_count]
    override suspend fun run(input: FloatNDArray): LabelsAndScores {
        val kernel = batchedKernelDot(input)
        val (scores, votes) = calculateScoresAndVotes(kernel)

        // probabilities shape: if haveProba [batchSize, classCount],
        // else if classCount == 2: [batchSize, 2]
        // else [batchSize, numClassifier]
        val probabilities = calculateProbabilities(scores)
        val finalScores = updateScoresInplace(probabilities)

        val labels = writeLabels(votes)

        return LabelsAndScores(labels, finalScores)
    }
}
