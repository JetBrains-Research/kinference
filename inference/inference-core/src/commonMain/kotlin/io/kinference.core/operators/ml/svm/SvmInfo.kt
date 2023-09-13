package io.kinference.core.operators.ml.svm

import io.kinference.core.operators.ml.utils.PostTransform
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.extensions.all.all
import io.kinference.trees.*
import io.kinference.utils.toIntArray

internal data class SvmInfo(
    val svmMode: SvmMode,
    // coefficients: if linear [classCount, featureCount]
    //               else      [classCount - 1, vectorCount]
    val coefficients: FloatNDArray,
    val kernelType: KernelType,
    val postTransformType: PostTransformType,
    val probA: FloatArray,
    val probB: FloatArray,
    val rho: FloatArray,
    val supportVectors: FloatNDArray, // [vectorCount, featuresCount]
    val vectorsPerClass: IntArray,
    val classCount: Int,
    val gamma: Float = 0f,
    val coefZero: Float = 0f,
    val degree: Float = 0f,
    val startingVector: IntArray,
) {

    val weightsAreAllPositive = coefficients.all { it >= 0f }

    val haveProba = probA.isNotEmpty()

    val numClassifier = classCount * (classCount - 1) / 2

    val postTransform = PostTransform[postTransformType]

    val writeAdditionalScores = if (svmMode == SvmMode.SVC && !haveProba) {
        if (postTransformType == PostTransformType.NONE)
            WriteAdditionalScores.WITHOUT_POST_TRANSFORM
        else
            WriteAdditionalScores.WITH_POST_TRANSFORM
    } else null

    init {
        if (writeAdditionalScores != null && classCount == 2 && postTransformType == PostTransformType.PROBIT) error("post_transform PROBIT isn't supported for binary case")
    }

    companion object {
        private const val DEFAULT_POST_TRANSFORM = "NONE"
        private const val DEFAULT_KERNEL_TYPE = "LINEAR"

        operator fun invoke(
            coefficients: FloatArray,
            kernelParams: FloatArray?,
            kernelTypeStr: String?,
            postTransformStr: String?,
            probA: FloatArray,
            probB: FloatArray,
            rho: FloatArray,
            supportVectors: FloatArray,
            vectorsPerClass: LongArray,
            classCount: Int = 1
        ) : SvmInfo {
            require(classCount > 0) { "Attribute either 'classlabels_ints' or 'classlabels_strings' must have at least one element" }
            val postTransform = PostTransformType.valueOf(postTransformStr ?: DEFAULT_POST_TRANSFORM)
            val kernelType = KernelType.valueOf(kernelTypeStr ?: DEFAULT_KERNEL_TYPE)

            require(probA.size == probB.size) { "Attributes prob_a and prob_b must have the same size" }

            val gamma = kernelParams?.getOrNull(0) ?: 0f
            val coefZero = kernelParams?.getOrNull(1) ?: 0f
            val degree = kernelParams?.getOrNull(2) ?: 0f

            var vectorCount = 0
            val startingVector = IntArray(vectorsPerClass.size)

            for ((idx, vectorsSize) in vectorsPerClass.withIndex()) {
                startingVector[idx] = vectorCount
                vectorCount += vectorsSize.toInt()
            }

            return if (vectorCount > 0) {
                require(classCount > 1) { "Attribute either 'classlabels_ints' or 'classlabels_strings' must have at least two elements for SVC" }

                val (supportVectorsTensor, coefficientsTensor) = configureSvcInfo(supportVectors, coefficients, vectorCount, classCount, kernelType)
                SvmInfo(
                    SvmMode.SVC,
                    coefficientsTensor,
                    kernelType,
                    postTransform,
                    probA,
                    probB,
                    rho,
                    supportVectorsTensor,
                    vectorsPerClass.toIntArray(),
                    classCount,
                    gamma,
                    coefZero,
                    degree,
                    startingVector
                )
            } else {
                val coefficientsTensor = configureLinearInfo(coefficients, classCount, kernelType)
                SvmInfo(
                    SvmMode.LINEAR,
                    coefficientsTensor,
                    KernelType.LINEAR,
                    postTransform,
                    probA,
                    probB,
                    rho,
                    FloatNDArray(),
                    vectorsPerClass.toIntArray(),
                    classCount,
                    gamma,
                    coefZero,
                    degree,
                    IntArray(0)
                )
            }
        }

        private data class SupportVectorsAndCoefficients(
            val supportVectors: FloatNDArray,
            val coefficients: FloatNDArray
        )

        private fun configureSvcInfo(supportVectors: FloatArray, coefficients: FloatArray, vectorCount: Int, numClasses: Int, kernelType: KernelType): SupportVectorsAndCoefficients {
            val featuresCount = supportVectors.size / vectorCount
            val supportVectorsTensor = if (kernelType == KernelType.RBF) {
                // Linear read with RBF
                FloatNDArray(vectorCount, featuresCount) { idx: Int -> supportVectors[idx] }
            } else {
                // Transpose read
                FloatNDArray(featuresCount, vectorCount) { (i, j): IntArray -> supportVectors[j * featuresCount + i] }
            }

            val coefficientsTensor = FloatNDArray(numClasses - 1, vectorCount) { idx: Int -> coefficients[idx] }

            return SupportVectorsAndCoefficients(supportVectorsTensor, coefficientsTensor)
        }

        private fun configureLinearInfo(coefficients: FloatArray, numClasses: Int, kernelType: KernelType): FloatNDArray {
            val featuresCount = coefficients.size / numClasses

            return if (kernelType == KernelType.RBF) {
                // Linear read with RBF
                FloatNDArray(numClasses, featuresCount) { idx: Int -> coefficients[idx] }
            } else {
                // Transpose read
                FloatNDArray(featuresCount, numClasses) { (i, j): IntArray -> coefficients[j * featuresCount + i] }
            }
        }
    }
}
