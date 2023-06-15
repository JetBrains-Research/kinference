package io.kinference.core.operators.ml.trees

import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray

//TODO: SOFTMAX_ZERO, LOGISTIC, PROBIT
enum class PostTransformType {
    NONE,
    SOFTMAX,
    SOFTMAX_ZERO,
    LOGISTIC,
    PROBIT
}

sealed class PostTransform {
    abstract suspend fun apply(array: MutableFloatNDArray): FloatNDArray

    object None : PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray) = array
    }

    object SoftmaxTransform : PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray): FloatNDArray {
            return array.softmax(axis = -1)
        }
    }

    companion object {
        operator fun get(name: PostTransformType) = when (name) {
            PostTransformType.NONE -> None
            PostTransformType.SOFTMAX -> SoftmaxTransform
            else -> error("Unsupported post-transformation: $name")
        }
    }
}
