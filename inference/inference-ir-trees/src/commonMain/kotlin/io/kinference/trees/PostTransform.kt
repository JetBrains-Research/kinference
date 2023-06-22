package io.kinference.trees

import io.kinference.ndarray.arrays.MutableNumberNDArray
import io.kinference.ndarray.arrays.NumberNDArray

//TODO: SOFTMAX_ZERO, LOGISTIC, PROBIT
enum class PostTransformType {
    NONE,
    SOFTMAX,
    SOFTMAX_ZERO,
    LOGISTIC,
    PROBIT
}

sealed class PostTransform {
    abstract suspend fun apply(array: MutableNumberNDArray): NumberNDArray

    object None : PostTransform() {
        override suspend fun apply(array: MutableNumberNDArray) = array
    }

    object SoftmaxTransform : PostTransform() {
        override suspend fun apply(array: MutableNumberNDArray): NumberNDArray {
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
