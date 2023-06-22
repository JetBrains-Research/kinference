package io.kinference.tfjs.operators.ml.trees

import io.kinference.ndarray.arrays.*

//TODO: SOFTMAX_ZERO, LOGISTIC, PROBIT
enum class PostTransformType {
    NONE,
    SOFTMAX,
    SOFTMAX_ZERO,
    LOGISTIC,
    PROBIT
}

sealed class PostTransform {
    abstract suspend fun apply(array: MutableNumberNDArrayTFJS): NumberNDArrayTFJS

    object None : PostTransform() {
        override suspend fun apply(array: MutableNumberNDArrayTFJS) = array
    }

    object SoftmaxTransform : PostTransform() {
        override suspend fun apply(array: MutableNumberNDArrayTFJS): NumberNDArrayTFJS {
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
