package io.kinference.core.operators.ml.trees

import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import kotlin.time.ExperimentalTime

//TODO: SOFTMAX_ZERO, LOGISTIC, PROBIT
@ExperimentalTime
sealed class PostTransform {
    abstract suspend fun apply(array: MutableFloatNDArray): FloatNDArray

    object None : PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray) = array
    }

    object SoftmaxTransform : PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray): FloatNDArray {
            // TODO: coroutines in softmax can't be used without context here
            return array.softmax(axis = -1)
        }
    }

    companion object {
        operator fun get(name: String) = when (name) {
            "NONE" -> None
            "SOFTMAX" -> SoftmaxTransform
            else -> error("Unsupported post-transformation: $name")
        }
    }
}
