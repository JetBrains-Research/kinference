package io.kinference.operators.ml.trees

import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import io.kinference.operators.activations.Softmax

//TODO: SOFTMAX, SOFTMAX_ZERO, LOGISTIC, PROBIT
sealed class PostTransform {
    abstract fun apply(array: MutableFloatNDArray): FloatNDArray

    object None : PostTransform() {
        override fun apply(array: MutableFloatNDArray) = array
    }

    object SoftmaxTransform : PostTransform() {
        override fun apply(array: MutableFloatNDArray): FloatNDArray {
            return Softmax.softmax(array, axis = -1) as FloatNDArray
        }
    }

    companion object {
        operator fun get(name: String) = when(name) {
            "NONE" -> None
            "SOFTMAX" -> SoftmaxTransform
            else -> error("Unsupported post-transformation: $name")
        }
    }
}
