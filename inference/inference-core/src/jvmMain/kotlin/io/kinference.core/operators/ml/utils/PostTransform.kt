package io.kinference.core.operators.ml.utils

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.logistic.logistic
import io.kinference.ndarray.extensions.probit.probit
import io.kinference.ndarray.extensions.softmax.softmax
import io.kinference.ndarray.extensions.softmax.softmaxZero
import io.kinference.trees.PostTransformType

sealed class PostTransform {
    abstract suspend fun apply(array: MutableFloatNDArray): FloatNDArray

    internal object None : PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray): FloatNDArray = array
    }

    internal object SoftmaxTransform : PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray): FloatNDArray {
            return softmax(array, array, axis = -1) as FloatNDArray
        }
    }

    internal object SoftmaxZeroTransform: PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray): FloatNDArray {
            return softmaxZero(array, array, axis = -1) as FloatNDArray
        }
    }

    internal object LogisticTransform: PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray): FloatNDArray {
            return logistic(array, array) as FloatNDArray
        }
    }

    internal object ProbitTransform: PostTransform() {
        override suspend fun apply(array: MutableFloatNDArray): FloatNDArray {
            return probit(array, array) as FloatNDArray
        }
    }

    companion object {
        internal operator fun get(name: PostTransformType): PostTransform {
            return when (name) {
                PostTransformType.NONE -> None
                PostTransformType.SOFTMAX -> SoftmaxTransform
                PostTransformType.SOFTMAX_ZERO -> SoftmaxZeroTransform
                PostTransformType.LOGISTIC -> LogisticTransform
                PostTransformType.PROBIT -> ProbitTransform
            }
        }
    }
}
