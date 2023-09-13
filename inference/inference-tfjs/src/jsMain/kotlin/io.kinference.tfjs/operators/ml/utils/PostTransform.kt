package io.kinference.tfjs.operators.ml.utils

import io.kinference.ndarray.arrays.MutableNumberNDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.trees.PostTransformType

sealed class PostTransform {
    abstract suspend fun apply(array: MutableNumberNDArray): NumberNDArray

    internal object None : PostTransform() {
        override suspend fun apply(array: MutableNumberNDArray): NumberNDArray = array
    }

    internal object SoftmaxTransform : PostTransform() {
        override suspend fun apply(array: MutableNumberNDArray): NumberNDArray {
            return array.softmax(axis = -1)
        }
    }

    companion object {
        internal operator fun get(name: PostTransformType): PostTransform {
            return when (name) {
                PostTransformType.NONE -> None
                PostTransformType.SOFTMAX -> SoftmaxTransform
                else -> error("Unsupported post-transformation: $name")
            }
        }
    }
}

