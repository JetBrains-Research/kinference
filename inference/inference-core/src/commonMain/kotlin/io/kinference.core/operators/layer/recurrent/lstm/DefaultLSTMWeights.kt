package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.NumberNDArray

class DefaultLSTMWeights(data: NumberNDArray): AbstractLSTMWeights(data) {
    override fun view(dim: Int): DefaultLSTMWeights = DefaultLSTMWeights(data.view(dim))
}
