package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.NumberNDArrayCore

class DefaultLSTMWeights(data: NumberNDArrayCore): AbstractLSTMWeights(data) {
    override fun view(dim: Int): DefaultLSTMWeights = DefaultLSTMWeights(data.view(dim))
}
