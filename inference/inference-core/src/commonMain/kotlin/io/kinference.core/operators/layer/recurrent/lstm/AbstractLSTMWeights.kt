package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.NumberNDArray


abstract class AbstractLSTMWeights(val data: NumberNDArray) {
    abstract fun view(dim: Int): AbstractLSTMWeights
}
