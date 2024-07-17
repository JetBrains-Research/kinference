package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.NumberNDArrayCore


abstract class AbstractLSTMWeights(val data: NumberNDArrayCore) {
    abstract fun view(dim: Int): AbstractLSTMWeights
}
