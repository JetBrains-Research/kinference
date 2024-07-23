package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.MutableNumberNDArrayCore
import io.kinference.ndarray.arrays.NumberNDArrayCore

abstract class AbstractLSTMInput(val data: NumberNDArrayCore) {
    abstract fun view(vararg dims: Int): AbstractLSTMInput
    abstract suspend fun dot(weights: AbstractLSTMWeights, destination: MutableNumberNDArrayCore)
    abstract suspend fun recreate(data: NumberNDArrayCore): AbstractLSTMInput
}
