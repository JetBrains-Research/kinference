package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.model.ExecutionContext
import io.kinference.ndarray.arrays.*

abstract class AbstractLSTMInput(val data: NumberNDArrayCore) {
    abstract fun view(vararg dims: Int): AbstractLSTMInput
    abstract fun dot(weights: AbstractLSTMWeights, destination: MutableNumberNDArrayCore, executionContext: ExecutionContext? = null)
    abstract fun recreate(data: NumberNDArrayCore): AbstractLSTMInput
}
