package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.graph.asCoroutineContext
import io.kinference.model.ExecutionContext
import io.kinference.ndarray.arrays.MutableNumberNDArrayCore
import io.kinference.ndarray.arrays.NumberNDArrayCore

class DefaultLSTMInput(data: NumberNDArrayCore): AbstractLSTMInput(data) {
    override fun view(vararg dims: Int): DefaultLSTMInput = DefaultLSTMInput(data.view(*dims))

    override fun dot(weights: AbstractLSTMWeights, destination: MutableNumberNDArrayCore, executionContext: ExecutionContext?) {
        when (weights) {
            is DefaultLSTMWeights -> data.dot(weights.data, destination, executionContext.asCoroutineContext())
            else -> error("Unsupported operation")
        }
    }

    override fun recreate(data: NumberNDArrayCore): DefaultLSTMInput = DefaultLSTMInput(data)
}
