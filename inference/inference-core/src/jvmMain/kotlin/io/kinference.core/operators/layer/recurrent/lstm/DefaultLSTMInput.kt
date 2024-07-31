package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.MutableNumberNDArrayCore
import io.kinference.ndarray.arrays.NumberNDArrayCore

class DefaultLSTMInput(data: NumberNDArrayCore): AbstractLSTMInput(data) {
    override fun view(vararg dims: Int): DefaultLSTMInput = DefaultLSTMInput(data.view(*dims))

    override suspend fun dot(weights: AbstractLSTMWeights, destination: MutableNumberNDArrayCore) {
        when (weights) {
            is DefaultLSTMWeights -> data.dot(weights.data, destination)
            else -> error("Unsupported operation")
        }
    }

    override suspend fun recreate(data: NumberNDArrayCore): DefaultLSTMInput = DefaultLSTMInput(data)
}
