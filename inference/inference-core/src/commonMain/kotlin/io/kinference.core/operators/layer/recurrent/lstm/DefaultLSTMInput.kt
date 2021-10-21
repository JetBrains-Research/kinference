package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.MutableNumberNDArray
import io.kinference.ndarray.arrays.NumberNDArray

class DefaultLSTMInput(data: NumberNDArray): AbstractLSTMInput(data) {
    override fun view(vararg axes: Int): DefaultLSTMInput = DefaultLSTMInput(data.view(*axes))

    override fun dot(weights: AbstractLSTMWeights, destination: MutableNumberNDArray) {
        when (weights) {
            is DefaultLSTMWeights -> data.dot(weights.data, destination)
            else -> error("Unsupported operation")
        }
    }

    override fun recreate(data: NumberNDArray): DefaultLSTMInput = DefaultLSTMInput(data)
}
