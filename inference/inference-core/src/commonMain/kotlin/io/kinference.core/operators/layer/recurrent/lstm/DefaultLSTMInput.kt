package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.graph.asCoroutineContext
import io.kinference.model.ExecutionContext
import io.kinference.ndarray.arrays.MutableNumberNDArray
import io.kinference.ndarray.arrays.NumberNDArray
import io.kinference.ndarray.extensions.view

class DefaultLSTMInput(data: NumberNDArray): AbstractLSTMInput(data) {
    override fun view(vararg axes: Int): DefaultLSTMInput = DefaultLSTMInput(data.view(*axes))

    override fun dot(weights: AbstractLSTMWeights, destination: MutableNumberNDArray, executionContext: ExecutionContext?) {
        when (weights) {
            is DefaultLSTMWeights -> data.dot(weights.data, destination, executionContext.asCoroutineContext())
            else -> error("Unsupported operation")
        }
    }

    override fun recreate(data: NumberNDArray): DefaultLSTMInput = DefaultLSTMInput(data)
}
