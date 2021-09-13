package io.kinference.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.MutableNumberNDArray
import io.kinference.ndarray.arrays.NumberNDArray

abstract class AbstractLSTMInput(val data: NumberNDArray) {
    abstract fun view(vararg dims: Int): AbstractLSTMInput
    abstract fun dot(weights: AbstractLSTMWeights, destination: MutableNumberNDArray)
    abstract fun recreate(data: NumberNDArray): AbstractLSTMInput
}
