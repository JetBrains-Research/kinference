package io.kinference.operators.layer.recurrent

import io.kinference.data.tensors.Tensor

abstract class RecurrentLayer(val hiddenSize: Int, val activations: List<String>, val direction: String) {
    abstract fun apply(inputList: List<Tensor?>): List<Tensor?>
}
