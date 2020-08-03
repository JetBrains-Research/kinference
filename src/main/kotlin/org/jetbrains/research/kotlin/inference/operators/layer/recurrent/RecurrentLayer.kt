package org.jetbrains.research.kotlin.inference.operators.layer.recurrent

import org.jetbrains.research.kotlin.inference.data.tensors.Tensor

abstract class RecurrentLayer(val hiddenSize: Int, val activations: List<String>, val direction: String) {
    abstract fun apply(inputList: List<Tensor?>): List<Tensor?>
}
