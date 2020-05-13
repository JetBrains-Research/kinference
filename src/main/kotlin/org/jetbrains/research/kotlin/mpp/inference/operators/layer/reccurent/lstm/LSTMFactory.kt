package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.RecurrentLayer
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class LSTMFactory<T : Number>(val attributes: Map<String, Attribute<*>>) : RecurrentLayer<T>() {
    override fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>> {
        return when(attributes["direction"]?.value){
            "forward" -> LSTMLayer<T>().apply(inputs)
            "bidirectional" -> BiLSTMLayer<T>().apply(inputs)
            else -> LSTMLayer<T>().apply(inputs)
        }
    }
}

