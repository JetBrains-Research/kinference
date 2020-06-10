package org.jetbrains.research.kotlin.mpp.inference.operators

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.*
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm.LSTM
import org.jetbrains.research.kotlin.mpp.inference.operators.math.Add
import org.jetbrains.research.kotlin.mpp.inference.operators.math.MatMul

object OperatorFactory {
    fun create(name: String?, attributes: Map<String, Attribute<Any>>): Operator = when (name) {
        "Add" -> Add(attributes)
        "MatMul" -> MatMul(attributes)
        "Identity" -> Identity(attributes)
        "Relu" -> Relu(attributes)
        "Sigmoid" -> Sigmoid(attributes)
        "Tanh" -> Tanh(attributes)
        "Softmax" -> Softmax(attributes)
        "LSTM" -> LSTM(attributes)
        "Transpose" -> Transpose(attributes)
        else -> error("Unsupported operator")
    }
}
