package org.jetbrains.research.kotlin.mpp.inference.operators

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXData
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.*
import org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm.LSTM
import org.jetbrains.research.kotlin.mpp.inference.operators.math.Add
import org.jetbrains.research.kotlin.mpp.inference.operators.math.MatMul
import org.jetbrains.research.kotlin.mpp.inference.operators.seq.ConcatFromSequence
import org.jetbrains.research.kotlin.mpp.inference.operators.seq.SplitToSequence
import org.jetbrains.research.kotlin.mpp.inference.operators.tensor.*
import org.jetbrains.research.kotlin.mpp.inference.types.TensorInfo
import org.jetbrains.research.kotlin.mpp.inference.types.ValueInfo

object OperatorFactory {
    fun create(name: String?, attributes: Map<String, Attribute<Any>>) = when (name) {
        "Add" -> Add(attributes)
        "MatMul" -> MatMul(attributes)
        "Identity" -> Identity(attributes)
        "Relu" -> Relu(attributes)
        "Sigmoid" -> Sigmoid(attributes)
        "Tanh" -> Tanh(attributes)
        "Softmax" -> Softmax(attributes)
        "LSTM" -> LSTM(attributes)
        "Transpose" -> Transpose(attributes)
        "Reshape" -> Reshape(attributes)
        "Split" -> Split(attributes)
        "Concat" -> Concat(attributes)
        "Squeeze" -> Squeeze(attributes)
        "SplitToSequence" -> SplitToSequence(attributes)
        "ConcatFromSequence" -> ConcatFromSequence(attributes)
        else -> error("Unsupported operator")
    } as Operator<ONNXData, ONNXData>
}
